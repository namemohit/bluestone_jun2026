"""Supabase-backed HITL store — direct Postgres (psycopg2) to the isolated `showroom` schema.

Drop-in for hitl.store.LocalStore: same method shapes, so dashboard/hitl_api.py and the loop are
unchanged — just swap the class. The dashboard BACKEND owns the DB connection; the browser only
ever talks to our FastAPI, so no keys live client-side and we don't depend on PostgREST schema
exposure. Crops are served locally for now (Storage upload is a later step for the Cloud Run host).

Config comes from configs/supabase.json or the root .env (SUPABASE_* / DB_URL). Values never echoed.
"""
from __future__ import annotations

import json
import pathlib
import threading
from contextlib import contextmanager

CONFIG = pathlib.Path("configs/supabase.json")
IST = "+05:30"
_CXLOCAL = threading.local()   # one reused Postgres connection PER THREAD (FastAPI runs sync routes in a threadpool).
                               # Avoids paying a fresh TLS+auth handshake (~0.7s) on every query — the page-load killer.


def _cfg() -> dict:
    if CONFIG.exists():
        return json.loads(CONFIG.read_text(encoding="utf-8"))
    import os
    try:
        from dotenv import load_dotenv
        load_dotenv()
    except Exception:
        pass
    url = os.environ.get("SUPABASE_URL")
    if not url and not os.environ.get("DB_URL"):
        raise RuntimeError("no Supabase config: add configs/supabase.json or SUPABASE_*/DB_URL to .env")
    return {"url": url or "", "anon_key": os.environ.get("SUPABASE_ANON_KEY", ""),
            "service_role_key": os.environ.get("SUPABASE_SERVICE_ROLE_KEY", ""),
            "db_url": os.environ.get("DB_URL", ""),
            "schema": os.environ.get("SHOWROOM_SCHEMA", "showroom"),
            "storage_bucket": os.environ.get("SHOWROOM_BUCKET", "showroom-crops")}


class SupabaseStore:
    def __init__(self, root: str = "outputs"):
        cfg = _cfg()
        self.dburl = cfg["db_url"]
        if not self.dburl:
            raise RuntimeError("DB_URL missing — SupabaseStore needs a direct Postgres URL")
        self.schema = cfg.get("schema", "showroom")
        self.root = pathlib.Path(root)

    def _connect(self):
        import psycopg2
        from psycopg2.extras import RealDictCursor
        return psycopg2.connect(self.dburl, connect_timeout=20, cursor_factory=RealDictCursor,
                                options=f"-c search_path={self.schema},public")

    @contextmanager
    def _cx(self):
        # Reuse this thread's open connection (no per-query handshake). Reconnect if it's missing, closed,
        # or points at a different DB. On ANY error, drop the connection so the next call starts clean.
        cx = getattr(_CXLOCAL, "cx", None)
        if cx is None or cx.closed or getattr(_CXLOCAL, "key", None) != self.dburl:
            if cx is not None:
                try:
                    cx.close()
                except Exception:
                    pass
            cx = self._connect()
            _CXLOCAL.cx = cx
            _CXLOCAL.key = self.dburl
        try:
            yield cx
            cx.commit()
        except Exception:
            try:
                cx.rollback()
            except Exception:
                pass
            try:
                cx.close()
            except Exception:
                pass
            _CXLOCAL.cx = None      # force a fresh connection next time (handles server-side drops/staleness)
            raise

    @staticmethod
    def _full_ts(date: str, hms: str) -> str | None:
        hms = (hms or "").strip()
        if not date or not hms:
            return None
        return f"{date} {hms}{IST}" if len(hms) <= 8 else hms  # "2026-06-03 18:09:57+05:30"

    # ---- push a processed window up (store + window + visits + events) ---
    def push_window(self, window: str, store_id: str, visits_json: dict, *,
                    upload_crops: bool = False, with_detections: bool = True) -> None:
        start = visits_json.get("window_start_ist") or ""
        date = start.split(" ")[0] if " " in start else (start[:10] or None)
        crop = (lambda p: p)  # local path for now; Storage upload wired at deploy time
        with self._cx() as cx, cx.cursor() as cur:
            cur.execute("insert into stores(id,name,clock_offset_s) values(%s,%s,20315) "
                        "on conflict(id) do nothing", (store_id, store_id))
            cur.execute(
                "insert into windows(id,store_id,date,start_ist,end_ist,status,params) "
                "values(%s,%s,%s,%s,%s,'ready',%s) on conflict(id) do update set "
                "status='ready', params=excluded.params, updated_at=now()",
                (window, store_id, date, self._full_ts(date, start.split(" ")[1] if " " in start else ""),
                 None, json.dumps(visits_json.get("params", {}))))
            # visits/events are DERIVED (regenerable) — replace this window's snapshot so removed
            # rows (e.g. a track reclassified as staff) disappear. labels are NOT touched.
            cur.execute("delete from visits where window_id=%s", (window,))
            cur.execute("delete from events where window_id=%s", (window,))
            for v in visits_json.get("visits", []):
                cur.execute(
                    "insert into visits(window_id,id,in_track,out_track,in_ist,out_ist,dwell_s,how,"
                    "confidence,uncertainty,status,in_crop_url,out_crop_url) "
                    "values(%(w)s,%(id)s,%(it)s,%(ot)s,%(ii)s,%(oi)s,%(d)s,%(how)s,%(c)s,%(u)s,%(s)s,%(ic)s,%(oc)s) "
                    "on conflict(window_id,id) do update set how=excluded.how, confidence=excluded.confidence, "
                    "uncertainty=excluded.uncertainty, status=excluded.status, updated_at=now()",
                    {"w": window, "id": v["id"], "it": v["in_track"], "ot": v["out_track"],
                     "ii": self._full_ts(date, v["in_ist"]), "oi": self._full_ts(date, v["out_ist"]),
                     "d": v["dwell_s"], "how": v["how"], "c": v.get("confidence"),
                     "u": v.get("uncertainty"), "s": v["status"],
                     "ic": crop(v["in_crop"]), "oc": crop(v["out_crop"])})
            for role, key in (("open", "open_sessions"), ("pre_exit", "pre_window_exits")):
                for e in visits_json.get(key, []):
                    cur.execute(
                        "insert into events(window_id,track,direction,ts_ist,crop_url,role) "
                        "values(%s,%s,%s,%s,%s,%s) on conflict(window_id,track,direction) "
                        "do update set role=excluded.role, crop_url=excluded.crop_url",
                        (window, e["track"], "in" if role == "open" else "out",
                         self._full_ts(date, e["ist"]), crop(e["crop"]), role))
            # detections (L1 raw, every camera) — regenerable snapshot for the detections view +
            # attendance (staff in_track -> first/last sighting). Read from the window's local L1 dirs.
            if with_detections:
                cur.execute("delete from detections where window_id=%s", (window,))
            wj = self.root / window / "window.json"
            if with_detections and wj.exists():
                wcfg = json.loads(wj.read_text(encoding="utf-8"))
                ddirs = [(wcfg.get("l1"), "C05")] + [(d, pathlib.Path(d).name.replace("L1_", ""))
                                                     for d in wcfg.get("interior", [])]
                rows = []
                for dd, cam in ddirs:
                    tj = pathlib.Path(dd) / "tracks.json" if dd else None
                    if not tj or not tj.exists():
                        continue
                    for t in json.loads(tj.read_text(encoding="utf-8")).get("tracks", []):
                        rows.append((window, cam, t["track"],
                                     self._full_ts(date, t.get("first_ist", "")),
                                     self._full_ts(date, t.get("last_ist") or t.get("first_ist", "")),
                                     round(t.get("last_ts", 0) - t.get("first_ts", 0), 1),
                                     t.get("frames"), (t.get("crop", "") or "").replace("\\", "/")))
                if rows:
                    cur.executemany(
                        "insert into detections(window_id,camera,track,first_ist,last_ist,dur_s,frames,crop_url) "
                        "values(%s,%s,%s,%s,%s,%s,%s,%s)", rows)

    # ---- reads (same shape as LocalStore) -------------------------------
    def get_visits(self, window: str) -> dict:
        with self._cx() as cx, cx.cursor() as cur:
            cur.execute(
                "select id,in_track,out_track,dwell_s::float,how,confidence::float,uncertainty::float,status,"
                "to_char(in_ist at time zone 'Asia/Kolkata','HH24:MI:SS') in_ist,"
                "to_char(out_ist at time zone 'Asia/Kolkata','HH24:MI:SS') out_ist,"
                "in_crop_url in_crop, out_crop_url out_crop from visits "
                "where window_id=%s order by uncertainty desc nulls last", (window,))
            visits = [dict(r) for r in cur.fetchall()]
            cur.execute("select track, to_char(ts_ist at time zone 'Asia/Kolkata','HH24:MI:SS') ist,"
                        " crop_url crop, role from events where window_id=%s and role in ('open','pre_exit') "
                        "order by ts_ist", (window,))
            evs = cur.fetchall()
        return {"visits": visits,
                "open_sessions": [{"track": e["track"], "ist": e["ist"], "crop": e["crop"]}
                                  for e in evs if e["role"] == "open"],
                "pre_window_exits": [{"track": e["track"], "ist": e["ist"], "crop": e["crop"]}
                                     for e in evs if e["role"] == "pre_exit"]}

    def list_windows(self) -> list[dict]:
        with self._cx() as cx, cx.cursor() as cur:
            cur.execute("select id from windows order by id")
            ids = [r["id"] for r in cur.fetchall()]
        # metrics for EVERY window in 2 batched round-trips, not 2 per window (this endpoint loads on
        # every page refresh — the per-window loop was ~2N Supabase calls = seconds of dropdown lag).
        vmap = self.get_visits_many(ids)
        lmap = self.get_labels_many(ids)
        import json as _json, pathlib as _pl, re as _re

        def _rng(wid):                                          # "HH:MM - HH:MM" from window.json label (real +Xmin, else 60)
            try:
                lbl = _json.loads((_pl.Path(self.root) / wid / "window.json").read_text(encoding="utf-8")).get("label", "")
            except Exception:
                return None
            m = _re.search(r"(\d{1,2}):(\d{2})", lbl)
            if not m:
                return None
            h, mn = int(m.group(1)), int(m.group(2))
            dm = _re.search(r"\+(\d+)\s*min", lbl)
            e = h * 60 + mn + (int(dm.group(1)) if dm else 60)
            return f"{h:02d}:{mn:02d} - {(e // 60) % 24:02d}:{e % 60:02d}"
        out = []
        for w in ids:
            visits = vmap.get(w, {}).get("visits", [])
            labels = {l["visit_id"]: l["verdict"] for l in lmap.get(w, []) if l["verdict"] != "reset"}
            reviewed = [v for v in visits if v["id"] in labels]
            confirmed = sum(1 for v in reviewed if labels[v["id"]] in ("confirm", "employee"))
            out.append({"window": w, "visits": len(visits), "reviewed": len(reviewed), "confirmed": confirmed,
                        "rejected": len(reviewed) - confirmed, "range": _rng(w),
                        "precision": round(confirmed / len(reviewed), 3) if reviewed else None,
                        "unreviewed": len(visits) - len(reviewed)})
        return out

    def get_labels(self, window: str) -> list[dict]:
        with self._cx() as cx, cx.cursor() as cur:
            cur.execute("select visit_id, verdict, reason, in_track, out_track, reviewer, employee_id "
                        "from latest_labels where window_id=%s", (window,))
            return [dict(r) for r in cur.fetchall()]

    # --- batched multi-window reads (one round-trip for the whole day, not one per hour) ---
    def get_visits_many(self, windows: list[str]) -> dict:
        out = {w: {"visits": [], "open_sessions": [], "pre_window_exits": []} for w in windows}
        if not windows:
            return out
        with self._cx() as cx, cx.cursor() as cur:
            cur.execute(
                "select window_id, id,in_track,out_track,dwell_s::float,how,confidence::float,uncertainty::float,status,"
                "to_char(in_ist at time zone 'Asia/Kolkata','HH24:MI:SS') in_ist,"
                "to_char(out_ist at time zone 'Asia/Kolkata','HH24:MI:SS') out_ist,"
                "in_crop_url in_crop, out_crop_url out_crop from visits "
                "where window_id = ANY(%s) order by uncertainty desc nulls last", (list(windows),))
            for r in cur.fetchall():
                r = dict(r)
                out.setdefault(r.pop("window_id"),
                               {"visits": [], "open_sessions": [], "pre_window_exits": []})["visits"].append(r)
            cur.execute("select window_id, track, to_char(ts_ist at time zone 'Asia/Kolkata','HH24:MI:SS') ist,"
                        " crop_url crop, role from events where window_id = ANY(%s) and role in ('open','pre_exit') "
                        "order by ts_ist", (list(windows),))
            for e in cur.fetchall():
                tgt = "open_sessions" if e["role"] == "open" else "pre_window_exits"
                out.setdefault(e["window_id"], {"visits": [], "open_sessions": [], "pre_window_exits": []})[tgt]\
                    .append({"track": e["track"], "ist": e["ist"], "crop": e["crop"]})
        return out

    def get_labels_many(self, windows: list[str]) -> dict:
        out = {w: [] for w in windows}
        if not windows:
            return out
        with self._cx() as cx, cx.cursor() as cur:
            cur.execute("select window_id, visit_id, verdict, reason, in_track, out_track, reviewer, employee_id "
                        "from latest_labels where window_id = ANY(%s)", (list(windows),))
            for r in cur.fetchall():
                r = dict(r)
                out.setdefault(r.pop("window_id"), []).append(r)
        return out

    def get_detections(self, window: str) -> list[dict]:
        with self._cx() as cx, cx.cursor() as cur:
            cur.execute("select camera, track, to_char(first_ist at time zone 'Asia/Kolkata','HH24:MI:SS') ist,"
                        " dur_s::float, frames, crop_url crop from detections where window_id=%s "
                        "order by first_ist", (window,))
            return [dict(r) for r in cur.fetchall()]

    # ---- employee roster + gallery (attendance) -------------------------
    def list_employees(self, store_id: str = "s14") -> list[dict]:
        try:
            with self._cx() as cx, cx.cursor() as cur:
                cur.execute("select id, code, name, staff_no from employees where store_id=%s order by id", (store_id,))
                return [dict(r) for r in cur.fetchall()]
        except Exception:                                          # pre-migration: no staff_no column yet -> degrade gracefully
            with self._cx() as cx, cx.cursor() as cur:
                cur.execute("select id, code, name from employees where store_id=%s order by id", (store_id,))
                return [dict(r) for r in cur.fetchall()]

    def create_employee(self, store_id: str = "s14", name=None) -> dict:
        with self._cx() as cx, cx.cursor() as cur:
            cur.execute("insert into employees(store_id, name) values(%s,%s) returning id", (store_id, name))
            eid = cur.fetchone()["id"]
            cur.execute("select coalesce(max(staff_no),0)+1 as n from employees where store_id=%s", (store_id,))
            sno = cur.fetchone()["n"]                                       # permanent staff number: max+1, never reused
            cur.execute("update employees set code=%s, staff_no=%s where id=%s", (f"S{eid}", sno, eid))
        return {"id": eid, "code": f"S{eid}", "name": name, "staff_no": sno}

    def rename_employee(self, emp_id: int, name: str) -> None:
        with self._cx() as cx, cx.cursor() as cur:
            cur.execute("update employees set name=%s where id=%s", (name, emp_id))

    def enroll_staff(self, employee_id: int, store_id: str, embedding, crop_url=None,
                     window=None, track=None) -> None:
        # embedding may be a numpy array (float32) -> .tolist() gives JSON-safe python floats
        emb_list = embedding.tolist() if hasattr(embedding, "tolist") else [float(x) for x in embedding]
        with self._cx() as cx, cx.cursor() as cur:
            cur.execute("insert into employee_gallery(employee_id,store_id,embedding,crop_url,source_window,source_track) "
                        "values(%s,%s,%s,%s,%s,%s)",
                        (employee_id, store_id, json.dumps(emb_list), crop_url, window, track))

    def get_gallery(self, store_id: str = "s14") -> list[dict]:
        with self._cx() as cx, cx.cursor() as cur:
            cur.execute("select employee_id, embedding, crop_url, source_window, source_track "
                        "from employee_gallery where store_id=%s", (store_id,))
            return [dict(r) for r in cur.fetchall()]

    def get_gallery_with_id(self, store_id: str = "s14") -> list[dict]:
        """Gallery rows WITH their primary key — needed to re-embed a row in place (UPDATE by id)
        when a new ReID model is promoted (keeps membership, swaps the vector into the new space)."""
        with self._cx() as cx, cx.cursor() as cur:
            cur.execute("select id, employee_id, embedding, crop_url, source_window, source_track "
                        "from employee_gallery where store_id=%s", (store_id,))
            return [dict(r) for r in cur.fetchall()]

    def update_gallery_embedding(self, row_id: int, embedding) -> None:
        """Replace ONE gallery row's embedding in place (no delete) — used to re-embed the staff
        gallery into a newly-promoted model's space so staff matching stays in one space."""
        emb_list = embedding.tolist() if hasattr(embedding, "tolist") else [float(x) for x in embedding]
        with self._cx() as cx, cx.cursor() as cur:
            cur.execute("update employee_gallery set embedding=%s where id=%s",
                        (json.dumps(emb_list), row_id))

    def get_gallery_meta(self, store_id: str = "s14") -> list[dict]:
        """Gallery rows WITHOUT the heavy embedding vector — for the review UI's crop/thumbnail lookups.
        The embedding (~512 floats/row) is only needed by L4 matching, not the dashboard, and transferring
        + parsing it on every page load cost ~2s."""
        with self._cx() as cx, cx.cursor() as cur:
            cur.execute("select employee_id, crop_url, source_window, source_track "
                        "from employee_gallery where store_id=%s", (store_id,))
            return [dict(r) for r in cur.fetchall()]

    def add_annotation(self, window: str, camera: str, track: int, category: str, *,
                       crop_url=None, employee_id=None, duplicate_of=None, embedding=None,
                       reviewer: str = "human") -> None:
        """Append a human ground-truth allocation for one detection (close-the-day + training)."""
        emb = None
        if embedding is not None:
            emb = json.dumps(embedding.tolist() if hasattr(embedding, "tolist") else [float(x) for x in embedding])
        with self._cx() as cx, cx.cursor() as cur:
            cur.execute("insert into annotations(window_id,camera,track,crop_url,category,employee_id,"
                        "duplicate_of,embedding,reviewer) values(%s,%s,%s,%s,%s,%s,%s,%s,%s)",
                        (window, camera, track, crop_url, category, employee_id, duplicate_of, emb, reviewer))

    def latest_annotations(self, window: str) -> list[dict]:
        with self._cx() as cx, cx.cursor() as cur:
            cur.execute("select camera, track, category, employee_id, duplicate_of, crop_url "
                        "from latest_annotations where window_id=%s", (window,))
            return [dict(r) for r in cur.fetchall()]

    def latest_annotations_bulk(self, windows: list[str]) -> dict[str, list[dict]]:
        """Many windows' latest annotations in ONE round-trip, grouped by window_id. Lets the day-wide
        C#/G#/S# numbering replace its per-window query fan-out (was ~1 RTT x 12 windows = the page-load lag)."""
        out: dict[str, list[dict]] = {w: [] for w in windows}
        if not windows:
            return out
        with self._cx() as cx, cx.cursor() as cur:
            cur.execute("select window_id, camera, track, category, employee_id, duplicate_of, crop_url "
                        "from latest_annotations where window_id = ANY(%s)", (list(windows),))
            for r in cur.fetchall():
                out.setdefault(r["window_id"], []).append(dict(r))
        return out

    def staff_matches(self, employee_id: int, store_id: str = "s14") -> dict:
        """Every sighting grouped to ONE employee across the whole day, for human confirmation:
        the enrolled crop + matches [{window, track, source('manual'|'auto'), crop, sim}] from the
        manual labels (DB) and the auto-recognised tracks (local visits.json), minus any already
        rejected (notstaff). Lets the reviewer see + reject every match, nothing hidden."""
        import pathlib
        gcrop, enrolled = {}, None
        with self._cx() as cx, cx.cursor() as cur:
            cur.execute("select crop_url, source_window, source_track from employee_gallery "
                        "where employee_id=%s", (employee_id,))
            for r in cur.fetchall():
                if r["crop_url"]:
                    enrolled = enrolled or r["crop_url"]
                    if r["source_track"] is not None:
                        gcrop[(r["source_window"], r["source_track"])] = r["crop_url"]
            cur.execute("select window_id, in_track from latest_labels where employee_id=%s "
                        "and verdict='employee' and in_track is not null", (employee_id,))
            manual = [(r["window_id"], r["in_track"]) for r in cur.fetchall()]
            cur.execute("select window_id, in_track from latest_labels where verdict='reject' "
                        "and visit_id like 'notstaff-%%' and in_track is not null")
            notstaff = {(r["window_id"], r["in_track"]) for r in cur.fetchall()}
        matches = []
        for win, tr in manual:
            if (win, tr) not in notstaff:
                matches.append({"window": win, "track": tr, "source": "manual",
                                "crop": gcrop.get((win, tr)) or enrolled, "sim": None, "weak": False})
        for vj in sorted(pathlib.Path(self.root).glob("*/visits.json")):
            win = vj.parent.name
            try:
                staff = json.loads(vj.read_text(encoding="utf-8")).get("staff", [])
            except Exception:
                continue
            for st in staff:
                if st.get("employee_id") == employee_id and (win, st.get("track")) not in notstaff:
                    matches.append({"window": win, "track": st["track"], "source": "auto",
                                    "crop": st.get("crop"), "sim": st.get("sim"),
                                    "sim2": st.get("sim2"), "weak": bool(st.get("weak"))})
        matches.sort(key=lambda m: (m.get("weak", False),         # weak suggestions last
                                    m["source"] != "manual",      # your manual marks first
                                    -(m.get("sim") or 0)))         # then strongest auto first
        return {"enrolled_crop": enrolled, "matches": matches}

    def attendance(self, store_id: str = "s14", date: str | None = None) -> list[dict]:
        """Per employee, across the day: first/last sighting, #sightings, #windows, and an hourly
        TIMELINE [{window, in, out, crop}]. Sightings = MANUAL marks (labels) + AUTO-recognised tracks
        (read local outputs/*/visits.json -> staff), each resolved to its door (C05) detection's
        first/last time + crop. Local-first so the live push pipeline is untouched."""
        import pathlib
        with self._cx() as cx, cx.cursor() as cur:
            cur.execute("select id, name, code from employees where store_id=%s order by id", (store_id,))
            emps = [dict(r) for r in cur.fetchall()]
            cur.execute("select employee_id, window_id, in_track from latest_labels "
                        "where employee_id is not null and verdict='employee' and in_track is not null")
            manual = [(r["employee_id"], r["window_id"], r["in_track"]) for r in cur.fetchall()]
        sightings = {}                                                  # (eid, window, track) -> source
        for eid, win, tr in manual:
            if not (date and not str(win).startswith(date)):
                sightings[(eid, win, tr)] = "manual"
        for vj in sorted(pathlib.Path(self.root).glob("*/visits.json")):  # auto from local L4 results
            win = vj.parent.name
            if date and not win.startswith(date):
                continue
            try:
                staff = json.loads(vj.read_text(encoding="utf-8")).get("staff", [])
            except Exception:
                continue
            for st in staff:
                if st.get("employee_id") is not None and st.get("track") is not None and not st.get("weak"):
                    sightings.setdefault((st["employee_id"], win, st["track"]), "auto")   # weak band isn't attendance
        det = {}                                                        # (window, track) -> door detection
        keys = {(w, t) for (_, w, t) in sightings}
        if keys:
            wins = tuple({w for w, t in keys})
            tracks = tuple({t for w, t in keys})
            with self._cx() as cx, cx.cursor() as cur:
                cur.execute("select window_id, track,"
                            " to_char(first_ist at time zone 'Asia/Kolkata','HH24:MI:SS') fi,"
                            " to_char(last_ist at time zone 'Asia/Kolkata','HH24:MI:SS') la, crop_url crop"
                            " from detections where camera='C05' and window_id in %s and track in %s",
                            (wins, tracks))
                for r in cur.fetchall():
                    det[(r["window_id"], r["track"])] = r
        by_emp: dict = {}                                               # eid -> window -> [detections]
        for (eid, win, tr) in sightings:
            d = det.get((win, tr))
            if d:
                by_emp.setdefault(eid, {}).setdefault(win, []).append(d)
        out = []
        for e in emps:
            wmap = by_emp.get(e["id"], {})
            timeline, alltimes = [], []
            for win in sorted(wmap):
                rows = wmap[win]
                fis = sorted(r["fi"] for r in rows if r["fi"])
                las = sorted(r["la"] for r in rows if r["la"])
                crop = next((r["crop"] for r in rows if r.get("crop")), None)
                if fis:
                    timeline.append({"window": win, "in": fis[0], "out": las[-1] if las else fis[-1],
                                     "crop": crop.replace("\\", "/") if crop else None})
                    alltimes += fis + las
            out.append({"id": e["id"], "code": e["code"], "name": e["name"],
                        "first_seen": min(alltimes) if alltimes else None,
                        "last_seen": max(alltimes) if alltimes else None,
                        "windows": len(timeline), "sightings": sum(len(v) for v in wmap.values()),
                        "timeline": timeline})
        return out

    # ---- writes ---------------------------------------------------------
    def add_label(self, window: str, visit_id: str, verdict: str, *, reason: str = "",
                  in_track=None, out_track=None, reviewer: str = "human", employee_id=None) -> dict:
        with self._cx() as cx, cx.cursor() as cur:
            cur.execute("insert into labels(window_id,visit_id,verdict,reason,in_track,out_track,reviewer,employee_id) "
                        "values(%s,%s,%s,%s,%s,%s,%s,%s)",
                        (window, visit_id, verdict, reason, in_track, out_track, reviewer, employee_id))
        return {"visit_id": visit_id, "verdict": verdict, "in_track": in_track, "out_track": out_track,
                "employee_id": employee_id}

    def feedback(self, window: str) -> dict:
        cannot, must, employees, not_staff, false = [], [], [], [], []
        for l in self.get_labels(window):
            it, ot, v = l.get("in_track"), l.get("out_track"), l["verdict"]
            vid = l.get("visit_id", "")
            if vid.startswith("notstaff-") and v == "reject" and it is not None:
                not_staff.append(it)                         # human: this track is NOT staff
            elif v == "false_detection" and it is not None:
                false.append(it)                             # not-a-person / pass-by -> drop from counts
            elif v == "reject" and it is not None and ot is not None:
                cannot.append([it, ot])
            elif v == "confirm" and it is not None and ot is not None:
                must.append([it, ot])
            elif v == "employee" and it is not None:
                employees.append(it)
        employees = [t for t in employees if t not in not_staff]   # not_staff overrides a staff mark
        return {"cannot_link": cannot, "must_link": must, "employees": employees,
                "not_staff": not_staff, "false": false}

    def write_feedback(self, window: str) -> str:
        d = self.root / window
        d.mkdir(parents=True, exist_ok=True)
        p = d / "feedback.json"
        p.write_text(json.dumps(self.feedback(window), indent=2), encoding="utf-8")
        return str(p)

    def metrics(self, window: str) -> dict:
        """Pure compute (read-only) — safe to call on every page load."""
        visits = self.get_visits(window).get("visits", [])
        labels = {l["visit_id"]: l["verdict"] for l in self.get_labels(window) if l["verdict"] != "reset"}
        reviewed = [v for v in visits if v["id"] in labels]
        confirmed = sum(1 for v in reviewed if labels[v["id"]] in ("confirm", "employee"))
        return {"visits": len(visits), "reviewed": len(reviewed), "confirmed": confirmed,
                "rejected": len(reviewed) - confirmed,
                "precision": round(confirmed / len(reviewed), 3) if reviewed else None,
                "unreviewed": len(visits) - len(reviewed)}

    def record_metrics(self, window: str) -> dict:
        """Append a point to the precision curve — called after a review action, not on reads."""
        m = self.metrics(window)
        with self._cx() as cx, cx.cursor() as cur:
            cur.execute("insert into metrics(window_id,visits,reviewed,confirmed,rejected,precision) "
                        "values(%s,%s,%s,%s,%s,%s)",
                        (window, m["visits"], m["reviewed"], m["confirmed"], m["rejected"], m["precision"]))
        return m

    def sync(self, window: str, store_id: str = "s14", with_detections: bool = True) -> None:
        """Push the freshly re-run local visits.json into the DB (source of truth for the UI).
        with_detections=False skips the heavy L1 detections re-push (use on label re-runs)."""
        vj = self.root / window / "visits.json"
        if vj.exists():
            self.push_window(window, store_id, json.loads(vj.read_text(encoding="utf-8")),
                             with_detections=with_detections)

    def upload_crops(self, window: str) -> int:
        """Upload this window's crop thumbnails to Supabase Storage so the cloud dashboard can
        show them. Called once after the GPU run; label re-runs don't change crops."""
        from hitl import storage
        vj = self.root / window / "visits.json"
        if not storage.configured() or not vj.exists():
            return 0
        data = json.loads(vj.read_text(encoding="utf-8"))
        paths = set()
        for v in data.get("visits", []):
            paths.update((v.get("in_crop"), v.get("out_crop")))
        for e in data.get("open_sessions", []) + data.get("pre_window_exits", []):
            paths.add(e.get("crop"))
        return sum(1 for p in paths if p and storage.upload_crop(p))

    # ===== training: model registry + gallery-rebuild source =====================
    def confirmed_staff(self, store_id: str = "s14", date: str | None = None) -> list[dict]:
        """Every HUMAN-confirmed staff sighting, for rebuilding the gallery: from annotations
        (category='staff') + labels (verdict='employee'). Each -> {employee_id, window, camera,
        track, crop_url, embedding(maybe None)}; caller fills missing embeddings from the cache."""
        like = f"{date}_%" if date else "%"
        out = []
        with self._cx() as cx, cx.cursor() as cur:
            cur.execute("select window_id, camera, track, crop_url, employee_id, embedding "
                        "from latest_annotations where category='staff' and employee_id is not null "
                        "and window_id like %s", (like,))
            for r in cur.fetchall():
                out.append({"employee_id": r["employee_id"], "window": r["window_id"], "camera": r["camera"],
                            "track": r["track"], "crop_url": r["crop_url"], "embedding": r["embedding"]})
            cur.execute("select window_id, in_track, employee_id from latest_labels "
                        "where verdict='employee' and employee_id is not null and in_track is not null "
                        "and window_id like %s", (like,))
            for r in cur.fetchall():
                out.append({"employee_id": r["employee_id"], "window": r["window_id"], "camera": "C05",
                            "track": r["in_track"], "crop_url": None, "embedding": None})
        return out

    def gallery_sources(self, store_id: str = "s14") -> set:
        with self._cx() as cx, cx.cursor() as cur:
            cur.execute("select source_window, source_track from employee_gallery where store_id=%s", (store_id,))
            return {(r["source_window"], r["source_track"]) for r in cur.fetchall()}

    def register_model_version(self, kind: str, params: dict, *, score=None, trained_on: int = 0,
                               notes: str = "", active: bool = True) -> int:
        with self._cx() as cx, cx.cursor() as cur:
            if active:
                cur.execute("update model_versions set active=false where active=true")
            cur.execute("insert into model_versions(kind,params,score,trained_on,active,notes) "
                        "values(%s,%s,%s,%s,%s,%s) returning version",
                        (kind, json.dumps(params), score, trained_on, active, notes))
            return cur.fetchone()["version"]

    def list_model_versions(self, limit: int = 50) -> list[dict]:
        with self._cx() as cx, cx.cursor() as cur:
            cur.execute("select version, kind, params, score, trained_on, active, notes, "
                        "to_char(created_at at time zone 'Asia/Kolkata','YYYY-MM-DD HH24:MI') created "
                        "from model_versions order by version desc limit %s", (limit,))
            return [dict(r) for r in cur.fetchall()]

    def active_model(self) -> dict | None:
        with self._cx() as cx, cx.cursor() as cur:
            cur.execute("select version, kind, params, score, trained_on, notes from model_versions "
                        "where active=true order by version desc limit 1")
            r = cur.fetchone()
            return dict(r) if r else None

    # ===== publish: the finalized client-facing snapshot =========================
    def publish_report(self, period: str, scope: str, report: dict, *, store_id: str = "s14",
                       model_version=None, reviewer: str = "human") -> int:
        with self._cx() as cx, cx.cursor() as cur:
            cur.execute("insert into published_reports(store_id,period,scope,report,model_version,published_by) "
                        "values(%s,%s,%s,%s,%s,%s) returning id",
                        (store_id, period, scope, json.dumps(report), model_version, reviewer))
            return cur.fetchone()["id"]

    def list_published(self, store_id: str = "s14") -> list[dict]:
        with self._cx() as cx, cx.cursor() as cur:
            cur.execute("select period, scope, model_version, "
                        "to_char(published_at at time zone 'Asia/Kolkata','YYYY-MM-DD HH24:MI') published_at "
                        "from latest_published where store_id=%s order by period desc, scope", (store_id,))
            return [dict(r) for r in cur.fetchall()]

    def published_history(self, store_id: str = "s14", limit: int = 50) -> list[dict]:
        """Every publish (append-only), newest first, with the key numbers pulled from each frozen
        report — the repository the Report tab shows to track progress over time."""
        with self._cx() as cx, cx.cursor() as cur:
            cur.execute("select period, scope, model_version, "
                        "to_char(published_at at time zone 'Asia/Kolkata','YYYY-MM-DD HH24:MI') published_at, "
                        "(report->'customers'->>'unique_customers')::int customers, "
                        "(report->'employees'->>'headcount')::int employees, "
                        "(report->'customers'->'groups'->>'count')::int groups "
                        "from published_reports where store_id=%s order by published_at desc limit %s",
                        (store_id, limit))
            return [dict(r) for r in cur.fetchall()]

    def get_published(self, period: str, scope: str = "day", store_id: str = "s14") -> dict | None:
        with self._cx() as cx, cx.cursor() as cur:
            cur.execute("select report, model_version, "
                        "to_char(published_at at time zone 'Asia/Kolkata','YYYY-MM-DD HH24:MI') published_at "
                        "from latest_published where store_id=%s and period=%s and scope=%s",
                        (store_id, period, scope))
            r = cur.fetchone()
            return dict(r) if r else None

    # --- person_contexts: durable per-day PID registry, frozen at /publish ---
    def save_person_contexts(self, date: str, rows: list[dict], *, store_id: str = "s14") -> int:
        """Replace the day's frozen PID snapshot. Each row: {kind, pid_no, group_no, employee_id,
        window_id, track, in_ist, out_ist, dwell_s, exit_src, meta}."""
        with self._cx() as cx, cx.cursor() as cur:
            cur.execute("delete from person_contexts where store_id=%s and date=%s", (store_id, date))
            for r in rows:
                cur.execute(
                    "insert into person_contexts(store_id,date,kind,pid_no,group_no,employee_id,"
                    "window_id,track,in_ist,out_ist,dwell_s,exit_src,meta) "
                    "values(%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)",
                    (store_id, date, r.get("kind", "customer"), r["pid_no"], r.get("group_no"),
                     r.get("employee_id"), r.get("window_id"), r.get("track"),
                     r.get("in_ist"), r.get("out_ist"), r.get("dwell_s"), r.get("exit_src"),
                     json.dumps(r.get("meta") or {})))
            return len(rows)

    def get_person_contexts(self, date: str, *, store_id: str = "s14") -> list[dict]:
        with self._cx() as cx, cx.cursor() as cur:
            cur.execute("select kind, pid_no, group_no, employee_id, window_id, track, "
                        "to_char(in_ist,'HH24:MI:SS') in_ist, to_char(out_ist,'HH24:MI:SS') out_ist, "
                        "dwell_s::float dwell_s, exit_src, meta from person_contexts "
                        "where store_id=%s and date=%s order by kind, pid_no", (store_id, date))
            return [dict(r) for r in cur.fetchall()]
