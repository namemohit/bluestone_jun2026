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
from contextlib import contextmanager

CONFIG = pathlib.Path("configs/supabase.json")
IST = "+05:30"


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

    @contextmanager
    def _cx(self):
        import psycopg2
        from psycopg2.extras import RealDictCursor
        cx = psycopg2.connect(self.dburl, connect_timeout=20, cursor_factory=RealDictCursor,
                              options=f"-c search_path={self.schema},public")
        try:
            yield cx
            cx.commit()
        except Exception:
            cx.rollback()
            raise
        finally:
            cx.close()

    @staticmethod
    def _full_ts(date: str, hms: str) -> str | None:
        hms = (hms or "").strip()
        if not date or not hms:
            return None
        return f"{date} {hms}{IST}" if len(hms) <= 8 else hms  # "2026-06-03 18:09:57+05:30"

    # ---- push a processed window up (store + window + visits + events) ---
    def push_window(self, window: str, store_id: str, visits_json: dict, *,
                    upload_crops: bool = False) -> None:
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
        return [{"window": w, **self.metrics(w)} for w in ids]

    def get_labels(self, window: str) -> list[dict]:
        with self._cx() as cx, cx.cursor() as cur:
            cur.execute("select visit_id, verdict, reason, in_track, out_track, reviewer, employee_id "
                        "from latest_labels where window_id=%s", (window,))
            return [dict(r) for r in cur.fetchall()]

    def get_detections(self, window: str) -> list[dict]:
        with self._cx() as cx, cx.cursor() as cur:
            cur.execute("select camera, track, to_char(first_ist at time zone 'Asia/Kolkata','HH24:MI:SS') ist,"
                        " dur_s::float, frames, crop_url crop from detections where window_id=%s "
                        "order by first_ist", (window,))
            return [dict(r) for r in cur.fetchall()]

    # ---- employee roster + gallery (attendance) -------------------------
    def list_employees(self, store_id: str = "s14") -> list[dict]:
        with self._cx() as cx, cx.cursor() as cur:
            cur.execute("select id, code, name from employees where store_id=%s order by id", (store_id,))
            return [dict(r) for r in cur.fetchall()]

    def create_employee(self, store_id: str = "s14", name=None) -> dict:
        with self._cx() as cx, cx.cursor() as cur:
            cur.execute("insert into employees(store_id, name) values(%s,%s) returning id", (store_id, name))
            eid = cur.fetchone()["id"]
            cur.execute("update employees set code=%s where id=%s", (f"S{eid}", eid))
        return {"id": eid, "code": f"S{eid}", "name": name}

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
            cur.execute("select employee_id, embedding from employee_gallery where store_id=%s", (store_id,))
            return [dict(r) for r in cur.fetchall()]

    def attendance(self, store_id: str = "s14", date: str | None = None) -> list[dict]:
        """Per employee: first/last sighting + how many windows they appear in (from staff labels
        joined to detection timestamps)."""
        with self._cx() as cx, cx.cursor() as cur:
            cur.execute(
                "select e.id, e.code, e.name,"
                " to_char(min(d.first_ist) at time zone 'Asia/Kolkata','HH24:MI:SS') first_seen,"
                " to_char(max(d.last_ist) at time zone 'Asia/Kolkata','HH24:MI:SS') last_seen,"
                " count(distinct l.window_id) windows, count(d.*) sightings"
                " from employees e"
                " left join latest_labels l on l.employee_id=e.id and l.verdict='employee'"
                " left join detections d on d.window_id=l.window_id and d.track=l.in_track"
                " where e.store_id=%s group by e.id, e.code, e.name order by e.id", (store_id,))
            return [dict(r) for r in cur.fetchall()]

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
        cannot, must, employees = [], [], []
        for l in self.get_labels(window):
            it, ot, v = l.get("in_track"), l.get("out_track"), l["verdict"]
            if v == "reject" and it is not None and ot is not None:
                cannot.append([it, ot])
            elif v == "confirm" and it is not None and ot is not None:
                must.append([it, ot])
            elif v == "employee" and it is not None:
                employees.append(it)
        return {"cannot_link": cannot, "must_link": must, "employees": employees}

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

    def sync(self, window: str, store_id: str = "s14") -> None:
        """Push the freshly re-run local visits.json into the DB (source of truth for the UI)."""
        vj = self.root / window / "visits.json"
        if vj.exists():
            self.push_window(window, store_id, json.loads(vj.read_text(encoding="utf-8")))

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
