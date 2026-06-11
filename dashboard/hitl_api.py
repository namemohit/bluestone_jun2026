"""Real-footage HITL console API (mounted alongside the existing dummy dashboard).

The learning loop, end to end:
  GET  /api/hitl/windows            -> processed slices (hours) available to review
  GET  /api/hitl/visits/{window}    -> visits (uncertainty-ranked) + open + pre-exit events
  GET  /api/hitl/crop?path=...       -> a crop thumbnail (sandboxed to outputs/)
  POST /api/hitl/label              -> tick/cross/staff/link -> compile feedback -> re-run L4
                                        (cached embeddings => instant) -> return updated visits

Each window dir holds: window.json (how to re-run L4), visits.json (machine result),
labels.json (human verdicts), feedback.json (compiled). Storage is behind hitl.store so a
Supabase backend swaps in later for the Cloud Run deployment with no API changes.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse, Response

from hitl.store import LocalStore

OUTPUTS = Path("outputs")


def _make_store():
    """Prefer the Supabase store (durable, shareable); fall back to local JSON if no creds."""
    try:
        from hitl.store_supabase import SupabaseStore
        return SupabaseStore(root=str(OUTPUTS)), "supabase"
    except Exception:
        return LocalStore(root=str(OUTPUTS)), "local"


store, BACKEND = _make_store()
# Cloud-hosted public dashboard sets HITL_READONLY=1: friends VIEW results, but labelling +
# re-runs (which need the local L1 data + GPU cache) are refused. Only the local console edits.
READONLY = bool(os.environ.get("HITL_READONLY"))
# Separate switch from read-only: hide customer faces. Off once the dashboard is password-protected.
HIDE_FACES = bool(os.environ.get("HIDE_FACES"))
router = APIRouter(prefix="/api/hitl", tags=["hitl"])


_STATE_VERSION = 0   # bumps on every write -> day-wide memos (annotations, tags) recompute exactly once after a change


def _guard_write():
    if READONLY:
        raise HTTPException(403, "this is the read-only public view — review on the local console")
    global _STATE_VERSION
    _STATE_VERSION += 1   # invalidate the day-wide annotation/tag memo so the next read reflects this write


def _safe_window(window: str) -> str:
    """Validate a window name WITHOUT requiring a local dir (the cloud reads from the DB)."""
    if not window or ".." in window or "/" in window or "\\" in window:
        raise HTTPException(400, f"bad window {window!r}")
    return window


def _window_dir(window: str) -> Path:
    d = OUTPUTS / window
    if not d.exists() or ".." in window or "/" in window or "\\" in window:
        raise HTTPException(404, f"unknown window {window!r}")
    return d


def _window_cfg(window: str) -> dict:
    p = _window_dir(window) / "window.json"
    if not p.exists():
        raise HTTPException(400, f"{window} has no window.json (don't know how to re-run L4)")
    return json.loads(p.read_text(encoding="utf-8"))


def _rerun_l4(window: str) -> None:
    """Re-run the matcher with the freshly-compiled feedback. Embeddings are cached, so this is
    a fast CPU pass, not a GPU job."""
    cfg = _window_cfg(window)
    fb = store.write_feedback(window)
    cmd = [sys.executable, "-m", "stack.l4_visits",
           "--l1", cfg["l1"], "--config", cfg["config"],
           "--out", str(_window_dir(window)), "--feedback", fb]
    if cfg.get("interior"):
        cmd += ["--interior", *cfg["interior"]]
        try:  # hand L4 the enrolled-staff gallery so known staff auto-tag on every re-run
            gal = store.get_gallery()
            if gal:
                galp = _window_dir(window) / "gallery.json"
                galp.write_text(json.dumps(gal), encoding="utf-8")
                cmd += ["--gallery", str(galp)]
                sp = (store.active_model() or {}).get("params") or {}  # trained thresholds, not the L4 defaults
                for flag, key in (("--staff-sim", "staff_sim"), ("--staff-auto-sim", "staff_auto_sim"),
                                  ("--staff-margin", "staff_margin")):
                    if sp.get(key):
                        cmd += [flag, str(sp[key])]
        except Exception:
            pass
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        raise HTTPException(500, f"L4 re-run failed: {r.stderr[-400:]}")
    store.sync(window, with_detections=False)  # labels don't change L1 -> skip the ~20s detections re-push
    store.record_metrics(window)               # append a point to the precision curve


@router.get("/windows")
def windows() -> dict:
    return {"backend": BACKEND, "readonly": READONLY, "hide_faces": HIDE_FACES,
            "windows": store.list_windows()}


@router.get("/visits/{window}")
def visits(window: str) -> dict:
    _safe_window(window)   # read from the DB; don't require a local dir (cloud has none)
    data = store.get_visits(window)
    labels = {l["visit_id"]: l for l in store.get_labels(window)}
    for v in data.get("visits", []):
        vd = labels.get(v["id"], {}).get("verdict")
        v["label"] = None if vd == "reset" else vd   # reset == undone == unlabelled
    # staff grouped by employee; each group carries crop IMAGES so the UI shows the actual people
    # (click to enlarge + verify the auto matches), not bare track numbers.
    emp_map = {e["id"]: e for e in store.list_employees()}
    rank = {e["id"]: (e.get("staff_no") or (i + 1)) for i, e in enumerate(sorted(emp_map.values(), key=lambda x: x["id"]))}  # permanent staff_no

    def _label(eid):  # display as "Staff #N" by enrollment order (store keeps the raw S<id>)
        return f"Staff #{rank[eid]}" if eid in rank else (emp_map.get(eid) or {}).get("code")

    not_staff = {l["in_track"] for l in labels.values()                 # human override: NOT staff
                 if str(l.get("visit_id", "")).startswith("notstaff-") and l.get("verdict") == "reject"
                 and l.get("in_track") is not None}
    groups: dict = {}
    manual_tracks = set()

    def _grp(eid):
        return groups.setdefault(eid, {"employee_id": eid, "code": _label(eid),
                                       "pid": (f"S{rank[eid]}" if eid in rank else None),   # daily S# rank
                                       "name": (emp_map.get(eid) or {}).get("name"),
                                       "tracks": [], "visit_ids": [], "auto_tracks": []})

    for l in labels.values():
        if (l.get("verdict") == "employee" and l.get("in_track") is not None
                and l["in_track"] not in not_staff):
            g = _grp(l.get("employee_id"))
            g["tracks"].append(l["in_track"])
            g["visit_ids"].append(l["visit_id"])
            manual_tracks.add(l["in_track"])            # a human click -> NOT an auto badge
    # merge gallery-recognised (auto) staff from the local L4 result (cloud read-only has no file)
    try:
        vj = OUTPUTS / window / "visits.json"
        auto = json.loads(vj.read_text(encoding="utf-8")).get("staff", []) if vj.exists() else []
    except Exception:
        auto = []
    for a in auto:
        if a.get("weak") or a["track"] in not_staff:    # weak band stays a customer; rejected auto -> hide
            continue
        g = _grp(a.get("employee_id"))
        if a["track"] not in g["tracks"]:
            g["tracks"].append(a["track"])
        if a["track"] not in manual_tracks:
            g["auto_tracks"].append(a["track"])         # only truly-auto tracks get the robot badge
    # one crop per track so the UI renders the grouped faces (interior crop preferred — clearest)
    track_crop = {}
    for v in data.get("visits", []):
        if v.get("in_crop"):
            track_crop[v["in_track"]] = v["in_crop"]
        if v.get("out_crop"):
            track_crop[v["out_track"]] = v["out_crop"]
    for e in data.get("open_sessions", []) + data.get("pre_window_exits", []):
        if e.get("crop"):
            track_crop.setdefault(e["track"], e["crop"])
    for a in auto:
        if a.get("crop"):
            track_crop[a["track"]] = a["crop"]
    # the enrolled crop (the image the human actually CLICKED) is the truest for a manual mark — it
    # beats the door crop, which can show the wrong person when the door<->interior bridge mismatched
    try:
        for ge in _gallery_meta():
            if ge.get("source_window") == window and ge.get("source_track") is not None and ge.get("crop_url"):
                track_crop[ge["source_track"]] = ge["crop_url"]
    except Exception:
        pass
    # fallback for manually-marked tracks that got filtered out of visits/open: the door (C05) crop
    missing = {t for g in groups.values() for t in g["tracks"] if t not in track_crop}
    if missing:
        for d in store.get_detections(window):
            if d.get("camera") == "C05" and d["track"] in missing and d.get("crop"):
                track_crop.setdefault(d["track"], d["crop"])
    for g in groups.values():
        autoset = set(g["auto_tracks"])
        g["crops"] = [{"track": t, "crop": track_crop.get(t), "auto": t in autoset} for t in g["tracks"]]
    data["staff"] = sorted(groups.values(), key=lambda g: (g["employee_id"] or 0))
    # L2: every door ENTRY (matched-visit INs + still-open INs), by timestamp
    data["entries"] = sorted(
        [{"track": v["in_track"], "ist": v["in_ist"], "crop": v["in_crop"], "matched": True}
         for v in data.get("visits", [])]
        + [{"track": e["track"], "ist": e["ist"], "crop": e["crop"], "matched": False}
           for e in data.get("open_sessions", [])],
        key=lambda x: x["ist"])
    data["metrics"] = store.metrics(window)
    cards = _cards_from_dets(window)                  # ONE classify pass -> both panels; SAME det crops as All-Detections (1:1)
    data["staff_cards"] = cards["staff_cards"]
    data["customer_cards"] = cards["customer_cards"]
    data["unassigned_cards"] = cards["unassigned_cards"]   # door crossings the HITL removed from customers (reason-tagged)
    data["handoff_suggest"] = [s for s in _handoff_suggestions(window.split("_")[0]) if s["window"] == window]
    return data


@router.get("/visits-day/{date}")
def visits_day(date: str) -> dict:
    """Whole-day rollup of the Review sections across the date's hourly windows (read-only overview)."""
    if not date or "/" in date or "\\" in date or ".." in date:
        raise HTTPException(400, "bad date")
    import glob
    windows = sorted({Path(p).parent.name for p in glob.glob(f"outputs/{date}_*/visits.json")})
    emp_map = {e["id"]: e for e in store.list_employees()}
    rank = {e["id"]: (e.get("staff_no") or (i + 1)) for i, e in enumerate(sorted(emp_map.values(), key=lambda x: x["id"]))}  # permanent staff_no
    agg = {"visits": [], "open_sessions": [], "pre_window_exits": [], "entries": [], "day": True}
    staff_by_emp = {}
    met = {"visits": 0, "reviewed": 0, "confirmed": 0, "rejected": 0, "unreviewed": 0}

    def _grp(eid):
        return staff_by_emp.setdefault(eid, {"employee_id": eid,
            "code": f"Staff #{rank[eid]}" if eid in rank else (emp_map.get(eid) or {}).get("code"),
            "name": (emp_map.get(eid) or {}).get("name"), "tracks": [], "visit_ids": [],
            "auto_tracks": [], "crops": []})

    vmap = store.get_visits_many(windows)               # one round-trip each for the whole day
    lmap = store.get_labels_many(windows)
    for win in windows:
        data = vmap.get(win, {"visits": [], "open_sessions": [], "pre_window_exits": []})
        labels = {l["visit_id"]: l for l in lmap.get(win, [])}
        not_staff = {l["in_track"] for l in labels.values()
                     if str(l.get("visit_id", "")).startswith("notstaff-") and l.get("verdict") == "reject"
                     and l.get("in_track") is not None}
        vis = data.get("visits", [])
        for v in vis:
            vd = labels.get(v["id"], {}).get("verdict")
            agg["visits"].append({**v, "label": None if vd == "reset" else vd, "window": win})
        for e in data.get("open_sessions", []):
            agg["open_sessions"].append({**e, "window": win})
        for e in data.get("pre_window_exits", []):
            agg["pre_window_exits"].append({**e, "window": win})
        agg["entries"] += [{"track": v["in_track"], "ist": v["in_ist"], "crop": v["in_crop"],
                            "matched": True, "window": win} for v in vis]
        agg["entries"] += [{"track": e["track"], "ist": e["ist"], "crop": e["crop"],
                            "matched": False, "window": win} for e in data.get("open_sessions", [])]
        manual = set()
        for l in labels.values():
            if l.get("verdict") == "employee" and l.get("in_track") is not None and l["in_track"] not in not_staff:
                _grp(l.get("employee_id"))["tracks"].append(l["in_track"])
                manual.add(l["in_track"])
        try:
            auto = json.loads((OUTPUTS / win / "visits.json").read_text(encoding="utf-8")).get("staff", [])
        except Exception:
            auto = []
        for a in auto:
            if a["track"] in not_staff:
                continue
            g = _grp(a.get("employee_id"))
            if a["track"] not in g["tracks"]:
                g["tracks"].append(a["track"])
            if a["track"] not in manual:
                g["auto_tracks"].append(a["track"])
            if a.get("crop"):
                g["crops"].append({"track": a["track"], "crop": a["crop"], "auto": a["track"] not in manual})
        reviewed = [v for v in vis if v["id"] in labels and labels[v["id"]]["verdict"] != "reset"]
        conf = sum(1 for v in reviewed if labels[v["id"]]["verdict"] in ("confirm", "employee"))
        met["visits"] += len(vis); met["reviewed"] += len(reviewed); met["confirmed"] += conf
        met["rejected"] += len(reviewed) - conf; met["unreviewed"] += len(vis) - len(reviewed)
    agg["staff"] = sorted(staff_by_emp.values(), key=lambda g: (g["employee_id"] or 0))
    agg["entries"].sort(key=lambda e: e.get("ist") or "")
    agg["visits"].sort(key=lambda v: v.get("in_ist") or "")
    met["precision"] = round(met["confirmed"] / met["reviewed"], 3) if met["reviewed"] else None
    agg["metrics"] = met
    cards = _cards_from_dets(date)                    # ONE classify pass -> whole-day staff + customer cards (1:1 with All-Detections)
    agg["staff_cards"] = cards["staff_cards"]
    agg["customer_cards"] = cards["customer_cards"]
    agg["unassigned_cards"] = cards["unassigned_cards"]    # door crossings the HITL removed from customers (reason-tagged)
    agg["handoff_suggest"] = _handoff_suggestions(date)
    return agg


@router.get("/detections/{window}")
def detections(window: str, grouped: int = 0) -> dict:
    """L1 — every human the cameras detected (door + interior). grouped=1 returns a RECONCILIATION
    view: each detection tagged street / accounted / uncounted (+ a staff flag) so the wall
    complements the Review tab instead of duplicating it."""
    _safe_window(window)
    if not grouped:
        return {"detections": store.get_detections(window)}
    try:
        dets = _classify_detections(window)
    except Exception:
        return {"grouped": False, "detections": store.get_detections(window)}
    return _grouped_detections(dets)


def _resolve_staff_names(dets: list) -> None:
    """Attach 'Staff #N · name' to each staff detection (day-global rank, so names match across hours)."""
    if any(d.get("employee_id") for d in dets):
        rank, names = _rank_labels(), {e["id"]: e.get("name") for e in store.list_employees("s14")}
        for d in dets:
            eid = d.get("employee_id")
            if eid:
                d["staff_name"] = (rank.get(eid) or f"#{eid}") + (" · " + names[eid] if names.get(eid) else "")


def _grouped_detections(dets: list) -> dict:
    """Bucket a (possibly multi-window) detection list into the reconciliation response shape.
    Shared by the per-hour /detections and the whole-day /detections-day endpoints."""
    _resolve_staff_names(dets)
    # display order (user-chosen): accounted people first, then already-counted, noise, then the work queue last.
    # 'inside' sits next to 'duplicate' — both mean "already counted at the door" (confirming inside writes duplicate).
    order = ["customer", "staff", "on_hold", "duplicate", "inside", "not_person", "passby", "to_review"]
    buckets: dict = {k: [] for k in order}
    for d in dets:
        buckets.setdefault(d.get("determination", "to_review"), []).append(d)
    for k in buckets:
        buckets[k].sort(key=lambda x: x.get("ist") or "")    # interleave hours chronologically within a bucket
    per_bucket = {k: {"confirmed": sum(1 for d in buckets.get(k, []) if d.get("confirmed")),
                      "total": len(buckets.get(k, []))} for k in order}
    return {"grouped": True, "total": len(dets), "order": order, "buckets": buckets,
            "per_bucket": per_bucket, "confirmed": sum(1 for d in dets if d.get("confirmed")),
            # the close now covers EVERY detection incl. 'inside' (a genuinely missed customer can hide there)
            "close_total": len(dets),
            "counts": {k: len(buckets.get(k, [])) for k in order}}


@router.get("/detections-day/{date}")
def detections_day(date: str) -> dict:
    """Whole-day reconciliation: every detection across the date's hourly windows, bucketed.
    Mirrors visits_day; each detection already carries its 'window' (set in _classify_detections)."""
    if not date or "/" in date or "\\" in date or ".." in date:
        raise HTTPException(400, "bad date")
    windows = _day_windows(date)
    tags = _daily_tags(date)                                          # compute the day-global C#/G#/S# numbering ONCE
    all_dets: list = []
    for win in windows:
        try:
            all_dets += _classify_detections(win, with_dims=False, tags=tags)   # skip crop-header reads -> fast day load
        except Exception:
            continue
    return _grouped_detections(all_dets)


def _point_in_poly(x, y, poly) -> bool:
    inside, n, j = False, len(poly), len(poly) - 1
    for i in range(n):
        xi, yi = poly[i]
        xj, yj = poly[j]
        if ((yi > y) != (yj > y)) and (x < (xj - xi) * (y - yi) / (yj - yi + 1e-12) + xi):
            inside = not inside
        j = i
    return inside


_CROP_DIM: dict = {}   # crop path -> (w, h) px, cached; crops never change so first read sticks


def _crop_dims(crop: str):
    """Pixel (w, h) of a person crop, for the size filter. Reads only the image header (lazy),
    caches by path. Crops are local to the labelling console; on the read-only Cloud Run view the
    file is absent -> (None, None) so the filter simply doesn't hide anything."""
    if not crop:
        return (None, None)
    if crop not in _CROP_DIM:
        wh = (None, None)
        try:
            from PIL import Image
            safe = os.path.normpath(crop).replace("\\", "/")
            if safe.startswith("outputs/") and os.path.exists(safe):
                with Image.open(safe) as im:
                    wh = im.size
        except Exception:
            wh = (None, None)
        _CROP_DIM[crop] = wh
    return _CROP_DIM[crop]


def _day_windows(date: str) -> list:
    """Real operating-hours windows for a date: outputs/<date>_HHMM (4-digit time). Excludes the
    pre-open '<date>_09' stub (store opens ~11:24) so the whole-day numbering/report/detections only
    count real customers. (Test slices like 'L4_hitl_test' never match the <date>_* glob anyway.)"""
    import glob
    import re as _re
    return sorted({Path(p).parent.name for p in glob.glob(f"outputs/{date}_*/visits.json")
                   if _re.search(r"_\d{4}$", Path(p).parent.name)})


_DAY_ANN: dict = {}        # (date, _STATE_VERSION) -> {window: [annotation rows]} — one batched query, reused until a write
_DAY_LABELS: dict = {}     # (date, _STATE_VERSION) -> {window: [label rows]} — same, for staff/reject labels
_DAILY_TAGS: dict = {}     # (date, _STATE_VERSION) -> the C#/G#/S# numbering — computed once per state, not per call
_HANDOFF: dict = {}        # (date, _STATE_VERSION) -> trajectory-handoff de-dup {fragment->canonical} + suggestions
_EXITS: dict = {}          # (date, _STATE_VERSION) -> {(win,in_track): {out_ist,dwell_s,source}} L4 exit resolution for open PIDs


def _prune(cache: dict) -> None:
    for k in list(cache):              # keep only the current write-version; drop stale snapshots
        if k[1] != _STATE_VERSION:
            del cache[k]


def _day_annotations(date: str) -> dict:
    """All of a date's windows' latest annotations in ONE Supabase round-trip, memoized until the next
    write. Replaces the per-window latest_annotations fan-out (12 RTTs) that dominated page-load time."""
    key = (date, _STATE_VERSION)
    hit = _DAY_ANN.get(key)
    if hit is not None:
        return hit
    wins = _day_windows(date)
    try:
        if hasattr(store, "latest_annotations_bulk"):
            out = store.latest_annotations_bulk(wins)
        else:
            out = {w: store.latest_annotations(w) for w in wins}   # LocalStore fallback (no batch method)
    except Exception:
        return {w: [] for w in wins}   # transient query failure -> serve empty but DON'T memoize a bad result
    _prune(_DAY_ANN)
    _DAY_ANN[key] = out
    return out


def _window_annotations(window: str) -> list:
    """Latest annotations for ONE window, served from the day-wide batch when possible (else a direct query
    for off-grid windows like the pre-open stub or test slices)."""
    day = _day_annotations(window.split("_")[0])
    return day[window] if window in day else store.latest_annotations(window)


def _day_labels(date: str) -> dict:
    """All of a date's windows' review labels (staff/reject) in ONE round-trip, memoized until the next write.
    Lets the whole-day reconciliation reuse a single labels query instead of one per hour."""
    key = (date, _STATE_VERSION)
    hit = _DAY_LABELS.get(key)
    if hit is not None:
        return hit
    wins = _day_windows(date)
    try:
        if hasattr(store, "get_labels_many"):
            out = store.get_labels_many(wins)
        else:
            out = {w: store.get_labels(w) for w in wins}          # LocalStore fallback (no batch method)
    except Exception:
        return {w: [] for w in wins}   # transient query failure -> serve empty but DON'T memoize a bad result
    _prune(_DAY_LABELS)
    _DAY_LABELS[key] = out
    return out


def _window_labels(window: str) -> list:
    """Review labels for ONE window, from the day-wide batch when possible (else a direct query)."""
    day = _day_labels(window.split("_")[0])
    return day[window] if window in day else store.get_labels(window)


def _entry_traj(window: str) -> dict:
    """C05 door-track trajectory endpoints {track: {f_ts, f_xy, l_ts, l_xy}} from the window's L1_entry tracks.json.
    A tracker ID-switch (fragment) leaves a tell-tale handoff: one track ends where the next begins, no overlap."""
    import glob
    tj = None
    for cand in [f"outputs/{window}/L1_entry/tracks.json", *glob.glob(f"outputs/{window}/*entry*/tracks.json")]:
        if os.path.exists(cand):
            try:
                tj = json.loads(Path(cand).read_text(encoding="utf-8"))
            except Exception:
                tj = None
            break
    out = {}
    for t in (tj.get("tracks", []) if tj else []):
        tr = t.get("traj", [])
        if tr:
            f, l = tr[0], tr[-1]
            out[t["track"]] = {"f_ts": f[0], "f_xy": (f[1], f[2]), "l_ts": l[0], "l_xy": (l[1], l[2])}
    return out


# trajectory-handoff thresholds: B begins ~same spot ~1s after A ends, no time overlap (an ID-switch, not two people)
_HO_GAP_TIGHT, _HO_DIST_TIGHT = 1.5, 0.12      # AUTO-merge band (high precision)
_HO_GAP_MAX, _HO_DIST_MAX = 2.5, 0.25          # SUGGEST band (a human confirms)
_HO_OVERLAP = 0.5                              # B alive >0.5s before A ends -> two people, NEVER merge (protects real groups)


def _handoff_merges(date: str) -> dict:
    """L3 strengthening / de-dup: collapse confirmed door entries that are ONE person split by a tracker ID-switch.
    A fragment = track B starts at ~the same (x,y) ~1s after track A ends with NO temporal overlap. Two tracks
    alive at the same instant are different people (a person can't be in two places), so a real group is never
    merged. Returns {merge:{(win,track):(win,canon)}  # fragment -> earliest member; suggest:[{window,a,b,gap,dist}]}."""
    import math
    cached = _HANDOFF.get((date, _STATE_VERSION))
    if cached is not None:
        return cached
    merge, suggest, merge_meta = {}, [], {}
    for win in _day_windows(date):
        try:
            ents = [e["track"] for e in _confirmed_entries(win)[0]]
        except Exception:
            ents = []
        if len(ents) < 2:
            continue
        kept_apart = {str(l.get("visit_id", "")) for l in _window_labels(win)
                      if str(l.get("visit_id", "")).startswith("distinct-")}   # human pressed "keep separate" -> never re-suggest this pair
        traj = _entry_traj(win)
        ep = {tk: traj[tk] for tk in ents if tk in traj}
        ks = sorted(ep, key=lambda k: ep[k]["f_ts"])
        for i, a in enumerate(ks):
            A = ep[a]
            for b in ks[i + 1:]:
                B = ep[b]
                gap = B["f_ts"] - A["l_ts"]                      # B starts this long after A ends
                if gap > _HO_GAP_MAX:
                    break                                       # ks sorted by start -> later b's only start later
                if B["f_ts"] < A["l_ts"] - _HO_OVERLAP:
                    continue                                    # temporal overlap -> two people, not a fragment
                dist = math.hypot(B["f_xy"][0] - A["l_xy"][0], B["f_xy"][1] - A["l_xy"][1])
                if -_HO_OVERLAP <= gap <= _HO_GAP_TIGHT and dist <= _HO_DIST_TIGHT:
                    merge[(win, b)] = merge.get((win, a), (win, a))      # AUTO-merge B -> A's canonical (chains A->B->C)
                    merge_meta[(win, b)] = {"gap": round(gap, 1), "dist": round(dist, 3)}   # why: the de-dup evidence, for the PID drawer
                elif (-_HO_OVERLAP <= gap <= _HO_GAP_MAX and dist <= _HO_DIST_MAX
                      and (win, b) not in merge and f"distinct-{a}-{b}" not in kept_apart):
                    suggest.append({"window": win, "a": a, "b": b, "gap": round(gap, 1), "dist": round(dist, 3)})
    result = {"merge": merge, "suggest": suggest, "merge_meta": merge_meta}
    _prune(_HANDOFF)
    _HANDOFF[(date, _STATE_VERSION)] = result
    return result


def _ist_from_secs(s) -> str:
    s = int(max(0, s))
    return f"{s // 3600:02d}:{(s % 3600) // 60:02d}:{s % 60:02d}"


def _rejected_exits(date: str) -> set:
    """Human 'this check-out is the wrong person' rejects: {(in_win, in_track, out_win, out_track)} from
    durable `rejexit-<intrack>-<outwin>-<outtrack>` labels (latest-wins; a 'reset' undoes). _exit_resolve
    excludes these (entry,out) pairs -> the wrong OUT frees + the entry falls to presumed."""
    out = {}
    for win in _day_windows(date):
        for l in _window_labels(win):
            vid = str(l.get("visit_id", ""))
            if not vid.startswith("rejexit-"):
                continue
            try:                                               # rejexit-<intrack>-<outwin>-<outtrack> (outwin has dashes -> split first/last)
                it, rest = vid[len("rejexit-"):].split("-", 1)
                ow, ot = rest.rsplit("-", 1)
                key = (win, int(it), ow, int(ot))
            except ValueError:
                continue
            if l.get("verdict") == "reset":
                out.pop(key, None)
            else:
                out[key] = True
    return set(out)


def _exit_resolve(date: str) -> dict:
    """L4 exit resolution (day-wide): close every still-inside customer PID. Pool the unmatched OUTs (per-window
    pre_window_exits) across the whole day and FIFO-match each to the earliest still-inside entry before it
    (within a max-dwell / ~2h bounded window); a residual open entry (its exit crossing was missed) is presumed
    to have left at last-seen + grace -- the store empties at close. ReID is an OPTIONAL gate: when reid_weight>0
    (from active_model().params) the eligible candidate with the best appearance match wins instead of FIFO --
    a no-op at weight 0, which is correct on this footage. Returns {(window, in_track): {out_ist, dwell_s, source}}
    for entries that had no matched visit exit."""
    cached = _EXITS.get((date, _STATE_VERSION))
    if cached is not None:
        return cached

    def secs(t):
        try:
            h, m, s = map(int, str(t).split(":"))
            return h * 3600 + m * 60 + s
        except Exception:
            return 0
    params = (store.active_model() or {}).get("params", {}) or {}
    MAX_DWELL = float(params.get("exit_max_dwell_s", 7200))    # ~2h bounded window + dwell cap
    GRACE = float(params.get("exit_grace_s", 60))
    reid_weight = float(params.get("reid_weight", 0.0))        # >0 -> ReID gate among time-eligible candidates (off on this footage)
    exit_floor = float(params.get("reid_exit_floor", 0.55))    # an OUT must LOOK like the entry to match it -> rejects wrong-person exits
    exit_same_cam = bool(params.get("exit_same_cam", True))    # compare entry C05 <-> OUT C05 door crop (not the C14 interior crop -> apples-to-apples)
    exit_margin = float(params.get("exit_match_margin", 0.05)) # best eligible open must beat 2nd-best by this, else presume (don't guess)
    exit_require_emb = bool(params.get("exit_require_emb", False))   # no-embedding OUT -> presumed, not a blind FIFO grab
    gender_gate = bool(params.get("exit_gender_gate", True))   # reject M<->F contradictions (cheap; LOW coverage - faces rare at the door)
    gender_conf = float(params.get("exit_gender_conf", 0.70))  # min face-gender confidence to enforce the gate
    merge = _handoff_merges(date)["merge"]
    rejected = _rejected_exits(date)                            # human 'wrong check-out' rejects -> never re-match these (entry,out) pairs
    opens, outs = [], []
    for win in _day_windows(date):
        ci = _entry_crop_info(win)
        try:
            vj = json.loads((OUTPUTS / win / "visits.json").read_text(encoding="utf-8"))
        except Exception:
            vj = {}
        link_last = {}                                         # entry in_track -> latest interior sighting time (richer last-seen)
        for l in vj.get("links", []):
            o = l.get("out_ist") or l.get("in_ist")
            if o:
                link_last[l["in_track"]] = max(link_last.get(l["in_track"], 0), secs(o))
        traj = _entry_traj(win)
        try:
            ents = _confirmed_entries(win)[0]
        except Exception:
            ents = []
        for e in ents:
            key = (e["window"], e["track"])
            if merge.get(key, key) != key:                     # absorbed fragment -> its canonical carries the PID
                continue
            info = ci.get(e["track"], {}) or {}
            if info.get("out_ist"):                            # already has a matched visit exit
                continue
            in_s = secs(e["ist"])
            last_s = in_s
            tr = traj.get(e["track"])
            if tr:
                last_s = max(last_s, in_s + (tr["l_ts"] - tr["f_ts"]))   # entry door-track's last frame
            last_s = max(last_s, link_last.get(e["track"], 0))           # interior last sighting
            opens.append({"key": key, "in_s": in_s, "last_s": last_s, "crop": info.get("crop")})
        for x in vj.get("pre_window_exits", []):
            if x.get("ist"):
                outs.append({"out_s": secs(x["ist"]), "ist": x["ist"], "window": win, "track": x.get("track"),
                             "crop": (x.get("crop") or "").replace("\\", "/"),
                             "door_crop": (x.get("door_crop") or "").replace("\\", "/")})
    opens.sort(key=lambda o: o["in_s"])                        # FIFO tiebreak: earliest-entered first
    outs.sort(key=lambda x: x["out_s"])
    import numpy as np
    exits, used = {}, [False] * len(opens)
    for x in outs:
        elig = [i for i, o in enumerate(opens)
                if not used[i] and o["in_s"] <= x["out_s"] and (x["out_s"] - o["in_s"]) <= MAX_DWELL
                and (o["key"][0], o["key"][1], x.get("window"), x.get("track")) not in rejected]   # Layer 3: human reject
        if not elig:
            continue
        if gender_gate:                                        # Layer 1: drop M<->F contradictions (a no-op unless BOTH crops have a confident face)
            xg, xgc = _demo_gender(x.get("door_crop") or x.get("crop"))
            if xg is None:
                xg2, xgc2 = _demo_gender(x.get("crop"))         # interior crop is likelier to carry a face than the door crop
                if xgc2 > xgc:
                    xg, xgc = xg2, xgc2
            if xg and xgc >= gender_conf:
                kept = []
                for i in elig:
                    eg, egc = _demo_gender(opens[i].get("crop"))
                    if eg and egc >= gender_conf and eg != xg:
                        continue                                # confident opposite genders -> this OUT is not this entry
                    kept.append(i)
                elig = kept
                if not elig:
                    continue
        ecrop = (x.get("door_crop") if exit_same_cam else None) or x.get("crop")   # Layer 2b: C05<->C05, not C05<->C14
        xe = _emb_for_crop(ecrop) if (exit_floor > 0 and ecrop) else None
        sim_used = margin_used = None
        if xe is not None:                                     # IDENTITY FLOOR + margin gate: an exit must LOOK like the entry it closes
            def _esim(i):
                ce = _emb_for_crop(opens[i].get("crop"))
                return float(np.dot(xe, ce)) if ce is not None else 1.0    # entry without an embedding -> don't block
            scored = sorted(((i, _esim(i)) for i in elig), key=lambda t: -t[1])
            scored = [(i, s) for i, s in scored if s >= exit_floor]        # drop clear mismatches (the C4 stranger)
            if not scored:
                continue                                                   # nobody looks like this exit -> leave entries open -> presumed
            if len(scored) >= 2 and (scored[0][1] - scored[1][1]) < exit_margin:
                continue                                                   # Layer 2a: too close to call between two opens -> presume, don't guess
            cand = scored[0][0]                                            # best-looking among the time-eligible
            sim_used = round(scored[0][1], 3)
            margin_used = round(scored[0][1] - scored[1][1], 3) if len(scored) >= 2 else None
        else:
            if exit_require_emb:                                            # Layer 2a: no OUT embedding -> presume, not a blind FIFO grab
                continue
            cand = elig[0]                                                  # legacy FIFO fallback
        used[cand] = True
        o = opens[cand]
        exits[o["key"]] = {"out_ist": x["ist"], "dwell_s": round(x["out_s"] - o["in_s"], 1),
                           "source": "matched", "out_crop": x.get("crop"),    # door-exit crossing crop -> check-out proof
                           "sim": sim_used, "margin": margin_used,            # Layer 3 evidence: how confident was this match
                           "cross_window": (o["key"][0] != x.get("window")),
                           "out_window": x.get("window"), "out_track": x.get("track")}
    for i, o in enumerate(opens):                              # residual opens -> presumed exit at last-seen + grace
        if used[i]:
            continue
        ex_s = o["last_s"] + GRACE
        exits[o["key"]] = {"out_ist": _ist_from_secs(ex_s), "dwell_s": round(max(0, ex_s - o["in_s"]), 1), "source": "presumed"}
    _prune(_EXITS)
    _EXITS[(date, _STATE_VERSION)] = exits
    return exits


def _exit_confirmed(date: str) -> set:
    """(window, in_track) pairs whose L4 presumed exit a human has accepted -> the 'exit inferred' chip clears.
    Stored as a durable 'exitok-<track>' confirm label (reuses the labels table, like 'notstaff-'/'false-')."""
    out = set()
    for win in _day_windows(date):
        for l in _window_labels(win):
            if (str(l.get("visit_id", "")).startswith("exitok-") and l.get("verdict") == "confirm"
                    and l.get("in_track") is not None):
                out.add((win, l["in_track"]))
    return out


def _frozen_contexts(date: str) -> dict:
    """Frozen #C/#G per canonical entry from a published-day person_contexts snapshot (empty if the day
    isn't published or the table isn't migrated yet). Lets _daily_tags lock the numbering after publish."""
    try:
        rows = store.get_person_contexts(date) if hasattr(store, "get_person_contexts") else []
    except Exception:
        return {}
    out = {}
    for r in rows:
        if r.get("kind") == "customer" and r.get("window_id") and r.get("track") is not None:
            out[(r["window_id"], r["track"])] = {"pid_no": r["pid_no"], "group_no": r.get("group_no")}
    return out


def _daily_tags(date: str) -> dict:
    """Day-global DISPLAY numbering (recomputed per request, passed down once): customers C1..n by arrival
    time across the whole day, their arrival group G1..n, and staff S1..n by enrollment rank. Reuses the
    report's machinery (_confirmed_entries + group_sessions) so #C count == report footfall and #G == the
    report's groups. Returns {cust:{(win,track):n}, grp:{(win,track):gid}, staff:{employee_id:rank}}."""
    import glob
    from types import SimpleNamespace
    from logic.grouping import group_sessions

    cached = _DAILY_TAGS.get((date, _STATE_VERSION))
    if cached is not None:
        return cached

    def secs(t):
        try:
            h, m, s = map(int, str(t).split(":"))
            return h * 3600 + m * 60 + s
        except Exception:
            return 0
    windows = _day_windows(date)
    entries = []
    for win in windows:
        try:
            entries += _confirmed_entries(win)[0]
        except Exception:
            pass
    entries.sort(key=lambda e: (secs(e["ist"]), e["window"], e["track"]))   # arrival order, DETERMINISTIC tiebreak -> stable C1..Cn across runs
    merge = _handoff_merges(date)["merge"]                      # trajectory-handoff de-dup: a fragment track -> its canonical (earliest) entry

    def _canon(e):                                             # one person split into several door tracks -> collapse to one PID
        return merge.get((e["window"], e["track"]), (e["window"], e["track"]))
    frozen = _frozen_contexts(date)                            # published-day snapshot -> freeze #C/#G (empty otherwise -> live numbering)
    pidn = {}                                                  # canonical -> C#  (number the DE-DUPED people by arrival)
    if frozen:
        nxt = max((v["pid_no"] for v in frozen.values()), default=0) + 1
        for e in entries:                                      # frozen PIDs keep their #C; any added since publish append above the max
            c = _canon(e)
            if c in pidn:
                continue
            fz = frozen.get(c)
            pidn[c] = fz["pid_no"] if fz else nxt
            if not fz:
                nxt += 1
    else:
        for e in entries:
            c = _canon(e)
            if c not in pidn:
                pidn[c] = len(pidn) + 1
    cust = {(e["window"], e["track"]): pidn[_canon(e)] for e in entries}    # fragments share their canonical's #C
    ist = {(e["window"], e["track"]): e["ist"] for e in entries}            # entry (door-crossing) ist -> E2 cards show the SAME in-time as the report
    canon_entries = [e for e in entries if _canon(e) == (e["window"], e["track"])]   # one (earliest) entry per PID, arrival order
    sess = [SimpleNamespace(is_employee=False, entry_ts=secs(e["ist"]), session_id=i)
            for i, e in enumerate(canon_entries)]
    gmap = group_sessions(sess)[0] if sess else {}             # {session_id -> group_id(1..n)} over the DE-DUPED people
    pid_grp = {(e["window"], e["track"]): gmap.get(i) for i, e in enumerate(canon_entries)}
    for c, fz in (frozen or {}).items():                       # frozen #G wins for already-published PIDs
        if fz.get("group_no") is not None and c in pid_grp:
            pid_grp[c] = fz["group_no"]
    grp = {(e["window"], e["track"]): pid_grp.get(_canon(e)) for e in entries}   # fragments inherit their canonical's group
    staff = _staff_no_map()                                    # employee id -> permanent staff_no (immutable; was a recomputed rank)
    result = {"cust": cust, "grp": grp, "staff": staff, "ist": ist}
    _prune(_DAILY_TAGS)
    _DAILY_TAGS[(date, _STATE_VERSION)] = result
    return result


def _handoff_suggestions(date: str) -> list:
    """Loose trajectory-handoff candidates for a human to adjudicate ('same person? merge / keep separate'),
    each enriched with both tracks' crop, #C and in-time. Auto-merged pairs and already-decided pairs (merged
    -> b drops from _confirmed_entries; kept-apart -> 'distinct-' label) never appear -- handled in _handoff_merges."""
    sugg = _handoff_merges(date)["suggest"]
    if not sugg:
        return []
    tags = _daily_tags(date)
    cust, ist = tags.get("cust", {}), tags.get("ist", {})
    cinfo: dict = {}
    out = []
    for s in sugg:
        win, a, b = s["window"], s["a"], s["b"]
        ci = cinfo.setdefault(win, _entry_crop_info(win))
        out.append({**s,
                    "a_crop": (ci.get(a, {}) or {}).get("crop"), "b_crop": (ci.get(b, {}) or {}).get("crop"),
                    "a_pid": cust.get((win, a)), "b_pid": cust.get((win, b)),
                    "a_ist": ist.get((win, a)), "b_ist": ist.get((win, b))})
    return out


def _classify_detections(window: str, with_dims: bool = True, tags: dict | None = None) -> list:
    """Tag each L1 detection by disposition. Reads the local tracks.json (which keeps the full traj),
    the zone config, and visits.json + labels to build the 'already accounted for' track set.
      accounted -> entered/exited/visit/open/pre/staff (shown in the Review tab)
      street    -> C05 track whose path is mostly inside street_mask (the noise filter dropped it)
      uncounted -> seen but never classified  (the review focus: possible missed customer)"""
    import re
    wdir = OUTPUTS / window
    wcfg = json.loads((wdir / "window.json").read_text(encoding="utf-8"))
    cfg = json.loads(Path(wcfg["config"]).read_text(encoding="utf-8"))
    street, sfrac = cfg.get("street_mask", []), cfg.get("street_drop_frac", 0.5)
    tags = tags or _daily_tags(window.split("_")[0])           # day-global C#/G#/S# numbering (shared by callers)

    def camtrack(crop):  # 'outputs/.../L1_C11/crops/trk_0035.jpg' -> ('C11', 35)
        m = re.search(r"L1_([^/\\]+)[/\\]crops[/\\]trk_(\d+)", str(crop or ""))
        return ("C05" if m.group(1) == "entry" else m.group(1), int(m.group(2))) if m else None

    accounted, staff_emp = set(), {}                        # staff_emp: (cam,track) -> employee_id
    try:
        vj = json.loads((wdir / "visits.json").read_text(encoding="utf-8"))
    except Exception:
        vj = {}
    for v in vj.get("visits", []):
        accounted.update({("C05", v["in_track"]), ("C05", v["out_track"])})
        for c in (v.get("in_crop"), v.get("out_crop")):
            if camtrack(c):
                accounted.add(camtrack(c))
    for e in vj.get("open_sessions", []) + vj.get("pre_window_exits", []):
        accounted.add(("C05", e["track"]))
        if camtrack(e.get("crop")):
            accounted.add(camtrack(e.get("crop")))
    weak_staff = set()                                          # gallery hit in the SUGGEST band -> stays a customer, flagged
    for st in vj.get("staff", []):
        ct = camtrack(st.get("crop"))
        if st.get("weak"):                                      # NOT accounted/staff_emp -> remains a customer to confirm
            weak_staff.add(("C05", st["track"]))
            if ct:
                weak_staff.add(ct)
            continue
        accounted.add(("C05", st["track"]))
        staff_emp[("C05", st["track"])] = st.get("employee_id")
        if ct:
            accounted.add(ct)
            staff_emp[ct] = st.get("employee_id")
    for l in vj.get("links", []):                               # interior auto-linked to its door entry
        accounted.add(("C05", l["in_track"]))
        accounted.add((l["cam"], l["track"]))                   # the interior track -> accounted (suggested customer)
    cust_entry = {}                                             # (cam,track) -> the door ENTRY track that owns this crop
    for v in vj.get("visits", []):
        ent = v["in_track"]
        cust_entry[("C05", v["in_track"])] = ent
        cust_entry[("C05", v["out_track"])] = ent
        for c in (v.get("in_crop"), v.get("out_crop")):
            if camtrack(c):
                cust_entry[camtrack(c)] = ent
    for e in vj.get("open_sessions", []) + vj.get("pre_window_exits", []):
        cust_entry[("C05", e["track"])] = e["track"]
        if camtrack(e.get("crop")):
            cust_entry[camtrack(e.get("crop"))] = e["track"]
    for l in vj.get("links", []):
        cust_entry[("C05", l["in_track"])] = l["in_track"]
        cust_entry[(l["cam"], l["track"])] = l["in_track"]
    labels = _window_labels(window)
    not_staff = {l["in_track"] for l in labels
                 if str(l.get("visit_id", "")).startswith("notstaff-") and l.get("verdict") == "reject"
                 and l.get("in_track") is not None}
    for l in labels:
        if l.get("verdict") == "employee" and l.get("in_track") is not None and l["in_track"] not in not_staff:
            accounted.add(("C05", l["in_track"]))
            staff_emp[("C05", l["in_track"])] = l.get("employee_id")

    for l in labels:                                            # human cross-camera validation (ReID trainer): a same-window
        vid = str(l.get("visit_id", ""))                        # interior crop LINKED to (xcamsame-) / UN-LINKED from (xcamdiff-) a door #C.
        if vid.startswith("xcamsame-") and l.get("verdict") == "confirm":
            try:
                at, cam_, bt = vid[len("xcamsame-"):].split("-"); at, bt = int(at), int(bt)
            except ValueError:
                continue
            accounted.add((cam_, bt)); cust_entry[(cam_, bt)] = at      # discovered interior crop -> attach to the door entry's #C
        elif vid.startswith("xcamdiff-") and l.get("verdict") == "reject":
            try:
                _at, cam_, bt = vid[len("xcamdiff-"):].split("-"); bt = int(bt)
            except ValueError:
                continue
            accounted.discard((cam_, bt)); cust_entry.pop((cam_, bt), None); staff_emp.pop((cam_, bt), None)  # mis-association -> back to 'inside'

    ann = {(a["camera"], a["track"]): a for a in _window_annotations(window)}  # human allocations (day-wide batch)
    for (cam, tk), a in ann.items():                           # camera-aware staff: an annotated staffer (ANY camera, incl. interior C11/C14)
        if a.get("category") == "staff" and a.get("employee_id") and tk not in not_staff:   # with an id -> resolve to #S, not 'S?'
            accounted.add((cam, tk))
            staff_emp[(cam, tk)] = a["employee_id"]
    parked = _parked_set(window)                                                    # held for later review
    l2set = {x["track"] for x in _l2_crossings(window)}                             # raw L2 door-IN crossings -> 🚪 marker on the det card
    dirs = [(wcfg.get("l1"), "C05")] + [(d, Path(d).name.replace("L1_", "")) for d in wcfg.get("interior", [])]
    out = []
    for d, cam in dirs:
        tj = Path(d) / "tracks.json" if d else None
        if not tj or not tj.exists():
            continue
        for t in json.loads(tj.read_text(encoding="utf-8")).get("tracks", []):
            key, traj = (cam, t["track"]), t.get("traj", [])
            if key in accounted:
                disp = "accounted"
            elif cam == "C05" and street and traj and \
                    sum(_point_in_poly(x, y, street) for _, x, y, _ in traj) / len(traj) > sfrac:
                disp = "street"
            elif cam == "C05":
                disp = "door"       # at the door, not counted -> possible missed entry (review focus)
            else:
                disp = "inside"     # interior, not matched to a visit -> who was in the store
            dur = round(t.get("last_ts", 0) - t.get("first_ts", 0), 1)
            ann_cat = (ann.get(key) or {}).get("category")
            cropr = (t.get("crop", "") or "").replace("\\", "/")
            cw, ch = _crop_dims(cropr) if with_dims else (None, None)   # day view skips per-crop header reads
            # --- anchored pid FIRST; the customer/staff bucket DERIVES from it (1:1 with E2, zero orphans) ---
            ent = cust_entry.get(key)
            if ent is None and cam == "C05":
                ent = t["track"]                                   # a C05 door track may itself be the entry
            entparked = (key in parked) or (ent is not None and ("C05", ent) in parked)   # parking moves the WHOLE person
            pid, group = None, None
            if key in staff_emp:
                eid = staff_emp.get(key)
                pid = f"S{tags['staff'][eid]}" if eid in tags["staff"] else "S?"   # unnamed staff still traceable
            elif ann_cat == "staff":
                pid = "S?"                                         # human-marked staff, not enrolled -> unassigned, still traceable
            elif ent is not None and not entparked:
                n = tags["cust"].get((window, ent))
                if n:
                    pid = f"C{n}"
                    g = tags["grp"].get((window, ent))
                    group = f"G{g}" if g else None
            # --- determination DERIVES from the pid (+ park + explicit non-customer call) so buckets == E2 ---
            if entparked:
                sugg, conf, det = "on_hold", ann_cat is not None, "on_hold"
            elif disp == "accounted":
                conf = ann_cat is not None
                if ann_cat in ("passby", "not_person", "duplicate"):
                    sugg = det = ann_cat                           # explicit human "not a customer" wins
                elif pid and pid[0] == "S":
                    sugg = det = "staff"
                elif pid and pid[0] == "C":
                    sugg = det = "customer"
                else:
                    sugg = det = "on_hold"                         # accounted but no #C/#S -> pending, never an orphan
            else:
                if disp == "street":
                    sugg = "passby"
                elif disp == "inside":
                    # only GENUINE noise -> not_person: a brief blip (<=4 frames) that's also unconfident, tiny,
                    # or static. Real interior people (median conf 0.80) -> to_review (not 'not a person').
                    pconf = t.get("peak_conf", 1.0) or 1.0
                    nfr = t.get("frames", 0) or 0
                    maxw = max((p[3] for p in traj if len(p) >= 4), default=0.0)
                    span = ((max((p[1] for p in traj), default=0) - min((p[1] for p in traj), default=0))
                            + (max((p[2] for p in traj), default=0) - min((p[2] for p in traj), default=0))) if traj else 0.0
                    noise = nfr <= 4 and (pconf < 0.55 or maxw < 0.06 or span < 0.02)
                    sugg = "not_person" if noise else "inside"
                else:                                             # door: a raw L2 IN crossing carries a #C (L2-
                    sugg = "customer" if (pid and pid[0] == "C") else "to_review"   # authoritative) -> customer; the
                conf = ann_cat is not None                        # rest (loiter/OUT-only) stay the human's enter/pass queue
                det = ann_cat if conf else sugg
            out.append({"camera": cam, "track": t["track"], "window": window, "ist": t.get("first_ist"), "dur_s": dur,
                        "crop": cropr, "crop_w": cw, "crop_h": ch,
                        "disposition": disp, "staff": key in staff_emp, "annotation": ann_cat,
                        "parked": key in parked, "suggested": sugg, "confirmed": conf,
                        "determination": det, "employee_id": staff_emp.get(key),
                        "pid": pid, "group": group, "weak_staff": key in weak_staff,
                        "l2_in": (cam == "C05" and t["track"] in l2set),   # 🚪 this det is one of the raw L2 door-IN crossings
                        "demo": _demo_label(cropr)})   # age/gender estimate (None if no face)
    return out


def _best_crop(crops: list):
    """Highest-pixel thumbnail among a card's crops (area via the cached _crop_dims)."""
    best, barea = None, -1
    for c in crops:
        wd, ht = _crop_dims(c["crop"])
        a = (wd or 0) * (ht or 0)
        if a > barea:
            barea, best = a, c["crop"]
    return best or (crops[0]["crop"] if crops else None)


def _cards_from_dets(window: str) -> dict:
    """Staff AND customer cards DERIVED from the same det crops as the All-Detections buckets, in ONE pass over
    the window(s) — so both panels stay 1:1 with their bucket (incl. unassigned 'S?'), and the day load pays a
    single _classify_detections pass, not two. `window` may be one 'YYYY-MM-DD_HHMM' or a whole date.
    Each card carries every crop + the highest-resolution thumbnail; staff also get C05 'floor-visits'
    (consecutive door crossings paired in->out; interior cameras are sightings, NOT crossings)."""
    wins = _day_windows(window) if "_" not in window else [window]
    names = {e["id"]: e.get("name") for e in store.list_employees()}
    staff: dict = {}
    cust: dict = {}
    unassigned: list = []                                  # L2 door crossings the HITL REMOVED from customers (pass-by/duplicate/not-person)
    for w in wins:
        try:
            dets = _classify_detections(w, with_dims=False)
        except Exception:
            continue
        l2set = {x["track"] for x in _l2_crossings(w)}     # raw IN crossings = the door-entry candidate pool for THIS window
        for d in dets:
            pid, det = d.get("pid"), d.get("determination")
            sight = {"crop": d.get("crop"), "camera": d["camera"], "ist": d.get("ist"),
                     "track": d["track"], "window": w, "demo": d.get("demo"),
                     # provenance: a human allocation/confirmation = "human" (✋); a pure pipeline call = "ai" (🤖)
                     "source": "human" if (d.get("confirmed") or d.get("annotation")) else "ai"}
            # a raw door crossing a human took OUT of the customer group -> the customer 'unassigned' pile, reason-tagged
            # (staff-reclassified crossings live on the staff side; on-hold stays pending -> neither is 'unassigned')
            if det in ("passby", "not_person", "duplicate") and d["camera"] == "C05" and d["track"] in l2set:
                unassigned.append({**sight, "reason": det})
            if not pid:
                continue
            if det == "staff":
                g = staff.setdefault(pid, {"pid": pid, "employee_id": d.get("employee_id"),
                                           "name": names.get(d.get("employee_id")), "crops": []})
                g["crops"].append(sight)
            elif det == "customer":
                g = cust.setdefault(pid, {"pid": pid, "group": d.get("group"), "crops": []})
                if d.get("group") and not g.get("group"):
                    g["group"] = d["group"]
                g["crops"].append(sight)

    staff_cards = []
    for pid in sorted(staff):
        g = staff[pid]
        g["crops"].sort(key=lambda c: c.get("ist") or "")   # chronological cadence — interleave cameras by time
        g["best_crop"] = _best_crop(g["crops"])
        crossings = sorted([c for c in g["crops"] if c["camera"] == "C05"], key=lambda c: c["ist"] or "")
        visits = []                                             # C05 door crossings -> consecutive in->out floor-visits
        for i in range(0, len(crossings), 2):
            a = crossings[i]
            b = crossings[i + 1] if i + 1 < len(crossings) else None
            visits.append({"in": a["ist"], "out": (b["ist"] if b else None), "still_inside": b is None,
                           "in_crop": a.get("crop"), "out_crop": (b.get("crop") if b else None)})
        g["floor_visits"] = visits
        g["in_crop"] = (crossings[0].get("crop") if crossings else g["best_crop"])    # check-in proof: first door crossing
        g["out_crop"] = (visits[-1]["out_crop"] if visits else None)                  # check-out proof: last crossing OUT (None if still inside)
        g["last_crop"] = (g["crops"][-1].get("crop") if g["crops"] else None)
        g["n_visits"] = len(visits)
        g["n_crossings"] = len(crossings)
        g["n_sightings"] = len(g["crops"])
        staff_cards.append(g)

    def _cnum(p):                                               # sort #C numerically so C2 precedes C10
        return int(p[1:]) if p[1:].isdigit() else 1 << 30
    date = window.split("_")[0]
    _tags = _daily_tags(date)
    _mrg = _handoff_merges(date)["merge"]                               # C# -> its CANONICAL (un-merged) door entry, so exit/in lookups hit the KEPT track
    _xconf = _exit_confirmed(date)                                      # presumed exits a human has accepted
    rev = {}
    for (w_, ent_), n_ in _tags["cust"].items():
        if _mrg.get((w_, ent_), (w_, ent_)) == (w_, ent_):              # the canonical member (a fragment maps elsewhere)
            rev[n_] = (w_, ent_)
    _mmeta = _handoff_merges(date)["merge_meta"]                        # de-dup provenance for the PID drawer (Phase 3)
    _frag_of: dict = {}                                                 # canonical (win,track) -> [auto-merged fragment members]
    for (fw, ft), (cw, ct) in _mrg.items():
        mm = _mmeta.get((fw, ft), {})
        _frag_of.setdefault((cw, ct), []).append({"track": ft, "source": "auto", "gap": mm.get("gap"), "dist": mm.get("dist")})
    _human_of: dict = {}                                               # canonical (win,track) -> [human 'duplicate' members]
    for w_ in wins:
        for a_ in _window_annotations(w_):
            if a_.get("category") == "duplicate" and a_.get("duplicate_of") is not None and a_.get("camera") == "C05":
                _human_of.setdefault((w_, a_["duplicate_of"]), []).append({"track": a_["track"], "source": "human", "gap": None, "dist": None})
    cinfo: dict = {}
    customer_cards = []
    for pid in sorted(cust, key=_cnum):
        g = cust[pid]
        g["crops"].sort(key=lambda c: c.get("ist") or "")   # chronological cadence — interleave cameras by time
        g["best_crop"] = _best_crop(g["crops"])
        ists = sorted([c["ist"] for c in g["crops"] if c["ist"]])
        g["first_ist"] = ists[0] if ists else None
        g["last_ist"] = ists[-1] if ists else None
        g["cameras"] = sorted({c["camera"] for c in g["crops"]})
        g["n_sightings"] = len(g["crops"])
        bc = next((c for c in g["crops"] if c["crop"] == g["best_crop"]), None)   # demo: best-crop's, else first available
        g["demo"] = (bc and bc.get("demo")) or next((c["demo"] for c in g["crops"] if c.get("demo")), None)
        n = int(pid[1:]) if pid[1:].isdigit() else None         # door in/out/dwell from the SAME _entry_crop_info the report uses
        ent = rev.get(n)
        g["entry"] = list(ent) if ent else None
        if ent:
            ew, et = ent
            ci = cinfo.setdefault(ew, _entry_crop_info(ew)).get(et, {})
            rx = _exit_resolve(date).get((ew, et), {})         # L4: resolved exit (matched/presumed) when the visit had no OUT
            g["out_ist"] = ci.get("out_ist") or rx.get("out_ist")
            g["dwell_s"] = ci.get("dwell_s") if ci.get("dwell_s") is not None else rx.get("dwell_s")
            g["exit_src"] = rx.get("source")
            g["exit_confirmed"] = (ew, et) in _xconf
            g["in_ist"] = _tags["ist"].get((ew, et)) or next((c["ist"] for c in g["crops"] if c["window"] == ew and c["track"] == et), g["first_ist"])
            _mem = _frag_of.get((ew, et), []) + _human_of.get((ew, et), [])      # L3 provenance: door tracks merged into this PID
            g["dedup"] = {"canonical": et, "members": _mem, "n_merged": len(_mem)}
            g["in_crop"] = next((c["crop"] for c in g["crops"] if c["camera"] == "C05"), ci.get("crop")) or g["best_crop"]   # check-in proof: door crop
            g["out_crop"] = ci.get("out_crop") or rx.get("out_crop")             # check-out proof: real door-exit crop (None if exit was presumed)
            g["last_crop"] = (g["crops"][-1]["crop"] if g["crops"] else None)    # fallback when out_crop is None (last seen)
        else:
            g["in_ist"], g["out_ist"], g["dwell_s"] = g["first_ist"], None, None
            g["dedup"] = {"canonical": None, "members": [], "n_merged": 0}
            g["in_crop"] = next((c["crop"] for c in g["crops"] if c["camera"] == "C05"), None) or g["best_crop"]
            g["out_crop"] = None
            g["last_crop"] = (g["crops"][-1]["crop"] if g["crops"] else None)
        customer_cards.append(g)
    # 'unassigned' customers: one card per removed door crossing, reason-tagged, in arrival order, so the
    # customer section closes the SAME way staff does -> count = accepted #C + unassigned.
    unassigned.sort(key=lambda u: u.get("ist") or "")
    unassigned_cards = [{"pid": "C?", "track": u["track"], "window": u["window"], "camera": u["camera"],
                         "ist": u.get("ist"), "best_crop": u.get("crop"), "crop": u.get("crop"),
                         "reason": u["reason"], "source": u.get("source"), "demo": u.get("demo"),
                         "crops": [u]} for u in unassigned]
    return {"staff_cards": staff_cards, "customer_cards": customer_cards,
            "unassigned_cards": unassigned_cards}


def _staff_from_dets(window: str) -> list:
    """Back-compat shim: staff cards only (callers that also want customer cards should use _cards_from_dets)."""
    return _cards_from_dets(window)["staff_cards"]


@router.get("/crop")
def crop(path: str):
    if HIDE_FACES:  # only when explicitly hiding faces (unprotected public view)
        raise HTTPException(403, "faces are hidden on this view")
    safe = os.path.normpath(path).replace("\\", "/")
    if not safe.startswith("outputs/") or ".." in safe:
        raise HTTPException(403, "crops are served only from outputs/")
    if os.path.exists(safe):
        return FileResponse(safe)
    # no local file (e.g. on Cloud Run) -> proxy the image from private Supabase Storage
    from hitl import storage
    data = storage.fetch_crop(storage.key_for(safe))
    if data:
        return Response(content=data, media_type="image/jpeg")
    raise HTTPException(404, "crop not found")


@router.get("/frame")
def frame(window: str, camera: str, track: int):
    """The FULL source frame at a detection's time, with its trajectory overlaid (green start ->
    red end) — context for 'did this person ENTER or just walk PAST?'. L1 keeps person crops only,
    so the frame is extracted on-demand from the window's video slice."""
    if HIDE_FACES:
        raise HTTPException(403, "frames are hidden on this view")
    _safe_window(window)
    wdir = OUTPUTS / window
    try:
        wcfg = json.loads((wdir / "window.json").read_text(encoding="utf-8"))
    except Exception:
        raise HTTPException(404, "no such window")
    cam = (camera or "").upper()
    if cam == "C05":
        l1dir, slice_name = Path(wcfg["l1"]), "entry.ts"
    else:
        l1dir = next((Path(d) for d in wcfg.get("interior", []) if Path(d).name == f"L1_{cam}"), None)
        slice_name = f"{cam}.ts"
    if not l1dir:
        raise HTTPException(404, "camera not in this window")
    try:
        tj = json.loads((l1dir / "tracks.json").read_text(encoding="utf-8"))
    except Exception:
        raise HTTPException(404, "no tracks for this camera")
    trk = next((t for t in tj.get("tracks", []) if t["track"] == track), None)
    if not trk:
        raise HTTPException(404, "no such track")
    slice_path = wdir / "slices" / slice_name
    if not slice_path.exists():
        raise HTTPException(404, "source slice not available for this window")
    import datetime
    lbl = wcfg.get("label", "").split(" IST")[0]
    try:                                                          # slice starts at the window's start IST
        dt = datetime.datetime.strptime(lbl, "%Y-%m-%d %H:%M:%S")
        win_start = (dt - datetime.datetime(1970, 1, 1)).total_seconds() - 5.5 * 3600
    except Exception:
        win_start = trk["first_ts"]
    import cv2
    import numpy as np
    mid_ts = (trk["first_ts"] + trk.get("last_ts", trk["first_ts"])) / 2
    offset = max(0.0, mid_ts - win_start)
    pre = max(0.0, offset - 3.0)                                  # HEVC: fast-seek to a keyframe, then
    cmd = ["ffmpeg", "-nostdin", "-loglevel", "error", "-ss", f"{pre:.3f}", "-i", str(slice_path),
           "-ss", f"{offset - pre:.3f}", "-frames:v", "1", "-f", "image2pipe", "-vcodec", "mjpeg",
           "-q:v", "3", "pipe:1"]                                 # accurate-decode forward -> a clean frame
    proc = subprocess.run(cmd, capture_output=True)
    fr = cv2.imdecode(np.frombuffer(proc.stdout, np.uint8), cv2.IMREAD_COLOR) if proc.stdout else None
    if fr is None:
        raise HTTPException(500, "could not extract the frame")
    H, W = fr.shape[:2]
    pts = [(int(p[1] * W), int(p[2] * H)) for p in trk.get("traj", []) if len(p) >= 3]
    for i in range(1, len(pts)):
        cv2.line(fr, pts[i - 1], pts[i], (0, 200, 255), 2)        # path -> amber
    if pts:
        cv2.circle(fr, pts[0], 8, (0, 220, 0), -1)                # start -> green
        cv2.circle(fr, pts[-1], 8, (0, 0, 235), -1)              # end -> red
    if cam == "C05":                                              # draw the door-crossing line -> "did they cross it?" audit
        try:
            zcfg = json.loads(Path(wcfg["config"]).read_text(encoding="utf-8"))
            el = zcfg.get("entry_line") or []
            if len(el) >= 2:
                p0 = (int(el[0][0] * W), int(el[0][1] * H)); p1 = (int(el[1][0] * W), int(el[1][1] * H))
                cv2.line(fr, p0, p1, (255, 0, 0), 3)              # entry line -> bright blue
                cv2.putText(fr, "ENTRY LINE", (min(p0[0], p1[0]), max(18, min(p0[1], p1[1]) - 8)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2, cv2.LINE_AA)
        except Exception:
            pass
    ok, buf = cv2.imencode(".jpg", fr, [int(cv2.IMWRITE_JPEG_QUALITY), 82])
    return Response(content=buf.tobytes(), media_type="image/jpeg")


@router.post("/label")
def label(body: dict) -> dict:
    _guard_write()
    window = body.get("window", "")
    _window_dir(window)
    verdict = body.get("verdict", "")
    emp_id = body.get("employee_id")
    store.add_label(window, body.get("visit_id", ""), verdict, reason=body.get("reason", ""),
                    in_track=body.get("in_track"), out_track=body.get("out_track"), employee_id=emp_id)
    # marking a track as staff overrides any earlier "not staff" on it (re-tag after a ✗ reject)
    if verdict == "employee" and body.get("in_track") is not None:
        store.add_label(window, f"notstaff-{body['in_track']}", "reset", reason="re-marked staff",
                        in_track=body.get("in_track"))
    if verdict == "employee" and emp_id and body.get("crop"):
        _enroll_from_cache(emp_id, body["crop"], window, body.get("in_track"))  # crop embedding -> gallery
    _rerun_l4(window)  # learning: re-route matches for free, then hand back the new result
    return visits(window)


@router.post("/unlabel")
def unlabel(body: dict) -> dict:
    """Undo a click: append a 'reset' label (latest wins) so the prior verdict no longer applies,
    then re-run. Nothing is ever destroyed — the history shows mark + undo."""
    _guard_write()
    window = body.get("window", "")
    _window_dir(window)
    store.add_label(window, body.get("visit_id", ""), "reset", reason="undo",
                    in_track=body.get("in_track"), out_track=body.get("out_track"))
    _rerun_l4(window)
    return visits(window)


@router.post("/link")
def link(body: dict) -> dict:
    """Human says an open IN and a pre-exit OUT are the same person -> must_link (recovers a
    missed visit). visit_id is synthesized from the two tracks."""
    _guard_write()
    window = body.get("window", "")
    _window_dir(window)
    it, ot = body.get("in_track"), body.get("out_track")
    if it is None or ot is None:
        raise HTTPException(400, "link needs in_track and out_track")
    store.add_label(window, f"{it}-{ot}", "confirm", reason=body.get("reason", "human link"),
                    in_track=it, out_track=ot)
    _rerun_l4(window)
    return visits(window)


@router.post("/unstaff-track")
def unstaff_track(body: dict) -> dict:
    """Per-track correction: the human says ONE track in a staff group is NOT that staffer (a customer
    the gallery mis-matched, or a wrong manual mark). Records a 'notstaff-<track>' reject -> L4 keeps
    that track a customer and the staff overlay drops it; the rest of the group is untouched."""
    _guard_write()
    window = body.get("window", "")
    _window_dir(window)
    track = body.get("track")
    if track is None:
        raise HTTPException(400, "unstaff-track needs a track")
    verdict = "reset" if body.get("restore") else "reject"      # restore=true undoes a prior removal (latest-wins)
    store.add_label(window, f"notstaff-{track}", verdict, reason="not staff", in_track=track)
    _rerun_l4(window)
    return visits(window)


_EMB_CACHE = {"data": None, "mtime": None}   # the OSNet emb map, loaded ONCE (mtime-invalidated)


def _load_emb_cache() -> dict:
    """Load + normalise outputs/osnet_emb_cache.pkl once, cached by mtime. It's MBs — re-reading + re-normalising
    it on every /allocate is what made staff assignment slow (twice per call: emb + enroll)."""
    p = "outputs/osnet_emb_cache.pkl"
    try:
        mt = os.path.getmtime(p)
    except Exception:
        return {}
    if mt != _EMB_CACHE["mtime"]:
        import pickle
        try:
            with open(p, "rb") as f:
                raw = pickle.load(f)
            _EMB_CACHE["data"] = {str(k).replace("\\", "/"): v for k, v in raw.items()}
        except Exception:
            _EMB_CACHE["data"] = {}
        _EMB_CACHE["mtime"] = mt
    return _EMB_CACHE["data"] or {}


def _emb_for_crop(crop):
    """OSNet embedding for a crop from the cached L4 emb map (no GPU); None if absent."""
    return _load_emb_cache().get(str(crop).replace("\\", "/")) if crop else None


# ===== training: gallery + threshold rebuild (learning-free, no GPU) ===============
def _cache_norm() -> dict:
    """Load the OSNet embedding cache ONCE with normalized keys (for many lookups)."""
    import pickle
    p = "outputs/osnet_emb_cache.pkl"
    if not os.path.exists(p):
        return {}
    try:
        with open(p, "rb") as f:
            cache = pickle.load(f)
    except Exception:
        return {}
    return {str(k).replace("\\", "/"): v for k, v in cache.items()}


def _l1_crop(window: str, camera: str, track) -> str | None:
    sub = "L1_entry" if (camera or "").upper() == "C05" else f"L1_{camera}"
    p = OUTPUTS / window / sub / "crops" / f"trk_{int(track):04d}.jpg"
    return str(p) if p.exists() else None


def _tune_threshold(by_emp: dict):
    """Suggest a staff-sim cutoff that separates same-employee (positive) from cross-employee
    (negative) cosine sims — midpoint of the hardest positive (5th pct) and hardest negative (95th
    pct), clamped to a safe band. Falls back to 0.6 when there isn't enough data."""
    import itertools
    import numpy as np
    emps = [(e, v) for e, v in by_emp.items() if v]
    pos, neg = [], []
    for _, embs in emps:
        for a, b in itertools.combinations(embs, 2):
            pos.append(float(np.dot(a, b)))
    for i in range(len(emps)):
        for j in range(i + 1, len(emps)):
            for a in emps[i][1]:
                for b in emps[j][1]:
                    neg.append(float(np.dot(a, b)))
    if len(pos) < 3 or len(neg) < 3:
        return 0.6, len(pos), len(neg)            # too little data -> keep the safe hand-picked default
    pos.sort(); neg.sort()
    hard_pos = pos[max(0, int(0.05 * len(pos)) - 1)]          # hardest same-employee pair (5th pct)
    hard_neg = neg[min(len(neg) - 1, int(0.95 * len(neg)))]   # hardest cross-employee pair (95th pct)
    mid = (hard_pos + hard_neg) / 2.0 if hard_pos > hard_neg else 0.6   # overlap -> not separable -> default
    # safe band: floor 0.55 guards against a CUSTOMER matching a staffer (the tuning pairs are staff-only,
    # so they can't see that risk); cap 0.72 keeps genuinely-present staff recognizable.
    thr = round(max(0.55, min(0.72, mid)), 2)
    return thr, len(pos), len(neg)


def _train_rebuild(date=None) -> dict:
    """Learning-free 'training': fold every human-confirmed staff crop into the gallery (more
    reference shots per employee -> sturdier auto-recognition) + re-tune the match threshold, then
    register an active model version. Uses cached embeddings only — no GPU, safe during processing."""
    import numpy as np
    store_id = "s14"
    sightings = store.confirmed_staff(store_id, date)
    cache = _cache_norm()
    have = store.gallery_sources(store_id)
    enrolled = 0
    for s in sightings:
        key = (s["window"], s["track"])
        if key in have:
            continue
        emb = None
        if s.get("embedding"):
            emb = np.asarray(s["embedding"], dtype="float32")
        else:
            crop = s.get("crop_url") or _l1_crop(s["window"], s["camera"], s["track"])
            v = cache.get(str(crop).replace("\\", "/")) if crop else None
            if v is not None:
                emb = np.asarray(v, dtype="float32")
        if emb is None:
            continue
        store.enroll_staff(s["employee_id"], store_id, emb,
                           crop_url=s.get("crop_url"), window=s["window"], track=s["track"])
        have.add(key)
        enrolled += 1
    by_emp = {}
    for g in store.get_gallery(store_id):                      # tune from the full post-enroll gallery
        try:
            by_emp.setdefault(g["employee_id"], []).append(np.asarray(g["embedding"], dtype="float32"))
        except Exception:
            pass
    thr, n_pos, n_neg = _tune_threshold(by_emp)
    n_emb = sum(len(v) for v in by_emp.values())
    params = {"staff_sim": thr, "employees": len(by_emp), "embeddings": n_emb,
              "enrolled_new": enrolled, "pos_pairs": n_pos, "neg_pairs": n_neg, "scope": date or "all-time"}
    ver = store.register_model_version(
        "gallery", params, trained_on=len(sightings),
        notes=f"gallery rebuild: {len(sightings)} confirmed sightings, +{enrolled} new embeddings"
              + (f" ({date})" if date else ""), active=True)
    return {"ok": True, "version": ver, "params": params,
            "summary": f"v{ver}: {len(by_emp)} staff · {n_emb} reference embeddings (+{enrolled} new) · "
                       f"threshold {thr} (from {n_pos} same / {n_neg} cross-employee pairs)"}


def _train_finetune(date=None) -> dict:
    """Launch the deep OSNet fine-tune as a detached background job; the tab polls /train-status/{job}.
    The job itself checks prerequisites (torchreid + enough crops + a free GPU) and reports honestly."""
    import time as _t
    job_id = "ft_" + _t.strftime("%Y%m%d_%H%M%S")
    jobf = OUTPUTS / "train_jobs" / f"{job_id}.json"
    jobf.parent.mkdir(parents=True, exist_ok=True)
    jobf.write_text(json.dumps({"status": "queued", "kind": "finetune", "job": job_id,
                                "progress": 0, "message": "starting…"}), encoding="utf-8")
    cmd = [sys.executable, "-m", "training.finetune_osnet", "--job", str(jobf)]
    if date:
        cmd += ["--date", date]
    subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)  # detached
    return {"ok": True, "job": job_id, "mode": "finetune",
            "message": "deep fine-tune started — watch progress in the Training tab"}


def _train_triplet(date=None) -> dict:
    """Launch contrastive/triplet fine-tune on human-confirmed same/different pairs.
    Exports the dataset first (to ensure pairs.jsonl is current), then fine-tunes."""
    import time as _t
    job_id = "tpl_" + _t.strftime("%Y%m%d_%H%M%S")
    jobf = OUTPUTS / "train_jobs" / f"{job_id}.json"
    jobf.parent.mkdir(parents=True, exist_ok=True)
    jobf.write_text(json.dumps({"status": "queued", "kind": "triplet", "job": job_id,
                                "progress": 0, "message": "starting…"}), encoding="utf-8")
    data_dir = str(Path("training_data") / (date or "all"))
    # Export dataset first (fast, no GPU), then fine-tune
    export_cmd = [sys.executable, "-m", "training.export_dataset", "--out", "training_data"]
    if date:
        export_cmd += ["--date", date]
    triplet_cmd = [sys.executable, "-m", "training.finetune_triplet",
                   "--data", data_dir, "--job", str(jobf)]

    def _run():
        import subprocess as _sp
        _sp.run(export_cmd, stdout=_sp.DEVNULL, stderr=_sp.DEVNULL)
        _sp.Popen(triplet_cmd, stdout=_sp.DEVNULL, stderr=_sp.DEVNULL)

    import threading
    threading.Thread(target=_run, daemon=True).start()
    return {"ok": True, "job": job_id, "mode": "triplet",
            "message": "contrastive fine-tune started — watch progress in the Training tab"}


@router.post("/train")
def train(body: dict) -> dict:
    """Initiate training. mode='rebuild' = gallery + threshold from confirmed labels (learning-free,
    instant, no GPU). mode='finetune' = deep OSNet identity fine-tune (background, needs GPU).
    mode='triplet' = contrastive fine-tune on human same/different pairs (background, needs GPU)."""
    _guard_write()
    mode = (body or {}).get("mode", "rebuild")
    date = (body or {}).get("date") or None
    if mode == "rebuild":
        return _train_rebuild(date)
    if mode == "finetune":
        return _train_finetune(date)
    if mode == "triplet":
        return _train_triplet(date)
    raise HTTPException(400, "mode must be 'rebuild', 'finetune', or 'triplet'")


@router.get("/train-status/{job}")
def train_status(job: str) -> dict:
    """Live progress for a background fine-tune job."""
    if "/" in job or "\\" in job or ".." in job:
        raise HTTPException(400, "bad job id")
    jobf = OUTPUTS / "train_jobs" / f"{job}.json"
    if not jobf.exists():
        raise HTTPException(404, "no such job")
    return json.loads(jobf.read_text(encoding="utf-8"))


@router.get("/training")
def training() -> dict:
    """Training tab: active model, version history, the current gallery snapshot, published periods."""
    from collections import Counter
    gal = store.get_gallery("s14")
    per = dict(Counter(g["employee_id"] for g in gal))
    rank = _rank_labels()
    return {"active": store.active_model(),
            "versions": store.list_model_versions(20),
            "gallery": {"employees": len(per), "embeddings": len(gal),
                        "per_employee": [{"employee": rank.get(e) or f"#{e}", "crops": n}
                                         for e, n in sorted(per.items())]},
            "published": store.list_published()}


def _reid_decided(date: str) -> dict:
    """All ReID pair verdicts: {pair_key: "same"|"different"}.
    A revoked verdict (verdict="reset") removes the key so the pair is undecided again."""
    out = {}
    for win in _day_windows(date):
        for l in _window_labels(win):
            vid = str(l.get("visit_id", ""))
            v = str(l.get("verdict", ""))
            if vid.startswith("reiddiff-"):
                if v == "reset":
                    out.pop(vid, None)
                else:
                    out[vid] = "different"
            elif vid.startswith("reidsame-"):
                if v == "reset":
                    out.pop(vid.replace("reidsame-", "reiddiff-", 1), None)
                    out.pop(vid, None)
        for a in _window_annotations(win):
            if a.get("category") == "duplicate" and a.get("duplicate_of") is not None:
                bt = a.get("track")
                at = a.get("duplicate_of")
                bw = win
                key = f"reiddiff-{at}-{bw}-{bt}"
                out[key] = "same"
    return out


@router.get("/reid/candidates")
def reid_candidates(date: str, threshold: float = 0.62) -> dict:
    """ReID trainer: 'same person?' candidate pairs among customers (returning-visitor de-dup), ranked by
    OSNet-embedding similarity (a customer's representative = mean of its embedded crops). Each pair gives
    both crops + #C + the entry (window,track) so a tick can MERGE and a cross can mark DISTINCT."""
    if not date or "/" in date or "\\" in date or ".." in date:
        raise HTTPException(400, "bad date")
    import numpy as np
    cards = _cards_from_dets(date)["customer_cards"]
    cache = _load_emb_cache()

    def rep(c):
        vs = [np.asarray(cache[k], dtype="float32") for s in c.get("crops", [])
              for k in [str(s.get("crop") or "").replace("\\", "/")] if k in cache]
        if not vs:
            return None
        m = np.mean(vs, axis=0); n = float(np.linalg.norm(m))
        return (m / n) if n > 0 else None

    reps = [(c, rep(c)) for c in cards]
    reps = [(c, v) for c, v in reps if v is not None]
    decided = _reid_decided(date)
    out = []
    for i in range(len(reps)):
        ca, va = reps[i]
        ea = ca.get("entry") or [None, None]
        for j in range(i + 1, len(reps)):
            cb, vb = reps[j]
            sim = float(np.dot(va, vb))
            if sim < threshold:
                continue
            eb = cb.get("entry") or [None, None]
            pair_key = f"reiddiff-{ea[1]}-{eb[0]}-{eb[1]}"
            verdict = decided.get(pair_key)
            out.append({"a_pid": ca["pid"], "a_crop": ca.get("best_crop"), "a_window": ea[0], "a_track": ea[1],
                        "a_demo": ca.get("demo"), "a_in": ca.get("in_ist"), "a_group": ca.get("group"),
                        "b_pid": cb["pid"], "b_crop": cb.get("best_crop"), "b_window": eb[0], "b_track": eb[1],
                        "b_demo": cb.get("demo"), "b_in": cb.get("in_ist"), "b_group": cb.get("group"),
                        "sim": round(sim, 3), "source": "reid", "verdict": verdict})
    out.sort(key=lambda x: -x["sim"])
    return {"candidates": out, "embedded_customers": len(reps), "total_customers": len(cards),
            "threshold": threshold}


def _parse_xcam(s: str):
    """'<door_track>-<cam>-<interior_track>' -> (door_track:int, cam:str, interior_track:int)."""
    try:
        at, cam, bt = s.split("-")
        return int(at), cam, int(bt)
    except ValueError:
        return None


def _xcam_decided(date: str) -> dict:
    """(window, door_track, cam, interior_track) -> 'same'|'different' from stored cross-camera verdicts
    (latest-wins: a 'reset' clears it). xcamsame-/xcamdiff- labels carry the door track + interior cam/track."""
    out: dict = {}
    for win in _day_windows(date):
        for l in _window_labels(win):
            vid = str(l.get("visit_id", "")); v = l.get("verdict")
            for pfx, val, keep in (("xcamsame-", "same", "confirm"), ("xcamdiff-", "different", "reject")):
                if vid.startswith(pfx):
                    p = _parse_xcam(vid[len(pfx):])
                    if not p:
                        continue
                    k = (win, *p)
                    if v == "reset":
                        out.pop(k, None)
                    elif v == keep:
                        out[k] = val
    return out


@router.get("/reid/cross-camera/{date}")
def reid_cross_camera(date: str, threshold: float = 0.55) -> dict:
    """Cross-camera pairs: VALIDATE the bridge's door<->interior associations (the time pairing it already
    did, in visits.json `links`) and DISCOVER interior (C11/C14) crops that look like a door customer but
    weren't auto-linked. Each pair = a door crop (C05) <-> an interior crop, with the door time, interior
    time, the time GAP, the OSNet sim, the owning #C, and any stored verdict. The trainer's tick links/keeps
    the association; cross un-links it (mis-association -> the interior crop drops back to 'inside')."""
    if not date or "/" in date or "\\" in date or ".." in date:
        raise HTTPException(400, "bad date")
    import numpy as np
    cache = _load_emb_cache()
    tags = _daily_tags(date)
    cards = _cards_from_dets(date)["customer_cards"]

    def rep(c):                                                # a #C's representative = mean of ALL its embedded crops (every camera)
        vs = [np.asarray(cache[k], dtype="float32") for s in c.get("crops", [])
              for k in [str(s.get("crop") or "").replace("\\", "/")] if k in cache]
        if not vs:
            return None
        m = np.mean(vs, axis=0); n = float(np.linalg.norm(m))
        return (m / n) if n > 0 else None

    pid_rep, pid_entry, pid_doorcrop, pid_inist = {}, {}, {}, {}
    for c in cards:
        r = rep(c)
        if r is None:
            continue
        pid_rep[c["pid"]] = r
        pid_entry[c["pid"]] = c.get("entry")
        pid_doorcrop[c["pid"]] = c.get("in_crop") or c.get("best_crop")
        pid_inist[c["pid"]] = c.get("in_ist")
    decided = _xcam_decided(date)

    def secs(t):
        try:
            h, m, s = map(int, str(t).split(":")); return h * 3600 + m * 60 + s
        except Exception:
            return None

    def gap(a, b):
        sa, sb = secs(a), secs(b)
        return abs(sa - sb) if sa is not None and sb is not None else None

    DISC_DEFAULT_DWELL = 900                                    # no clean exit -> bound discovery to 15 min after entry
    pid_win = {}                                               # #C -> (in_secs, end_secs): a real interior sighting must fall in here
    for c in cards:
        if c["pid"] not in pid_rep:
            continue
        i = secs(c.get("in_ist"))
        if i is None:
            continue
        end = secs(c.get("out_ist"))
        if end is None:
            end = i + int(c["dwell_s"]) if c.get("dwell_s") else i + DISC_DEFAULT_DWELL
        pid_win[c["pid"]] = (i, end)

    done, cand, seen, crop_key, ist_key = [], [], set(), {}, {}
    for win in _day_windows(date):
        try:
            vj = json.loads((OUTPUTS / win / "visits.json").read_text(encoding="utf-8"))
        except Exception:
            vj = {}
        try:
            dets = _classify_detections(win, with_dims=False)
        except Exception:
            dets = []
        det_ist = {(d["camera"], d["track"]): d.get("ist") for d in dets}
        for d in dets:                                         # remember every det's crop+time so a decided pair can be re-shown after it leaves the pool
            crop_key[(win, d["camera"], d["track"])] = (d.get("crop") or "").replace("\\", "/")
            ist_key[(win, d["camera"], d["track"])] = d.get("ist")
        win_pids = [p for p in pid_rep if pid_entry.get(p) and pid_entry[p][0] == win]    # #C whose door entry is THIS window
        for l in vj.get("links", []):                          # --- the bridge's DONE cross-camera associations -> validate
            at, cam, bt = l.get("in_track"), l.get("cam"), l.get("track")
            seen.add((win, cam, bt))
            n = tags["cust"].get((win, at))
            iist = det_ist.get((cam, bt))
            done.append({"kind": "done", "pid": (f"C{n}" if n else None), "window": win, "a_track": at,
                         "cam": cam, "b_track": bt, "door_crop": (l.get("door_crop") or "").replace("\\", "/"),
                         "int_crop": (l.get("int_crop") or "").replace("\\", "/"), "door_ist": l.get("in_ist"),
                         "int_ist": iist, "gap_s": gap(iist, l.get("in_ist")), "sim": l.get("sim"),
                         "source": l.get("source"), "verdict": decided.get((win, at, cam, bt))})
        if not win_pids:
            continue
        for d in dets:                                         # --- DISCOVER: unlinked interior crops that match a door #C
            if d["camera"] not in ("C11", "C14") or d.get("determination") not in ("inside", "to_review"):
                continue
            if (win, d["camera"], d["track"]) in seen:
                continue
            ev = cache.get(str(d.get("crop") or "").replace("\\", "/"))
            if ev is None:
                continue
            ev = np.asarray(ev, dtype="float32")
            isecs = secs(d.get("ist"))
            best = None
            for p in win_pids:
                w = pid_win.get(p)                             # the interior sighting must fall within THIS person's visit (in -> out)
                if w and isecs is not None and not (w[0] - 30 <= isecs <= w[1] + 120):
                    continue
                sim = float(np.dot(pid_rep[p], ev))
                if sim >= threshold and (best is None or sim > best[1]):
                    best = (p, sim)
            if not best:
                continue
            p, sim = best
            at = pid_entry[p][1]
            cand.append({"kind": "candidate", "pid": p, "window": win, "a_track": at, "cam": d["camera"],
                         "b_track": d["track"], "door_crop": pid_doorcrop.get(p),
                         "int_crop": (d.get("crop") or "").replace("\\", "/"), "door_ist": pid_inist.get(p),
                         "int_ist": d.get("ist"), "gap_s": gap(d.get("ist"), pid_inist.get(p)),
                         "sim": round(sim, 3), "source": "discover", "verdict": decided.get((win, at, d["camera"], d["track"]))})
    emitted = {(x["window"], x["a_track"], x["cam"], x["b_track"]) for x in done + cand}
    for (win, at, cam, bt), v in decided.items():             # a ✓-linked candidate leaves the 'inside' pool -> re-add it so the verdict stays visible + undoable
        if (win, at, cam, bt) in emitted:
            continue
        n = tags["cust"].get((win, at)); pid = f"C{n}" if n else None
        done.append({"kind": "linked", "pid": pid, "window": win, "a_track": at, "cam": cam, "b_track": bt,
                     "door_crop": crop_key.get((win, "C05", at)) or (pid_doorcrop.get(pid) if pid else None),
                     "int_crop": crop_key.get((win, cam, bt)), "door_ist": (pid_inist.get(pid) if pid else None),
                     "int_ist": ist_key.get((win, cam, bt)), "gap_s": None, "sim": None, "source": "human", "verdict": v})
    cand.sort(key=lambda x: -x["sim"])
    per, capped = {}, []                                       # keep the best 3 interior look-alikes per #C -> a reviewable list
    for x in cand:
        if per.get(x["pid"], 0) >= 3:
            continue
        per[x["pid"]] = per.get(x["pid"], 0) + 1
        capped.append(x)
    return {"done": done, "candidates": capped, "threshold": threshold,
            "n_done": len(done), "n_candidates": len(capped)}


@router.post("/reid/cross-decision")
def reid_cross_decision(body: dict) -> dict:
    """Validate ONE cross-camera pair. same=True LINKS the interior crop to the door #C (a 'same' training
    pair); same=False UN-LINKS it -> the interior crop drops back to 'inside' (a 'different' training pair).
    Both are same-window (door + interior in one hour). The customer COUNT is untouched (footfall = C05)."""
    _guard_write()
    win, at, cam, bt = body.get("window"), body.get("a_track"), body.get("cam"), body.get("b_track")
    same = bool(body.get("same"))
    if win is None or at is None or not cam or bt is None:
        raise HTTPException(400, "cross-decision needs window + a_track + cam + b_track")
    _window_dir(win)
    if same:
        store.add_label(win, f"xcamsame-{at}-{cam}-{bt}", "confirm", in_track=at, reason=f"xcam-link:{cam}:{bt}")
        store.add_label(win, f"xcamdiff-{at}-{cam}-{bt}", "reset", in_track=at)        # clear any prior 'different'
    else:
        store.add_label(win, f"xcamdiff-{at}-{cam}-{bt}", "reject", in_track=at, reason=f"xcam-unlink:{cam}:{bt}")
        store.add_label(win, f"xcamsame-{at}-{cam}-{bt}", "reset", in_track=at)        # clear any prior 'same'
    return {"ok": True, "stale": True}                         # affects _classify -> the day re-renders


@router.post("/reid/cross-revoke")
def reid_cross_revoke(body: dict) -> dict:
    """Undo a cross-camera verdict (append-only reset, latest-wins) -> the pair returns to unvalidated."""
    _guard_write()
    win, at, cam, bt = body.get("window"), body.get("a_track"), body.get("cam"), body.get("b_track")
    if win is None or at is None or not cam or bt is None:
        raise HTTPException(400, "cross-revoke needs window + a_track + cam + b_track")
    _window_dir(win)
    store.add_label(win, f"xcamsame-{at}-{cam}-{bt}", "reset", in_track=at)
    store.add_label(win, f"xcamdiff-{at}-{cam}-{bt}", "reset", in_track=at)
    return {"ok": True, "stale": True}


@router.post("/reid/decision")
def reid_decision(body: dict) -> dict:
    """ReID trainer verdict on a candidate pair. same=True MERGES b into a (b -> 'duplicate' annotation:
    drops from the count + a 'same' training pair). same=False records a durable 'reiddiff-' label so the
    pair never re-suggests (a 'different' training pair). Both feed the future OSNet fine-tune."""
    _guard_write()
    aw, at = body.get("a_window"), body.get("a_track")
    bw, bt = body.get("b_window"), body.get("b_track")
    same = bool(body.get("same"))
    if at is None or bt is None:
        raise HTTPException(400, "reid-decision needs a + b tracks")
    date = (aw or bw or "").split("_")[0]
    pair_key = f"reiddiff-{at}-{bw}-{bt}"
    existing = _reid_decided(date).get(pair_key) if date else None
    if existing == ("same" if same else "different"):
        return {"ok": True, "stale": False, "already": True}
    if same:
        _window_dir(bw)
        _alloc_one(bw, body.get("camera", "C05"), bt, "duplicate", body.get("b_crop"), None, duplicate_of=at)
    else:
        _window_dir(aw)
        store.add_label(aw, f"reiddiff-{at}-{bw}-{bt}", "reject", in_track=at, reason=f"reid-distinct:{bw}:{bt}")
    return {"ok": True, "stale": same}


@router.post("/reid/revoke")
def reid_revoke(body: dict) -> dict:
    """Undo a ReID verdict. same=True un-merges b (re-annotate it 'customer' -> latest_annotations wins over the
    'duplicate', + reset the false-<track> label -> b is a customer again, count restored). same=False resets the
    'reiddiff-' label so the pair re-suggests. Append-only + latest-wins, so nothing is destroyed (history shows mark+undo)."""
    _guard_write()
    aw, at = body.get("a_window"), body.get("a_track")
    bw, bt = body.get("b_window"), body.get("b_track")
    same = bool(body.get("same"))
    if at is None or bt is None:
        raise HTTPException(400, "reid-revoke needs a + b tracks")
    if same:
        _window_dir(bw)
        store.add_annotation(bw, body.get("camera", "C05"), bt, "customer",
                             crop_url=body.get("b_crop"), embedding=_emb_for_crop(body.get("b_crop")))
        store.add_label(bw, f"false-{bt}", "reset", in_track=bt)
        store.add_label(bw, f"reidsame-{at}-{bw}-{bt}", "reset", in_track=at)
    else:
        _window_dir(aw)
        store.add_label(aw, f"reiddiff-{at}-{bw}-{bt}", "reset", in_track=at)
    return {"ok": True, "stale": same}


@router.get("/reid/matches")
def reid_matches(date: str, pid: str, threshold: float = None, cap: int = 80) -> dict:
    """L3 strengthening: every crop in the day the TRAINED matcher says looks like THIS person
    (>= the calibrated reid_threshold from the active model), tagged with its owner #C/#S — the PID
    drawer's look-alike strip. Confirming a customer match merges the PIDs (same /reid/decision flow)."""
    if not date or "/" in date or "\\" in date or ".." in date:
        raise HTTPException(400, "bad date")
    import numpy as np
    if threshold is None:
        params = (store.active_model() or {}).get("params", {}) or {}
        threshold = float(params.get("reid_threshold", 0.62))   # the trained cut, not a guess
    cards = _cards_from_dets(date)
    pool = cards["customer_cards"]                             # HARD RULE: tagged staff are a KNOWN identity -> never candidates
    me = next((c for c in pool + cards["staff_cards"] if str(c.get("pid")) == str(pid)), None)
    if me is None:
        raise HTTPException(404, "no such pid")
    cache = _load_emb_cache()
    mine = [np.asarray(cache[k], dtype="float32") for s in me.get("crops", [])
            for k in [str(s.get("crop") or "").replace("\\", "/")] if k in cache]
    if not mine:
        return {"pid": pid, "threshold": round(threshold, 3), "matches": [],
                "note": "no embedded crops for this PID yet"}
    rep = np.mean(mine, axis=0)
    n = float(np.linalg.norm(rep))
    rep = rep / n if n > 0 else rep                            # vectors are L2-normalized -> plain dot = cosine
    me_ent = me.get("entry") or [None, None]
    decided = _reid_decided(date)
    is_staff = str(pid).startswith("S")
    staff_decided = _staff_marks(date) if is_staff else {}     # so a re-opened staff drawer shows already-marked candidates
    me_emp = me.get("employee_id")
    best = {}                                                  # one row per OTHER customer: their single best-matching crop
    for c in pool:
        if str(c.get("pid")) == str(pid):
            continue
        ent = c.get("entry") or [None, None]
        for s in c.get("crops", []):
            v = cache.get(str(s.get("crop") or "").replace("\\", "/"))
            if v is None:
                continue
            sim = float(np.dot(rep, np.asarray(v, dtype="float32")))
            if sim < threshold:
                continue
            prev = best.get(c["pid"])
            if prev is None or sim > prev["sim"]:
                if is_staff:
                    sv = staff_decided.get((me_emp, ent[0], ent[1]))
                    verdict = sv["label"] if sv else None
                else:
                    verdict = decided.get(f"reiddiff-{me_ent[1]}-{ent[0]}-{ent[1]}")
                best[c["pid"]] = {"kind": "customer", "pid": c["pid"], "name": c.get("name"),
                                  "crop": s.get("crop"), "camera": s.get("camera"), "ist": s.get("ist"),
                                  "window": ent[0], "track": ent[1], "sim": round(sim, 3),
                                  "verdict": verdict}
    out = sorted(best.values(), key=lambda x: -x["sim"])
    return {"pid": pid, "threshold": round(threshold, 3), "matches": out[:cap], "total": len(out)}


def _staff_marks(date: str) -> dict:
    """Per-PID staff look-alike verdicts from the drawer's '🎯 Train on look-alikes for #S' strip:
    {(emp,window,track): {label:'same'|'different', staff_crop}}. staffsame-/staffdiff-<emp>-<track>;
    verdict='reset' undoes (latest-wins). The candidate (window,track) is a customer door entry."""
    out: dict = {}
    for win in _day_windows(date):
        for l in _window_labels(win):
            vid = str(l.get("visit_id", ""))
            if not (vid.startswith("staffsame-") or vid.startswith("staffdiff-")):
                continue
            emp, it = l.get("employee_id"), l.get("in_track")
            if emp is None or it is None:
                continue
            key = (emp, win, it)
            if str(l.get("verdict", "")) == "reset":
                out.pop(key, None)
            else:
                out[key] = {"label": "same" if vid.startswith("staffsame-") else "different",
                            "staff_crop": (l.get("reason") or "").replace("\\", "/")}
    return out


@router.post("/reid/staff-mark")
def reid_staff_mark(body: dict) -> dict:
    """Record a per-PID staff look-alike verdict (drawer): same → this candidate IS the staff (also enrolls
    into the gallery for auto-recognition); different → it is NOT. Counts as a ReID marking (pending training),
    shows in 'My markings', undoable. revoke=true clears it (latest-wins reset)."""
    _guard_write()
    emp, window, track = body.get("employee_id"), body.get("window", ""), body.get("track")
    if emp is None or track is None:
        raise HTTPException(400, "reid/staff-mark needs employee_id + track")
    _window_dir(window)
    camera = body.get("camera") or "C05"
    crop = body.get("crop")
    if body.get("revoke"):
        store.add_label(window, f"staffsame-{emp}-{track}", "reset", in_track=track, employee_id=emp)
        store.add_label(window, f"staffdiff-{emp}-{track}", "reset", in_track=track, employee_id=emp)
        store.add_annotation(window, camera, track, "customer")   # latest-wins: overrides the staff annotation -> back to a customer
        _rerun_l4(window)
        return {"ok": True, "stale": True}
    staff_crop = (body.get("staff_crop") or "").replace("\\", "/")
    if body.get("same"):
        store.add_label(window, f"staffsame-{emp}-{track}", "confirm", reason=staff_crop, in_track=track, employee_id=emp)
        # ✓ = this IS the staff member: a staff annotation reclassifies the crop into the Staff bucket (cleanly
        # revertible via a customer annotation), + enroll the embedding for future-day auto-recognition.
        store.add_annotation(window, camera, track, "staff", crop_url=crop, employee_id=emp, embedding=_emb_for_crop(crop))
        if crop:
            _enroll_from_cache(emp, crop, window, track)
        _rerun_l4(window)                                         # so it lands in the Staff group now
    else:
        store.add_label(window, f"staffdiff-{emp}-{track}", "reject", reason=staff_crop, in_track=track, employee_id=emp)
    return {"ok": True, "stale": True}


@router.post("/reid/reject-exit")
def reject_exit(body: dict) -> dict:
    """Human: 'this check-out is the WRONG person' (the Review Queue's visit_exit ✗). Writes a durable
    `rejexit-<intrack>-<outwin>-<outtrack>` reject so _exit_resolve never re-matches that (entry,out) pair ->
    the wrong OUT frees + the entry falls to a presumed exit. revoke=true clears it (latest-wins reset).
    Mirrors /reid/cross-decision + /unstaff-track. _guard_write bumps _STATE_VERSION -> the exit memo re-runs."""
    _guard_write()
    if body.get("same") is True:                               # ✓ 'check-out is correct' -> nothing to persist (dismiss only)
        return {"ok": True, "stale": False}
    win, track = body.get("window", ""), body.get("track")
    ow, ot = body.get("out_window", ""), body.get("out_track")
    if track is None or ot is None:
        raise HTTPException(400, "reject-exit needs window+track (entry) and out_window+out_track")
    _window_dir(win)
    verdict = "reset" if body.get("revoke") else "reject"
    store.add_label(win, f"rejexit-{track}-{ow}-{ot}", verdict, in_track=track,
                    reason=f"{'undo wrong-exit' if body.get('revoke') else 'wrong-exit'}:{ow}:{ot}")
    return {"ok": True, "stale": True}


@router.get("/reid/markings")
def reid_markings(date: str) -> dict:
    """All human-confirmed ReID pair markings for the markings log in the Logs tab.
    Returns crop URLs + label + similarity + undo keys so the frontend can display + undo each one."""
    if not date or "/" in date or "\\" in date or ".." in date:
        raise HTTPException(400, "bad date")
    cache = _load_emb_cache()
    cinfo: dict = {}

    def _crop(win, track):
        info = cinfo.setdefault(win, _entry_crop_info(win))
        c = info.get(int(track) if str(track).isdigit() else track)
        return (c or {}).get("crop", "") if isinstance(c, dict) else (c or "")

    def _sim(va, vb):
        import numpy as np
        if va is None or vb is None:
            return None
        return round(float(np.dot(np.asarray(va, "float32"), np.asarray(vb, "float32"))), 3)

    track_win: dict = {}
    for win in _day_windows(date):
        for tk in _entry_crop_info(win):
            track_win.setdefault(tk, win)

    pairs = []
    seen: set = set()
    for win in _day_windows(date):
        for l in _window_labels(win):
            vid = str(l.get("visit_id", ""))
            if vid.startswith("reidsame-") or vid.startswith("reiddiff-"):
                try:
                    at, rest = vid.split("-", 1)[1].split("-", 1)
                    bw, bt = rest.rsplit("-", 1)
                except ValueError:
                    continue
                key = (win, at, bw, bt)
                if key in seen:
                    continue
                seen.add(key)
                ca = _crop(win, at); cb = _crop(bw, bt)
                va = cache.get(str(ca).replace("\\", "/")); vb = cache.get(str(cb).replace("\\", "/"))
                pairs.append({"label": "same" if vid.startswith("reidsame-") else "different",
                              "source": "reidsame" if vid.startswith("reidsame-") else "reiddiff",
                              "a_window": win, "a_track": at, "b_window": bw, "b_track": bt,
                              "a_crop": ca, "b_crop": cb, "sim": _sim(va, vb),
                              "created": l.get("created")})
            elif vid.startswith("distinct-"):
                try:
                    a, b = vid[len("distinct-"):].split("-")
                except ValueError:
                    continue
                key = (win, a, win, b)
                if key in seen:
                    continue
                seen.add(key)
                ca = _crop(win, a); cb = _crop(win, b)
                va = cache.get(str(ca).replace("\\", "/")); vb = cache.get(str(cb).replace("\\", "/"))
                pairs.append({"label": "different", "source": "distinct",
                              "a_window": win, "a_track": a, "b_window": win, "b_track": b,
                              "a_crop": ca, "b_crop": cb, "sim": _sim(va, vb),
                              "created": l.get("created")})
            elif (vid.startswith("xcamsame-") or vid.startswith("xcamdiff-")) and l.get("verdict") != "reset":
                pfx = "xcamsame-" if vid.startswith("xcamsame-") else "xcamdiff-"
                try:
                    at_s, cam, bt_s = vid[len(pfx):].split("-")
                except ValueError:
                    continue
                key = (win, at_s, cam, bt_s)
                if key in seen:
                    continue
                seen.add(key)
                ca = _crop(win, at_s)
                cb = _interior_crop_path(win, cam, bt_s) or ""
                va = cache.get(str(ca).replace("\\", "/")); vb = cache.get(str(cb).replace("\\", "/"))
                pairs.append({"label": "same" if pfx == "xcamsame-" else "different",
                              "source": "xcam", "cam": cam,
                              "a_window": win, "a_track": at_s, "b_window": win, "b_track": bt_s,
                              "a_crop": ca, "b_crop": cb, "sim": _sim(va, vb),
                              "created": l.get("created")})
        for a in _window_annotations(win):
            if a.get("category") == "duplicate" and a.get("duplicate_of") is not None and a.get("track") is not None:
                at = a["duplicate_of"]; bt = a["track"]
                aw = track_win.get(at, win)
                key = (aw, at, win, bt)
                if key in seen:
                    continue
                seen.add(key)
                ca = _crop(aw, at)
                cb = (a.get("crop_url") or "").replace("\\", "/") or _crop(win, bt)
                va = cache.get(str(ca).replace("\\", "/")); vb = cache.get(str(cb).replace("\\", "/"))
                pairs.append({"label": "same", "source": "merge",
                              "a_window": aw, "a_track": at, "b_window": win, "b_track": bt,
                              "a_crop": ca, "b_crop": cb, "sim": _sim(va, vb),
                              "created": a.get("created")})
    for (emp, win, track), v in _staff_marks(date).items():     # per-PID staff look-alike verdicts (drawer)
        sc = v.get("staff_crop") or ""                          # staff rep crop stored in the label's reason
        cc = _crop(win, track)                                  # candidate (a customer door entry)
        va = cache.get(str(sc).replace("\\", "/")); vb = cache.get(str(cc).replace("\\", "/"))
        pairs.append({"label": v["label"], "source": "staff", "emp": emp,
                      "a_window": win, "a_track": track, "b_window": win, "b_track": track,
                      "a_crop": sc, "b_crop": cc, "sim": _sim(va, vb), "created": None})
    removals, seen_rm = [], set()                                # staff-crop removals (per-crop ✗ / unstaff-track), latest-wins, undoable
    for win in _day_windows(date):
        rm = {}
        for l in _window_labels(win):
            vid = str(l.get("visit_id", ""))
            if vid.startswith("notstaff-") and l.get("in_track") is not None:
                rm[l["in_track"]] = str(l.get("verdict", ""))
        for tk, vd in rm.items():
            if vd != "reject" or (win, tk) in seen_rm:          # latest 'reset' un-removes it
                continue
            seen_rm.add((win, tk))
            removals.append({"window": win, "track": tk, "crop": _crop(win, tk), "label": "removed"})
    n_same = sum(1 for p in pairs if p["label"] == "same")
    n_diff = len(pairs) - n_same
    return {"pairs": pairs, "n_same": n_same, "n_diff": n_diff, "total": len(pairs),
            "removals": removals, "n_removed": len(removals)}


@router.get("/reid/status")
def reid_status(date: str) -> dict:
    """How many ReID pairs you've marked (same = duplicate annotations, different = reiddiff-/distinct-)
    and how many are NEW since the last calibration -> drives the 'train now (N new)' affordance. Light
    (label/annotation counts only, no embeddings), so it's cheap to poll on every drawer open."""
    if not date or "/" in date or "\\" in date or ".." in date:
        raise HTTPException(400, "bad date")
    n_same = n_diff = 0
    for win in _day_windows(date):
        for l in _window_labels(win):
            vid = str(l.get("visit_id", ""))
            if vid.startswith("reidsame-"):
                n_same += 1
            elif vid.startswith("reiddiff-") or vid.startswith("distinct-"):
                n_diff += 1
            elif vid.startswith("xcamsame-") and l.get("verdict") != "reset":   # cross-camera link = a 'same' pair
                n_same += 1
            elif vid.startswith("xcamdiff-") and l.get("verdict") != "reset":   # cross-camera un-link = a 'different' pair
                n_diff += 1
        for a in _window_annotations(win):
            if a.get("category") == "duplicate" and a.get("duplicate_of") is not None:
                n_same += 1
    for v in _staff_marks(date).values():                       # per-PID staff look-alike verdicts (drawer)
        if v["label"] == "same":
            n_same += 1
        else:
            n_diff += 1
    total = n_same + n_diff
    last_trained, last_ver, last_at = 0, None, None
    for v in store.list_model_versions(40):                     # the most recent reid calibration (has params.impact)
        if (v.get("params") or {}).get("impact"):
            last_trained, last_ver, last_at = (v.get("trained_on") or 0), v.get("version"), v.get("created")
            break
    return {"n_same": n_same, "n_diff": n_diff, "total": total, "last_trained_on": last_trained,
            "pending": max(0, total - last_trained), "last_version": last_ver, "last_trained_at": last_at}


def _review_items(date: str) -> list:
    """The single review queue: every UNDECIDED uncertain item across the surfaces, normalized + scored.
    A LIVE VIEW — reads the same producers the ReID belt / All-Detections use and writes through their
    existing endpoints, so nothing is stored/duplicated and an item decided anywhere drops out next call
    (verdict-filtered). Ranked so the human clears the highest report-impact first (returning-visitor
    merges remove a #C → fix the unique-customer overcount)."""
    items: list = []
    thr = float((store.active_model() or {}).get("params", {}).get("reid_threshold", 0.62) or 0.62)
    try:                                                            # 1) same-camera returning-visitor merges — TOP
        for p in reid_candidates(date, threshold=thr).get("candidates", []):
            if p.get("verdict"):
                continue
            items.append({"type": "reid", "score": float(p.get("sim") or 0),
                          "crops": [p.get("a_crop"), p.get("b_crop")], "sim": p.get("sim"),
                          "left": f"#{p.get('a_pid')} · {p.get('a_in') or ''}",
                          "right": f"#{p.get('b_pid')} · {p.get('b_in') or ''}",
                          "impact": "same person → merge (−1 customer)",
                          "action": {"url": "/api/hitl/reid/decision", "verb": "POST", "kind": "samediff",
                                     "body": {"a_window": p.get("a_window"), "a_track": p.get("a_track"),
                                              "b_window": p.get("b_window"), "b_track": p.get("b_track"),
                                              "b_crop": p.get("b_crop")}}})
    except Exception:
        pass
    try:                                                            # 2) cross-camera door↔interior validation
        for p in reid_cross_camera(date).get("candidates", []):
            if p.get("verdict"):
                continue
            items.append({"type": "xcam", "score": float(p.get("sim") or 0) * 0.9, "cam": p.get("cam"),
                          "crops": [p.get("door_crop"), p.get("int_crop")], "sim": p.get("sim"),
                          "left": f"#{p.get('pid') or '?'} · 🚪 {p.get('door_ist') or ''}",
                          "right": f"{p.get('cam')} · {p.get('int_ist') or ''}",
                          "impact": "link interior crop to this #C",
                          "action": {"url": "/api/hitl/reid/cross-decision", "verb": "POST", "kind": "samediff",
                                     "body": {"window": p.get("window"), "a_track": p.get("a_track"),
                                              "cam": p.get("cam"), "b_track": p.get("b_track")}}})
    except Exception:
        pass
    sm_decided = {(w, t) for (_e, w, t) in _staff_marks(date)}      # weak-staff already resolved via /reid/staff-mark
    for win in _day_windows(date):
        try:                                                        # 3) weak staff (0.60–0.72) — visits.json carries emp_id + sim
            sj = json.loads((OUTPUTS / win / "visits.json").read_text(encoding="utf-8")).get("staff", [])
        except Exception:
            sj = []
        for st in sj:
            tk = st.get("track")
            if not st.get("weak") or st.get("employee_id") is None or tk is None or (win, tk) in sm_decided:
                continue
            items.append({"type": "weak_staff", "score": float(st.get("sim") or 0.66),
                          "crops": [st.get("crop")], "sim": st.get("sim"),
                          "left": f"weak staff · emp #{st.get('employee_id')}", "right": st.get("ist") or "",
                          "impact": "confirm staff vs keep customer",
                          "action": {"url": "/api/hitl/reid/staff-mark", "verb": "POST", "kind": "samediff",
                                     "body": {"employee_id": st.get("employee_id"), "window": win,
                                              "track": tk, "crop": st.get("crop")}}})
        try:                                                        # 4) to_review / on_hold dets (un-categorised)
            dets = _classify_detections(win, with_dims=False)
        except Exception:
            dets = []
        for d in dets:
            if d.get("confirmed") or d.get("determination") not in ("to_review", "on_hold"):
                continue
            items.append({"type": "review_det", "score": 0.2 + min(0.15, (d.get("dur_s") or 0) / 600.0),
                          "crops": [d.get("crop")], "sim": None,
                          "left": f"{d.get('camera')} · {d.get('ist') or ''}", "right": d.get("determination"),
                          "impact": "categorize this detection",
                          "action": {"url": "/api/hitl/allocate", "verb": "POST", "kind": "category",
                                     "body": {"window": win, "camera": d.get("camera"), "track": d.get("track"),
                                              "crop": d.get("crop")},
                                     "categories": ["customer", "staff", "passby", "not_person"]}})
    try:                                                            # 5) WRONG CHECK-OUT? uncertain L4 exit matches -> human verifies
        ap = (store.active_model() or {}).get("params", {}) or {}
        efloor = float(ap.get("reid_exit_floor", 0.55))
        eband = float(ap.get("exit_review_band", 0.10)); emarg = float(ap.get("exit_review_margin", 0.08))
        edwell = float(ap.get("exit_review_dwell_s", 3600))
        tagc = _daily_tags(date)["cust"]
        for (iw, it), ex in _exit_resolve(date).items():
            if ex.get("source") != "matched":
                continue
            sim, mg, dw = ex.get("sim"), ex.get("margin"), (ex.get("dwell_s") or 0)
            borderline = sim is not None and sim < efloor + eband           # only just cleared the floor
            thin = mg is not None and mg < emarg                            # toss-up between two opens
            longx = bool(ex.get("cross_window")) and dw > edwell            # long cross-window dwell
            if not (borderline or thin or longx):
                continue
            n = tagc.get((iw, it))
            ecrop = (_entry_crop_info(iw).get(it) or {}).get("crop")
            items.append({"type": "visit_exit",
                          "score": 0.55 + (0.10 if borderline else 0) + (0.05 if longx else 0),   # ranks above weak_staff/review_det
                          "crops": [ecrop, ex.get("out_crop")], "sim": sim,
                          "left": (f"#C{n} · 🚪 in" if n else "🚪 in"),
                          "right": f"🚪 out {ex.get('out_ist') or ''} · {round(dw / 60)}m dwell",
                          "impact": "wrong check-out corrupts dwell + proof",
                          "action": {"url": "/api/hitl/reid/reject-exit", "verb": "POST", "kind": "samediff",
                                     "body": {"window": iw, "track": it,
                                              "out_window": ex.get("out_window"), "out_track": ex.get("out_track")}}})
    except Exception:
        pass
    items.sort(key=lambda x: -x["score"])
    return items


def _review_counts(date: str, items: list | None = None) -> dict:
    """Queue length + by-type breakdown. Shared by /review/queue and /accuracy so the two never diverge."""
    items = _review_items(date) if items is None else items
    by_type: dict = {}
    for it in items:
        by_type[it["type"]] = by_type.get(it["type"], 0) + 1
    return {"uncertain": len(items), "by_type": by_type}


_REVIEW_DET_CAP = 25       # render-cap the low-value un-categorised-door tail so the queue stays actionable


@router.get("/review/queue")
def review_queue(date: str) -> dict:
    """Single ranked review queue (Phase 1): all undecided uncertain items across surfaces, score-desc so
    the human clears the highest report-impact first (returning-visitor merges → fix the overcount). Live
    view — nothing stored; clearing an item uses the existing endpoints, so it reflects everywhere and drops
    out on next load. The bulky `review_det` tail is render-capped, but `counts` keep the TRUE totals (so
    the Phase-2 'sent to review' metric is honest)."""
    if not date or "/" in date or "\\" in date or ".." in date:
        raise HTTPException(400, "bad date")
    full = _review_items(date)
    counts = _review_counts(date, full)                            # TRUE totals (uncapped) for the metric
    counts["cleared"] = reid_status(date).get("total", 0)          # decided-so-far (reuse the markings tally)
    rendered, rd = [], 0                                            # cap only the review_det tail in the rendered list
    for it in full:
        if it["type"] == "review_det":
            rd += 1
            if rd > _REVIEW_DET_CAP:
                continue
        rendered.append(it)
    counts["rendered"] = len(rendered)
    counts["review_det_total"] = counts["by_type"].get("review_det", 0)
    counts["review_det_capped_at"] = _REVIEW_DET_CAP
    return {"items": rendered, "counts": counts, "date": date}


def _auto_baseline(date: str):
    """The pipeline's PRE-human report, snapshotted ONCE at initial processing (outputs/<date>_auto_baseline.json,
    written by the day-runner, never clobbered by _rerun_l4). None for days processed before this feature -> the
    metric falls back to the override-based estimate. This is the only honest auto baseline: visits.json itself is
    NOT one (it's regenerated with the human-enrolled gallery + must_link on every _rerun_l4)."""
    try:
        return json.loads((OUTPUTS / f"{date}_auto_baseline.json").read_text(encoding="utf-8"))
    except Exception:
        return None


@router.get("/accuracy/{date}")
def accuracy(date: str) -> dict:
    """Per-day TRUST instrument (Phase 2): how accurate the auto pass was + how much HITL effort the day needed.
    `separation` (the model's same/diff gap + its trend over trainings) is the leading indicator — when it crosses
    ~0.15, auto-merge becomes safe. `effort` = corrections the human made. If an auto baseline was snapshotted at
    processing, also the EXACT auto-vs-final headline. Kept light (no queue recompute) so the §1 line is snappy."""
    if not date or "/" in date or "\\" in date or ".." in date:
        raise HTTPException(400, "bad date")
    import numpy as np
    same, diff = _reid_labeled_pairs(date)
    same_avg = round(float(np.mean(same)), 3) if same else None
    diff_avg = round(float(np.mean(diff)), 3) if diff else None
    sep = round(same_avg - diff_avg, 3) if (same_avg is not None and diff_avg is not None) else None
    trend = []
    for v in reversed(store.list_model_versions(40)):              # oldest -> newest separation points
        im = (v.get("params") or {}).get("impact") or {}
        if im.get("separation") is not None:
            trend.append({"version": v.get("version"), "separation": im["separation"], "created": v.get("created")})
    report = _day_report(date)
    cust = report["customers"]["unique_customers"]
    staff = report["employees"]["headcount"]
    overrides = reid_status(date).get("total", 0)                  # corrections the human made (markings)
    auto = _auto_baseline(date)
    if auto is None and overrides == 0:                            # fresh day, no HITL yet -> THIS report IS the auto
        try:                                                       # baseline; snapshot it once (never re-written after)
            (OUTPUTS / f"{date}_auto_baseline.json").write_text(json.dumps(report), encoding="utf-8")
            auto = report
        except Exception:
            pass
    acc = None
    if auto:                                                       # exact auto-vs-final headline (June-4-forward)
        a_c = (auto.get("customers") or {}).get("unique_customers")
        a_s = (auto.get("employees") or {}).get("headcount")
        if a_c is not None:
            d_c, d_s = abs(a_c - cust), abs((a_s or 0) - staff)
            acc = {"auto_customers": a_c, "final_customers": cust, "auto_staff": a_s, "final_staff": staff,
                   "pct": round(1 - (d_c + d_s) / max(1, a_c + (a_s or 0)), 3)}
    auto_decisions = (acc["auto_customers"] + (acc["auto_staff"] or 0)) if acc else (cust + staff)
    # est_accuracy is ONLY honest with an auto baseline (auto-vs-final). Without one, every marking is NOT an
    # "error" — most ✗ 'different' verdicts CONFIRM the auto pass was right — so we don't fabricate a number.
    est = acc["pct"] if acc else None
    verdict = ("separation crossing the safe band (~0.15) — auto-merge is becoming viable; verify on a baseline day before enabling"
               if (sep or 0) >= 0.15
               else "same & different still overlap — keep clearing the queue + fine-tune before any auto-merge")
    return {"date": date,
            "separation": {"current": sep, "same_avg": same_avg, "diff_avg": diff_avg, "trend": trend},
            "effort": {"decisions": overrides, "auto_decisions": auto_decisions},   # decisions = the HITL work done
            "accuracy": acc, "est_accuracy": est, "report": {"customers": cust, "staff": staff},
            "verdict": verdict, "has_baseline": auto is not None}


_INT_CROP_CACHE: dict = {}


def _interior_crop_path(window, cam, track):
    """Interior (C11/C14) crop path for (window, cam, track) from that camera's L1 tracks.json. Cached
    (L1 outputs are immutable). For cross-camera ReID pairs whose interior side _entry_crop_info can't resolve."""
    key = (window, cam)
    info = _INT_CROP_CACHE.get(key)
    if info is None:
        info = {}
        try:
            tj = json.loads((OUTPUTS / window / f"L1_{cam}" / "tracks.json").read_text(encoding="utf-8"))
            for t in tj.get("tracks", []):
                cr = (t.get("crop") or "").replace("\\", "/")
                if cr:
                    info[t["track"]] = cr
        except Exception:
            pass
        _INT_CROP_CACHE[key] = info
    try:
        return info.get(int(track))
    except (TypeError, ValueError):
        return None


def _reid_labeled_pairs(date: str):
    """The human-confirmed ReID pairs as OSNet cosine sims: same = 'duplicate' annotations (b->a) +
    cross-camera 'xcamsame-' links; different = 'reiddiff-'/'distinct-' + cross-camera 'xcamdiff-' labels.
    Window ids contain '-', so parse at|bw|bt by first/last dash."""
    import numpy as np
    cache = _load_emb_cache()
    cinfo: dict = {}

    def sim(aw, at, bw, bt):
        try:
            at, bt = int(at), int(bt)
        except Exception:
            return None
        ca = (cinfo.setdefault(aw, _entry_crop_info(aw)).get(at) or {}).get("crop") if aw else None
        cb = (cinfo.setdefault(bw, _entry_crop_info(bw)).get(bt) or {}).get("crop") if bw else None
        va = cache.get(str(ca or "").replace("\\", "/"))
        vb = cache.get(str(cb or "").replace("\\", "/"))
        if va is None or vb is None:
            return None
        return float(np.dot(np.asarray(va, "float32"), np.asarray(vb, "float32")))

    track_win = {}                                              # entry track -> its window, to recover a_window for legacy 'duplicate' merges
    for c in _cards_from_dets(date)["customer_cards"]:
        e = c.get("entry")
        if e:
            track_win.setdefault(e[1], e[0])
    same, diff = [], []
    for win in _day_windows(date):
        for l in _window_labels(win):
            vid = str(l.get("visit_id", ""))
            if vid.startswith("reidsame-") or vid.startswith("reiddiff-"):
                try:
                    at, rest = vid.split("-", 1)[1].split("-", 1); bw, bt = rest.rsplit("-", 1)
                except ValueError:
                    continue
                s = sim(win, at, bw, bt)
                if s is not None:
                    (same if vid.startswith("reidsame-") else diff).append(s)
            elif vid.startswith("distinct-"):
                try:
                    a, b = vid[len("distinct-"):].split("-")
                except ValueError:
                    continue
                s = sim(win, a, win, b)
                if s is not None:
                    diff.append(s)
            elif (vid.startswith("xcamsame-") or vid.startswith("xcamdiff-")) and l.get("verdict") != "reset":
                pfx = "xcamsame-" if vid.startswith("xcamsame-") else "xcamdiff-"   # cross-camera door<->interior pair
                try:
                    at, cam, bt = vid[len(pfx):].split("-"); at = int(at)
                except ValueError:
                    continue
                ca = (cinfo.setdefault(win, _entry_crop_info(win)).get(at) or {}).get("crop")
                cb = _interior_crop_path(win, cam, bt)
                va = cache.get(str(ca or "").replace("\\", "/")); vb = cache.get(str(cb or "").replace("\\", "/"))
                if va is not None and vb is not None:
                    s = float(np.dot(np.asarray(va, "float32"), np.asarray(vb, "float32")))
                    (same if pfx == "xcamsame-" else diff).append(s)
        for a in _window_annotations(win):                     # legacy 'same' = duplicate annotations
            if a.get("category") == "duplicate" and a.get("duplicate_of") is not None:
                s = sim(track_win.get(a["duplicate_of"]), a["duplicate_of"], win, a.get("track"))
                if s is not None:
                    same.append(s)
    for (emp, win, track), v in _staff_marks(date).items():    # per-PID staff look-alike verdicts: staff_crop vs candidate door crop
        cc = (cinfo.setdefault(win, _entry_crop_info(win)).get(track) or {}).get("crop")
        va = cache.get(str(v.get("staff_crop") or "").replace("\\", "/"))
        vb = cache.get(str(cc or "").replace("\\", "/"))
        if va is not None and vb is not None:
            s = float(np.dot(np.asarray(va, "float32"), np.asarray(vb, "float32")))
            (same if v["label"] == "same" else diff).append(s)
    return same, diff


@router.post("/reid/train")
def reid_train(body: dict) -> dict:
    """Learning-free 'training' on the human ReID verdicts: calibrate the similarity threshold that best
    separates the confirmed SAME from DIFFERENT pairs, register it as a model version, and report the
    before->after impact (separation + accuracy + the merge effect on the count). No GPU; safe + instant."""
    _guard_write()
    date = (body or {}).get("date")
    if not date or "/" in date or "\\" in date or ".." in date:
        raise HTTPException(400, "bad date")
    import numpy as np
    same, diff = _reid_labeled_pairs(date)
    n = len(same) + len(diff)
    if n < 2:
        return {"ok": False, "message": "not enough confirmed pairs yet — tick ✓ / cross ✗ a few more look-alikes first"}
    cur = (store.active_model() or {}).get("params", {}) or {}
    old_thr = float(cur.get("reid_threshold", 0.72))

    def acc(t):
        return (sum(1 for s in same if s >= t) + sum(1 for d in diff if d < t)) / n

    best_thr, best_acc = old_thr, acc(old_thr)
    for c in sorted(set(same + diff)):                         # scan every cut point just below a sample
        t = round(c - 1e-4, 4); a = acc(t)
        if a > best_acc:
            best_acc, best_thr = a, round(c, 3)
    same_avg = round(float(np.mean(same)), 3) if same else None
    diff_avg = round(float(np.mean(diff)), 3) if diff else None
    sep = round(same_avg - diff_avg, 3) if (same_avg is not None and diff_avg is not None) else None
    cust_now = len(_cards_from_dets(date)["customer_cards"])
    verdict = ("clean separation — your labels sharpen the matcher" if (sep or 0) > 0.1
               else "same & different overlap — OSNet is weak on this CCTV; a deep fine-tune (more labels) is the real fix")
    impact = {"n_same": len(same), "n_diff": len(diff), "same_avg": same_avg, "diff_avg": diff_avg,
              "separation": sep, "old_threshold": round(old_thr, 3), "old_acc": round(acc(old_thr), 3),
              "best_threshold": round(best_thr, 3), "best_acc": round(best_acc, 3),
              "customers_now": cust_now, "verdict": verdict, "date": date}
    # Count ALL marked pairs (embedding-independent) so trained_on matches reid_status total
    total_marked = 0
    for win in _day_windows(date):
        for l in _window_labels(win):
            vid = str(l.get("visit_id", ""))
            if vid.startswith(("reidsame-", "reiddiff-", "distinct-")):
                total_marked += 1
            elif vid.startswith(("xcamsame-", "xcamdiff-")) and l.get("verdict") != "reset":
                total_marked += 1
        for a in _window_annotations(win):
            if a.get("category") == "duplicate" and a.get("duplicate_of") is not None:
                total_marked += 1
    total_marked += len(_staff_marks(date))                     # per-PID staff look-alike verdicts count too
    params = {**cur, "reid_threshold": round(best_thr, 3), "reid_calibrated_pairs": n, "impact": impact}
    ver = store.register_model_version(
        "thresholds", params, trained_on=total_marked,
        notes=f"reid calibration: {len(same)} same / {len(diff)} different pairs -> threshold "
              f"{round(best_thr, 3)} ({round(best_acc * 100)}% separation)", active=True)
    return {"ok": True, "version": ver, **impact}


_ALLOC_CATS = ("customer", "staff", "not_person", "passby", "duplicate")


def _existing_employee_id(window, track):
    """Which employee a track is already attributed to (auto-recognised or previously marked) — so a
    plain 'confirm staff' keeps that identity instead of writing an unattributed staff label."""
    for l in store.get_labels(window):
        if l.get("verdict") == "employee" and l.get("in_track") == track and l.get("employee_id"):
            return l["employee_id"]
    try:
        for s in json.loads((OUTPUTS / window / "visits.json").read_text(encoding="utf-8")).get("staff", []):
            if s.get("track") == track and s.get("employee_id"):
                return s["employee_id"]
    except Exception:
        pass
    return None


def _alloc_one(window, camera, track, category, crop, emp_id, duplicate_of=None) -> bool:
    """Record ONE allocation — durable annotation (+ training crop) and, for a door (C05) track, the
    matcher label that drives the count. Does NOT re-run L4. Returns True if it touched a C05 track
    (so the caller knows an L4 re-run is needed)."""
    if category == "staff" and not emp_id:
        emp_id = _existing_employee_id(window, track)          # a plain confirm keeps the known identity
    store.add_annotation(window, camera, track, category, crop_url=crop, employee_id=emp_id,
                         duplicate_of=duplicate_of, embedding=_emb_for_crop(crop))
    if category == "staff" and emp_id and crop:
        _enroll_from_cache(emp_id, crop, window, track)        # enroll regardless of camera
    if camera == "C05":                                        # only door tracks drive the entry count
        if category == "staff":
            store.add_label(window, f"open-{track}", "employee", in_track=track, employee_id=emp_id)
            store.add_label(window, f"notstaff-{track}", "reset", in_track=track)
            store.add_label(window, f"false-{track}", "reset", in_track=track)
        elif category in ("not_person", "passby", "duplicate"):
            store.add_label(window, f"false-{track}", "false_detection", in_track=track, reason=category)
        elif category == "customer":
            store.add_label(window, f"false-{track}", "reset", in_track=track)
        return True
    return False


@router.post("/allocate")
def allocate(body: dict) -> dict:
    """Close-the-day: a human assigns ONE detection to a category (annotation + matcher label)."""
    _guard_write()
    window = body.get("window", "")
    _window_dir(window)
    camera, track = body.get("camera", ""), body.get("track")
    category, crop, emp_id = body.get("category", ""), body.get("crop"), body.get("employee_id")
    if track is None or category not in _ALLOC_CATS:
        raise HTTPException(400, "allocate needs a track and a valid category")
    touched = _alloc_one(window, camera, track, category, crop, emp_id, body.get("duplicate_of"))
    _unpark(window, [(camera, track)])      # a decision releases any hold
    # NB: the bucket move is annotation-driven (instant, client-side); the customer COUNT is refreshed
    # lazily via POST /recompute, so a click never blocks on the ~1-5s L4 re-run.
    return {"ok": True, "stale": touched}   # stale=True (a C05 track) -> the count needs a recompute


@router.post("/allocate-bulk")
def allocate_bulk(body: dict) -> dict:
    """Allocate MANY detections to ONE category in a single pass — write every annotation + label,
    then re-run L4 just ONCE. Far faster than N separate /allocate calls (each of which re-ran L4)."""
    _guard_write()
    window = body.get("window", "")
    _window_dir(window)
    category, emp_id = body.get("category", ""), body.get("employee_id")
    items = body.get("items", [])
    if not items or category not in _ALLOC_CATS:
        raise HTTPException(400, "allocate-bulk needs items + a valid category")
    touched, n = False, 0
    for it in items:
        track = it.get("track")
        if track is None:
            continue
        if _alloc_one(window, it.get("camera", ""), track, category, it.get("crop"),
                      emp_id, it.get("duplicate_of")):
            touched = True
        n += 1
    _unpark(window, [(it.get("camera", ""), it["track"]) for it in items if it.get("track") is not None])
    return {"ok": True, "n": n, "stale": touched}              # count recomputed lazily via /recompute


@router.post("/handoff-decision")
def handoff_decision(body: dict) -> dict:
    """L3 HITL: a human adjudicates a loose trajectory-handoff suggestion ('same person?').
      same=True  -> MERGE b into a: write b as a 'duplicate' annotation (so it drops from the customer
                    count, exactly like the All-Detections 'duplicate' bucket) -> also a 'same' training pair.
      same=False -> KEEP SEPARATE: durable 'distinct-<a>-<b>' label so the pair is never re-suggested
                    (and stands as a 'different' training pair). Reuses the labels table, like 'notstaff-'/'false-'."""
    _guard_write()
    window = body.get("window", "")
    _window_dir(window)
    a, b, same = body.get("a"), body.get("b"), bool(body.get("same"))
    if a is None or b is None:
        raise HTTPException(400, "handoff-decision needs tracks a + b")
    if same:
        _alloc_one(window, body.get("camera", "C05"), b, "duplicate", body.get("b_crop"), None, duplicate_of=a)
    else:
        store.add_label(window, f"distinct-{a}-{b}", "reject", in_track=b, reason=f"distinct:{a}")
    return {"ok": True, "stale": same}                          # same -> a C05 track left the count; refresh lazily


@router.post("/confirm-exit")
def confirm_exit(body: dict) -> dict:
    """L4 HITL: a human accepts a presumed (inferred) exit for a still-inside customer PID. Durable
    'exitok-<track>' confirm label -> the 'exit inferred' chip flips to 'exit confirmed'."""
    _guard_write()
    window = body.get("window", "")
    _window_dir(window)
    track = body.get("track")
    if track is None:
        raise HTTPException(400, "confirm-exit needs a track")
    store.add_label(window, f"exitok-{track}", "confirm", in_track=track, reason="exit_confirmed")
    return {"ok": True}


@router.post("/recompute")
def recompute(body: dict) -> dict:
    """Refresh the customer COUNT for a window after annotation edits (the bucket moves are already
    live; this just re-runs L2-L4 to refresh the headline number). Deliberate + synchronous — the one
    place L4 runs after a reconciliation edit, so per-click stays instant and visits.json never races."""
    _guard_write()
    window = body.get("window", "")
    _window_dir(window)
    _rerun_l4(window)
    return {"ok": True, **_window_counts(window)}


def _parked_path(window) -> Path:
    return _window_dir(window) / "parked.json"


def _parked_set(window) -> set:
    """(camera, track) pairs held for later review — local review scratch, NOT a training annotation."""
    p = _parked_path(window)
    if not p.exists():
        return set()
    try:
        return {(c, int(t)) for c, t in json.loads(p.read_text(encoding="utf-8"))}
    except Exception:
        return set()


def _park_write(window, s) -> None:
    _parked_path(window).write_text(json.dumps(sorted([[c, t] for c, t in s])), encoding="utf-8")


def _unpark(window, keys) -> None:
    s = _parked_set(window)
    if not s:
        return
    before = len(s)
    for k in keys:
        s.discard((k[0], int(k[1])))
    if len(s) != before:
        _park_write(window, s)


@router.post("/park")
def park(body: dict) -> dict:
    """Hold detections for later review (on=true) or release them (on=false). Parked detections move
    to their own group at the top and drop out of the resolved count until decided. A decision
    (allocate) auto-releases the hold."""
    _guard_write()
    window = body.get("window", "")
    _window_dir(window)
    on = bool(body.get("on", True))
    s = _parked_set(window)
    for it in body.get("items", []):
        track = it.get("track")
        if track is None:
            continue
        key = (it.get("camera", ""), int(track))
        if on:
            s.add(key)
        else:
            s.discard(key)
    _park_write(window, s)
    return {"ok": True, "parked": len(s)}


def _enroll_from_cache(employee_id, crop, window, track):
    """Add the marked crop's OSNet embedding (from the cached L4 emb map, no GPU) to the gallery."""
    emb = _load_emb_cache().get(str(crop).replace("\\", "/"))
    if emb is not None:
        try:
            store.enroll_staff(employee_id, "s14", emb, crop_url=crop, window=window, track=track)
        except Exception:
            pass


def _gallery_meta():
    """Gallery rows for the UI WITHOUT embeddings (≈2s/load saved); falls back if the store lacks the lean method."""
    return store.get_gallery_meta() if hasattr(store, "get_gallery_meta") else store.get_gallery()


def _staff_no_map() -> dict:
    """employee id -> permanent staff_no (immutable; S1 always means ONE person). Falls back to
    enrollment rank if the staff_no column hasn't been migrated yet, so this is safe pre-migration."""
    emps = sorted(store.list_employees(), key=lambda x: x["id"])
    return {e["id"]: (e.get("staff_no") or (i + 1)) for i, e in enumerate(emps)}


def _rank_labels() -> dict:
    """employee id -> 'Staff #N' by permanent staff_no (display-only; the store keeps the raw S<id>)."""
    return {eid: f"Staff #{n}" for eid, n in _staff_no_map().items()}


@router.get("/employees")
def employees() -> dict:
    emps = sorted(store.list_employees(), key=lambda x: x["id"])
    thumb: dict = {}
    try:                                   # a representative enrolled crop per staffer, for the picker thumbnails
        for g in _gallery_meta():
            if g.get("employee_id") and g.get("crop_url"):
                thumb.setdefault(g["employee_id"], g["crop_url"])
    except Exception:
        pass
    return {"employees": [{**e, "code": f"Staff #{i + 1}", "crop": thumb.get(e["id"])}
                          for i, e in enumerate(emps)]}


@router.post("/employees")
def create_employee(body: dict) -> dict:
    _guard_write()
    return store.create_employee(name=(body.get("name") or None))


@router.patch("/employees/{emp_id}")
def rename_employee(emp_id: int, body: dict) -> dict:
    _guard_write()
    store.rename_employee(emp_id, body.get("name", ""))
    return {"ok": True}


@router.get("/attendance")
def attendance(store_id: str = "s14", date: str | None = None) -> dict:
    lab = _rank_labels()
    rows = store.attendance(store_id, date)
    for r in rows:
        r["code"] = lab.get(r["id"], r.get("code"))
    return {"attendance": rows}


@router.get("/staff-matches")
def staff_matches(employee_id: int) -> dict:
    """Every sighting grouped to one staffer across the day (manual + auto) for human confirmation."""
    out = store.staff_matches(employee_id)
    out["code"] = _rank_labels().get(employee_id, f"S{employee_id}")
    rank = _staff_no_map()
    out["pid"] = f"S{rank.get(employee_id, employee_id)}"             # daily S# rank for the card title
    return out


_RESOLVED_NOT_CUSTOMER = ("staff", "not_person", "passby", "duplicate")

_L2_XING: dict = {}        # window -> [{track, ist}] raw IN crossings (L1 + zone config are immutable for a processed day)


def _l2_crossings(window: str) -> list:
    """Raw L2 door-line IN crossings for a window — the SAME geometry ingest's L2 uses (entry_line +
    street_mask), but WITHOUT the time-window dedup, so every genuine door entry shows (broad, L2-
    authoritative coverage; the user accepts more false positives over missed entries). Returns
    [{track, ist}] for C05 L1 tracks whose path crosses the entry line onto the inside; street-mask
    noise is dropped (same filter as _classify_detections, so a street track never becomes a customer)."""
    hit = _L2_XING.get(window)
    if hit is not None:
        return hit
    try:
        wcfg = json.loads((OUTPUTS / window / "window.json").read_text(encoding="utf-8"))
        cfg = json.loads(Path(wcfg["config"]).read_text(encoding="utf-8"))
        tracks = json.loads((Path(wcfg["l1"]) / "tracks.json").read_text(encoding="utf-8")).get("tracks", [])
    except Exception:
        _L2_XING[window] = []
        return []
    p1, p2 = tuple(cfg["entry_line"][0]), tuple(cfg["entry_line"][1])
    street, sfrac = cfg.get("street_mask", []), cfg.get("street_drop_frac", 0.5)
    min_h = cfg.get("min_bbox_h", 0.0)
    inside_sign = -1 if cfg.get("inside") == "right" else 1     # 'right' of the line = the store side = IN

    def side(a, b, pt):
        d = (b[0] - a[0]) * (pt[1] - a[1]) - (b[1] - a[1]) * (pt[0] - a[0])
        return 1 if d > 0 else (-1 if d < 0 else 0)

    def ccw(A, B, C):
        return (C[1] - A[1]) * (B[0] - A[0]) > (B[1] - A[1]) * (C[0] - A[0])

    def seg_int(A, B, C, D):
        return ccw(A, C, D) != ccw(B, C, D) and ccw(A, B, C) != ccw(A, B, D)

    out = []
    for t in tracks:
        traj = t.get("traj", [])
        if not traj:
            continue
        if street and sum(_point_in_poly(x, y, street) for _, x, y, _ in traj) / len(traj) > sfrac:
            continue                                            # mostly through-glass street -> noise
        if min_h > 0 and max((p[3] for p in traj), default=0.0) < min_h:
            continue
        prev = prev_sign = None
        for ts, x, y, h in traj:
            cur = side(p1, p2, (x, y))
            if (prev is not None and prev_sign not in (None, 0) and cur != 0
                    and cur != prev_sign and seg_int(prev, (x, y), p1, p2)):
                if cur == inside_sign:                          # crossed onto the inside -> an IN entry
                    out.append({"track": t["track"], "ist": t.get("first_ist")})
                break
            if cur != 0:
                prev_sign = cur
            prev = (x, y)
    _L2_XING[window] = out
    return out


def _confirmed_entries(window):
    """(window,track,ist) for each human-confirmed customer door entry + matched-visit dwells, PLUS every
    raw L2 door-line IN crossing (L2-authoritative footfall) that isn't already a matched visit/its exit,
    staff, parked, or human-tagged not-a-customer."""
    try:
        data = json.loads((OUTPUTS / window / "visits.json").read_text(encoding="utf-8"))
    except Exception:
        return [], []
    ann = {a["track"]: a["category"] for a in _window_annotations(window) if a["camera"] == "C05"}
    auto = {}                                                  # door-IN tracks (post-rerun) -> ist
    for v in data.get("visits", []):
        auto[v["in_track"]] = v["in_ist"]
    for e in data.get("open_sessions", []):
        auto[e["track"]] = e["ist"]
    out_tracks = {v["out_track"] for v in data.get("visits", [])}              # the EXIT of a counted visit...
    extra = [t for t, c in ann.items() if c == "customer" and t not in auto and t not in out_tracks]   # ...isn't a new customer (no double-count)
    det_ist = {d["track"]: d["ist"] for d in store.get_detections(window)
               if d.get("camera") == "C05"} if extra else {}
    parked = _parked_set(window)                                              # held-for-review = PENDING -> not counted (matches the All-Detections on_hold bucket)
    labels = _window_labels(window)                                           # staff door crossings AREN'T customers -> mirror _classify_detections' staff set exactly:
    not_staff = {l["in_track"] for l in labels if str(l.get("visit_id", "")).startswith("notstaff-")
                 and l.get("verdict") == "reject" and l.get("in_track") is not None}
    staff_tr = {st["track"] for st in data.get("staff", []) if not st.get("weak")}      # (a) gallery auto-recognised (non-weak)
    staff_tr |= {l["in_track"] for l in labels if l.get("verdict") == "employee"        # (b) label-assigned #S
                 and l.get("in_track") is not None and l["in_track"] not in not_staff}  # (annotated category='staff' is already dropped via _RESOLVED_NOT_CUSTOMER)
    entries = [{"window": window, "track": t, "ist": ist} for t, ist in auto.items()
               if ann.get(t) not in _RESOLVED_NOT_CUSTOMER and ("C05", t) not in parked and t not in staff_tr]
    entries += [{"window": window, "track": t, "ist": det_ist[t]} for t in extra
                if det_ist.get(t) and ("C05", t) not in parked and t not in staff_tr]
    # L2-authoritative footfall: every RAW door-line IN crossing is a customer entry, unless it's already a
    # matched visit (auto) / that visit's exit, staff, parked, or a human said "not a customer". The smart
    # trajectory-handoff de-dup in _handoff_merges still collapses one person split across two door tracks.
    seen_tr = {e["track"] for e in entries} | out_tracks
    for x in _l2_crossings(window):
        t = x["track"]
        if (t in seen_tr or t in auto or t in staff_tr or ("C05", t) in parked
                or ann.get(t) in _RESOLVED_NOT_CUSTOMER):
            continue
        entries.append({"window": window, "track": t, "ist": x.get("ist")})
        seen_tr.add(t)
    dwell = [v["dwell_s"] for v in data.get("visits", [])
             if v.get("dwell_s") and ann.get(v["in_track"]) not in _RESOLVED_NOT_CUSTOMER and v["in_track"] not in staff_tr]
    return entries, dwell


def _window_counts(window) -> dict:
    """Counts that move when training changes the gallery/threshold — for the re-run delta."""
    vj = OUTPUTS / window / "visits.json"
    if not vj.exists():
        return {"customers": 0, "entries": 0, "auto_staff": 0, "visits": 0}
    c = json.loads(vj.read_text(encoding="utf-8")).get("counts", {})
    visits, still = c.get("visits", 0), c.get("still_inside", 0)
    try:
        customers = len(_confirmed_entries(window)[0])
    except Exception:
        customers = visits + still
    return {"customers": customers, "entries": visits + still,
            "auto_staff": c.get("auto_staff", 0), "visits": visits}


@router.post("/rerun")
def rerun(body: dict) -> dict:
    """Re-run L2-L4 (cached L1, no GPU) on the chosen Review filter — one hour or a whole date —
    with the ACTIVE gallery + threshold, and return the before->after delta of the key counts.
    'window' = 'YYYY-MM-DD' (whole day) or 'YYYY-MM-DD_HHMM' (one hour)."""
    _guard_write()
    target = (body or {}).get("window", "")
    if not target:
        raise HTTPException(400, "rerun needs a 'window' (a date for whole-day, or YYYY-MM-DD_HHMM)")
    is_day = "_" not in target
    if is_day:
        import glob
        windows = sorted({Path(p).parent.name for p in glob.glob(f"outputs/{target}_*/visits.json")})
    else:
        windows = [target]
    keys = ("customers", "entries", "auto_staff", "visits")
    rows, tb, ta = [], {k: 0 for k in keys}, {k: 0 for k in keys}
    for w in windows:
        before = _window_counts(w)
        err = None
        try:
            _rerun_l4(w)
        except HTTPException as e:
            err = str(e.detail)[:200]
        after = _window_counts(w)
        rows.append({"window": w, "before": before, "after": after,
                     "delta": {k: after[k] - before[k] for k in keys}, "error": err})
        for k in keys:
            tb[k] += before[k]; ta[k] += after[k]
    return {"ok": True, "scope": "day" if is_day else "hour", "n_windows": len(windows),
            "windows": rows,
            "total": {"before": tb, "after": ta, "delta": {k: ta[k] - tb[k] for k in keys}},
            "model": store.active_model()}


def _entry_crop_info(window):
    """track -> {crop, out_ist, dwell_s} from a window's visits.json (enriches the report's customer list)."""
    try:
        data = json.loads((OUTPUTS / window / "visits.json").read_text(encoding="utf-8"))
    except Exception:
        return {}
    info = {}
    for v in data.get("visits", []):
        info[v["in_track"]] = {"crop": (v.get("in_crop") or "").replace("\\", "/"),
                               "out_crop": (v.get("out_crop") or "").replace("\\", "/"),
                               "out_ist": v.get("out_ist"), "dwell_s": v.get("dwell_s")}
    for e in data.get("open_sessions", []):
        info.setdefault(e["track"], {"crop": (e.get("crop") or "").replace("\\", "/"),
                                     "out_crop": None, "out_ist": None, "dwell_s": None})
    try:  # door-crop fallback: a human-added "missed customer" (annotation only, not in visits/open) still gets a face
        wcfg = json.loads((OUTPUTS / window / "window.json").read_text(encoding="utf-8"))
        for t in json.loads((Path(wcfg["l1"]) / "tracks.json").read_text(encoding="utf-8")).get("tracks", []):
            cr = (t.get("crop") or "").replace("\\", "/")
            if not cr:
                continue
            cur = info.get(t["track"])
            if cur is None:
                info[t["track"]] = {"crop": cr, "out_crop": None, "out_ist": None, "dwell_s": None}
            elif not cur.get("crop"):
                cur["crop"] = cr
    except Exception:
        pass
    return info


_DEMO = {"data": {}, "mtime": -1.0}   # path-keyed age/gender cache (outputs/demographics_cache.json), hot-reloaded


def _demo_label(crop) -> str | None:
    """'~30s · M' from the cached face-based age/gender estimate for a crop path; None when there's no
    face or no cache. Reloads the cache file when it changes (re-run the batch -> labels update live)."""
    if not crop:
        return None
    f = OUTPUTS / "demographics_cache.json"
    try:
        mt = f.stat().st_mtime
    except Exception:
        return None
    if mt != _DEMO["mtime"]:
        try:
            _DEMO["data"] = json.loads(f.read_text(encoding="utf-8"))
        except Exception:
            _DEMO["data"] = {}
        _DEMO["mtime"] = mt
    e = _DEMO["data"].get(str(crop).replace("\\", "/"))
    if not e or not e.get("face"):
        return None
    age, g = e.get("age"), e.get("gender")
    gi = "M" if g == "male" else "F" if g == "female" else "?"
    return f"{('~' + str((age // 10) * 10) + 's') if age else '?'} · {gi}"


def _demo_gender(crop):
    """(('male'|'female')|None, conf) from the cached face-gender for a crop — for the exit-match gender gate.
    (None, 0.0) when there's no face/estimate. Reuses the same hot-reloaded _DEMO cache as _demo_label."""
    if not crop:
        return None, 0.0
    f = OUTPUTS / "demographics_cache.json"
    try:
        mt = f.stat().st_mtime
    except Exception:
        return None, 0.0
    if mt != _DEMO["mtime"]:
        try:
            _DEMO["data"] = json.loads(f.read_text(encoding="utf-8"))
        except Exception:
            _DEMO["data"] = {}
        _DEMO["mtime"] = mt
    e = _DEMO["data"].get(str(crop).replace("\\", "/"))
    if not e or not e.get("face"):
        return None, 0.0
    g = e.get("gender")
    return (g if g in ("male", "female") else None), float(e.get("gender_conf") or 0.0)


def _passby_count(window) -> int:
    """Fast pass-by count for the report's capture rate: C05 door tracks whose path is mostly inside
    the street_mask and that aren't an accounted visit. Reads only C05 tracks.json + visits.json —
    no Supabase, no interior cams — so it's ~12x cheaper than a full _classify_detections pass."""
    try:
        wdir = OUTPUTS / window
        wcfg = json.loads((wdir / "window.json").read_text(encoding="utf-8"))
        cfg = json.loads(Path(wcfg["config"]).read_text(encoding="utf-8"))
        street, sfrac = cfg.get("street_mask", []), cfg.get("street_drop_frac", 0.5)
        if not street:
            return 0
        tracks = json.loads((Path(wcfg["l1"]) / "tracks.json").read_text(encoding="utf-8")).get("tracks", [])
        vj = json.loads((wdir / "visits.json").read_text(encoding="utf-8"))
    except Exception:
        return 0
    accounted = set()
    for v in vj.get("visits", []):
        accounted.update({v["in_track"], v["out_track"]})
    for e in vj.get("open_sessions", []) + vj.get("pre_window_exits", []):
        accounted.add(e["track"])
    for st in vj.get("staff", []):
        accounted.add(st.get("track"))
    n = 0
    for t in tracks:
        traj = t.get("traj", [])
        if t["track"] not in accounted and traj and \
                sum(_point_in_poly(x, y, street) for _, x, y, _ in traj) / len(traj) > sfrac:
            n += 1
    return n


def _day_report(date: str, windows=None) -> dict:
    """The B2B deliverable: human-confirmed unique customers + groups + dwell, and per-employee
    check-in/out. Reuses logic.grouping.group_sessions + store.attendance. Pass `windows` to scope
    to a subset (e.g. a single hour for an hour-level publish). The 'customers'/'employees' fields
    are kept stable (publish flow + tests); rich detail is added in extra keys."""
    import glob
    import statistics
    from collections import Counter
    from types import SimpleNamespace
    from logic.grouping import group_sessions
    entries, dwell = [], []
    if windows is None:
        windows = _day_windows(date)
    for win in windows:
        e, d = _confirmed_entries(win)
        entries += e
        dwell += d

    def secs(t):
        try:
            h, m, s = map(int, str(t).split(":"))
            return h * 3600 + m * 60 + s
        except Exception:
            return 0
    entries.sort(key=lambda e: (secs(e["ist"]), e["window"], e["track"]))   # DETERMINISTIC + identical order to _daily_tags -> report C# == E2 C#
    _mrg = _handoff_merges(date)["merge"]                                    # drop trajectory-handoff fragments -> de-duped count + members, matching _daily_tags
    entries = [e for e in entries if _mrg.get((e["window"], e["track"]), (e["window"], e["track"])) == (e["window"], e["track"])]
    sess = [SimpleNamespace(is_employee=False, entry_ts=secs(e["ist"]), session_id=i)
            for i, e in enumerate(entries)]
    _gres = group_sessions(sess, group_gap_sec=15.0) if sess else ({}, [])
    gmap, groups = _gres[0], _gres[1]              # gmap: session_id -> group_id (1..n) -> the #G tag
    att = store.attendance("s14", date)
    rank = _rank_labels()
    rank_num = _staff_no_map()
    timesheets = [{"employee": rank.get(a["id"], a.get("code")) + (" · " + a["name"] if a.get("name") else ""),
                   "check_in": a.get("first_seen"), "check_out": a.get("last_seen"),
                   "hours": a.get("windows")} for a in att if a.get("sightings")]

    # ---- rich detail (ADDITIVE; the stable fields above are untouched) ----
    cinfo = {win: _entry_crop_info(win) for win in windows}
    exitr = _exit_resolve(date)                                        # L4: resolved exit (matched/presumed) for entries with no visit OUT
    _xconf = _exit_confirmed(date)                                     # presumed exits a human has accepted -> 'exit inferred' chip clears
    rich = []                                                          # one per entry, aligned with sess session_ids
    for idx, e in enumerate(entries):                                  # idx == sess session_id -> C#/G# alignment
        ci = cinfo.get(e["window"], {}).get(e["track"], {})
        rx = exitr.get((e["window"], e["track"]), {})
        rich.append({"window": e["window"], "track": e["track"], "in_ist": e["ist"],
                     "pid": f"C{idx+1}", "group": (f"G{gmap[idx]}" if gmap.get(idx) else None),
                     "demo": _demo_label(ci.get("crop")),
                     "out_ist": ci.get("out_ist") or rx.get("out_ist"),
                     "dwell_s": ci.get("dwell_s") if ci.get("dwell_s") is not None else rx.get("dwell_s"),
                     "exit_src": rx.get("source"),                     # 'matched' | 'presumed' | None (real visit OUT)
                     "exit_confirmed": (e["window"], e["track"]) in _xconf,
                     "crop": ci.get("crop"), "out_crop": ci.get("out_crop") or rx.get("out_crop")})
    dwell = [r["dwell_s"] for r in rich if r.get("dwell_s") is not None]   # dwell-KPI over EVERY customer's resolved dwell (matched visit OR L4 exit), not just the matched visits
    customer_groups = []
    for g in groups:
        mems = sorted([rich[sid] for sid in g["session_ids"]], key=lambda m: secs(m["in_ist"]))
        if len(mems) > 1:
            spread = max(secs(m["in_ist"]) for m in mems) - min(secs(m["in_ist"]) for m in mems)
            reason = f"arrived together (within {spread}s)"
        else:
            reason = "arrived alone"
        customer_groups.append({"group_id": g["group_id"], "size": g["size"], "reason": reason, "members": mems})
    hot_leads = sorted([r for r in rich if r.get("dwell_s")], key=lambda r: r["dwell_s"], reverse=True)[:8]
    dwell_segments = {"browsers": sum(1 for d in dwell if d < 120),          # <2 min
                      "engaged": sum(1 for d in dwell if 120 <= d < 600),    # 2-10 min
                      "serious": sum(1 for d in dwell if d >= 600)}          # >10 min = hot
    bh = Counter()
    for e in entries:
        hh = str(e["ist"])[:2]
        if hh.isdigit():
            bh[hh] += 1
    footfall_by_hour = [{"hour": h + ":00", "entries": bh[h]} for h in sorted(bh)]
    peak_occupancy = 0
    for win in windows:
        try:
            c = json.loads((OUTPUTS / win / "visits.json").read_text(encoding="utf-8")).get("counts", {})
            peak_occupancy = max(peak_occupancy, c.get("peak_occupancy", 0) or 0)
        except Exception:
            pass
    passersby = sum(_passby_count(win) for win in windows)            # window-conversion: walked past vs entered (fast: C05 + visits.json only)
    footfall = len(entries)
    capture_rate = round(100 * footfall / (footfall + passersby), 1) if (footfall + passersby) else None
    gal = {}                                                         # enrolled face per staffer = the reliable thumbnail (door crop can be the wrong person)
    try:
        for ge in _gallery_meta():
            if ge.get("employee_id") and ge.get("crop_url"):
                gal.setdefault(ge["employee_id"], (ge["crop_url"] or "").replace("\\", "/"))
    except Exception:
        pass
    winset = set(windows)
    staff_detail = []                                                # per staffer: each in/out -> dwell, total, span (scoped to `windows`)
    for a in att:
        tl, total, times = [], 0, []
        for w in a.get("timeline", []):
            if w["window"] not in winset:                            # scope to this report's hours (per-hour report shows that hour's staff)
                continue
            dw = max(0, secs(w.get("out")) - secs(w.get("in")))
            total += dw
            times += [w.get("in"), w.get("out")]
            tl.append({"window": w["window"], "in": w.get("in"), "out": w.get("out"),
                       "dwell_min": round(dw / 60, 1), "crop": w.get("crop")})
        if not tl:
            continue
        times = [t for t in times if t]
        fi, lo = (min(times), max(times)) if times else (None, None)
        span = (secs(lo) - secs(fi)) if (fi and lo) else 0
        staff_detail.append({"employee_id": a["id"], "pid": f"S{rank_num.get(a['id'], a['id'])}",
                             "code": rank.get(a["id"], a.get("code")), "name": a.get("name"),
                             "crop": gal.get(a["id"]) or next((t["crop"] for t in tl if t.get("crop")), None),
                             "first_in": fi, "last_out": lo,
                             "span_min": round(span / 60, 1), "total_dwell_min": round(total / 60, 1),
                             "sightings": tl})

    return {
        "date": date, "windows": len(windows),
        "customers": {
            "unique_customers": len(entries),
            "groups": {"count": len(groups), "sizes": sorted((g["size"] for g in groups), reverse=True),
                       "solo": sum(1 for g in groups if g["size"] == 1),
                       "grouped": sum(1 for g in groups if g["size"] > 1)},
            "dwell_min": {"mean": round(statistics.mean(dwell) / 60, 1) if dwell else None,
                          "median": round(statistics.median(dwell) / 60, 1) if dwell else None},
        },
        "employees": {"headcount": len(timesheets), "timesheets": timesheets},
        # ---- rich sections for the redesigned report (additive) ----
        "kpis": {"footfall": footfall, "passersby": passersby, "capture_rate": capture_rate,
                 "peak_occupancy": peak_occupancy, "groups": len(groups),
                 "avg_dwell_min": round(statistics.mean(dwell) / 60, 1) if dwell else None,
                 "staff_on_duty": len(timesheets)},
        "footfall_by_hour": footfall_by_hour, "customer_groups": customer_groups,
        "hot_leads": hot_leads, "dwell_segments": dwell_segments, "staff_detail": staff_detail,
    }


@router.get("/report/{date}")
def report(date: str) -> dict:
    """Closed-day report for a date (YYYY-MM-DD): the shareable B2B summary."""
    if not date or "/" in date or "\\" in date or ".." in date:
        raise HTTPException(400, "bad date")
    return _day_report(date)


@router.get("/report-window/{window}")
def report_window(window: str) -> dict:
    """The SAME report scoped to one hour — each hour's review shows its slice of the day total
    (the day report is a sum of these hourly reports)."""
    _safe_window(window)
    return _day_report(window.split("_")[0], windows=[window])


def _report_for(period: str, scope: str) -> dict:
    if scope == "hour":
        rep = _day_report(period.split("_")[0], windows=[period])
    else:
        rep = _day_report(period)
    rep["period"], rep["scope"] = period, scope
    return rep


def _freeze_person_contexts(date: str, rep: dict) -> int:
    """Snapshot the day's #C/#G (one row per canonical customer PID) + on-duty staff (permanent S#)
    into person_contexts, so _daily_tags locks the numbering for this now-published day."""
    rows = []
    for g in rep.get("customer_groups", []):
        for m in g.get("members", []):
            pn, gn = str(m.get("pid") or "")[1:], str(m.get("group") or "")[1:]
            rows.append({"kind": "customer", "pid_no": int(pn) if pn.isdigit() else 0,
                         "group_no": int(gn) if gn.isdigit() else None,
                         "window_id": m.get("window"), "track": m.get("track"),
                         "in_ist": m.get("in_ist"), "out_ist": m.get("out_ist"),
                         "dwell_s": m.get("dwell_s"), "exit_src": m.get("exit_src")})
    for s in rep.get("staff_detail", []):
        sn = str(s.get("pid") or "")[1:]
        rows.append({"kind": "staff", "pid_no": int(sn) if sn.isdigit() else 0,
                     "employee_id": s.get("employee_id"), "in_ist": s.get("first_in"),
                     "out_ist": s.get("last_out")})
    return store.save_person_contexts(date, rows)


@router.post("/publish")
def publish(body: dict) -> dict:
    """Freeze the report for a period into a published snapshot the read-only public dashboard
    serves to the B2B client. 'period' = 'YYYY-MM-DD' (scope=day) or 'YYYY-MM-DD_HHMM' (scope=hour)."""
    _guard_write()
    period = (body or {}).get("period", "")
    scope = (body or {}).get("scope", "day")
    if not period or "/" in period or "\\" in period or ".." in period:
        raise HTTPException(400, "bad period")
    if scope not in ("day", "hour"):
        raise HTTPException(400, "scope must be 'day' or 'hour'")
    report = _report_for(period, scope)
    mv = (store.active_model() or {}).get("version")
    pid = store.publish_report(period, scope, report, model_version=mv)
    frozen_n = 0
    if scope == "day" and hasattr(store, "save_person_contexts"):     # freeze #C/#G into the durable PID registry
        try:
            frozen_n = _freeze_person_contexts(period, report)
            _guard_write()                                            # invalidate the day memo so the frozen numbers take effect now
        except Exception:
            frozen_n = 0
    return {"ok": True, "id": pid, "period": period, "scope": scope, "model_version": mv,
            "frozen_pids": frozen_n,
            "unique_customers": report["customers"]["unique_customers"],
            "headcount": report["employees"]["headcount"]}


@router.get("/published")
def published() -> dict:
    """Periods finalized for the client (read-only public dashboard reads these)."""
    return {"published": store.list_published()}


@router.get("/publish-history")
def publish_history() -> dict:
    """Full publish history (newest first) with key numbers — the Report-tab repository."""
    return {"history": store.published_history()}
