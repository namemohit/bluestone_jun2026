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


def _guard_write():
    if READONLY:
        raise HTTPException(403, "this is the read-only public view — review on the local console")


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
                thr = ((store.active_model() or {}).get("params") or {}).get("staff_sim")
                if thr:                                # use the trained threshold, not the default 0.6
                    cmd += ["--staff-sim", str(thr)]
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
    rank = {e["id"]: i + 1 for i, e in enumerate(sorted(emp_map.values(), key=lambda x: x["id"]))}

    def _label(eid):  # display as "Staff #N" by enrollment order (store keeps the raw S<id>)
        return f"Staff #{rank[eid]}" if eid in rank else (emp_map.get(eid) or {}).get("code")

    not_staff = {l["in_track"] for l in labels.values()                 # human override: NOT staff
                 if str(l.get("visit_id", "")).startswith("notstaff-") and l.get("verdict") == "reject"
                 and l.get("in_track") is not None}
    groups: dict = {}
    manual_tracks = set()

    def _grp(eid):
        return groups.setdefault(eid, {"employee_id": eid, "code": _label(eid),
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
        if a["track"] in not_staff:                     # user said this auto-match is wrong -> hide it
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
        for ge in store.get_gallery():
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
    return data


@router.get("/visits-day/{date}")
def visits_day(date: str) -> dict:
    """Whole-day rollup of the Review sections across the date's hourly windows (read-only overview)."""
    if not date or "/" in date or "\\" in date or ".." in date:
        raise HTTPException(400, "bad date")
    import glob
    windows = sorted({Path(p).parent.name for p in glob.glob(f"outputs/{date}_*/visits.json")})
    emp_map = {e["id"]: e for e in store.list_employees()}
    rank = {e["id"]: i + 1 for i, e in enumerate(sorted(emp_map.values(), key=lambda x: x["id"]))}
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
            # 'inside' = auto-handled (already counted at the door) -> excluded from the close-the-day denominator
            "close_total": sum(1 for d in dets if d.get("determination") != "inside"),
            "counts": {k: len(buckets.get(k, [])) for k in order}}


@router.get("/detections-day/{date}")
def detections_day(date: str) -> dict:
    """Whole-day reconciliation: every detection across the date's hourly windows, bucketed.
    Mirrors visits_day; each detection already carries its 'window' (set in _classify_detections)."""
    if not date or "/" in date or "\\" in date or ".." in date:
        raise HTTPException(400, "bad date")
    import glob
    windows = sorted({Path(p).parent.name for p in glob.glob(f"outputs/{date}_*/visits.json")})
    all_dets: list = []
    for win in windows:
        try:
            all_dets += _classify_detections(win, with_dims=False)   # skip crop-header reads -> fast day load (size filter is a per-hour tool)
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


def _classify_detections(window: str, with_dims: bool = True) -> list:
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
    for st in vj.get("staff", []):
        accounted.add(("C05", st["track"]))
        staff_emp[("C05", st["track"])] = st.get("employee_id")
        if camtrack(st.get("crop")):
            accounted.add(camtrack(st.get("crop")))
            staff_emp[camtrack(st.get("crop"))] = st.get("employee_id")
    for l in vj.get("links", []):                               # interior auto-linked to its door entry
        accounted.add(("C05", l["in_track"]))
        accounted.add((l["cam"], l["track"]))                   # the interior track -> accounted (suggested customer)
    labels = store.get_labels(window)
    not_staff = {l["in_track"] for l in labels
                 if str(l.get("visit_id", "")).startswith("notstaff-") and l.get("verdict") == "reject"
                 and l.get("in_track") is not None}
    for l in labels:
        if l.get("verdict") == "employee" and l.get("in_track") is not None and l["in_track"] not in not_staff:
            accounted.add(("C05", l["in_track"]))
            staff_emp[("C05", l["in_track"])] = l.get("employee_id")

    ann = {(a["camera"], a["track"]): a for a in store.latest_annotations(window)}  # human allocations
    parked = _parked_set(window)                                                    # held for later review
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
            if key in parked:                                     # held -> on_hold (a decision auto-unparks)
                sugg, conf, det = "on_hold", False, "on_hold"
            else:
                if disp == "accounted":
                    sugg = "staff" if key in staff_emp else "customer"
                elif disp == "street":
                    sugg = "passby"
                elif disp == "inside":
                    # only GENUINE noise -> not_person: a brief blip (<=4 frames) that's also unconfident,
                    # tiny, or static. Real interior people (median conf 0.80, decent size) -> to_review,
                    # so we no longer mislabel ~90% of them as 'not a person' (the old dur<3s rule did).
                    pconf = t.get("peak_conf", 1.0) or 1.0
                    nfr = t.get("frames", 0) or 0
                    maxw = max((p[3] for p in traj if len(p) >= 4), default=0.0)
                    span = ((max((p[1] for p in traj), default=0) - min((p[1] for p in traj), default=0))
                            + (max((p[2] for p in traj), default=0) - min((p[2] for p in traj), default=0))) if traj else 0.0
                    noise = nfr <= 4 and (pconf < 0.55 or maxw < 0.06 or span < 0.02)
                    sugg = "not_person" if noise else "inside"   # real interior, not matched -> Inside bucket
                else:                                             # door -> the human must decide enter/pass
                    sugg = "to_review"
                conf = ann_cat is not None                        # confirmed = a human annotation exists
                det = ann_cat if conf else sugg
            cropr = (t.get("crop", "") or "").replace("\\", "/")
            cw, ch = _crop_dims(cropr) if with_dims else (None, None)   # day view skips per-crop header reads
            out.append({"camera": cam, "track": t["track"], "window": window, "ist": t.get("first_ist"), "dur_s": dur,
                        "crop": cropr, "crop_w": cw, "crop_h": ch,
                        "disposition": disp, "staff": key in staff_emp, "annotation": ann_cat,
                        "parked": key in parked, "suggested": sugg, "confirmed": conf,
                        "determination": det, "employee_id": staff_emp.get(key)})
    return out


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
    store.add_label(window, f"notstaff-{track}", "reject", reason="not staff", in_track=track)
    _rerun_l4(window)
    return visits(window)


def _emb_for_crop(crop):
    """OSNet embedding for a crop pulled from the L4 cache (no GPU); None if absent."""
    import pickle
    if not crop or not os.path.exists("outputs/osnet_emb_cache.pkl"):
        return None
    try:
        with open("outputs/osnet_emb_cache.pkl", "rb") as f:
            cache = pickle.load(f)
    except Exception:
        return None
    return {str(k).replace("\\", "/"): v for k, v in cache.items()}.get(str(crop).replace("\\", "/"))


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


@router.post("/train")
def train(body: dict) -> dict:
    """Initiate training. mode='rebuild' = gallery + threshold from confirmed labels (learning-free,
    instant, no GPU). mode='finetune' = deep OSNet fine-tune (background job, needs a free GPU)."""
    _guard_write()
    mode = (body or {}).get("mode", "rebuild")
    date = (body or {}).get("date") or None
    if mode == "rebuild":
        return _train_rebuild(date)
    if mode == "finetune":
        return _train_finetune(date)
    raise HTTPException(400, "mode must be 'rebuild' or 'finetune'")


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
    """Pull the marked crop's OSNet embedding from the L4 cache (no GPU) and add it to the gallery."""
    import pickle
    cache_path = "outputs/osnet_emb_cache.pkl"
    if not os.path.exists(cache_path):
        return
    try:
        with open(cache_path, "rb") as f:
            cache = pickle.load(f)
    except Exception:
        return
    norm = {str(k).replace("\\", "/"): v for k, v in cache.items()}  # cache keys are mixed-separator
    emb = norm.get(str(crop).replace("\\", "/"))
    if emb is not None:
        try:
            store.enroll_staff(employee_id, "s14", emb, crop_url=crop, window=window, track=track)
        except Exception:
            pass


def _rank_labels() -> dict:
    """employee id -> 'Staff #N' by enrollment order (display-only; the store keeps the raw S<id>)."""
    emps = sorted(store.list_employees(), key=lambda x: x["id"])
    return {e["id"]: f"Staff #{i + 1}" for i, e in enumerate(emps)}


@router.get("/employees")
def employees() -> dict:
    emps = sorted(store.list_employees(), key=lambda x: x["id"])
    thumb: dict = {}
    try:                                   # a representative enrolled crop per staffer, for the picker thumbnails
        for g in store.get_gallery():
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
    return out


_RESOLVED_NOT_CUSTOMER = ("staff", "not_person", "passby", "duplicate")


def _confirmed_entries(window):
    """(window,track,ist) for each human-confirmed customer door entry + matched-visit dwells."""
    try:
        data = json.loads((OUTPUTS / window / "visits.json").read_text(encoding="utf-8"))
    except Exception:
        return [], []
    ann = {a["track"]: a["category"] for a in store.latest_annotations(window) if a["camera"] == "C05"}
    auto = {}                                                  # door-IN tracks (post-rerun) -> ist
    for v in data.get("visits", []):
        auto[v["in_track"]] = v["in_ist"]
    for e in data.get("open_sessions", []):
        auto[e["track"]] = e["ist"]
    extra = [t for t, c in ann.items() if c == "customer" and t not in auto]   # human-added missed
    det_ist = {d["track"]: d["ist"] for d in store.get_detections(window)
               if d.get("camera") == "C05"} if extra else {}
    entries = [{"window": window, "track": t, "ist": ist} for t, ist in auto.items()
               if ann.get(t) not in _RESOLVED_NOT_CUSTOMER]
    entries += [{"window": window, "track": t, "ist": det_ist[t]} for t in extra if det_ist.get(t)]
    dwell = [v["dwell_s"] for v in data.get("visits", [])
             if v.get("dwell_s") and ann.get(v["in_track"]) not in _RESOLVED_NOT_CUSTOMER]
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


def _day_report(date: str, windows=None) -> dict:
    """The B2B deliverable: human-confirmed unique customers + groups + dwell, and per-employee
    check-in/out. Reuses logic.grouping.group_sessions + store.attendance. Pass `windows` to scope
    to a subset (e.g. a single hour for an hour-level publish)."""
    import glob
    import statistics
    from types import SimpleNamespace
    from logic.grouping import group_sessions
    entries, dwell = [], []
    if windows is None:
        windows = sorted({Path(p).parent.name for p in glob.glob(f"outputs/{date}_*/visits.json")})
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
    entries.sort(key=lambda e: secs(e["ist"]))
    sess = [SimpleNamespace(is_employee=False, entry_ts=secs(e["ist"]), session_id=i)
            for i, e in enumerate(entries)]
    groups = group_sessions(sess, group_gap_sec=15.0)[1] if sess else []
    att = store.attendance("s14", date)
    rank = _rank_labels()
    timesheets = [{"employee": rank.get(a["id"], a.get("code")) + (" · " + a["name"] if a.get("name") else ""),
                   "check_in": a.get("first_seen"), "check_out": a.get("last_seen"),
                   "hours": a.get("windows")} for a in att if a.get("sightings")]
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
    }


@router.get("/report/{date}")
def report(date: str) -> dict:
    """Closed-day report for a date (YYYY-MM-DD): the shareable B2B summary."""
    if not date or "/" in date or "\\" in date or ".." in date:
        raise HTTPException(400, "bad date")
    return _day_report(date)


def _report_for(period: str, scope: str) -> dict:
    if scope == "hour":
        rep = _day_report(period.split("_")[0], windows=[period])
    else:
        rep = _day_report(period)
    rep["period"], rep["scope"] = period, scope
    return rep


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
    return {"ok": True, "id": pid, "period": period, "scope": scope, "model_version": mv,
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
