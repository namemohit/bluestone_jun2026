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
        except Exception:
            pass
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        raise HTTPException(500, f"L4 re-run failed: {r.stderr[-400:]}")
    store.sync(window)            # push the re-routed result into the DB (no-op for local)
    store.record_metrics(window)  # append a point to the precision curve


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


@router.get("/detections/{window}")
def detections(window: str) -> dict:
    """L1 — every human the cameras detected (door + interior), incl. staff."""
    _safe_window(window)
    return {"detections": store.get_detections(window)}


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


@router.post("/label")
def label(body: dict) -> dict:
    _guard_write()
    window = body.get("window", "")
    _window_dir(window)
    verdict = body.get("verdict", "")
    emp_id = body.get("employee_id")
    store.add_label(window, body.get("visit_id", ""), verdict, reason=body.get("reason", ""),
                    in_track=body.get("in_track"), out_track=body.get("out_track"), employee_id=emp_id)
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
    return {"employees": [{**e, "code": f"Staff #{i + 1}"} for i, e in enumerate(emps)]}


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
