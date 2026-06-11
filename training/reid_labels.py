"""Reconstruct the human-confirmed ReID pairs (✓ same / ✗ different) as MODEL-AGNOSTIC crop-path pairs,
so the labelling effort is reusable by ANY future ReID backbone (re-embed the JPEGs, train) — not just OSNet.

Pure: (label/annotation rows from the store + local visits.json) -> crop image paths. NO embeddings, NO
OSNet. Mirrors dashboard/hitl_api.py `_reid_labeled_pairs` + `_entry_crop_info`, but emits paths not sims.
Kept dependency-light (no FastAPI dashboard import) so the standalone training scripts can reuse it.
"""
from __future__ import annotations

import glob
import json
import re
from pathlib import Path

OUTPUTS = Path("outputs")

_CROPINFO: dict = {}


def day_windows(date: str | None = None) -> list[str]:
    """Operating-hours windows (outputs/<date>_HHMM, 4-digit) for one date, or ALL dates when date is None.
    Excludes the pre-open '_09' stub and test slices (same filter as the dashboard's _day_windows)."""
    pat = f"outputs/{date}_*/visits.json" if date else "outputs/*/visits.json"
    return sorted({Path(p).parent.name for p in glob.glob(pat)
                   if re.search(r"_\d{4}$", Path(p).parent.name)})


def _entry_crop_info(window: str) -> dict:
    """track -> door crop path, from the window's visits.json (+ tracks.json door-crop fallback) — the
    same resolution dashboard/hitl_api.py `_entry_crop_info` uses, minus the report extras. Memoized."""
    if window in _CROPINFO:
        return _CROPINFO[window]
    info: dict = {}
    try:
        data = json.loads((OUTPUTS / window / "visits.json").read_text(encoding="utf-8"))
    except Exception:
        _CROPINFO[window] = info
        return info
    for v in data.get("visits", []):
        if v.get("in_crop"):
            info[v["in_track"]] = v["in_crop"].replace("\\", "/")
    for e in data.get("open_sessions", []):
        if e.get("crop"):
            info.setdefault(e["track"], e["crop"].replace("\\", "/"))
    try:                                                       # door-crop fallback: any C05 track -> its crop
        wcfg = json.loads((OUTPUTS / window / "window.json").read_text(encoding="utf-8"))
        for t in json.loads((Path(wcfg["l1"]) / "tracks.json").read_text(encoding="utf-8")).get("tracks", []):
            cr = (t.get("crop") or "").replace("\\", "/")
            if cr and not info.get(t["track"]):
                info[t["track"]] = cr
    except Exception:
        pass
    _CROPINFO[window] = info
    return info


def _crop_for(window, track):
    try:
        track = int(track)
    except (TypeError, ValueError):
        return None
    return _entry_crop_info(window).get(track) or None


_INTCROP: dict = {}


def _interior_crop_for(window: str, cam: str, track):
    """Interior (C11/C14) crop path for (window, cam, track), from that camera's L1 tracks.json. Memoized."""
    key = (window, cam)
    if key not in _INTCROP:
        info: dict = {}
        try:
            tj = json.loads((OUTPUTS / window / f"L1_{cam}" / "tracks.json").read_text(encoding="utf-8"))
            for t in tj.get("tracks", []):
                cr = (t.get("crop") or "").replace("\\", "/")
                if cr:
                    info[t["track"]] = cr
        except Exception:
            pass
        _INTCROP[key] = info
    try:
        track = int(track)
    except (TypeError, ValueError):
        return None
    return _INTCROP[key].get(track)


def _parse_xcam3(s: str):
    """'<door_track>-<cam>-<interior_track>' -> (door_track:int, cam:str, interior_track:int)."""
    try:
        at, cam, bt = s.split("-")
        return int(at), cam, int(bt)
    except ValueError:
        return None


def human_pairs(store, date: str | None = None) -> list[dict]:
    """Every human-confirmed ReID pair for one date (or all-time when date is None), as crop-path rows:
      {label:'same'|'different', crop_a, crop_b, window_a, track_a, window_b, track_b, reviewer, reason}

      same      = a 'duplicate' annotation (latest-wins, so a REVOKED merge re-annotated 'customer' is
                  correctly excluded); b is the merged track, a is its `duplicate_of`.
      different = a 'reiddiff-<at>-<bw>-<bt>' or 'distinct-<a>-<b>' label whose latest verdict is 'reject'.

    Pairs whose crop can't be resolved on disk are skipped. No embeddings — a future backbone re-embeds
    the JPEGs. Window ids contain '-', so reiddiff ids parse at | bw | bt by first/last dash."""
    wins = day_windows(date)
    labels = (store.get_labels_many(wins) if hasattr(store, "get_labels_many")
              else {w: store.get_labels(w) for w in wins})
    anns = (store.latest_annotations_bulk(wins) if hasattr(store, "latest_annotations_bulk")
            else {w: store.latest_annotations(w) for w in wins})
    track_win: dict = {}                                       # door-entry track -> its window (recover a_window for merges)
    for w in wins:
        for tk in _entry_crop_info(w):
            track_win.setdefault(tk, w)

    out: list = []
    seen: set = set()

    def add(label, aw, at, bw, bt, reviewer, reason, crop_b=None, cam_b="C05"):
        try:
            at, bt = int(at), int(bt)
        except (TypeError, ValueError):
            return
        key = (label, aw, at, cam_b, bw, bt)
        if key in seen:
            return
        ca = _crop_for(aw, at)
        cb = crop_b or (_interior_crop_for(bw, cam_b, bt) if cam_b != "C05" else _crop_for(bw, bt))
        if not ca or not cb:
            return
        seen.add(key)
        out.append({"label": label, "crop_a": ca, "crop_b": cb,
                    "window_a": aw, "track_a": at, "window_b": bw, "track_b": bt,
                    "camera_a": "C05", "camera_b": cam_b, "cross_camera": cam_b != "C05",
                    "reviewer": reviewer or "human", "reason": reason or ""})

    for w in wins:
        for l in labels.get(w, []):
            if l.get("verdict") != "reject":                  # latest-wins: a 'reset' (undo) drops the pair
                continue
            vid = str(l.get("visit_id", ""))
            if vid.startswith("reiddiff-"):
                try:
                    at, rest = vid.split("-", 1)[1].split("-", 1)
                    bw, bt = rest.rsplit("-", 1)
                except ValueError:
                    continue
                add("different", w, at, bw, bt, l.get("reviewer"), l.get("reason"))
            elif vid.startswith("distinct-"):
                try:
                    a, b = vid[len("distinct-"):].split("-")
                except ValueError:
                    continue
                add("different", w, a, w, b, l.get("reviewer"), l.get("reason"))
        for l in labels.get(w, []):                            # cross-camera door<->interior verdicts (same-window, C05<->C11/C14)
            vid = str(l.get("visit_id", "")); v = l.get("verdict")
            if vid.startswith("xcamsame-") and v == "confirm":
                p = _parse_xcam3(vid[len("xcamsame-"):])
                if p:
                    add("same", w, p[0], w, p[2], l.get("reviewer"), l.get("reason"), cam_b=p[1])
            elif vid.startswith("xcamdiff-") and v == "reject":
                p = _parse_xcam3(vid[len("xcamdiff-"):])
                if p:
                    add("different", w, p[0], w, p[2], l.get("reviewer"), l.get("reason"), cam_b=p[1])
        for a in anns.get(w, []):
            if a.get("category") == "duplicate" and a.get("duplicate_of") is not None and a.get("track") is not None:
                aw = track_win.get(a["duplicate_of"], w)       # a_window unknown in the row -> recover; default same window
                add("same", aw, a["duplicate_of"], w, a["track"], a.get("reviewer"),
                    "reid: confirmed same person (merge)",
                    crop_b=((a.get("crop_url") or "").replace("\\", "/") or None))
    return out
