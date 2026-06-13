"""Extract the TWO canonical crops for every C11 line-crossing tracklet in an `l2_crossings` namespace
(built by `reprocess_pure --person-source l2_crossings`):

  1. LINE-MOMENT crop  — the person at the exact frame their foot crossed the door line (`in_ts`/`out_ts`).
                         Proof of the crossing.
  2. BEST-FACE crop    — the clearest face along the track's trajectory (max InsightFace det_score).
                         Clothing-invariant ID; its 512-d ArcFace embedding is written into the Builder's
                         face cache (outputs/face_emb_cache.json) for free.

Reuses the same frame-grab + bbox geometry as training.build_c11_tracklets (grab/bbox_px/start_epoch) so the
crops land exactly where the trajectory says. Updates each crossings.json entry in place with `line_crop`,
`face_crop`, `face_score`. Idempotent: a crossing whose crops already exist is skipped.

  python -m batch.extract_crossing_crops --date 2026-06-03-dx          # CPU (default)
  python -m batch.extract_crossing_crops --date 2026-06-03-dx --gpu    # needs onnxruntime-gpu
"""
from __future__ import annotations

import argparse
import glob
import json
import pathlib

import cv2
import numpy as np

from training.build_c11_tracklets import bbox_px, grab, start_epoch
from vision.demographics import InsightFaceEstimator

FACE_CACHE = pathlib.Path("outputs/face_emb_cache.json")


def _aspect(track: dict) -> float:
    cp = track.get("crop")
    if cp and pathlib.Path(cp).exists():
        im = cv2.imread(cp)
        if im is not None and im.shape[0] > 0:
            return im.shape[1] / im.shape[0]
    return 0.45


def _nearest_pt(traj: list, ts: float):
    return min(traj, key=lambda p: abs(p[0] - ts)) if traj else None


def _crop_at(slice_path: str, st: float, pt, aspect: float, min_side: int):
    """Grab the frame at pt's timestamp and crop the person bbox; None if unusable."""
    fr = grab(slice_path, max(0.0, pt[0] - st))
    if fr is None:
        return None
    H, W = fr.shape[:2]
    x1, y1, x2, y2 = bbox_px(pt, W, H, aspect)
    if x2 - x1 < min_side or y2 - y1 < min_side:
        return None
    crop = fr[y1:y2, x1:x2]
    if crop.size == 0 or float(crop.std()) < 12.0:        # blank / black / out-of-frame
        return None
    return crop


def main() -> None:
    ap = argparse.ArgumentParser(description="Line-moment + best-face crops for an l2_crossings namespace")
    ap.add_argument("--date", required=True, help="namespace date incl. tag, e.g. 2026-06-03-dx")
    ap.add_argument("--gpu", action="store_true", help="InsightFace on CUDA (needs onnxruntime-gpu)")
    ap.add_argument("--face-views", type=int, default=6, help="frames scanned along the track for the best face")
    ap.add_argument("--min-side", type=int, default=44, help="skip crops smaller than this (px)")
    args = ap.parse_args()

    xj_paths = sorted(glob.glob(f"outputs/{args.date}_*/crossings.json"))
    if not xj_paths:
        print(f"no crossings.json under outputs/{args.date}_*  — run reprocess_pure --person-source l2_crossings first")
        return

    est = InsightFaceEstimator(gpu=args.gpu)
    cache = json.loads(FACE_CACHE.read_text(encoding="utf-8")) if FACE_CACHE.exists() else {}
    FACE_CACHE.parent.mkdir(parents=True, exist_ok=True)
    tot_line = tot_face = 0

    for xp in xj_paths:
        ns = pathlib.Path(xp).parent
        data = json.loads(pathlib.Path(xp).read_text(encoding="utf-8"))
        xs = data.get("crossings", [])
        wcfg = json.loads((ns / "window.json").read_text(encoding="utf-8"))
        try:
            tj = json.loads((pathlib.Path(wcfg["l1"]) / "tracks.json").read_text(encoding="utf-8"))
        except Exception as e:
            print(f"[xc] {ns.name}: no tracks.json ({type(e).__name__}) — skip"); continue
        st = start_epoch(tj)
        tdict = {t["track"]: t for t in tj.get("tracks", [])}
        slice_path = str(ns / "slices" / "entry.ts")
        if not pathlib.Path(slice_path).exists():
            slice_path = f"outputs/{wcfg.get('source_window')}/slices/C11.ts"
        cropdir = ns / "L2_C11" / "crops"
        cropdir.mkdir(parents=True, exist_ok=True)

        n_line = n_face = 0
        for x in xs:
            trk = x["track"]
            t = tdict.get(trk)
            if not t or not t.get("traj"):
                continue
            traj = t["traj"]
            aspect = _aspect(t)
            line_fp = cropdir / f"line_{trk}.jpg"
            face_fp = cropdir / f"face_{trk}.jpg"

            # 1) line-moment crop at the crossing ts (prefer IN; else OUT)
            cross_ts = x.get("in_ts") if x.get("in_ts") is not None else x.get("out_ts")
            if not line_fp.exists() and cross_ts is not None:
                pt = _nearest_pt(traj, cross_ts)
                cr = _crop_at(slice_path, st, pt, aspect, args.min_side) if pt else None
                if cr is not None:
                    cv2.imwrite(str(line_fp), cr, [int(cv2.IMWRITE_JPEG_QUALITY), 92]); n_line += 1
            if line_fp.exists():
                x["line_crop"] = str(line_fp).replace("\\", "/")

            # 2) best-face crop = max det_score over evenly-spaced trajectory frames
            if not face_fp.exists():
                best = None  # (score, crop, emb)
                for i in np.linspace(0, len(traj) - 1, args.face_views).astype(int):
                    cr = _crop_at(slice_path, st, traj[int(i)], aspect, args.min_side)
                    if cr is None:
                        continue
                    r = est.estimate(cr, with_emb=True)
                    if r.get("face") and r.get("face_emb") and (best is None or r["face_score"] > best[0]):
                        best = (r["face_score"], cr, r["face_emb"])
                if best is not None:
                    cv2.imwrite(str(face_fp), best[1], [int(cv2.IMWRITE_JPEG_QUALITY), 92]); n_face += 1
                    cache[str(face_fp).replace("\\", "/")] = {"emb": best[2], "score": round(float(best[0]), 4)}
            if face_fp.exists():
                x["face_crop"] = str(face_fp).replace("\\", "/")
                fc = cache.get(str(face_fp).replace("\\", "/"))
                if fc:
                    x["face_score"] = fc.get("score")

        pathlib.Path(xp).write_text(json.dumps(data), encoding="utf-8")
        FACE_CACHE.write_text(json.dumps(cache), encoding="utf-8")          # incremental: resumable
        print(f"[xc] {ns.name}: +{n_line} line, +{n_face} face  ({len(xs)} tracklets)", flush=True)
        tot_line += n_line; tot_face += n_face

    n_total = sum(len(json.loads(pathlib.Path(p).read_text(encoding='utf-8')).get('crossings', [])) for p in xj_paths)
    print(f"[xc] DONE: {tot_line} line + {tot_face} face crops over {n_total} tracklets "
          f"({round(100 * tot_face / max(1, n_total))}% have a usable face) -> {FACE_CACHE}")


if __name__ == "__main__":
    main()
