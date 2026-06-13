"""Cache a 512-d ArcFace FACE embedding for every C11 crop of a day — the clothing-invariant half of the
Face⊕Body fusion ReID. Runs InsightFace buffalo_l once per crop; ~half the CCTV crops have a usable face
(the rest are skipped). Path-keyed cache the dashboard's fusion model reads at cluster time.

  python -m batch.extract_face_emb --date 2026-06-03          # CPU (default)
  python -m batch.extract_face_emb --date 2026-06-03 --gpu    # needs onnxruntime-gpu

Variant C/D reuse the SAME base L1 crops, so extracting the base date's C11 crops covers them too (the cache
is keyed by the real crop path). Idempotent/incremental: only embeds crops not already cached.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import pathlib

from vision.demographics import InsightFaceEstimator

CACHE = pathlib.Path("outputs/face_emb_cache.json")


def _card_crops(date: str, cam: str) -> set[str]:
    """Only the crops that appear in this day's Builder cards (staff+customer) — the set fusion needs.
    Far smaller than the full L1 sweep."""
    import dashboard.hitl_api as H
    H._STATE_VERSION += 1
    out: set[str] = set()
    try:
        a = H._cards_from_dets(date)
        for c in a["customer_cards"] + a["staff_cards"]:
            for s in c.get("crops", []):
                cr = (s.get("crop") or "").replace("\\", "/")
                if f"L1_{cam}" in cr:
                    out.add(cr)
    except Exception as e:
        print(f"  (cards for {date} unavailable: {type(e).__name__})")
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description="Cache ArcFace face embeddings for a day's C11 card-crops")
    ap.add_argument("--date", required=True, help="base date YYYY-MM-DD")
    ap.add_argument("--gpu", action="store_true", help="use CUDA (needs onnxruntime-gpu)")
    ap.add_argument("--cam", default="C11", help="camera dir to embed (default C11)")
    ap.add_argument("--all", action="store_true",
                    help="embed EVERY L1_<cam> crop (slow); default = only crops in the cards (base + -c + -d)")
    args = ap.parse_args()

    if args.all:
        crops = set(p.replace("\\", "/") for p in glob.glob(f"outputs/{args.date}_*/L1_{args.cam}/crops/*.jpg"))
    else:
        crops = set()
        for d in (args.date, f"{args.date}-c", f"{args.date}-d"):     # variants reuse the same base crops
            crops |= _card_crops(d, args.cam)
    crops = sorted(crops)
    if not crops:
        print(f"no {args.cam} crops for {args.date}"); return

    cache = json.loads(CACHE.read_text(encoding="utf-8")) if CACHE.exists() else {}
    todo = [c for c in crops if c not in cache]
    print(f"{len(crops)} {args.cam} card-crops; {len(todo)} to embed ({len(crops) - len(todo)} cached)", flush=True)
    if todo:
        est = InsightFaceEstimator(gpu=args.gpu)
        CACHE.parent.mkdir(parents=True, exist_ok=True)
        for i, c in enumerate(todo):
            r = est.estimate(os.path.normpath(c), with_emb=True)
            cache[c] = ({"emb": r["face_emb"], "score": r.get("face_score", 0.0)}
                        if r.get("face") and r.get("face_emb") else None)
            if (i + 1) % 20 == 0 or i + 1 == len(todo):
                CACHE.write_text(json.dumps(cache), encoding="utf-8")   # incremental: resumable
                print(f"  {i + 1}/{len(todo)}", flush=True)
    faces = sum(1 for c in crops if cache.get(c))
    print(f"done -> {CACHE}  |  faces: {faces}/{len(crops)} ({round(100*faces/max(1,len(crops)))}%)")


if __name__ == "__main__":
    main()
