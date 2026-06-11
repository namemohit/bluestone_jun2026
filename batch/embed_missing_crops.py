"""Embed every day crop missing from the OSNet cache (notably the C05 door crops the L4 bridge
skipped — it only embedded interior crops). One-time GPU pass so the ReID trainer can find
similar images for EVERY person. Appends to outputs/osnet_emb_cache.pkl (load -> extend -> save).

Run:  python -m batch.embed_missing_crops 2026-06-03
"""
from __future__ import annotations

import glob
import os
import pickle
import sys

import cv2

from stack.reid import osnet_embed

CACHE = "outputs/osnet_emb_cache.pkl"


def main() -> None:
    date = sys.argv[1] if len(sys.argv) > 1 else "2026-06-03"
    cache = pickle.load(open(CACHE, "rb")) if os.path.exists(CACHE) else {}
    have = {str(k).replace("\\", "/") for k in cache}
    todo = [p for p in glob.glob(f"outputs/{date}_*/L1_*/crops/*.jpg")
            if p.replace("\\", "/") not in have]
    print(f"[embed] {len(todo)} crops to embed (cache has {len(cache)})", flush=True)
    done = 0
    for i, p in enumerate(todo):
        im = cv2.imread(p)
        if im is None:
            continue
        try:
            v = osnet_embed(im)
        except Exception:
            continue
        if v is not None:
            cache[p] = v
            done += 1
        if (i + 1) % 300 == 0:
            pickle.dump(cache, open(CACHE, "wb"))
            print(f"[embed] {i + 1}/{len(todo)} (+{done})", flush=True)
    pickle.dump(cache, open(CACHE, "wb"))
    print(f"[embed] DONE - embedded {done}, cache now {len(cache)}", flush=True)


if __name__ == "__main__":
    main()
