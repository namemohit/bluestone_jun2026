"""Full re-embed of the OSNet cache into the CURRENTLY-ACTIVE embedding space.

When a contrastive fine-tune is promoted (outputs/reid_active_weights.txt points at the new
weights), the existing outputs/osnet_emb_cache.pkl is still in the OLD space — every cosine in
it is meaningless against new-space vectors. This re-embeds EVERY cached crop in place (same key)
through stack.reid.osnet_embed, which now loads the promoted model. Keys whose crop file is gone
are DROPPED, never kept, so the cache stays purely one space (mixing spaces silently corrupts
similarity). The running dashboard mtime-invalidates and picks the new pickle up automatically.

  python -m batch.reembed_all                 # re-embed every key in the cache, in place
  python -m batch.reembed_all --date 2026-06-03   # also scan the day's crops for any new files

Back up outputs/osnet_emb_cache.pkl FIRST (the promote flow does this) — this overwrites it.
"""
from __future__ import annotations

import argparse
import glob
import os
import pickle
import time

import cv2

from stack import reid as R

CACHE = "outputs/osnet_emb_cache.pkl"


def main() -> None:
    ap = argparse.ArgumentParser(description="Re-embed the whole OSNet cache into the active space")
    ap.add_argument("--date", default=None, help="also scan outputs/<date>_*/L1_*/crops for new files")
    ap.add_argument("--checkpoint", type=int, default=500, help="flush the pickle every N crops")
    args = ap.parse_args()

    aw = R._active_weights()
    active = f"{aw[0]}:{aw[1]}" if aw else None
    R.reset_osnet()
    print(f"[reembed] active weights: {active or '(stock boxmot OSNet)'}", flush=True)

    old = pickle.load(open(CACHE, "rb")) if os.path.exists(CACHE) else {}
    keys = list(old)
    if args.date:
        have = {str(k).replace("\\", "/") for k in old}
        extra = [p for p in glob.glob(f"outputs/{args.date}_*/L1_*/crops/*.jpg")
                 if p.replace("\\", "/") not in have]
        keys += extra
        print(f"[reembed] +{len(extra)} new crops from {args.date} scan", flush=True)

    print(f"[reembed] {len(keys)} crops to re-embed (old cache had {len(old)})", flush=True)

    new: dict = {}
    done = missing = failed = 0
    t0 = time.time()
    tmp = CACHE + ".tmp"                      # checkpoint to a temp file; swap the LIVE cache once, atomically
    for i, k in enumerate(keys):
        path = str(k)
        im = cv2.imread(path)
        if im is None:                       # file gone -> DROP (never keep an old-space vector)
            missing += 1
            continue
        try:
            v = R.osnet_embed(im)
        except Exception:
            failed += 1
            continue
        if v is not None:
            new[k] = v
            done += 1
        if (i + 1) % args.checkpoint == 0:
            pickle.dump(new, open(tmp, "wb"))
            rate = (i + 1) / max(1e-6, time.time() - t0)
            print(f"[reembed] {i + 1}/{len(keys)} (+{done}) {rate:.1f}/s", flush=True)

    pickle.dump(new, open(tmp, "wb"))
    os.replace(tmp, CACHE)                    # atomic swap: dashboard sees the full new-space cache in one step
    # stamp the space so we can detect accidental mixing later
    open("outputs/osnet_emb_cache.space.txt", "w", encoding="utf-8").write(
        f"weights={active or 'stock'}\nentries={len(new)}\nstamped={time.strftime('%Y-%m-%d %H:%M:%S')}\n")
    print(f"[reembed] DONE - embedded {done}, dropped {missing} missing, {failed} failed; "
          f"cache now {len(new)} (was {len(old)})", flush=True)


if __name__ == "__main__":
    main()
