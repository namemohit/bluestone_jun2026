"""Before/after: cluster the C11-day bucket cards with the ACTIVE model vs the new C11-trained checkpoint.

Embeds every bucket-card crop with BOTH models, prints the unique-count-vs-threshold curve (a STABLE
plateau = clean separation; a wild swing = the over-merge collapse) + the stranger FLOOD (random
different-card sims; lower = better separation), and SAVES the C11-model embeddings to a side cache
(outputs/c11_model_emb_cache.pkl) so the Cluster tab can toggle between models with no live promotion.

  python -m training.compare_c11_model --ckpt "outputs/models/osnet_c11_tracklet_*.pt" --date 2026-06-03-c
"""
from __future__ import annotations

import argparse
import glob
import pickle
import random
from pathlib import Path

import cv2
import numpy as np

import dashboard.hitl_api as H
from stack import reid

OUT_CACHE = "outputs/c11_model_emb_cache.pkl"


def rep(card, cc):
    vs = [np.asarray(cc[p], "float32") for s in card.get("crops", [])
          for p in [(s.get("crop") or "").replace("\\", "/")] if p in cc]
    if not vs:
        return None
    m = np.mean(vs, axis=0); n = float(np.linalg.norm(m))
    return (m / n) if n > 0 else None


def sweep(cards, cc, label):
    embs = [e for e in (rep(c, cc) for c in cards) if e is not None]
    if not embs:
        print(f"\n{label}: no embeddings"); return
    M = np.vstack(embs); S = M @ M.T
    n = len(embs)
    print(f"\n{label}: {n} customer cards")
    for thr in (0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80):
        p = list(range(n))
        def f(x):
            while p[x] != x:
                p[x] = p[p[x]]; x = p[x]
            return x
        for a in range(n):
            for b in range(a + 1, n):
                if S[a][b] >= thr:
                    p[f(a)] = f(b)
        uniq = len({f(i) for i in range(n)})
        print(f"   thr={thr:.2f} -> {uniq:3d} unique")
    rng = random.Random(0)
    rs = []
    for _ in range(3000):
        a, b = rng.randrange(n), rng.randrange(n)
        if a != b:
            rs.append(float(S[a][b]))
    print(f"   stranger sims: mean={np.mean(rs):.3f}  p90={np.percentile(rs,90):.3f}  p99={np.percentile(rs,99):.3f}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--date", default="2026-06-03-c")
    args = ap.parse_args()
    ckpt = sorted(glob.glob(args.ckpt))[-1] if "*" in args.ckpt else args.ckpt
    print(f"checkpoint: {ckpt}")

    H._STATE_VERSION += 1
    active = H._load_emb_cache()
    cards = H._cards_from_dets(args.date)
    cust, staff = cards["customer_cards"], cards["staff_cards"]

    crops = sorted({(s.get("crop") or "").replace("\\", "/")
                    for c in cust + staff for s in c.get("crops", []) if s.get("crop")})
    print(f"embedding {len(crops)} bucket-card crops with the C11 model (CPU)…", flush=True)
    c11 = {}
    for i, p in enumerate(crops):
        im = cv2.imread(p)
        if im is None:
            continue
        c11[p] = reid._triplet_embed(im, ckpt)
        if (i + 1) % 100 == 0:
            print(f"  {i+1}/{len(crops)}", flush=True)
    pickle.dump(c11, open(OUT_CACHE, "wb"))
    print(f"saved C11-model cache: {len(c11)} crops -> {OUT_CACHE}")

    sweep(cust, active, "=== ACTIVE model (BEFORE) ===")
    sweep(cust, c11, "=== C11-trained model (AFTER) ===")


if __name__ == "__main__":
    main()
