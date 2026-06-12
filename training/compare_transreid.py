"""The "see the difference" money table. On ONE held-out set (disjoint VAL tracklets, random strangers),
score every ReID contender and print a single comparison, then run the bucket-card cluster sweep for the
two that matter (active OSNet vs fine-tuned TransReID) — the real deliverable, where OSNet collapsed.

  python -m training.compare_transreid --ckpt "outputs/models/transreid_c11_tracklet_*.pt"
"""
from __future__ import annotations

import argparse
import glob
import json
import random
from pathlib import Path

import cv2
import numpy as np

from stack import reid, transreid


def held_out_pairs(seed=0, n=300):
    man = [json.loads(l) for l in Path("training_data/c11_tracklets/manifest.jsonl").read_text().splitlines() if l.strip()]
    val = [t for t in man if t["split"] == "val" and len(t["crops"]) >= 2]
    rng = random.Random(seed)
    same, diff = [], []
    for _ in range(n):
        t = rng.choice(val); a, b = rng.sample(t["crops"], 2); same.append((a["path"], b["path"]))
    for _ in range(n):
        t, u = rng.sample(val, 2); diff.append((rng.choice(t["crops"])["path"], rng.choice(u["crops"])["path"]))
    return same, diff


def sep(embfn, same, diff):
    cache = {}
    def E(p):
        if p not in cache:
            im = cv2.imread(p); v = embfn(im) if im is not None else None
            if v is not None:
                n = np.linalg.norm(v); v = v / n if n > 0 else v
            cache[p] = v
        return cache[p]
    s = [float(np.dot(E(a), E(b))) for a, b in same if E(a) is not None and E(b) is not None]
    d = [float(np.dot(E(a), E(b))) for a, b in diff if E(a) is not None and E(b) is not None]
    return float(np.mean(s)), float(np.mean(d)), float(np.mean(s) - np.mean(d))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--date", default="2026-06-03-c")
    args = ap.parse_args()
    ckpt = sorted(glob.glob(args.ckpt))[-1] if "*" in args.ckpt else args.ckpt
    ft_model, ft_sie = transreid.load_finetuned(ckpt)
    print(f"fine-tuned checkpoint: {ckpt}  (SIE={'on' if ft_sie else 'off'})\n", flush=True)

    same, diff = held_out_pairs()
    rows = [
        ("OSNet (active / stock)", lambda im: reid.osnet_embed(im)),
        ("TransReID (stock MSMT, SIE off)", lambda im: transreid.transreid_embed(im, use_sie=False)),
        ("TransReID (C11 fine-tuned)", lambda im: transreid.embed_with(ft_model, im, ft_sie)),
    ]
    print("=== held-out C11 tracklets · random strangers ===")
    print(f"  {'model':34} {'same':>7} {'stranger':>9} {'SEP':>8}")
    for name, fn in rows:
        sm, dm, g = sep(fn, same, diff)
        print(f"  {name:34} {sm:7.3f} {dm:9.3f} {g:+8.3f}", flush=True)


if __name__ == "__main__":
    main()
