"""Embed each Cluster-tab bucket card with the fine-tuned TransReID on RE-EXTRACTED crops (the model's
training domain). Mirror of `training.build_c11_card_reps` but with the TransReID backbone. For each card
we find its (window, track), re-extract a few views from that window's C11 slice, embed each with the
fine-tuned TransReID, and store the mean as the card's representative. Also runs a cluster sweep + stranger
FLOOD vs the ACTIVE OSNet reps so we can see whether TransReID separates where OSNet collapsed.

Saves {`<window>|<track>` -> embedding} to outputs/transreid_card_reps.pkl, which the cluster endpoint's
?model=transreid path reads.

  python -m training.build_transreid_card_reps --ckpt "outputs/models/transreid_c11_tracklet_*.pt" --date 2026-06-03-c
"""
from __future__ import annotations

import argparse
import glob
import json
import pickle
import random
import re
from pathlib import Path

import cv2
import numpy as np

import dashboard.hitl_api as H
from stack import transreid
from training.build_c11_tracklets import bbox_px, grab, start_epoch


def real_window_track(c):
    for s in c.get("crops", []):
        m = re.search(r"outputs/([^/]+)/L1_C11/crops/trk_(\d+)\.jpg",
                      (s.get("crop") or "").replace("\\", "/"))
        if m:
            return m.group(1), int(m.group(2))
    return None, None


def card_reps(cards, model, use_sie, views):
    tj_cache: dict = {}
    reps: dict = {}
    for c in cards:
        cid = H._card_id(c)
        rw, trk = real_window_track(c)
        if rw is None:
            continue
        if rw not in tj_cache:
            try:
                tj = json.loads(Path(f"outputs/{rw}/L1_C11/tracks.json").read_text(encoding="utf-8"))
                tj_cache[rw] = (tj, start_epoch(tj))
            except Exception:
                tj_cache[rw] = (None, None)
        tj, st = tj_cache[rw]
        if tj is None:
            continue
        t = next((x for x in tj["tracks"] if x["track"] == trk), None)
        if not t or not t.get("traj"):
            continue
        aspect = 0.45
        cp = t.get("crop")
        if cp and Path(cp).exists():
            im = cv2.imread(cp)
            if im is not None and im.shape[0] > 0:
                aspect = im.shape[1] / im.shape[0]
        slice_path = f"outputs/{rw}/slices/C11.ts"
        traj = t["traj"]
        vs = []
        for i in np.linspace(0, len(traj) - 1, views).astype(int):
            pt = traj[int(i)]
            fr = grab(slice_path, max(0.0, pt[0] - st))
            if fr is None:
                continue
            Hh, Ww = fr.shape[:2]
            x1, y1, x2, y2 = bbox_px(pt, Ww, Hh, aspect)
            if x2 - x1 < 30 or y2 - y1 < 30:
                continue
            crop = fr[y1:y2, x1:x2]
            if crop.size == 0 or float(crop.std()) < 12.0:
                continue
            e = transreid.embed_with(model, crop, use_sie)
            if e is not None:
                vs.append(e)
        if vs:
            m = np.mean(vs, axis=0); n = float(np.linalg.norm(m))
            reps[f"{cid[0]}|{cid[1]}"] = (m / n if n > 0 else m).astype("float32")
    return reps


def sweep(reps, label):
    keys = list(reps)
    if not keys:
        print(f"\n{label}: no reps"); return
    M = np.vstack([reps[k] for k in keys]); S = M @ M.T
    n = len(keys)
    print(f"\n{label}: {n} cards")
    for thr in (0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90):
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
    rs = [float(S[rng.randrange(n)][rng.randrange(n)]) for _ in range(3000)]
    print(f"   stranger sims: mean={np.mean(rs):.3f}  p90={np.percentile(rs,90):.3f}  p99={np.percentile(rs,99):.3f}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--date", default="2026-06-03-c")
    ap.add_argument("--views", type=int, default=4)
    args = ap.parse_args()
    ckpt = sorted(glob.glob(args.ckpt))[-1] if "*" in args.ckpt else args.ckpt
    model, use_sie = transreid.load_finetuned(ckpt)
    print(f"checkpoint: {ckpt}  (SIE={'on' if use_sie else 'off'})", flush=True)

    H._STATE_VERSION += 1
    allc = H._cards_from_dets(args.date)
    cust, staff = allc["customer_cards"], allc["staff_cards"]
    reps = card_reps(cust + staff, model, use_sie, args.views)
    pickle.dump(reps, open("outputs/transreid_card_reps.pkl", "wb"))
    print(f"saved {len(reps)}/{len(cust)+len(staff)} card reps -> outputs/transreid_card_reps.pkl", flush=True)

    # customer-only sweep (the hard case where OSNet collapsed)
    cust_reps = {k: v for k, v in reps.items()
                 for cid in [H._card_id(c) for c in cust] if k == f"{cid[0]}|{cid[1]}"}
    sweep(cust_reps, "=== TransReID fine-tuned card reps (customers) ===")


if __name__ == "__main__":
    main()
