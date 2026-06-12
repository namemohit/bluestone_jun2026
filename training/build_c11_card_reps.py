"""Embed each Cluster-tab bucket card with the C11 model on RE-EXTRACTED crops (the model's training
domain — NOT the L1 'best crops', which are out-of-distribution and collapse). For each card we find its
(window, track) from a crop path, re-extract a few views from that window's C11 slice, embed each with the
C11 checkpoint, and store the mean as the card's representative. Saves {`<window>|<track>` -> embedding} to
outputs/c11_card_reps.pkl, which the cluster endpoint's ?model=c11 path reads.

  python -m training.build_c11_card_reps --ckpt "outputs/models/osnet_c11_tracklet_*.pt" --date 2026-06-03-c
"""
from __future__ import annotations

import argparse
import glob
import json
import pickle
import re
from pathlib import Path

import cv2
import numpy as np

import dashboard.hitl_api as H
from stack import reid
from training.build_c11_tracklets import bbox_px, grab, start_epoch


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--date", default="2026-06-03-c")
    ap.add_argument("--views", type=int, default=3)
    args = ap.parse_args()
    ckpt = sorted(glob.glob(args.ckpt))[-1] if "*" in args.ckpt else args.ckpt
    print(f"checkpoint: {ckpt}", flush=True)

    H._STATE_VERSION += 1
    allc = H._cards_from_dets(args.date)
    cards = [(c, "customer") for c in allc["customer_cards"]] + [(c, "staff") for c in allc["staff_cards"]]

    def real_window_track(c):
        for s in c.get("crops", []):
            m = re.search(r"outputs/([^/]+)/L1_C11/crops/trk_(\d+)\.jpg",
                          (s.get("crop") or "").replace("\\", "/"))
            if m:
                return m.group(1), int(m.group(2))
        return None, None

    tj_cache: dict = {}
    reps: dict = {}
    for c, _bucket in cards:
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
        for i in np.linspace(0, len(traj) - 1, args.views).astype(int):
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
            e = reid._triplet_embed(crop, ckpt)
            vs.append(e / (np.linalg.norm(e) + 1e-9))
        if vs:
            m = np.mean(vs, axis=0); n = float(np.linalg.norm(m))
            reps[f"{cid[0]}|{cid[1]}"] = (m / n if n > 0 else m).astype("float32")
    pickle.dump(reps, open("outputs/c11_card_reps.pkl", "wb"))
    print(f"saved {len(reps)}/{len(cards)} card reps -> outputs/c11_card_reps.pkl", flush=True)


if __name__ == "__main__":
    main()
