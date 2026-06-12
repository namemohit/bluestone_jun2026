"""Build a multi-view tracklet ReID dataset from C11 crops (re-extracted from the slices).

L1 saves only ONE crop per track (the largest detection), but ReID training needs several views
per identity. So for each C11 tracklet we re-extract `--views` evenly-spaced crops from that window's
C11.ts slice, reconstructing the bbox from the trajectory ([ts, foot_x=center-x, foot_y=bottom-y,
bbox_h=height]). Each tracklet = one pseudo-identity; ~`--val-frac` of tracklets are held out as VAL
(disjoint identities — never trained on, the explicit guard against the prior in-sample overfit).

  python -m training.build_c11_tracklets --date 2026-06-03
  python -m training.build_c11_tracklets --date 2026-06-03 --limit-windows 1 --max-tracklets 30   # quick test
"""
from __future__ import annotations

import argparse
import glob
import json
import random
import subprocess
from datetime import datetime, timedelta, timezone
from pathlib import Path

import cv2
import numpy as np

IST = timezone(timedelta(hours=5, minutes=30))


def start_epoch(tracks: dict) -> float:
    return datetime.strptime(tracks["start"], "%Y-%m-%d %H:%M:%S").replace(tzinfo=IST).timestamp()


def grab(slice_path: str, offset: float):
    """Accurate single frame at `offset` seconds via two-pass ffmpeg seek (keyframe pre-roll, then
    accurate decode) — the same method the /frame endpoint uses; cv2's POS_MSEC seek corrupts HEVC."""
    pre = max(0.0, offset - 3.0)
    cmd = ["ffmpeg", "-nostdin", "-loglevel", "error", "-ss", f"{pre:.3f}", "-i", str(slice_path),
           "-ss", f"{offset - pre:.3f}", "-frames:v", "1", "-f", "image2pipe", "-vcodec", "mjpeg",
           "-q:v", "2", "pipe:1"]
    p = subprocess.run(cmd, capture_output=True)
    if not p.stdout:
        return None
    return cv2.imdecode(np.frombuffer(p.stdout, np.uint8), cv2.IMREAD_COLOR)


def bbox_px(pt, W, H, aspect):
    _ts, fx, fy, bh = pt
    h = bh * H
    w = h * aspect
    cx, by = fx * W, fy * H
    return (int(max(0, cx - w / 2)), int(max(0, by - h)),
            int(min(W, cx + w / 2)), int(min(H, by)))


def main() -> None:
    ap = argparse.ArgumentParser(description="Re-extract multi-view tracklet crops from C11 slices")
    ap.add_argument("--date", default="2026-06-03")
    ap.add_argument("--views", type=int, default=5, help="crops re-extracted per tracklet")
    ap.add_argument("--min-pts", type=int, default=12, help="min trajectory points to qualify")
    ap.add_argument("--max-tracklets", type=int, default=1400, help="cap (richest-trajectory first)")
    ap.add_argument("--val-frac", type=float, default=0.2)
    ap.add_argument("--min-side", type=int, default=44, help="skip crops smaller than this (px)")
    ap.add_argument("--out", default="training_data/c11_tracklets")
    ap.add_argument("--limit-windows", type=int, default=0, help="testing: only first N windows")
    args = ap.parse_args()
    random.seed(0)
    out = Path(args.out)

    tj_paths = sorted(glob.glob(f"outputs/{args.date}_*/L1_C11/tracks.json"))
    if args.limit_windows:
        tj_paths = tj_paths[:args.limit_windows]

    cands, meta = [], {}
    for tp in tj_paths:
        win = Path(tp).parent.parent.name
        tj = json.loads(Path(tp).read_text(encoding="utf-8"))
        meta[win] = (tj, start_epoch(tj))
        for t in tj["tracks"]:
            if len(t.get("traj", [])) >= args.min_pts:
                cands.append((len(t["traj"]), win, t["track"]))
    cands.sort(reverse=True)                                   # richest trajectories first
    if args.max_tracklets:
        cands = cands[:args.max_tracklets]
    print(f"[ds] {len(cands)} tracklets >= {args.min_pts} pts across {len(tj_paths)} windows", flush=True)

    by_win: dict = {}
    for _n, win, trk in cands:
        by_win.setdefault(win, []).append(trk)

    man, n_crops = [], 0
    for win, trks in by_win.items():
        tj, st = meta[win]
        tdict = {t["track"]: t for t in tj["tracks"]}
        slice_path = f"outputs/{win}/slices/C11.ts"
        if not Path(slice_path).exists():
            print(f"[ds]   WARN no C11.ts for {win}", flush=True)
            continue
        nwin = 0
        for trk in trks:
            t = tdict[trk]
            traj = t["traj"]
            aspect = 0.45                                       # bbox aspect from the saved best crop
            cp = t.get("crop")
            if cp and Path(cp).exists():
                im = cv2.imread(cp)
                if im is not None and im.shape[0] > 0:
                    aspect = im.shape[1] / im.shape[0]
            tid = f"{win}_{trk}"
            saved = []
            for k, i in enumerate(np.linspace(0, len(traj) - 1, args.views).astype(int)):
                pt = traj[int(i)]
                fr = grab(slice_path, max(0.0, pt[0] - st))     # accurate ffmpeg seek -> clean frame
                if fr is None:
                    continue
                H, W = fr.shape[:2]
                x1, y1, x2, y2 = bbox_px(pt, W, H, aspect)
                if x2 - x1 < args.min_side or y2 - y1 < args.min_side:
                    continue
                crop = fr[y1:y2, x1:x2]
                if crop.size == 0 or float(crop.std()) < 12.0:   # skip blank/black/out-of-frame crops
                    continue
                d = out / tid
                d.mkdir(parents=True, exist_ok=True)
                fp = d / f"{k}.jpg"
                cv2.imwrite(str(fp), crop, [int(cv2.IMWRITE_JPEG_QUALITY), 92])
                saved.append({"path": str(fp).replace("\\", "/"), "ts": round(float(pt[0]), 2)})
                n_crops += 1
            if len(saved) >= 2:                                # need >=2 views for a positive pair
                man.append({"tracklet": tid, "window": win, "track": trk,
                            "first_ts": float(t["first_ts"]),
                            "last_ts": float(t.get("last_ts", t["first_ts"])),
                            "crops": saved})
                nwin += 1
        print(f"[ds]   {win}: {nwin} tracklets", flush=True)

    random.shuffle(man)
    nval = int(len(man) * args.val_frac)
    for i, m in enumerate(man):
        m["split"] = "val" if i < nval else "train"
    out.mkdir(parents=True, exist_ok=True)
    (out / "manifest.jsonl").write_text("\n".join(json.dumps(m) for m in man), encoding="utf-8")
    ntr = sum(1 for m in man if m["split"] == "train")
    print(f"[ds] DONE: {len(man)} tracklets ({ntr} train / {len(man) - ntr} val), "
          f"{n_crops} crops -> {out}", flush=True)


if __name__ == "__main__":
    main()
