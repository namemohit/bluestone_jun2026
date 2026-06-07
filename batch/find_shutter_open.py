"""Find the moment the showroom shutter lifts (store opens) in the entrance clip — so processing
starts at the REAL opening instead of a guessed hour, and the dead pre-open footage is skipped.

When the rolling shutter is DOWN the door region is uniform grey metal slats (low variance);
when it LIFTS, daylight + street + cars flood in (variance and brightness jump). We scan a time
window coarsely, score the door ROI, and report the transition. A verification strip is saved so
the call can be eyeballed. Reusable for any day.

  python -m batch.find_shutter_open --clip data/footage_day/2026-06-03/C05_day.ts \
      --clip-start "2026-06-03 09:00:00" --from-t 3600 --to-t 12000 --step 15
"""
from __future__ import annotations

import argparse
import glob
import os
import subprocess
from datetime import datetime, timedelta, timezone

import cv2
import numpy as np

IST = timezone(timedelta(hours=5, minutes=30))


def ist(clip_start: datetime, t: float) -> str:
    return (clip_start + timedelta(seconds=t)).strftime("%Y-%m-%d %H:%M:%S")


def main() -> None:
    ap = argparse.ArgumentParser(description="Detect the shutter-open (store-open) moment")
    ap.add_argument("--clip", required=True)
    ap.add_argument("--clip-start", required=True, help="IST of clip t=0, 'YYYY-MM-DD HH:MM:SS'")
    ap.add_argument("--from-t", type=float, default=0)
    ap.add_argument("--to-t", type=float, default=14400)
    ap.add_argument("--step", type=float, default=15)
    ap.add_argument("--roi", default="0.36,0.10,1.0,0.92", help="x0,y0,x1,y1 fractions of the door region")
    ap.add_argument("--workdir", default="data/shutter_scan")
    ap.add_argument("--reuse", action="store_true", help="reuse already-extracted frames (skip ffmpeg)")
    args = ap.parse_args()

    os.makedirs(args.workdir, exist_ok=True)
    clip_start = datetime.strptime(args.clip_start, "%Y-%m-%d %H:%M:%S").replace(tzinfo=IST)
    print(f"[scan] {args.clip}  t=[{args.from_t:g},{args.to_t:g}]  step={args.step:g}s "
          f"(IST {ist(clip_start, args.from_t)[11:]}-{ist(clip_start, args.to_t)[11:]})")
    if not args.reuse:
        for f in glob.glob(os.path.join(args.workdir, "f*.jpg")):
            os.remove(f)
        dur = args.to_t - args.from_t
        subprocess.run(["ffmpeg", "-y", "-ss", str(args.from_t), "-i", args.clip, "-t", str(dur),
                        "-vf", f"fps=1/{args.step}", "-q:v", "4",
                        os.path.join(args.workdir, "f%05d.jpg")], capture_output=True)
    frames = sorted(glob.glob(os.path.join(args.workdir, "f*.jpg")))
    if not frames:
        raise SystemExit("no frames extracted")

    x0, y0, x1, y1 = (float(v) for v in args.roi.split(","))
    rows = []  # (t, mean, std)
    for i, fp in enumerate(frames):
        im = cv2.imread(fp)
        if im is None:
            continue
        h, w = im.shape[:2]
        roi = cv2.cvtColor(im[int(y0 * h):int(y1 * h), int(x0 * w):int(x1 * w)], cv2.COLOR_BGR2GRAY)
        rows.append((args.from_t + i * args.step, float(roi.mean()), float(roi.std()), fp))

    t = np.array([r[0] for r in rows])
    # open-score: street detail (std) + daylight (mean), each normalized 0..1
    std = np.array([r[2] for r in rows]); mean = np.array([r[1] for r in rows])
    def norm(a): return (a - a.min()) / (np.ptp(a) + 1e-6)
    score = 0.6 * norm(std) + 0.4 * norm(mean)
    lo, hi = np.percentile(score, 15), np.percentile(score, 85)
    thresh = (lo + hi) / 2

    open_i = None
    for i in range(len(score) - 3):
        if score[i] < thresh <= score[i + 1] and score[i + 1:i + 4].mean() > thresh:
            open_i = i + 1
            break
    if open_i is None:  # fallback: biggest single step up
        open_i = int(np.argmax(np.diff(score))) + 1

    open_t = float(t[open_i])
    print(f"\n[OPEN] shutter lifts at t={open_t:g}s  ->  IST {ist(clip_start, open_t)}")
    print(f"       (score {score[open_i-1]:.2f} -> {score[open_i]:.2f}, thresh {thresh:.2f})")

    # verification strip: 6 frames spanning the transition
    lo_i, hi_i = max(0, open_i - 4), min(len(rows) - 1, open_i + 4)
    picks = np.linspace(lo_i, hi_i, 6).astype(int)
    strip = []
    for j in picks:
        im = cv2.imread(rows[j][3])
        im = cv2.resize(im, (320, 180))
        tag = "OPEN>" if j >= open_i else ""
        cv2.putText(im, f"{tag}{ist(clip_start, rows[j][0])[11:]}", (6, 170),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0) if j >= open_i else (0, 200, 255), 2)
        strip.append(im)
    cv2.imwrite(os.path.join(args.workdir, "transition.jpg"), np.hstack(strip))
    print(f"       verification strip -> {os.path.join(args.workdir, 'transition.jpg')}")


if __name__ == "__main__":
    main()
