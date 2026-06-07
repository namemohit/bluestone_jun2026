"""L1 - all detected humans in the entry camera, as a scannable digest.

Runs YOLO + BoT-SORT over a clip; every person-TRACK (one continuous appearance)
becomes ONE reviewable record: the best (largest) crop + IST first/last-seen + duration.
Recall-first by design: includes everyone -- customers, staff, and street pedestrians
seen through the glass. False positives are EXPECTED; L2 filters them out.

Outputs (under --out, default outputs/L1):
  crops/trk_<id>.jpg   one best crop per person-track
  index.csv            one row per track (track, first/last IST, dur, frames, bbox height, conf)
  contact_sheet.jpg    grid of all crops + labels -- scan this instead of the video

Run:
  python -m stack.l1_humans --video data/footage_rtsp/c05_6pm_ist.ts \
      --start "2026-06-03 18:00:00" --fps 6 --out outputs/L1
"""
from __future__ import annotations

import argparse
import csv
import json
import os
from datetime import datetime, timezone, timedelta

import cv2
import numpy as np

from ingest.video_source import VideoStream
from vision.track_source import YoloTrackSource

IST = timezone(timedelta(hours=5, minutes=30))


def fmt(ts: float) -> str:
    return datetime.fromtimestamp(ts, IST).strftime("%H:%M:%S")


def build_contact_sheet(thumbs, path, cols=6, tw=180, th=240, pad=8, labelh=24):
    if not thumbs:
        cv2.imwrite(path, np.full((120, 480, 3), 35, np.uint8))
        return
    rows = (len(thumbs) + cols - 1) // cols
    W = cols * (tw + pad) + pad
    H = rows * (th + labelh + pad) + pad
    sheet = np.full((H, W, 3), 28, np.uint8)
    for i, (crop, label) in enumerate(thumbs):
        r, c = divmod(i, cols)
        x = pad + c * (tw + pad)
        y = pad + r * (th + labelh + pad)
        ch, cw = crop.shape[:2]
        s = min(tw / cw, th / ch)
        nw, nh = max(1, int(cw * s)), max(1, int(ch * s))
        sheet[y:y + nh, x:x + nw] = cv2.resize(crop, (nw, nh))
        cv2.putText(sheet, label, (x, y + th + 17),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, (200, 255, 200), 1, cv2.LINE_AA)
    cv2.imwrite(path, sheet)


def main() -> None:
    ap = argparse.ArgumentParser(description="L1: all detected humans -> scannable digest")
    ap.add_argument("--video", required=True)
    ap.add_argument("--weights", default="yolo11m.pt")
    ap.add_argument("--device", default="0")
    ap.add_argument("--imgsz", type=int, default=736)
    ap.add_argument("--conf", type=float, default=0.30)
    ap.add_argument("--fps", type=float, default=6.0)
    ap.add_argument("--start", default="2026-06-03 18:00:00", help="IST wall-clock of frame 0")
    ap.add_argument("--min-frames", type=int, default=3, help="drop tracks seen fewer times (noise)")
    ap.add_argument("--max-frames", type=int, default=0)
    ap.add_argument("--out", default="outputs/L1")
    args = ap.parse_args()

    start_ts = datetime.strptime(args.start, "%Y-%m-%d %H:%M:%S").replace(tzinfo=IST).timestamp()
    vs = VideoStream(args.video, fps_sample=args.fps, timestamp="media", start_ts=start_ts)
    det = {"weights": args.weights, "conf": args.conf, "iou": 0.5, "classes": [0],
           "imgsz": args.imgsz, "device": args.device, "tracker": "botsort.yaml"}
    src = YoloTrackSource(vs, det)

    cropdir = os.path.join(args.out, "crops")
    os.makedirs(cropdir, exist_ok=True)

    tracks: dict[int, dict] = {}
    n = 0
    print(f"[L1] {args.video}  weights={args.weights} device={args.device} fps={args.fps}")
    for ft in src:
        for t in ft.tracks:
            x1, y1, x2, y2 = (int(v) for v in t.bbox)
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(ft.width, x2), min(ft.height, y2)
            if x2 <= x1 or y2 <= y1:
                continue
            area = (x2 - x1) * (y2 - y1)
            r = tracks.get(t.track_id)
            if r is None:
                r = {"first": ft.ts, "last": ft.ts, "n": 0, "area": 0,
                     "crop": None, "conf": 0.0, "h": 0, "traj": []}
                tracks[t.track_id] = r
            r["last"] = ft.ts
            r["n"] += 1
            r["conf"] = max(r["conf"], t.conf)
            r["h"] = max(r["h"], y2 - y1)
            # foot-point (bottom-center) + bbox height, all normalized -> feeds L2 filtering
            r["traj"].append((round(ft.ts, 2),
                              round(((x1 + x2) / 2) / ft.width, 4),
                              round(y2 / ft.height, 4),
                              round((y2 - y1) / ft.height, 4)))
            if area > r["area"] and ft.frame is not None:
                r["area"] = area
                r["crop"] = ft.frame[y1:y2, x1:x2].copy()
        n += 1
        if args.max_frames and n >= args.max_frames:
            break

    rows, thumbs, track_data = [], [], []
    for tid, r in sorted(tracks.items(), key=lambda kv: kv[1]["first"]):
        if r["n"] < args.min_frames or r["crop"] is None:
            continue
        cp = os.path.join(cropdir, f"trk_{tid:04d}.jpg")
        cv2.imwrite(cp, r["crop"])
        dur = r["last"] - r["first"]
        rows.append({"track": tid, "first_ist": fmt(r["first"]), "last_ist": fmt(r["last"]),
                     "dur_s": round(dur, 1), "frames": r["n"], "bbox_h_px": r["h"],
                     "peak_conf": round(r["conf"], 2), "crop": cp})
        thumbs.append((r["crop"], f"#{tid}  {fmt(r['first'])}  {round(dur)}s"))
        track_data.append({"track": tid, "first_ts": round(r["first"], 2),
                           "last_ts": round(r["last"], 2), "first_ist": fmt(r["first"]),
                           "frames": r["n"], "peak_conf": round(r["conf"], 2),
                           "crop": cp, "traj": r["traj"]})

    with open(os.path.join(args.out, "index.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["track", "first_ist", "last_ist", "dur_s",
                                          "frames", "bbox_h_px", "peak_conf", "crop"])
        w.writeheader()
        w.writerows(rows)

    # machine-readable: trajectories for L2/L4 to filter (foot-point = [ts, x, y, bbox_h], normalized)
    with open(os.path.join(args.out, "tracks.json"), "w", encoding="utf-8") as f:
        json.dump({"video": args.video, "fps": args.fps, "start": args.start,
                   "traj_format": ["ts", "foot_x", "foot_y", "bbox_h"], "tracks": track_data}, f)

    build_contact_sheet(thumbs, os.path.join(args.out, "contact_sheet.jpg"))
    print(f"[L1] {len(rows)} person-tracks (raw, unfiltered) -> {args.out}/")
    print(f"[L1]   index.csv | crops/ | contact_sheet.jpg")


if __name__ == "__main__":
    main()
