"""Real single-camera count on a recorded clip: YOLO + BoT-SORT -> directional line counter.

This is the GPU path for the entry/exit cameras. Feed it a transcoded H.264 clip, a
normalized counting line, and which side is "inside"; it writes an annotated video + an
events JSON and prints the IN / OUT / NET summary.

Line convention (image coords, y is DOWN): draw the line P1(left x) -> P2(right x).
A point BELOW the line (bottom of frame) is the "right" side; ABOVE is "left".
For an entry cam where people walk DOWN into the store, inside="right" => IN = entry.

Example (C05 entry, line across the door threshold, store side = below = right):
  python run_count.py --video data/footage_rtsp/c05_5min.mp4 --cam C05 \
      --line 0.26,0.70,0.62,0.66 --inside right \
      --weights yolo11m.pt --device 0 --imgsz 960 --conf 0.30 --fps 8 \
      --start "2026-06-03 10:00:00" --out outputs/c05_5min
"""
from __future__ import annotations

import argparse
import time
from datetime import datetime, timezone, timedelta

from ingest.video_source import VideoStream
from ingest.motion import MotionGate
from vision.track_source import YoloTrackSource
from vision.pipeline import Pipeline

IST = timezone(timedelta(hours=5, minutes=30))


def parse_start(s: str) -> float:
    if not s:
        return 0.0
    dt = datetime.strptime(s, "%Y-%m-%d %H:%M:%S").replace(tzinfo=IST)
    return dt.timestamp()


def main() -> None:
    ap = argparse.ArgumentParser(description="Real one-camera count on a recorded clip")
    ap.add_argument("--video", required=True)
    ap.add_argument("--cam", default="C05")
    ap.add_argument("--line", required=True, help="x1,y1,x2,y2 normalized (0..1), P1=left x -> P2=right x")
    ap.add_argument("--inside", default="right",
                    help="which side counts as IN: left|right ('right'=below/bottom of frame)")
    ap.add_argument("--weights", default="yolo11m.pt")
    ap.add_argument("--device", default="0", help="'0' for GPU, 'cpu' for CPU")
    ap.add_argument("--imgsz", type=int, default=960)
    ap.add_argument("--conf", type=float, default=0.30)
    ap.add_argument("--iou", type=float, default=0.5)
    ap.add_argument("--fps", type=float, default=8.0, help="sample FPS (downsample)")
    ap.add_argument("--min-track-len", type=int, default=3)
    ap.add_argument("--start", default="", help='wall-clock of frame 0, "YYYY-MM-DD HH:MM:SS" (IST)')
    ap.add_argument("--out", default="outputs/count")
    ap.add_argument("--max-frames", type=int, default=0)
    ap.add_argument("--save-video", action="store_true",
                    help="also write the annotated video (SLOW: CPU draw+encode). Off by default.")
    ap.add_argument("--motion", action="store_true",
                    help="motion-gate: skip YOLO on static frames (frame differencing)")
    args = ap.parse_args()

    x1, y1, x2, y2 = (float(v) for v in args.line.split(","))
    cam_cfg = {
        "counting_lines": [
            {"id": "door", "points": [[x1, y1], [x2, y2]], "inside": args.inside}
        ]
    }
    det_cfg = {
        "weights": args.weights,
        "conf": args.conf,
        "iou": args.iou,
        "classes": [0],            # person
        "imgsz": args.imgsz,
        "device": args.device,
        "tracker": "botsort.yaml",
    }

    start_ts = parse_start(args.start)
    vs = VideoStream(args.video, fps_sample=args.fps, timestamp="media", start_ts=start_ts)
    gate = MotionGate() if args.motion else None
    src = YoloTrackSource(vs, det_cfg, motion_gate=gate)
    pipe = Pipeline(
        args.cam, cam_cfg,
        counting_cfg={"track_point": "bottom_center", "min_track_len": args.min_track_len},
    )

    print(f"[count] {args.cam}  weights={args.weights} device={args.device} imgsz={args.imgsz} "
          f"conf={args.conf} fps={args.fps}")
    print(f"[count] line={args.line} inside={args.inside}  video={args.video}")
    t0 = time.time()
    summary, events = pipe.run(
        src,
        out_video=(args.out + ".mp4") if args.save_video else None,
        events_path=args.out + ".json",
        out_fps=args.fps,
        max_frames=args.max_frames,
    )
    dt = time.time() - t0

    def fmt(ts: float) -> str:
        return datetime.fromtimestamp(ts, IST).strftime("%H:%M:%S") if ts else "-"

    print("\n========== RESULT ==========")
    print(f"  IN  (entries): {summary['totals']['in']}")
    print(f"  OUT (exits)  : {summary['totals']['out']}")
    print(f"  NET inside   : {summary['totals']['net']}")
    print(f"  crossing events: {len(events)}")
    for e in events:
        print(f"    {fmt(e.ts)}  {e.direction.upper():3s}  track#{e.track_id}")
    print(f"  processed in {dt:.1f}s -> {args.out}.mp4 / {args.out}.json")
    print("============================")


if __name__ == "__main__":
    main()
