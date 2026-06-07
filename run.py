"""Single-camera pipeline runner.

Examples:
  # GPU-free synthetic demo (no model needed)
  python run.py --synthetic --out outputs/synthetic_demo.mp4

  # Real camera from config (NVR RTSP), write annotated video + events JSON
  python run.py --config configs/cameras.yaml --camera cam_entry --out outputs/cam_entry.mp4

  # Real camera from a local video file (override source)
  python run.py --config configs/cameras.yaml --camera cam_entry --source clip.mp4
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def synthetic_camera_cfg() -> dict:
    return {
        "id": "synthetic",
        "role": "entry",
        "counting_lines": [
            {"id": "mid", "points": [[0.05, 0.5], [0.95, 0.5]], "inside": "left"}
        ],
    }


def _events_default(out_video: str | None) -> str | None:
    if not out_video:
        return None
    return str(Path(out_video).with_suffix(".events.json"))


def main() -> None:
    ap = argparse.ArgumentParser(description="Showroom single-camera counting pipeline")
    ap.add_argument("--config", help="path to cameras.yaml")
    ap.add_argument("--camera", help="camera id from the config")
    ap.add_argument("--source", help="override source: file path or rtsp:// URL")
    ap.add_argument("--synthetic", action="store_true", help="run the GPU-free synthetic demo")
    ap.add_argument("--out", help="annotated output video path (.mp4)")
    ap.add_argument("--events", help="events JSON output path (default: alongside --out)")
    ap.add_argument("--max-frames", type=int, default=0)
    ap.add_argument("--fps-sample", type=float, default=0.0, help="process N fps (0 = use config/native)")
    ap.add_argument("--motion-gate", action="store_true", help="skip detection on static frames")
    ap.add_argument("--no-db", action="store_true", help="never write to the database")
    args = ap.parse_args()

    from vision.pipeline import Pipeline

    # ---- synthetic demo -------------------------------------------------
    if args.synthetic:
        from vision.track_source import SyntheticTrackSource

        cam = synthetic_camera_cfg()
        ts_src = SyntheticTrackSource()
        out = args.out or "outputs/synthetic_demo.mp4"
        events = args.events or _events_default(out)
        pipe = Pipeline("synthetic", cam, {"track_point": "bottom_center", "min_track_len": 3})
        summary, _ = pipe.run(ts_src, out_video=out, events_path=events, out_fps=ts_src.fps,
                              max_frames=args.max_frames)
        print(json.dumps(summary, indent=2))
        print(f"expected (synthetic scenario): {SyntheticTrackSource.EXPECTED_COUNTS}")
        print(f"video:  {out}\nevents: {events}")
        return

    # ---- real pipeline --------------------------------------------------
    if not args.config or not args.camera:
        ap.error("provide --config and --camera (or use --synthetic)")

    from common.config import get_camera, load_config
    from ingest.motion import MotionGate
    from ingest.nvr import build_rtsp_url, masked
    from ingest.video_source import VideoStream
    from vision.track_source import YoloTrackSource

    cfg = load_config(args.config)
    cam = get_camera(cfg, args.camera)
    counting = cfg.get("counting", {})

    src = args.source or build_rtsp_url(cfg.get("nvr", {}), cam)
    shown = masked(src) if str(src).lower().startswith("rtsp") else src
    print(f"[source] {shown}")

    fps_sample = args.fps_sample or float(cam.get("fps_sample", 0) or 0)
    vs = VideoStream(src, fps_sample=fps_sample)
    motion = MotionGate() if args.motion_gate else None
    ts_src = YoloTrackSource(vs, cfg.get("detector", {}), motion_gate=motion)

    writer = None
    if not args.no_db and cfg.get("storage", {}).get("enabled"):
        from storage.db import EventWriter

        writer = EventWriter(cfg["storage"]["dsn"])
        writer.ensure_camera(cam)

    events = args.events or _events_default(args.out)
    pipe = Pipeline(args.camera, cam, counting, writer=writer)
    summary, _ = pipe.run(
        ts_src,
        out_video=args.out,
        events_path=events,
        out_fps=fps_sample or 15.0,
        max_frames=args.max_frames,
    )
    print(json.dumps(summary, indent=2))
    if events:
        print(f"events: {events}")
    if writer is not None:
        writer.close()


if __name__ == "__main__":
    main()
