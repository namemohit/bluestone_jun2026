"""On-site edge agent.

For each camera it pulls the NVR RTSP (locally, over the LAN), runs the motion gate, and writes
**motion-triggered clips** which it uploads to a sink (GCS in production, local for dev). Only
active footage leaves the store -> low bandwidth, NVR never exposed, clips deleted after the
nightly batch processes them.

  # dev: segment a sample file into a local folder
  python -m edge.agent --config configs/cameras.yaml --camera cam_entry --source clip.mp4 --sink local:clips

  # prod: all cameras from the NVR -> GCS
  python -m edge.agent --config configs/cameras.yaml --sink gcs:my-bucket/showroom-42
"""
from __future__ import annotations

import argparse
import tempfile
import threading
from pathlib import Path

import cv2

from common.config import get_camera, load_config
from edge.motion_clip import segment_motion_clips  # noqa: F401  (re-export for convenience)
from edge.sink import make_sink
from ingest.motion import MotionGate
from ingest.nvr import build_rtsp_url, masked
from ingest.video_source import VideoStream


def run_camera(nvr: dict, camera: dict, sink, *, fps_sample: float = 8.0,
               clip_seconds: float = 120.0, grace_sec: float = 2.0,
               max_frames: int = 0, source: str | None = None) -> list[str]:
    """Pull one camera, segment motion clips, upload them. Returns uploaded clip names."""
    src = source or camera.get("source") or build_rtsp_url(nvr, camera)
    shown = masked(src) if str(src).lower().startswith("rtsp") else src
    print(f"[{camera['id']}] source {shown}")

    vs = VideoStream(src, fps_sample=fps_sample)
    frames = vs.frames()
    if max_frames:
        frames = _limit(frames, max_frames)

    with tempfile.TemporaryDirectory() as tmp:
        clips = segment_motion_clips(
            frames, tmp, prefix=camera["id"], fps=fps_sample or 8.0,
            grace_sec=grace_sec, max_clip_sec=clip_seconds, gate=MotionGate(),
        )
        names = [sink.put(c, Path(c).name) for c in clips]
    print(f"[{camera['id']}] uploaded {len(names)} clip(s)")
    return names


def _limit(it, n):
    for i, x in enumerate(it):
        if i >= n:
            break
        yield x


def main() -> None:
    ap = argparse.ArgumentParser(description="Edge agent: RTSP -> motion clips -> sink")
    ap.add_argument("--config", required=True)
    ap.add_argument("--camera", help="one camera id (default: all in the config)")
    ap.add_argument("--source", help="override source (file path) — for testing")
    ap.add_argument("--sink", default="local:clips", help="local:/path or gcs:bucket/prefix")
    ap.add_argument("--fps-sample", type=float, default=8.0)
    ap.add_argument("--clip-seconds", type=float, default=120.0)
    ap.add_argument("--max-frames", type=int, default=0)
    args = ap.parse_args()

    cfg = load_config(args.config)
    nvr = cfg.get("nvr", {})
    sink = make_sink(args.sink)
    cams = [get_camera(cfg, args.camera)] if args.camera else cfg.get("cameras", [])

    threads = []
    for cam in cams:
        t = threading.Thread(target=run_camera, args=(nvr, cam, sink),
                             kwargs=dict(fps_sample=args.fps_sample, clip_seconds=args.clip_seconds,
                                         max_frames=args.max_frames, source=args.source))
        t.start()
        threads.append(t)
    for t in threads:
        t.join()


if __name__ == "__main__":
    main()
