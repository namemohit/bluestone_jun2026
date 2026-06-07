"""Segment a frame stream into motion-triggered mp4 clips."""
from __future__ import annotations

from pathlib import Path

import cv2

from ingest.motion import MotionGate


def segment_motion_clips(frames, out_dir, *, prefix: str = "clip", fps: float = 8.0,
                         grace_sec: float = 2.0, max_clip_sec: float = 120.0, gate=None) -> list[str]:
    """Consume (frame_idx, ts, frame); write motion-active spans as mp4 clips into out_dir.

    A clip opens when motion starts, keeps recording through brief still gaps (<= grace_sec),
    closes after the gap or at max_clip_sec, and is named <prefix>_<ts_ms>.mp4.
    Returns the list of written clip paths.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    gate = gate or MotionGate()

    writer = None
    clip_path = None
    clip_start_ts = None
    last_motion_ts = None
    written: list[str] = []
    fps = float(fps) if fps and fps > 0 else 8.0

    def _open(ts, frame):
        nonlocal writer, clip_path, clip_start_ts
        h, w = frame.shape[:2]
        clip_path = out_dir / f"{prefix}_{int(ts * 1000)}.mp4"
        writer = cv2.VideoWriter(str(clip_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
        clip_start_ts = ts

    def _close():
        nonlocal writer, clip_path
        if writer is not None:
            writer.release()
            written.append(str(clip_path))
            writer = None
            clip_path = None

    for _idx, ts, frame in frames:
        active = gate.is_active(frame)
        if active:
            if writer is None:
                _open(ts, frame)
            writer.write(frame)
            last_motion_ts = ts
            if ts - clip_start_ts >= max_clip_sec:
                _close()
        elif writer is not None:
            if last_motion_ts is not None and ts - last_motion_ts <= grace_sec:
                writer.write(frame)  # keep recording through a brief still gap
            else:
                _close()
    _close()
    return written
