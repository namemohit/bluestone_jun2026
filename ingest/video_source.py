"""Video ingestion for files and RTSP/NVR streams.

Yields (frame_idx, ts, frame) tuples. Timestamps:
  - RTSP  -> wall-clock time.time() (host + NVR clocks should be NTP-synced).
  - files -> media time: start_ts + original_frame_index / native_fps (reproducible).

Handles FPS downsampling and RTSP reconnect.
"""
from __future__ import annotations

import time
from typing import Iterator, Optional

import cv2
import numpy as np


class VideoStream:
    def __init__(
        self,
        src: str | int,
        fps_sample: float = 0.0,
        timestamp: str = "auto",  # auto | wall | media
        reconnect: bool = True,
        max_reconnect: int = 5,
        reconnect_delay: float = 2.0,
        start_ts: float = 0.0,
    ):
        self.src = src
        self.fps_sample = float(fps_sample)
        self.reconnect = reconnect
        self.max_reconnect = max_reconnect
        self.reconnect_delay = reconnect_delay
        self.start_ts = start_ts

        self.is_rtsp = isinstance(src, str) and src.lower().startswith("rtsp")
        if timestamp == "auto":
            self.timestamp = "wall" if self.is_rtsp else "media"
        else:
            self.timestamp = timestamp

        self._cap: Optional[cv2.VideoCapture] = None
        self.native_fps: float = 0.0
        self.width = 0
        self.height = 0

    # -- capture lifecycle -------------------------------------------------
    def _open(self) -> bool:
        cap = cv2.VideoCapture(self.src)
        if self.is_rtsp:
            # keep latency low: don't buffer stale frames
            try:
                cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            except Exception:
                pass
        if not cap.isOpened():
            cap.release()
            return False
        self._cap = cap
        fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
        self.native_fps = fps if fps and fps > 0 else 25.0
        self.width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
        self.height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
        return True

    def release(self) -> None:
        if self._cap is not None:
            self._cap.release()
            self._cap = None

    # -- iteration ---------------------------------------------------------
    def frames(self) -> Iterator[tuple[int, float, np.ndarray]]:
        if not self._open():
            raise RuntimeError(f"Could not open video source: {self.src!r}")

        # media-mode subsampling step
        step = 1
        if self.fps_sample > 0 and self.timestamp == "media":
            step = max(1, round(self.native_fps / self.fps_sample))

        orig_idx = -1   # index in the original stream
        emit_idx = -1   # index among emitted (kept) frames
        next_wall_emit = 0.0
        reconnects = 0

        while True:
            ok, frame = self._cap.read()
            if not ok or frame is None:
                if self.is_rtsp and self.reconnect and reconnects < self.max_reconnect:
                    reconnects += 1
                    self.release()
                    time.sleep(self.reconnect_delay)
                    if self._open():
                        continue
                break
            orig_idx += 1

            # FPS downsampling
            if self.timestamp == "media":
                if orig_idx % step != 0:
                    continue
                ts = self.start_ts + orig_idx / self.native_fps
            else:  # wall clock (rtsp)
                now = time.time()
                if self.fps_sample > 0:
                    if now < next_wall_emit:
                        continue
                    next_wall_emit = now + 1.0 / self.fps_sample
                ts = now

            emit_idx += 1
            if self.width == 0:
                self.height, self.width = frame.shape[:2]
            yield emit_idx, ts, frame

        self.release()
