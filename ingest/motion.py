"""Lightweight motion gate via frame differencing.

Used to skip running the (expensive) detector on static frames, saving GPU/bandwidth.
Optional — off by default so slow movers are never dropped; enable for always-on RTSP.
"""
from __future__ import annotations

import cv2
import numpy as np


class MotionGate:
    def __init__(self, downscale: int = 4, pixel_threshold: int = 20, min_area_frac: float = 0.0008):
        self.downscale = max(1, int(downscale))
        self.pixel_threshold = int(pixel_threshold)
        self.min_area_frac = float(min_area_frac)
        self._prev: np.ndarray | None = None

    def is_active(self, frame: np.ndarray) -> bool:
        """True if enough pixels changed since the last frame (or the first frame)."""
        small = cv2.resize(
            frame,
            (max(1, frame.shape[1] // self.downscale), max(1, frame.shape[0] // self.downscale)),
            interpolation=cv2.INTER_AREA,
        )
        gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY) if small.ndim == 3 else small
        if self._prev is None:
            self._prev = gray
            return True
        diff = cv2.absdiff(gray, self._prev)
        self._prev = gray
        changed = int(np.count_nonzero(diff > self.pixel_threshold))
        frac = changed / float(gray.size)
        return frac >= self.min_area_frac
