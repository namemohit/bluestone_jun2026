"""Shared lightweight dataclasses used across the pipeline.

Kept dependency-free (no numpy/torch import at runtime) so geometry, counting and
tests can run in a minimal environment.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Optional, Sequence

if TYPE_CHECKING:  # only for type checkers; no runtime import cost
    import numpy as np

Point = tuple[float, float]
BBox = tuple[float, float, float, float]  # x1, y1, x2, y2 in pixels


@dataclass
class Track:
    """A single tracked person in one frame (output of detect+track)."""

    track_id: int
    bbox: BBox
    conf: float = 1.0
    cls: int = 0

    @property
    def center(self) -> Point:
        x1, y1, x2, y2 = self.bbox
        return ((x1 + x2) / 2.0, (y1 + y2) / 2.0)

    @property
    def bottom_center(self) -> Point:
        """Feet position — the right reference point for ground-plane line crossing."""
        x1, y1, x2, y2 = self.bbox
        return ((x1 + x2) / 2.0, y2)

    def point(self, mode: str = "bottom_center") -> Point:
        return self.bottom_center if mode == "bottom_center" else self.center


@dataclass
class FrameTracks:
    """All tracks for one frame, plus timing + (optional) the image itself."""

    frame_idx: int
    ts: float  # epoch seconds, wall-clock (host + NVR clocks should be NTP-synced)
    tracks: list[Track] = field(default_factory=list)
    frame: Optional["np.ndarray"] = None  # BGR image, or None (headless/synthetic-events)
    width: int = 0
    height: int = 0
