"""Domain events emitted by the pipeline."""
from __future__ import annotations

from dataclasses import asdict, dataclass

from common.types import Point


@dataclass
class CrossingEvent:
    """A person's tracked point crossed a counting line in a given direction."""

    camera_id: str
    line_id: str
    track_id: int
    ts: float  # epoch seconds (wall clock)
    direction: str  # "in" (into the showroom) | "out"
    point: Point  # pixel location where the crossing was detected
    frame_idx: int = -1
    confidence: float = 1.0

    def to_dict(self) -> dict:
        d = asdict(self)
        d["point"] = [float(self.point[0]), float(self.point[1])]
        return d
