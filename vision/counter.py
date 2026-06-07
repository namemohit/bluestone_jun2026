"""Directional line-crossing counter.

Per-track state machine: remembers which side of each counting line a track was last on,
and emits an `in`/`out` CrossingEvent when the track's movement segment crosses the line
segment. Requiring the *movement segment* to intersect the *line segment* (not just an
infinite line) rejects most jitter and out-of-segment noise.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

from common.types import Point
from logic.events import CrossingEvent
from vision.geometry import (
    inside_sign_from_label,
    segments_intersect,
    side,
    to_pixels,
)


@dataclass
class CountingLine:
    id: str
    p1: Point
    p2: Point
    inside_sign: int  # a track crossing onto this side is counted as "in"


def build_lines(camera_cfg: dict, width: int, height: int) -> list[CountingLine]:
    """Build pixel-space counting lines from a camera config block + frame size."""
    lines: list[CountingLine] = []
    for ln in camera_cfg.get("counting_lines") or []:
        pts = to_pixels(ln["points"], width, height)
        lines.append(
            CountingLine(
                id=ln["id"],
                p1=pts[0],
                p2=pts[1],
                inside_sign=inside_sign_from_label(ln.get("inside", "left")),
            )
        )
    return lines


class LineCounter:
    def __init__(self, lines: list[CountingLine], min_track_len: int = 3):
        self.lines = lines
        self.min_track_len = max(1, int(min_track_len))
        self.counts: dict[str, dict[str, int]] = {
            ln.id: {"in": 0, "out": 0} for ln in lines
        }
        self._last_point: dict[int, Point] = {}
        self._last_sign: dict[tuple[str, int], int] = {}
        self._seen: dict[int, int] = {}

    def update(
        self,
        track_id: int,
        point: Point,
        ts: float,
        frame_idx: int = -1,
        camera_id: str = "",
    ) -> list[CrossingEvent]:
        """Feed one track's current position; returns any crossing events triggered."""
        events: list[CrossingEvent] = []
        seen = self._seen.get(track_id, 0) + 1
        self._seen[track_id] = seen
        prev_point = self._last_point.get(track_id)

        for ln in self.lines:
            cur = side(ln.p1, ln.p2, point)
            key = (ln.id, track_id)
            prev = self._last_sign.get(key)
            if cur == 0:
                # Exactly on the line — keep previous side, decide on next move.
                continue
            crossed = (
                prev_point is not None
                and prev is not None
                and prev != 0
                and cur != prev
                and seen >= self.min_track_len
                and segments_intersect(prev_point, point, ln.p1, ln.p2)
            )
            if crossed:
                direction = "in" if cur == ln.inside_sign else "out"
                self.counts[ln.id][direction] += 1
                events.append(
                    CrossingEvent(
                        camera_id=camera_id,
                        line_id=ln.id,
                        track_id=track_id,
                        ts=ts,
                        direction=direction,
                        point=point,
                        frame_idx=frame_idx,
                    )
                )
            self._last_sign[key] = cur

        self._last_point[track_id] = point
        return events

    def drop_missing(self, active_ids: Iterable[int]) -> None:
        """Release state for tracks that are no longer active (bounds memory)."""
        active = set(active_ids)
        for tid in [t for t in self._last_point if t not in active]:
            self._last_point.pop(tid, None)
            self._seen.pop(tid, None)
        for k in [k for k in self._last_sign if k[1] not in active]:
            self._last_sign.pop(k, None)

    def totals(self) -> dict[str, int]:
        """Aggregate in/out/net across all lines."""
        tin = sum(c["in"] for c in self.counts.values())
        tout = sum(c["out"] for c in self.counts.values())
        return {"in": tin, "out": tout, "net": tin - tout}
