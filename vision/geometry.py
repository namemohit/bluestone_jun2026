"""2D geometry helpers for line-crossing detection.

Conventions (IMPORTANT — image coordinates have x right, y DOWN):
  - A counting line is a directed segment P1 -> P2.
  - `cross(p1, p2, q)` > 0  => q lies to the RIGHT of travel direction P1->P2 (toward
    larger y / bottom of frame); < 0 => LEFT (toward top of frame).
  - `side()` returns the sign: +1 (right), -1 (left), 0 (on the line).
  - So config `inside: left`  -> inside_sign = -1
            `inside: right` -> inside_sign = +1
  These map left/right to "which way you'd turn walking along P1->P2". The zone-preview
  tool shades the inside region so you can confirm/flip it visually.
"""
from __future__ import annotations

from typing import Sequence

from common.types import Point

EPS = 1e-9


def cross(p1: Point, p2: Point, q: Point) -> float:
    """Signed area (z of cross product) of (P2-P1) x (Q-P1)."""
    return (p2[0] - p1[0]) * (q[1] - p1[1]) - (p2[1] - p1[1]) * (q[0] - p1[0])


def side(p1: Point, p2: Point, q: Point) -> int:
    """Which side of directed line P1->P2 the point Q is on: +1 right, -1 left, 0 on-line."""
    v = cross(p1, p2, q)
    if v > EPS:
        return 1
    if v < -EPS:
        return -1
    return 0


def _orient(a: Point, b: Point, c: Point) -> int:
    return side(a, b, c)


def _on_segment(a: Point, b: Point, c: Point) -> bool:
    """Assuming a,b,c are collinear, is c within the bounding box of segment a-b?"""
    return (
        min(a[0], b[0]) - EPS <= c[0] <= max(a[0], b[0]) + EPS
        and min(a[1], b[1]) - EPS <= c[1] <= max(a[1], b[1]) + EPS
    )


def segments_intersect(p1: Point, p2: Point, p3: Point, p4: Point) -> bool:
    """True if segment p1-p2 intersects segment p3-p4 (proper or collinear-overlap)."""
    d1 = _orient(p3, p4, p1)
    d2 = _orient(p3, p4, p2)
    d3 = _orient(p1, p2, p3)
    d4 = _orient(p1, p2, p4)
    if d1 != d2 and d3 != d4:
        return True
    if d1 == 0 and _on_segment(p3, p4, p1):
        return True
    if d2 == 0 and _on_segment(p3, p4, p2):
        return True
    if d3 == 0 and _on_segment(p1, p2, p3):
        return True
    if d4 == 0 and _on_segment(p1, p2, p4):
        return True
    return False


def to_pixels(norm_points: Sequence[Sequence[float]], width: int, height: int) -> list[Point]:
    """Convert normalized [0,1] coords to pixel coords for a given frame size."""
    return [(float(x) * width, float(y) * height) for x, y in norm_points]


def inside_sign_from_label(label: str) -> int:
    """Map an `inside: left|right` config label to the side sign used by the counter."""
    label = (label or "left").strip().lower()
    if label == "right":
        return 1
    if label == "left":
        return -1
    raise ValueError(f"inside must be 'left' or 'right', got {label!r}")
