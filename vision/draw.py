"""Shared overlay drawing (used by the pipeline and the zone-preview tool)."""
from __future__ import annotations

import math

import cv2

from vision.geometry import side

LINE_COLOR = (0, 215, 255)   # amber
INSIDE_COLOR = (80, 220, 80)  # green


def draw_counting_lines(vis, lines, counts: dict | None = None):
    """Draw each counting line with its P1->P2 direction arrow, an 'IN' arrow pointing
    to the inside side, and (optionally) live in/out tallies."""
    for ln in lines:
        p1 = (int(ln.p1[0]), int(ln.p1[1]))
        p2 = (int(ln.p2[0]), int(ln.p2[1]))
        cv2.line(vis, p1, p2, LINE_COLOR, 3)
        cv2.arrowedLine(vis, p1, p2, LINE_COLOR, 2, tipLength=0.04)

        mid = ((p1[0] + p2[0]) // 2, (p1[1] + p2[1]) // 2)
        dx, dy = (p2[0] - p1[0]), (p2[1] - p1[1])
        length = math.hypot(dx, dy) or 1.0
        nx, ny = -dy / length, dx / length
        if side(ln.p1, ln.p2, (mid[0] + nx * 10, mid[1] + ny * 10)) != ln.inside_sign:
            nx, ny = -nx, -ny
        tip = (int(mid[0] + nx * 45), int(mid[1] + ny * 45))
        cv2.arrowedLine(vis, mid, tip, INSIDE_COLOR, 2, tipLength=0.3)
        cv2.putText(vis, "IN", (tip[0] + 4, tip[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.5, INSIDE_COLOR, 2)

        label = ln.id
        if counts is not None and ln.id in counts:
            c = counts[ln.id]
            label = f"{ln.id}  in:{c['in']} out:{c['out']}"
        cv2.putText(vis, label, (p1[0], max(20, p1[1] - 8)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, LINE_COLOR, 2)
    return vis
