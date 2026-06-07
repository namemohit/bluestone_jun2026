"""Single-camera pipeline: track source -> directional counter -> overlay + events.

Backend-agnostic: feed it any FrameTracks iterator (synthetic or YOLO).
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import cv2

from common.types import FrameTracks
from vision.counter import LineCounter, build_lines
from vision.draw import draw_counting_lines


class Pipeline:
    def __init__(
        self,
        camera_id: str,
        camera_cfg: dict,
        counting_cfg: Optional[dict] = None,
        writer=None,
        draw: bool = True,
    ):
        counting_cfg = counting_cfg or {}
        self.camera_id = camera_id
        self.camera_cfg = camera_cfg
        self.track_point = counting_cfg.get("track_point", "bottom_center")
        self.min_track_len = int(counting_cfg.get("min_track_len", 3))
        self.max_idle = float(counting_cfg.get("max_idle_sec", 5.0))
        self.writer = writer
        self.draw = draw
        self.counter: Optional[LineCounter] = None
        self._w = 0
        self._h = 0

    def _ensure_counter(self, w: int, h: int) -> None:
        if self.counter is None:
            lines = build_lines(self.camera_cfg, w, h)
            self.counter = LineCounter(lines, self.min_track_len)
            self._w, self._h = w, h

    def run(
        self,
        track_source,
        out_video: Optional[str] = None,
        events_path: Optional[str] = None,
        out_fps: float = 15.0,
        max_frames: int = 0,
        log_every: int = 60,
    ) -> tuple[dict, list]:
        events: list = []
        last_seen: dict[int, float] = {}
        writer_vid = None
        n = 0

        for ft in track_source:
            w = ft.width or (ft.frame.shape[1] if ft.frame is not None else self._w)
            h = ft.height or (ft.frame.shape[0] if ft.frame is not None else self._h)
            self._ensure_counter(w, h)

            active = []
            for t in ft.tracks:
                pt = t.point(self.track_point)
                for e in self.counter.update(t.track_id, pt, ft.ts, ft.frame_idx, self.camera_id):
                    events.append(e)
                    if self.writer is not None:
                        self.writer.write_event(e)
                last_seen[t.track_id] = ft.ts
                active.append(t.track_id)

            # periodic prune of stale track state (keeps memory bounded; tolerant of
            # brief occlusion since BoT-SORT reuses ids)
            if ft.frame_idx % 30 == 0 and last_seen:
                keep = [tid for tid, t0 in last_seen.items() if ft.ts - t0 <= self.max_idle]
                self.counter.drop_missing(keep)
                last_seen = {tid: last_seen[tid] for tid in keep}

            if ft.frame is not None and out_video:
                vis = self._render(ft)
                if writer_vid is None:
                    Path(out_video).parent.mkdir(parents=True, exist_ok=True)
                    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                    writer_vid = cv2.VideoWriter(str(out_video), fourcc, float(out_fps), (w, h))
                writer_vid.write(vis)

            n += 1
            if max_frames and n >= max_frames:
                break

        if writer_vid is not None:
            writer_vid.release()

        summary = self.summary()
        if events_path:
            Path(events_path).parent.mkdir(parents=True, exist_ok=True)
            payload = {
                "camera_id": self.camera_id,
                "summary": summary,
                "events": [e.to_dict() for e in events],
            }
            Path(events_path).write_text(json.dumps(payload, indent=2), encoding="utf-8")

        return summary, events

    def summary(self) -> dict:
        if self.counter is None:
            return {"lines": {}, "totals": {"in": 0, "out": 0, "net": 0}}
        totals = self.counter.totals()
        return {
            "camera_id": self.camera_id,
            "lines": self.counter.counts,
            "totals": totals,
            "entries": totals["in"],  # single-cam entry estimate (footfall at this camera)
        }

    # -- rendering ---------------------------------------------------------
    def _render(self, ft: FrameTracks):
        vis = ft.frame.copy()
        draw_counting_lines(vis, self.counter.lines, self.counter.counts)

        # tracks
        for t in ft.tracks:
            x1, y1, x2, y2 = (int(v) for v in t.bbox)
            cv2.rectangle(vis, (x1, y1), (x2, y2), (255, 180, 0), 2)
            px, py = (int(v) for v in t.point(self.track_point))
            cv2.circle(vis, (px, py), 4, (0, 0, 255), -1)
            cv2.putText(vis, f"#{t.track_id}", (x1, max(14, y1 - 6)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 180, 0), 2)

        # HUD
        tot = self.counter.totals()
        cv2.rectangle(vis, (0, 0), (vis.shape[1], 30), (0, 0, 0), -1)
        cv2.putText(
            vis, f"cam:{self.camera_id}  IN:{tot['in']}  OUT:{tot['out']}  NET:{tot['net']}",
            (8, 21), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2,
        )
        return vis
