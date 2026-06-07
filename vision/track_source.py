"""Track sources: a common `FrameTracks` iterator with two backends.

- SyntheticTrackSource: scripted walkers crossing a mid-line. No model/GPU. Renders frames
  so the full ingest->count->overlay->events path is exercised. Default scenario => in=3, out=2.
- YoloTrackSource: ultralytics YOLO + BoT-SORT over an ingest VideoStream (lazy import, GPU path).
"""
from __future__ import annotations

from typing import Iterator, Optional

import numpy as np

from common.types import FrameTracks, Track


class SyntheticTrackSource:
    # (track_id, x_norm, y_start_norm, y_end_norm, frame_start, frame_end)
    DEFAULT_WALKERS = [
        (1, 0.20, 0.92, 0.08, 0, 60),   # bottom->top  => IN
        (2, 0.40, 0.92, 0.08, 10, 70),  # IN
        (3, 0.60, 0.92, 0.08, 20, 80),  # IN
        (4, 0.80, 0.08, 0.92, 30, 90),  # top->bottom  => OUT
        (5, 0.50, 0.08, 0.92, 60, 119), # OUT
    ]
    EXPECTED_COUNTS = {"in": 3, "out": 2}

    def __init__(
        self,
        width: int = 1280,
        height: int = 720,
        fps: float = 15.0,
        n_frames: int = 120,
        walkers: Optional[list] = None,
        start_ts: float = 0.0,
    ):
        self.width = width
        self.height = height
        self.fps = fps
        self.n_frames = n_frames
        self.walkers = walkers if walkers is not None else self.DEFAULT_WALKERS
        self.start_ts = start_ts

    def __iter__(self) -> Iterator[FrameTracks]:
        return self.frames()

    def frames(self) -> Iterator[FrameTracks]:
        W, H = self.width, self.height
        bw, bh = int(0.07 * W), int(0.20 * H)
        for f in range(self.n_frames):
            frame = np.full((H, W, 3), 30, np.uint8)
            frame[:] = (40, 35, 30)  # dim BGR background
            ts = self.start_ts + f / self.fps
            tracks: list[Track] = []
            for (tid, x, y0, y1, fs, fe) in self.walkers:
                if fs <= f <= fe:
                    a = (f - fs) / max(1, (fe - fs))
                    cx = x * W
                    cy = (y0 + (y1 - y0) * a) * H
                    tracks.append(
                        Track(
                            track_id=tid,
                            bbox=(cx - bw / 2, cy - bh / 2, cx + bw / 2, cy + bh / 2),
                            conf=0.99,
                            cls=0,
                        )
                    )
            yield FrameTracks(frame_idx=f, ts=ts, tracks=tracks, frame=frame, width=W, height=H)


class YoloTrackSource:
    """Wrap an ingest VideoStream with YOLO detection + BoT-SORT tracking (persistent IDs)."""

    def __init__(self, video_stream, detector_cfg: dict, motion_gate=None):
        self.vs = video_stream
        self.cfg = detector_cfg or {}
        self.motion_gate = motion_gate
        self._model = None

    @property
    def fps(self) -> float:
        s = float(getattr(self.vs, "fps_sample", 0) or 0)
        return s if s > 0 else float(getattr(self.vs, "native_fps", 0) or 15.0)

    def _load(self):
        from ultralytics import YOLO  # lazy: heavy import only when actually used

        self._model = YOLO(self.cfg.get("weights", "yolo11n.pt"))

    def __iter__(self) -> Iterator[FrameTracks]:
        return self.frames()

    def frames(self) -> Iterator[FrameTracks]:
        if self._model is None:
            self._load()
        classes = self.cfg.get("classes", [0])
        conf = float(self.cfg.get("conf", 0.35))
        iou = float(self.cfg.get("iou", 0.5))
        imgsz = int(self.cfg.get("imgsz", 960))
        device = self.cfg.get("device", None)
        tracker = self.cfg.get("tracker", "botsort.yaml")

        for idx, ts, frame in self.vs.frames():
            h, w = frame.shape[:2]
            if self.motion_gate is not None and not self.motion_gate.is_active(frame):
                yield FrameTracks(frame_idx=idx, ts=ts, tracks=[], frame=frame, width=w, height=h)
                continue

            result = self._model.track(
                frame,
                persist=True,
                classes=classes,
                conf=conf,
                iou=iou,
                imgsz=imgsz,
                device=device,
                tracker=tracker,
                verbose=False,
            )[0]

            tracks: list[Track] = []
            boxes = getattr(result, "boxes", None)
            if boxes is not None and boxes.id is not None:
                xyxy = boxes.xyxy.cpu().numpy()
                ids = boxes.id.int().cpu().numpy()
                confs = boxes.conf.cpu().numpy()
                for (x1, y1, x2, y2), tid, c in zip(xyxy, ids, confs):
                    tracks.append(
                        Track(int(tid), (float(x1), float(y1), float(x2), float(y2)), float(c), 0)
                    )
            yield FrameTracks(frame_idx=idx, ts=ts, tracks=tracks, frame=frame, width=w, height=h)
