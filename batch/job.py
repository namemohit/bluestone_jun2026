"""Nightly batch job (Cloud Run Job entrypoint).

Reads the day's motion clips from the sink (GCS in prod), runs the per-camera vision pipeline on
each, aggregates crossing events into a day report, and optionally deletes the raw clips.

  # real (needs GPU + models): clips in GCS -> report
  python -m batch.job --config configs/cameras.yaml --sink gcs:my-bucket/showroom-42 --delete-after

  # GPU-free plumbing demo (synthetic detections)
  python -m batch.job --config configs/cameras.yaml --sink local:clips --synthetic
"""
from __future__ import annotations

import argparse
import json
import tempfile
from collections import defaultdict
from pathlib import Path

from common.config import load_config
from vision.pipeline import Pipeline

MID_LINE = {"id": "mid", "points": [[0.05, 0.5], [0.95, 0.5]], "inside": "left"}


def camera_of(clip_name: str) -> str:
    """`cam_entry_1730000000000.mp4` -> `cam_entry`."""
    return clip_name.rsplit(".", 1)[0].rsplit("_", 1)[0]


def real_track_source(clip_path: str, camera_cfg: dict, detector_cfg: dict):
    from ingest.video_source import VideoStream  # lazy
    from vision.track_source import YoloTrackSource

    return YoloTrackSource(VideoStream(clip_path), detector_cfg)


def run_batch(sink, config: dict, *, track_source_factory=None, work_dir: str | None = None,
              delete_after: bool = False, out_path: str | None = None) -> dict:
    cameras = {c["id"]: c for c in config.get("cameras", [])}
    counting = config.get("counting", {})
    detector = config.get("detector", {})
    work_dir = work_dir or tempfile.mkdtemp()
    Path(work_dir).mkdir(parents=True, exist_ok=True)

    by_cam: dict[str, dict] = defaultdict(lambda: {"in": 0, "out": 0})
    processed = 0
    for name in sink.list():
        cam_id = camera_of(name)
        cam = cameras.get(cam_id)
        if cam is None:
            continue  # clip for an unknown camera
        local = str(Path(work_dir) / name)
        sink.get(name, local)
        ts_src = (track_source_factory(local, cam) if track_source_factory
                  else real_track_source(local, cam, detector))
        pipe = Pipeline(cam_id, cam, counting)
        _, events = pipe.run(ts_src, out_video=None, events_path=None)
        for e in events:
            if e.direction in ("in", "out"):
                by_cam[cam_id][e.direction] += 1
        processed += 1
        if delete_after:
            sink.delete(name)

    tin = sum(v["in"] for v in by_cam.values())
    tout = sum(v["out"] for v in by_cam.values())
    report = {"clips_processed": processed, "by_camera": dict(by_cam),
              "totals": {"in": tin, "out": tout, "net": tin - tout}}
    if out_path:
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        Path(out_path).write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def main() -> None:
    ap = argparse.ArgumentParser(description="Nightly batch: clips -> day report")
    ap.add_argument("--config", required=True)
    ap.add_argument("--sink", default="local:clips")
    ap.add_argument("--out", default="outputs/day_report.json")
    ap.add_argument("--delete-after", action="store_true")
    ap.add_argument("--synthetic", action="store_true", help="GPU-free plumbing demo")
    args = ap.parse_args()

    from edge.sink import make_sink

    cfg = load_config(args.config)
    sink = make_sink(args.sink)

    factory = None
    if args.synthetic:
        from vision.track_source import SyntheticTrackSource
        for cam in cfg.get("cameras", []):
            if not cam.get("counting_lines"):
                cam["counting_lines"] = [MID_LINE]
        factory = lambda _clip, _cam: SyntheticTrackSource()  # noqa: E731

    report = run_batch(sink, cfg, track_source_factory=factory, out_path=args.out,
                       delete_after=args.delete_after)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
