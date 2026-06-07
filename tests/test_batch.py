"""Nightly batch job: clip grouping + orchestration (stub track source, GPU-free)."""
from batch.job import camera_of, run_batch
from edge.sink import LocalSink
from vision.track_source import SyntheticTrackSource


def test_camera_of():
    assert camera_of("cam_entry_1730000000000.mp4") == "cam_entry"
    assert camera_of("cam_common_42.mp4") == "cam_common"


def test_run_batch_aggregates_events(tmp_path):
    sink = LocalSink(str(tmp_path / "clips"))
    for n in ("cam_entry_1.mp4", "cam_entry_2.mp4"):
        p = tmp_path / n
        p.write_bytes(b"x")
        sink.put(str(p), n)

    config = {
        "cameras": [{"id": "cam_entry", "role": "entry",
                     "counting_lines": [{"id": "mid", "points": [[0.05, 0.5], [0.95, 0.5]], "inside": "left"}]}],
        "counting": {"track_point": "bottom_center", "min_track_len": 3},
    }
    report = run_batch(sink, config, work_dir=str(tmp_path / "w"),
                       track_source_factory=lambda clip, cam: SyntheticTrackSource())
    assert report["clips_processed"] == 2
    # SyntheticTrackSource => 3 in / 2 out per clip, x2 clips
    assert report["by_camera"]["cam_entry"] == {"in": 6, "out": 4}
    assert report["totals"] == {"in": 6, "out": 4, "net": 2}
