"""Edge agent: motion-clip segmentation + local sink."""
from pathlib import Path

import numpy as np

from edge.motion_clip import segment_motion_clips
from edge.sink import LocalSink


def _frames():
    """30 frames: still, then a moving box (motion), then still."""
    for i in range(30):
        f = np.zeros((120, 160, 3), np.uint8)
        if 8 <= i <= 20:
            x = 10 + (i - 8) * 8
            f[40:80, x:x + 20] = 255
        yield i, float(i) / 8.0, f


def test_segment_motion_clips(tmp_path):
    clips = segment_motion_clips(_frames(), tmp_path, prefix="cam_entry", fps=8.0)
    assert len(clips) >= 1
    for c in clips:
        p = Path(c)
        assert p.exists() and p.stat().st_size > 0
        assert p.name.startswith("cam_entry_") and p.suffix == ".mp4"


def test_local_sink(tmp_path):
    src = tmp_path / "a.mp4"
    src.write_bytes(b"data")
    sink = LocalSink(str(tmp_path / "sink"))
    sink.put(str(src), "cam_entry_1.mp4")
    assert sink.list() == ["cam_entry_1.mp4"]
    dest = tmp_path / "got.mp4"
    sink.get("cam_entry_1.mp4", str(dest))
    assert dest.read_bytes() == b"data"
    sink.delete("cam_entry_1.mp4")
    assert sink.list() == []
