"""Directional counting behavior: in/out, no-double-count, segment bound, min length."""
from vision.counter import CountingLine, LineCounter

# Horizontal line across the middle; inside_sign=-1 => "inside" is the TOP (small y).
LINE = CountingLine(id="door", p1=(0.0, 50.0), p2=(100.0, 50.0), inside_sign=-1)


def _feed(counter, track_id, points):
    events = []
    for i, p in enumerate(points):
        events += counter.update(track_id, p, ts=float(i), frame_idx=i, camera_id="cam")
    return events


def test_counts_entry_when_moving_to_inside():
    c = LineCounter([LINE], min_track_len=3)
    # bottom (outside) -> top (inside)
    ev = _feed(c, 1, [(50, 90), (50, 60), (50, 40), (50, 10)])
    assert c.counts["door"] == {"in": 1, "out": 0}
    assert len(ev) == 1 and ev[0].direction == "in"


def test_counts_exit_when_moving_to_outside():
    c = LineCounter([LINE], min_track_len=3)
    # top (inside) -> bottom (outside)
    ev = _feed(c, 1, [(50, 10), (50, 40), (50, 60), (50, 90)])
    assert c.counts["door"] == {"in": 0, "out": 1}
    assert ev[0].direction == "out"


def test_no_double_count_when_lingering_after_crossing():
    c = LineCounter([LINE], min_track_len=3)
    _feed(c, 1, [(50, 90), (50, 60), (50, 40), (50, 30), (50, 35), (50, 20)])
    assert c.counts["door"] == {"in": 1, "out": 0}


def test_round_trip_counts_one_each_direction():
    c = LineCounter([LINE], min_track_len=3)
    _feed(c, 1, [(50, 90), (50, 60), (50, 40), (50, 10), (50, 40), (50, 90)])
    assert c.counts["door"] == {"in": 1, "out": 1}
    assert c.totals() == {"in": 1, "out": 1, "net": 0}


def test_min_track_len_suppresses_very_short_tracks():
    c = LineCounter([LINE], min_track_len=3)
    # crosses on the 2nd update (seen=2 < 3) -> suppressed
    _feed(c, 1, [(50, 60), (50, 40)])
    assert c.counts["door"] == {"in": 0, "out": 0}


def test_crossing_outside_segment_not_counted():
    # Short line spanning x in [0,10]; a track crossing y=50 at x=50 is outside the segment.
    short = CountingLine(id="seg", p1=(0.0, 50.0), p2=(10.0, 50.0), inside_sign=-1)
    c = LineCounter([short], min_track_len=1)
    _feed(c, 7, [(50, 90), (50, 60), (50, 40), (50, 10)])
    assert c.counts["seg"] == {"in": 0, "out": 0}


def test_two_tracks_independent():
    c = LineCounter([LINE], min_track_len=3)
    _feed(c, 1, [(20, 90), (20, 60), (20, 40), (20, 10)])  # in
    _feed(c, 2, [(80, 10), (80, 40), (80, 60), (80, 90)])  # out
    assert c.counts["door"] == {"in": 1, "out": 1}


def test_drop_missing_releases_state():
    c = LineCounter([LINE], min_track_len=3)
    _feed(c, 1, [(50, 90), (50, 60)])
    c.drop_missing(active_ids=[])  # track 1 gone
    assert 1 not in c._last_point and 1 not in c._seen
