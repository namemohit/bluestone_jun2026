"""Group detection: arrival-window clustering, employees excluded."""
from types import SimpleNamespace

from logic.grouping import group_sessions


def _s(sid, ts, is_emp=False):
    return SimpleNamespace(session_id=sid, entry_ts=ts, is_employee=is_emp)


def test_groups_by_arrival_window():
    sessions = [
        _s(1, 100), _s(2, 103),            # group A (within 15s)
        _s(3, 300), _s(4, 305), _s(5, 308),  # group B
        _s(6, 1000),                        # solo
        _s(99, 304, is_emp=True),           # employee -> excluded
    ]
    mapping, groups = group_sessions(sessions, group_gap_sec=15.0)
    sizes = sorted(g["size"] for g in groups)
    assert sizes == [1, 2, 3]
    assert mapping[1] == mapping[2]
    assert mapping[3] == mapping[4] == mapping[5]
    assert 99 not in mapping  # employee excluded


def test_no_runaway_chaining():
    # spaced 10s apart but window is from the FIRST member -> shouldn't chain into one big group
    sessions = [_s(i, i * 10.0) for i in range(6)]  # ts 0,10,20,30,40,50
    _, groups = group_sessions(sessions, group_gap_sec=15.0)
    # group1 from t=0 covers 0,10 (<=15); next from 20 covers 20,30; next 40 covers 40,50
    assert [g["size"] for g in groups] == [2, 2, 2]
