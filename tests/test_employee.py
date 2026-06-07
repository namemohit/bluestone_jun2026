"""Employee gallery match/classifier + timesheet aggregation."""
from types import SimpleNamespace

from fusion.employee import EmployeeGallery, build_timesheets
from fusion.synthetic_visits import at_cosine, identity_vector
import numpy as np


def test_gallery_match_and_classifier():
    g = EmployeeGallery(threshold=0.7)
    emp = identity_vector(50)
    g.enroll(50, emp)
    rng = np.random.default_rng(0)
    # a near view of the employee matches
    is_emp, score, eid = g.classifier()(at_cosine(emp, 0.95, rng))
    assert is_emp is True and eid == 50 and score >= 0.7
    # a stranger does not
    is_emp2, _, eid2 = g.classifier()(identity_vector(123))
    assert is_emp2 is False and eid2 is None


def _emp_session(eid, t_in, t_out):
    return SimpleNamespace(
        is_employee=True, matched=True, employee_id=eid,
        entry_ts=t_in, exit_ts=t_out, dwell_sec=t_out - t_in,
    )


def test_build_timesheets_inside_outside():
    sessions = [
        _emp_session(50, 200, 500),   # 300s
        _emp_session(50, 600, 900),   # 300s
        _emp_session(51, 250, 400),   # 150s
        SimpleNamespace(is_employee=False, matched=True, employee_id=None,
                        entry_ts=0, exit_ts=10, dwell_sec=10),  # customer ignored
    ]
    sheets = build_timesheets(sessions, day_start=0.0, day_end=1000.0)
    assert set(sheets) == {50, 51}
    assert sheets[50]["time_inside_sec"] == 600
    assert sheets[50]["sessions"] == 2
    assert sheets[50]["time_outside_sec"] == 400  # 1000 span - 600 inside
    assert sheets[51]["time_inside_sec"] == 150
