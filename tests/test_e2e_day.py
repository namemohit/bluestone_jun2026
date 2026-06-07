"""End-to-end: the full batch day pipeline over dummy data must match ground truth."""
from run_day import run


def test_day_report_matches_ground_truth():
    report, gt = run()
    c = report["customers"]
    assert c["unique_customers"] == gt["unique_customers"]
    assert c["groups"]["count"] == gt["n_groups"]
    assert c["groups"]["sizes"] == gt["group_sizes"]
    assert c["demographics"]["gender"] == gt["gender"]
    assert c["demographics"]["age_bucket"] == gt["age_bucket"]

    e = report["employees"]
    assert e["headcount"] == gt["n_employees"]
    assert {k: v["sessions"] for k, v in e["timesheets"].items()} == gt["employee_sessions"]

    # clean deterministic day -> nothing should be flagged
    assert report["review_queue"]["needs_review"] == 0
    assert report["review_queue"]["unmatched_exits"] == 0


def test_employee_timesheet_inside_outside():
    report, _ = run()
    ts = report["employees"]["timesheets"]
    assert ts[50]["time_inside_sec"] == 600.0
    assert ts[50]["time_outside_sec"] == 1800.0  # 2400 day span - 600 inside
    assert ts[51]["time_inside_sec"] == 500.0
    assert ts[51]["time_outside_sec"] == 1900.0
