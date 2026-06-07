"""Accuracy harness: event summing, segment filtering, MAPE + pass/fail."""
from evaluation.harness import evaluate, sum_events

EVENTS = [
    {"ts": 1.0, "direction": "in"},
    {"ts": 2.0, "direction": "in"},
    {"ts": 3.0, "direction": "out"},
    {"ts": 10.0, "direction": "in"},
]


def test_sum_events_unbounded():
    assert sum_events(EVENTS, None, None) == {"in": 3, "out": 1}


def test_sum_events_segment_filter():
    assert sum_events(EVENTS, 0.0, 5.0) == {"in": 2, "out": 1}


def test_evaluate_perfect_match_passes():
    truth = {"name": "t", "segments": [{"id": "full", "truth": {"in": 3, "out": 1}}]}
    rep = evaluate(truth, {"events": EVENTS}, gate=5.0)
    assert rep["mape"] == 0.0 and rep["passed"] is True


def test_evaluate_off_by_one_fails_tight_gate():
    # truth in=4 (pred 3 -> 25% err), out=1 (0%) => MAPE 12.5% > 5%
    truth = {"name": "t", "segments": [{"id": "full", "truth": {"in": 4, "out": 1}}]}
    rep = evaluate(truth, {"events": EVENTS}, gate=5.0)
    assert rep["passed"] is False
    assert abs(rep["mape"] - 12.5) < 1e-6
