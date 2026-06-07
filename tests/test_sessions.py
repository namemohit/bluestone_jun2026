"""Entry<->exit session matcher: matching, dwell, employee split, review/unmatched/timeout."""
import numpy as np

from fusion.sessions import SessionManager, cosine_gallery_classifier
from fusion.synthetic_visits import at_cosine, identity_vector, make_scenario


def _run(mgr, events):
    last = 0.0
    for e in events:
        last = max(last, e["ts"])
        if e["kind"] == "entry":
            mgr.on_entry(e["emb"], e["ts"], ref={**e["ref"], "identity": e["identity"]})
        else:
            mgr.on_exit(e["emb"], e["ts"], ref=e["ref"])
    mgr.expire(last + 1)
    return mgr


# ---- scenario-level (integration) ----------------------------------------
def _scenario_mgr():
    events, expected, gallery = make_scenario(seed=0)
    mgr = SessionManager(employee_classifier=cosine_gallery_classifier(gallery, 0.7))
    _run(mgr, events)
    return mgr, expected


def test_scenario_summary_split():
    mgr, _ = _scenario_mgr()
    s = mgr.summary()
    assert s["customer_visits"] == 7
    assert s["employee_sessions"] == 1
    assert s["needs_review"] == 1
    assert s["unmatched_exits"] == 1
    assert s["still_inside_or_timed_out"] == 1
    assert s["max_occupancy"] == 8


def test_dwell_exact_for_matched():
    mgr, expected = _scenario_mgr()
    for sess in mgr.closed_sessions:
        if sess.matched:
            ident = sess.entry_ref.get("identity")
            assert abs(sess.dwell_sec - expected[ident]) < 1e-6


def test_employee_tagged_and_excluded():
    mgr, _ = _scenario_mgr()
    emp = [s for s in mgr.closed_sessions if s.is_employee and s.matched]
    assert len(emp) == 1 and emp[0].entry_ref.get("identity") == 50


def test_lookalike_flagged_for_review():
    mgr, _ = _scenario_mgr()
    rev = [s for s in mgr.closed_sessions if s.needs_review]
    assert len(rev) == 1 and rev[0].entry_ref.get("identity") == 7


# ---- unit-level ----------------------------------------------------------
def test_basic_match_and_dwell():
    mgr = SessionManager()
    v = identity_vector(1)
    mgr.on_entry(v, 100.0)
    out = mgr.on_exit(v, 460.0)
    assert out.matched and not out.needs_review
    assert out.session.dwell_sec == 360.0


def test_review_band_between_thresholds():
    mgr = SessionManager(match_threshold=0.8, review_threshold=0.6)
    v = identity_vector(2)
    mgr.on_entry(v, 0.0)
    out = mgr.on_exit(at_cosine(v, 0.70, np.random.default_rng(0)), 100.0)
    assert out.matched and out.needs_review


def test_unmatched_when_below_review():
    mgr = SessionManager(review_threshold=0.6)
    mgr.on_entry(identity_vector(3), 0.0)
    out = mgr.on_exit(identity_vector(123456), 50.0)  # ~orthogonal
    assert out.unmatched and out.session is None


def test_window_expires_old_entries():
    mgr = SessionManager(window_sec=600)
    v = identity_vector(4)
    mgr.on_entry(v, 0.0)
    out = mgr.on_exit(v, 1000.0)  # beyond window -> entry expired -> unmatched
    assert out.unmatched
    assert mgr.summary()["still_inside_or_timed_out"] == 1


def test_occupancy_tracks_peak():
    mgr = SessionManager()
    mgr.on_entry(identity_vector(1), 0.0)
    mgr.on_entry(identity_vector(2), 1.0)
    assert mgr.occupancy() == 2 and mgr.max_occupancy == 2
    mgr.on_exit(identity_vector(1), 2.0)
    assert mgr.occupancy() == 1 and mgr.max_occupancy == 2


def test_employee_classifier_tags_entry():
    emp = identity_vector(50)
    mgr = SessionManager(employee_classifier=cosine_gallery_classifier([emp], 0.7))
    assert mgr.on_entry(emp, 0.0).is_employee is True
    assert mgr.on_entry(identity_vector(1), 1.0).is_employee is False
