"""Review-card builder: right kinds/counts, JSON-safe dicts with control specs."""
from hitl.loop import cold_start_params, collect_corrections
from hitl.review import build_review_cards
from run_day import run_pipeline


def test_build_review_cards_kinds_and_count():
    _, _, mgr = run_pipeline(params=cold_start_params())
    cards = build_review_cards(mgr, collect_corrections(mgr))
    assert len(cards) == 7
    kinds = [c["kind"] for c in cards]
    assert kinds.count("employee") == 4
    assert kinds.count("demographics") == 2
    assert kinds.count("group") == 1
    for c in cards:
        assert {"id", "kind", "title", "thumb", "prediction_label", "confidence", "control"} <= set(c)
        assert "type" in c["control"]
    # employee cards offer a staff picker
    emp = next(c for c in cards if c["kind"] == "employee")
    assert emp["control"]["employee_options"]  # non-empty
