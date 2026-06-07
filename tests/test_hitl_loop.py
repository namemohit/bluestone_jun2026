"""Self-improving loop: cold-start -> human labels -> retrain -> eval-gate -> improvement."""
from fusion.synthetic_day import EMPLOYEES
from hitl.corrections import LabelStore
from hitl.loop import run_improvement_cycle
from hitl.review import ReviewItem, select_for_review, simulate_human_corrections
from run_day import PipelineParams, run_pipeline
from training.eval_gate import gate
from training.trainer import train_candidate


def test_loop_improves_and_promotes():
    res = run_improvement_cycle(seed=0, target=0.95, rounds=1)
    h = res["history"]
    assert h[0]["score"] == 0.0          # cold start: everything wrong
    assert h[1]["score"] == 1.0          # after enrolling corrected employees
    assert h[1]["promoted"] is True
    assert res["active_version"] == 2
    assert res["final_score"] == 1.0


def test_loop_fixes_all_models():
    res = run_improvement_cycle(seed=0, target=0.95, rounds=1)
    assert all(v is False for v in res["history"][0]["checks"].values())  # cold start: all wrong
    assert all(v is True for v in res["history"][1]["checks"].values())   # after retrain: all right
    assert res["history"][1]["labels"] == 7  # 4 employee + 2 demographics + 1 group


def test_gate_logic():
    assert gate(0.5, 0.5)[0] is False                 # no improvement
    assert gate(0.90, 0.0, target=0.95)[0] is False   # below target
    assert gate(0.96, 0.5, target=0.95)[0] is True


def test_train_candidate_noop_without_labels():
    cand = train_candidate(PipelineParams(employee_gallery={}), LabelStore())
    assert cand.employee_gallery == {}


def test_simulate_corrections_targets_employees_only():
    _, _, mgr = run_pipeline(params=PipelineParams(employee_gallery={}))
    corr = simulate_human_corrections(mgr, set(EMPLOYEES.keys()))
    assert len(corr) == 4  # 2 employees x 2 sessions, all miscounted at cold start
    assert all(c.field == "is_employee" and c.after is True for c in corr)
    assert all(c.employee_id in EMPLOYEES for c in corr)


def test_select_for_review_uncertainty_order():
    items = [ReviewItem("a", "x", 0.95, {}, {}),
             ReviewItem("b", "x", 0.80, {}, {}),
             ReviewItem("c", "x", 0.60, {}, {})]
    picked = select_for_review(items, budget=2, threshold=0.8)
    assert [p.item_id for p in picked] == ["b", "a"]  # closest to threshold first
