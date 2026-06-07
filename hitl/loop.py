"""The self-improving loop — one full data-engine cycle across MULTIPLE models.

Round 0 cold-starts a deliberately-untrained pipeline:
  - empty employee gallery   -> staff miscounted as customers
  - demographics bias on      -> some gender/age wrong
  - too-small group gap       -> groups over-split
A (simulated) human corrects each, the trainer updates all three models from those
human-verified labels, and the eval gate promotes the candidate only if it measurably
improves on the frozen golden set.
"""
from __future__ import annotations

from fusion.synthetic_day import COLD_DEMO_BIAS, CUSTOMER_GROUPS, EMPLOYEES
from hitl.corrections import LabelStore
from hitl.review import (
    build_review_queue,
    simulate_demographics_corrections,
    simulate_grouping_corrections,
    simulate_human_corrections,
)
from run_day import PipelineParams, run_pipeline
from training.eval_gate import gate
from training.metrics import day_score
from training.registry import ModelRegistry
from training.trainer import train_candidate


def cold_start_params() -> PipelineParams:
    """An untrained pipeline that breaks the employee, demographics, and grouping models."""
    return PipelineParams(employee_gallery={}, group_gap_sec=3.0, demographic_bias=True)


def collect_corrections(mgr):
    """All human-verified corrections a reviewer would make for the day."""
    return (
        simulate_human_corrections(mgr, set(EMPLOYEES.keys()))
        + simulate_demographics_corrections(mgr, COLD_DEMO_BIAS)
        + simulate_grouping_corrections(mgr, CUSTOMER_GROUPS)
    )


def run_improvement_cycle(seed: int = 0, target: float = 0.95, rounds: int = 1) -> dict:
    registry = ModelRegistry()
    store = LabelStore()
    history = []

    params = cold_start_params()
    report, gt, mgr = run_pipeline(seed=seed, params=params)
    score, checks = day_score(report, gt)
    mv = registry.register(params, {"score": score, "checks": checks})
    registry.promote(mv.version)
    history.append({"round": 0, "score": score, "checks": checks, "promoted": True,
                    "labels": 0, "review_items": len(build_review_queue(mgr))})
    current_score = score

    for r in range(1, rounds + 1):
        store.add_round(collect_corrections(mgr))
        candidate = train_candidate(params, store)
        creport, _, cmgr = run_pipeline(seed=seed, params=candidate)
        cscore, cchecks = day_score(creport, gt)
        promote, reason = gate(cscore, current_score, target=target)
        history.append({"round": r, "score": cscore, "checks": cchecks,
                        "promoted": promote, "reason": reason, "labels": len(store)})
        if promote:
            mv = registry.register(candidate, {"score": cscore, "checks": cchecks})
            registry.promote(mv.version)
            params, mgr, current_score = candidate, cmgr, cscore

    return {
        "history": history,
        "active_version": registry.active_version,
        "final_score": current_score,
        "n_versions": len(registry.versions),
    }
