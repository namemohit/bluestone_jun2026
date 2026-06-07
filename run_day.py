"""Batch 'nightly job' over a full day of dummy data — runs the entire pipeline:

  tracklets -> cross-camera fusion (global ids) -> entry/exit sessions (+employee-first)
            -> demographics -> grouping -> employee timesheets -> day report.

The pipeline is parameterized by `PipelineParams` (employee gallery + thresholds + grouping
gap), so a 'learned' model can be swapped in/out — this is what the HITL loop trains.

  python run_day.py
"""
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, field

from fusion.employee import EmployeeGallery, build_timesheets
from fusion.mtmc import fuse
from fusion.sessions import SessionManager
from fusion.synthetic_day import COLD_DEMO_BIAS, make_day
from logic.grouping import group_sessions
from logic.report import build_report
from vision.demographics import StubDemographicsEstimator, aggregate


@dataclass
class PipelineParams:
    employee_gallery: dict = field(default_factory=dict)  # employee_id -> [embeddings]
    group_gap_sec: float = 15.0
    match_threshold: float = 0.80
    review_threshold: float = 0.60
    employee_threshold: float = 0.70
    demographic_overrides: dict = field(default_factory=dict)  # identity -> {gender/age_bucket}
    demographic_bias: bool = False  # simulate an untrained (wrong) demographics model
    fusion_time_window: float = 5.0
    fusion_sim_threshold: float = 0.6


def default_params(gallery_embs: dict) -> PipelineParams:
    """Fully-enrolled params (one embedding per employee) — the 'trained' baseline."""
    return PipelineParams(employee_gallery={eid: [emb] for eid, emb in gallery_embs.items()})


def run_pipeline(seed: int = 0, params: PipelineParams | None = None, day_end: float = 2400.0):
    """Run the full day pipeline. Returns (report, ground_truth, SessionManager)."""
    tracklets, ground_truth, gallery_embs = make_day(seed=seed)
    if params is None:
        params = default_params(gallery_embs)

    gallery = EmployeeGallery(threshold=params.employee_threshold)
    for eid, embs in params.employee_gallery.items():
        for e in embs:
            gallery.enroll(eid, e)

    # 1) cross-camera fusion -> appearances (global ids)
    appearances = fuse(tracklets, time_window=params.fusion_time_window,
                       sim_threshold=params.fusion_sim_threshold)

    # 2) entry/exit sessions, employee-first classification
    mgr = SessionManager(
        window_sec=7200,
        match_threshold=params.match_threshold,
        review_threshold=params.review_threshold,
        employee_classifier=gallery.classifier(),
    )
    for a in sorted(appearances, key=lambda x: x.ts):
        if a.crossing == "in":
            ident = a.identities[0] if a.identities else None
            mgr.on_entry(a.embedding, a.ts,
                         ref={"global_id": a.global_id, "identity": ident}, attrs=a.attrs)
        elif a.crossing == "out":
            mgr.on_exit(a.embedding, a.ts, ref={"global_id": a.global_id})
    if appearances:
        mgr.expire(max(a.ts for a in appearances) + 1.0)

    # 3) demographics for matched customers (bias = untrained error; overrides = learned fixes)
    est = StubDemographicsEstimator()
    customers = [s for s in mgr.closed_sessions if s.matched and not s.is_employee]
    estimates = []
    for s in customers:
        d = est.estimate(s.attrs)
        ident = s.entry_ref.get("identity")
        if params.demographic_bias and ident in COLD_DEMO_BIAS:
            d = {**d, **COLD_DEMO_BIAS[ident]}
        if ident in params.demographic_overrides:
            d = {**d, **params.demographic_overrides[ident]}
        estimates.append(d)
    demographics = aggregate(estimates)

    # 4) grouping
    _, groups = group_sessions(mgr.closed_sessions, group_gap_sec=params.group_gap_sec)

    # 5) employee timesheets
    timesheets = build_timesheets(mgr.closed_sessions, day_start=0.0, day_end=day_end)

    report = build_report(mgr, groups, demographics, timesheets)
    return report, ground_truth, mgr


def run(seed: int = 0, day_end: float = 2400.0):
    report, gt, _ = run_pipeline(seed=seed, day_end=day_end)
    return report, gt


def main() -> None:
    ap = argparse.ArgumentParser(description="Batch day pipeline over dummy data")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    report, gt = run(seed=args.seed)
    print("================ DAY REPORT ================")
    print(json.dumps(report, indent=2))
    print("\n================ GROUND TRUTH ==============")
    print(json.dumps(gt, indent=2))


if __name__ == "__main__":
    main()
