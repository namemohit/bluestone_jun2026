"""Scoring a day report against ground truth — drives the eval gate + drift monitoring."""
from __future__ import annotations


def day_score(report: dict, gt: dict):
    """Return (score in [0,1], per-check booleans) over the key day outputs."""
    checks = {
        "unique_customers": report["customers"]["unique_customers"] == gt["unique_customers"],
        "employee_headcount": report["employees"]["headcount"] == gt["n_employees"],
        "group_count": report["customers"]["groups"]["count"] == gt["n_groups"],
        "gender": report["customers"]["demographics"]["gender"] == gt["gender"],
        "age_bucket": report["customers"]["demographics"]["age_bucket"] == gt["age_bucket"],
    }
    score = sum(1 for v in checks.values() if v) / len(checks)
    return score, {k: bool(v) for k, v in checks.items()}
