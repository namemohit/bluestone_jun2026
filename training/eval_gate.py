"""Promotion gate: a candidate model is promoted only if it beats the current one AND clears
the target on the frozen golden set. (Shadow/canary before full rollout in production.)"""
from __future__ import annotations


def gate(candidate_score: float, current_score: float, target: float = 0.95):
    promote = candidate_score >= target and candidate_score > current_score
    reason = (
        f"candidate={candidate_score:.2f} vs current={current_score:.2f} "
        f"(target {target:.2f}) -> {'PROMOTE' if promote else 'REJECT'}"
    )
    return promote, reason
