"""Produce a candidate model from human-verified corrections — across MULTIPLE models.

From one round of corrections this updates three "models" at once:
  - employee classifier  <- `is_employee` corrections  (enroll embeddings into the gallery)
  - demographics model   <- `demographics` corrections (per-identity learned overrides)
  - grouping model       <- `group` corrections        (widen group_gap to cover corrected groups)

The same hook is where real fine-tuning of the detector / ReID / MiVOLO head plugs in for
production, using the accumulated LabelStore as the per-model dataset.
"""
from __future__ import annotations

from run_day import PipelineParams


def train_candidate(params: PipelineParams, label_store) -> PipelineParams:
    # 1) employee classifier: enroll corrected employees into the gallery
    gallery = {eid: list(embs) for eid, embs in params.employee_gallery.items()}
    for c in label_store.for_field("is_employee"):
        if c.after is True and c.embedding is not None and c.employee_id is not None:
            gallery.setdefault(c.employee_id, []).append(c.embedding)

    # 2) demographics: per-identity overrides from corrected gender/age
    overrides = {k: dict(v) for k, v in params.demographic_overrides.items()}
    for c in label_store.for_field("demographics"):
        if c.subject is not None and isinstance(c.after, dict):
            overrides[c.subject] = {**overrides.get(c.subject, {}), **c.after}

    # 3) grouping: widen the arrival gap so it covers the human-confirmed groups
    group_gap = params.group_gap_sec
    spans = []
    for c in label_store.for_field("group"):
        for grp in (c.after or []):
            times = [t for (_ident, t) in grp]
            if len(times) >= 2:
                spans.append(max(times) - min(times))
    if spans:
        group_gap = max(group_gap, max(spans))

    return PipelineParams(
        employee_gallery=gallery,
        group_gap_sec=group_gap,
        match_threshold=params.match_threshold,
        review_threshold=params.review_threshold,
        employee_threshold=params.employee_threshold,
        demographic_overrides=overrides,
        demographic_bias=params.demographic_bias,
    )
