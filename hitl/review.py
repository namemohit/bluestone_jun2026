"""Review queue construction + active-learning sampling + a simulated reviewer.

`build_review_queue` surfaces uncertain/conflicting decisions; `select_for_review` ranks by
uncertainty (closest to the decision threshold). `simulate_human_corrections` stands in for a
human during demos/tests, using ground-truth identity to correct mislabeled employees.
"""
from __future__ import annotations

from dataclasses import dataclass

from hitl.corrections import Correction


@dataclass
class ReviewItem:
    item_id: str
    kind: str  # session_match | unmatched_exit | employee | demographics
    confidence: float
    predicted: dict
    payload: dict


def build_review_queue(mgr) -> list[ReviewItem]:
    items: list[ReviewItem] = []
    for s in mgr.closed_sessions:
        if s.matched and s.needs_review:
            items.append(ReviewItem(
                item_id=f"sess-{s.session_id}", kind="session_match",
                confidence=s.match_confidence,
                predicted={"is_employee": s.is_employee},
                payload={"global_id": s.entry_ref.get("global_id")},
            ))
    for i, ux in enumerate(mgr.unmatched_exits):
        items.append(ReviewItem(
            item_id=f"unmatched-{i}", kind="unmatched_exit",
            confidence=float(ux.get("best_sim", 0.0)), predicted={}, payload=ux,
        ))
    return items


def select_for_review(items, budget: int, threshold: float = 0.8) -> list[ReviewItem]:
    """Uncertainty sampling: review the items the model is least sure about first."""
    ranked = sorted(items, key=lambda it: abs(it.confidence - threshold))
    return ranked[: max(0, budget)]


def simulate_human_corrections(mgr, employee_ids) -> list[Correction]:
    """Stand-in reviewer: label sessions of known employees that were classified as customers.

    In production a human recognizes the employee from the session thumbnail; here we use the
    synthetic ground-truth identity carried on the session.
    """
    employee_ids = set(employee_ids)
    corrections: list[Correction] = []
    for s in mgr.closed_sessions:
        ident = s.entry_ref.get("identity")
        if ident in employee_ids and not s.is_employee:
            corrections.append(Correction(
                item_id=f"sess-{s.session_id}", field="is_employee",
                before=False, after=True, embedding=s.embedding,
                employee_id=ident, subject=ident,
            ))
    return corrections


def simulate_demographics_corrections(mgr, cold_bias) -> list[Correction]:
    """Reviewer corrects the cold-start demographic errors back to the truth (session.attrs)."""
    corrections: list[Correction] = []
    for s in mgr.closed_sessions:
        if not (s.matched and not s.is_employee):
            continue
        ident = s.entry_ref.get("identity")
        if ident in cold_bias:
            truth = {k: s.attrs.get(k) for k in cold_bias[ident]}
            corrections.append(Correction(
                item_id=f"demo-{s.session_id}", field="demographics",
                before=dict(cold_bias[ident]), after=truth, subject=ident,
            ))
    return corrections


def simulate_grouping_corrections(mgr, groups_by_identity) -> list[Correction]:
    """Reviewer confirms the correct customer groups (with arrival times) for the day."""
    ts_by_id = {
        s.entry_ref.get("identity"): s.entry_ts
        for s in mgr.closed_sessions if s.matched and not s.is_employee
    }
    after = []
    for grp in groups_by_identity:
        members = [(i, ts_by_id[i]) for i in grp if i in ts_by_id]
        if members:
            after.append(members)
    return [Correction(item_id="group-day", field="group", before=None, after=after)]


def build_review_cards(mgr, corrections) -> list[dict]:
    """Human-friendly review cards (JSON, no embeddings): thumbnail seed + readable predicted
    label + confidence + a control spec the dashboard renders (toggles / dropdowns)."""
    by_sid = {f"sess-{s.session_id}": s for s in mgr.closed_sessions}
    emp_options = sorted({c.employee_id for c in corrections
                          if c.field == "is_employee" and c.employee_id is not None})
    cards: list[dict] = []
    for c in corrections:
        vid = c.item_id.split("-")[-1]
        seed = c.subject if c.subject is not None else vid
        if c.field == "is_employee":
            s = by_sid.get(c.item_id)
            score = round(float(getattr(s, "employee_score", 0.0)), 2) if s else 0.0
            cards.append({
                "id": c.item_id, "kind": "employee",
                "title": f"Visit {vid} — staff or customer?",
                "thumb": {"seed": seed, "tag": f"#{vid}"},
                "prediction_label": "Customer",
                "confidence": score,
                "control": {"type": "employee",
                            "suggested": {"is_employee": True, "employee_id": c.employee_id},
                            "employee_options": emp_options},
            })
        elif c.field == "demographics":
            pred = c.before or {}
            cards.append({
                "id": c.item_id, "kind": "demographics",
                "title": f"Visit {vid} — confirm age & gender",
                "thumb": {"seed": seed, "tag": f"#{vid}"},
                "prediction_label": ", ".join(f"{k}: {v}" for k, v in pred.items()) or "—",
                "confidence": 0.5,
                "control": {"type": "demographics",
                            "suggested": dict(c.after or {}),
                            "gender_options": ["female", "male", "unknown"],
                            "age_options": ["0-12", "13-19", "20-34", "35-54", "55+"]},
            })
        elif c.field == "group":
            cards.append({
                "id": c.item_id, "kind": "group",
                "title": "Confirm customer groups for the day",
                "thumb": {"seed": "grp", "tag": "GRP"},
                "prediction_label": "current grouping over-splits arrivals",
                "confidence": 0.5,
                "control": {"type": "group", "suggested": {"groups": len(c.after or [])}},
            })
    return cards
