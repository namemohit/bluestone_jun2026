"""Entry<->exit visit sessions + dwell + occupancy, with the customer/employee split.

Flow:
  on_entry(emb, ts)  -> opens a VisitSession (employee-classified first, if a gallery hook is set)
  on_exit(emb, ts)   -> matches ONLY open sessions within `window_sec` by cosine:
                          >= match_threshold  -> close + dwell
                          >= review_threshold -> close + needs_review
                          else                -> unmatched exit (flagged)
  expire(now)        -> times out stale open sessions (person still inside / missed exit)

Matching is by embedding, so it works across different track ids / cameras. Employee sessions
are tagged and reported separately (Employees tab) and excluded from customer counts.
"""
from __future__ import annotations

import statistics
from dataclasses import asdict, dataclass, field
from typing import Callable, Optional

import numpy as np

from fusion.tracklet import cosine, l2_normalize

# emb -> (is_employee, score)
EmployeeClassifier = Callable[[np.ndarray], "tuple[bool, float]"]


@dataclass
class VisitSession:
    session_id: int
    entry_ts: float
    embedding: np.ndarray
    is_employee: bool = False
    employee_score: float = 0.0
    employee_id: Optional[int] = None
    entry_ref: dict = field(default_factory=dict)
    attrs: dict = field(default_factory=dict)
    exit_ts: Optional[float] = None
    exit_ref: Optional[dict] = None
    match_confidence: float = 0.0
    matched: bool = False
    needs_review: bool = False
    status: str = "open"  # open | closed | timed_out
    thumbnail: Optional[str] = None

    @property
    def dwell_sec(self) -> Optional[float]:
        return None if self.exit_ts is None else self.exit_ts - self.entry_ts

    def to_dict(self) -> dict:
        d = {k: v for k, v in asdict(self).items() if k != "embedding"}
        d["dwell_sec"] = self.dwell_sec
        return d


@dataclass
class ExitOutcome:
    matched: bool
    unmatched: bool
    needs_review: bool
    confidence: float
    session: Optional[VisitSession]


def cosine_gallery_classifier(gallery_embeddings, threshold: float = 0.7) -> EmployeeClassifier:
    """Employee-first check: is this entry embedding close to any enrolled employee embedding?"""
    gallery = [l2_normalize(np.asarray(e, float)) for e in gallery_embeddings]

    def classify(emb):
        if not gallery:
            return (False, 0.0)
        best = max(cosine(emb, g) for g in gallery)
        return (best >= threshold, float(best))

    return classify


class SessionManager:
    def __init__(
        self,
        window_sec: float = 7200,
        match_threshold: float = 0.80,
        review_threshold: float = 0.60,
        employee_classifier: Optional[EmployeeClassifier] = None,
    ):
        self.window_sec = window_sec
        self.match_threshold = match_threshold
        self.review_threshold = review_threshold
        self.employee_classifier = employee_classifier
        self.open_sessions: list[VisitSession] = []
        self.closed_sessions: list[VisitSession] = []
        self.unmatched_exits: list[dict] = []
        self._next_id = 1
        self._max_occupancy = 0

    def on_entry(self, embedding, ts, ref=None, thumbnail=None, is_employee=None, attrs=None) -> VisitSession:
        emb = l2_normalize(np.asarray(embedding, float))
        emp, score, emp_id = False, 0.0, None
        if is_employee is not None:  # explicit override (e.g. from HITL label)
            emp = bool(is_employee)
        elif self.employee_classifier is not None:  # employee-first gallery check
            res = self.employee_classifier(emb)
            emp, score = bool(res[0]), float(res[1])
            emp_id = res[2] if len(res) > 2 else None
        s = VisitSession(
            session_id=self._next_id,
            entry_ts=float(ts),
            embedding=emb,
            is_employee=emp,
            employee_score=score,
            employee_id=emp_id,
            entry_ref=ref or {},
            attrs=attrs or {},
            thumbnail=thumbnail,
        )
        self._next_id += 1
        self.open_sessions.append(s)
        self._max_occupancy = max(self._max_occupancy, len(self.open_sessions))
        return s

    def on_exit(self, embedding, ts, ref=None) -> ExitOutcome:
        ts = float(ts)
        self.expire(ts)
        emb = l2_normalize(np.asarray(embedding, float))
        best, best_sim = None, -1.0
        for s in self.open_sessions:
            if 0 <= ts - s.entry_ts <= self.window_sec:
                sim = cosine(emb, s.embedding)
                if sim > best_sim:
                    best_sim, best = sim, s

        if best is not None and best_sim >= self.review_threshold:
            best.exit_ts = ts
            best.exit_ref = ref or {}
            best.match_confidence = best_sim
            best.matched = True
            best.needs_review = best_sim < self.match_threshold
            best.status = "closed"
            self.open_sessions.remove(best)
            self.closed_sessions.append(best)
            return ExitOutcome(True, False, best.needs_review, best_sim, best)

        self.unmatched_exits.append({"ts": ts, "ref": ref or {}, "best_sim": max(best_sim, 0.0)})
        return ExitOutcome(False, True, True, max(best_sim, 0.0), None)

    def expire(self, now) -> None:
        keep = []
        for s in self.open_sessions:
            if now - s.entry_ts > self.window_sec:
                s.status = "timed_out"
                self.closed_sessions.append(s)
            else:
                keep.append(s)
        self.open_sessions = keep

    def occupancy(self) -> int:
        return len(self.open_sessions)

    @property
    def max_occupancy(self) -> int:
        return self._max_occupancy

    def summary(self) -> dict:
        matched = [s for s in self.closed_sessions if s.matched]
        customers = [s for s in matched if not s.is_employee]
        employees = [s for s in matched if s.is_employee]
        cust_dwell = [s.dwell_sec for s in customers]
        timed_out = [s for s in self.closed_sessions if s.status == "timed_out"]
        return {
            "customer_visits": len(customers),
            "employee_sessions": len(employees),
            "mean_customer_dwell_sec": round(statistics.mean(cust_dwell), 1) if cust_dwell else None,
            "median_customer_dwell_sec": round(statistics.median(cust_dwell), 1) if cust_dwell else None,
            "max_occupancy": self._max_occupancy,
            "needs_review": len([s for s in matched if s.needs_review]),
            "unmatched_exits": len(self.unmatched_exits),
            "still_inside_or_timed_out": len(self.open_sessions) + len(timed_out),
        }
