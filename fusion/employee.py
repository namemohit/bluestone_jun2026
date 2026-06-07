"""Employee gallery (enroll/match) + timesheet aggregation.

The gallery is checked FIRST at entry (face-anchored in production; body embeddings here).
`classifier()` returns a 3-tuple (is_employee, score, employee_id) consumed by SessionManager.
`build_timesheets` turns matched employee sessions into per-employee in/out + dwell.
"""
from __future__ import annotations

from collections import defaultdict
from typing import Optional

import numpy as np

from fusion.tracklet import cosine, l2_normalize


class EmployeeGallery:
    def __init__(self, threshold: float = 0.70):
        self.threshold = threshold
        self._emb: dict[int, list] = {}  # employee_id -> [embeddings]

    def enroll(self, employee_id: int, embedding) -> None:
        self._emb.setdefault(employee_id, []).append(l2_normalize(np.asarray(embedding, float)))

    def match(self, embedding) -> "tuple[Optional[int], float]":
        emb = l2_normalize(np.asarray(embedding, float))
        best_id, best_sim = None, -1.0
        for eid, embs in self._emb.items():
            for e in embs:
                sim = cosine(emb, e)
                if sim > best_sim:
                    best_sim, best_id = sim, eid
        if best_id is not None and best_sim >= self.threshold:
            return best_id, float(best_sim)
        return None, float(max(best_sim, 0.0))

    def classifier(self):
        def classify(emb):
            eid, sim = self.match(emb)
            return (eid is not None, sim, eid)

        return classify


def build_timesheets(sessions, day_start: Optional[float] = None, day_end: Optional[float] = None) -> dict:
    """Per-employee timesheet from matched employee sessions.

    time_inside = sum of session dwells; time_outside = operating span − inside (if span given).
    """
    by_emp: dict[int, list] = defaultdict(list)
    for s in sessions:
        if s.is_employee and s.matched and s.employee_id is not None:
            by_emp[s.employee_id].append(s)

    sheets = {}
    for eid, ss in by_emp.items():
        ss.sort(key=lambda s: s.entry_ts)
        inside = sum(s.dwell_sec for s in ss if s.dwell_sec)
        events = [{"in": s.entry_ts, "out": s.exit_ts, "dwell_sec": s.dwell_sec} for s in ss]
        sheet = {
            "employee_id": eid,
            "sessions": len(ss),
            "time_inside_sec": inside,
            "in_out": events,
        }
        if day_start is not None and day_end is not None:
            sheet["time_outside_sec"] = max(0.0, (day_end - day_start) - inside)
        sheets[eid] = sheet
    return sheets
