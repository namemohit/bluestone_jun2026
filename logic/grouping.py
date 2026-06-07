"""Group detection — cluster customers who arrived together.

Phase-1 heuristic: window from each group's first arrival; sessions whose entry is within
`group_gap_sec` of that first arrival form a group (singletons allowed). Arrival-time proximity
is the dominant signal for sparse retail traffic; richer features (co-movement, joint exit) and
an HITL-trained classifier come later. Employees are excluded.
"""
from __future__ import annotations


def group_sessions(sessions, group_gap_sec: float = 15.0):
    """Return (mapping {session_id: group_id}, groups [{group_id,size,session_ids}])."""
    customers = sorted(
        [s for s in sessions if not s.is_employee],
        key=lambda s: s.entry_ts,
    )
    groups: list[list] = []
    i = 0
    while i < len(customers):
        start = customers[i]
        members = [start]
        j = i + 1
        while j < len(customers) and customers[j].entry_ts - start.entry_ts <= group_gap_sec:
            members.append(customers[j])
            j += 1
        groups.append(members)
        i = j

    mapping: dict[int, int] = {}
    out = []
    for gid, members in enumerate(groups, start=1):
        for s in members:
            mapping[s.session_id] = gid
        out.append({
            "group_id": gid,
            "size": len(members),
            "session_ids": [s.session_id for s in members],
        })
    return mapping, out
