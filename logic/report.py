"""Assemble the day report — the four outputs split into Customer / Employee tabs."""
from __future__ import annotations

import statistics


def build_report(mgr, groups, demographics_agg, timesheets) -> dict:
    customers = [s for s in mgr.closed_sessions if s.matched and not s.is_employee]
    dwell = [s.dwell_sec for s in customers if s.dwell_sec is not None]
    summ = mgr.summary()
    return {
        "customers": {
            "unique_customers": len(customers),
            "groups": {
                "count": len(groups),
                "sizes": sorted((g["size"] for g in groups), reverse=True),
            },
            "demographics": demographics_agg,
            "dwell_sec": {
                "mean": round(statistics.mean(dwell), 1) if dwell else None,
                "median": round(statistics.median(dwell), 1) if dwell else None,
                "min": min(dwell) if dwell else None,
                "max": max(dwell) if dwell else None,
            },
        },
        "employees": {
            "headcount": len(timesheets),
            "timesheets": timesheets,
        },
        "review_queue": {
            "needs_review": summ["needs_review"],
            "unmatched_exits": summ["unmatched_exits"],
        },
    }
