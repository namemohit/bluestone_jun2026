"""A simulated day of dummy data for the full batch pipeline.

Generates per-camera Tracklets (entry/exit/common) for customers (some arriving in groups) and
employees (multiple in/out cycles), each with a latent identity + demographics, plus the
ground truth to assert against. No models/GPU — embeddings are controlled-cosine identity vectors.
"""
from __future__ import annotations

import numpy as np

from fusion.mtmc import Tracklet
from fusion.synthetic_visits import at_cosine, identity_vector

# id, arrival_s, dwell_s, gender, age_bucket   (arrivals chosen so groups are obvious)
CUSTOMERS = [
    (1, 100, 600, "female", "20-34"),  # group A
    (2, 104, 620, "male", "35-54"),    # group A
    (3, 300, 500, "female", "13-19"),  # group B
    (4, 305, 520, "male", "20-34"),    # group B
    (5, 309, 540, "female", "20-34"),  # group B
    (6, 1000, 800, "male", "55+"),     # solo
    (7, 2000, 300, "female", "0-12"),  # solo
]

# employee_id -> [(t_in, t_out), ...]
EMPLOYEES = {
    50: [(200, 500), (1500, 1800)],
    51: [(250, 450), (1600, 1900)],
}

# Customers who arrived together (by identity) — grouping ground truth.
CUSTOMER_GROUPS = [[1, 2], [3, 4, 5], [6], [7]]

# A deliberate cold-start demographics error (a model that's wrong until corrected by a human).
COLD_DEMO_BIAS = {5: {"gender": "male"}, 7: {"age_bucket": "20-34"}}


def _event(tracklets, counters, rng, identity, ts, crossing, role_cam, attrs=None):
    """One crossing event = a role-cam (entry/exit) tracklet + a common-cam corroboration."""
    idv = identity_vector(identity)
    counters[role_cam] = counters.get(role_cam, 0) + 1
    tracklets.append(Tracklet(role_cam, role_cam, counters[role_cam], ts,
                              at_cosine(idv, 0.97, rng), crossing=crossing,
                              attrs=attrs or {}, identity=identity))
    counters["common"] = counters.get("common", 0) + 1
    tracklets.append(Tracklet("common", "common", counters["common"], ts + 1.0,
                              at_cosine(idv, 0.95, rng), crossing=None,
                              attrs=attrs or {}, identity=identity))


def make_day(seed: int = 0, base_ts: float = 0.0):
    """Return (tracklets, ground_truth, employee_gallery_embeddings)."""
    rng = np.random.default_rng(seed)
    tracklets: list[Tracklet] = []
    counters: dict[str, int] = {}

    for cid, arrival, dwell, gender, age in CUSTOMERS:
        attrs = {"gender": gender, "age_bucket": age}
        _event(tracklets, counters, rng, cid, base_ts + arrival, "in", "entry", attrs)
        _event(tracklets, counters, rng, cid, base_ts + arrival + dwell, "out", "exit", attrs)

    gallery = {}
    for eid, cycles in EMPLOYEES.items():
        gallery[eid] = identity_vector(eid)  # clean enrollment vector
        for (t_in, t_out) in cycles:
            _event(tracklets, counters, rng, eid, base_ts + t_in, "in", "entry")
            _event(tracklets, counters, rng, eid, base_ts + t_out, "out", "exit")

    ground_truth = {
        "unique_customers": len(CUSTOMERS),
        "n_groups": 4,
        "group_sizes": [3, 2, 1, 1],
        "n_employees": len(EMPLOYEES),
        "employee_sessions": {eid: len(c) for eid, c in EMPLOYEES.items()},
        "gender": {"female": 4, "male": 3},
        "age_bucket": {"20-34": 3, "35-54": 1, "13-19": 1, "55+": 1, "0-12": 1},
        "groups_by_identity": CUSTOMER_GROUPS,
        "customer_arrivals": {cid: base_ts + arrival for (cid, arrival, *_rest) in CUSTOMERS},
    }
    return tracklets, ground_truth, gallery
