"""Deterministic synthetic entry/exit events for the session matcher (GPU-free).

Each "person" has a latent identity unit-vector. Entry embedding = the identity vector;
exit embedding = a vector at a CONTROLLED cosine to it (so tests are exact, not noisy).
Includes: normal matched customers, an enrolled employee, a lookalike (-> needs_review),
an unmatched exit, and a still-inside customer.
"""
from __future__ import annotations

import numpy as np

from fusion.tracklet import l2_normalize


def identity_vector(identity: int, dim: int = 128):
    """Stable pseudo-random unit vector for an identity (deterministic via seed)."""
    rng = np.random.default_rng(identity)
    return l2_normalize(rng.standard_normal(dim))


def at_cosine(idv, target_cos, rng):
    """Return a unit vector whose cosine to `idv` is exactly `target_cos`."""
    d = idv.shape[0]
    w = rng.standard_normal(d)
    w = w - np.dot(w, idv) * idv  # component orthogonal to idv
    u = l2_normalize(w)
    target_cos = float(np.clip(target_cos, -1.0, 1.0))
    return l2_normalize(target_cos * idv + np.sqrt(max(0.0, 1.0 - target_cos**2)) * u)


def make_scenario(n_customers: int = 6, dim: int = 128, base_ts: float = 0.0, seed: int = 0,
                  exit_cos: float = 0.98):
    """Return (events, expected_dwell, employee_gallery).

    events: time-sorted list of {kind: 'entry'|'exit', ts, emb, identity, ref}
    expected_dwell: {identity: dwell_sec} for sessions that should match
    employee_gallery: list of enrolled employee embeddings
    """
    rng = np.random.default_rng(seed)
    events: list[dict] = []
    expected: dict[int, float] = {}

    # normal customers — should all match cleanly
    for i in range(1, n_customers + 1):
        idv = identity_vector(i, dim)
        entry = base_ts + i * 60.0
        dwell = 300.0 + i * 120.0
        expected[i] = dwell
        events.append({"kind": "entry", "ts": entry, "emb": idv, "identity": i,
                       "ref": {"cam": "entry", "track": i}})
        events.append({"kind": "exit", "ts": entry + dwell, "emb": at_cosine(idv, exit_cos, rng),
                       "identity": i, "ref": {"cam": "exit", "track": 1000 + i}})

    # enrolled employee — should be tagged employee, excluded from customer count
    emp_id = 50
    emp_vec = identity_vector(emp_id, dim)
    employee_gallery = [emp_vec]
    e_entry, e_dwell = base_ts + 120.0, 600.0
    expected[emp_id] = e_dwell
    events.append({"kind": "entry", "ts": e_entry, "emb": emp_vec, "identity": emp_id,
                   "ref": {"cam": "entry", "track": 90}})
    events.append({"kind": "exit", "ts": e_entry + e_dwell, "emb": at_cosine(emp_vec, exit_cos, rng),
                   "identity": emp_id, "ref": {"cam": "exit", "track": 1090}})

    # lookalike — exit only 0.70 similar to entry -> matched but needs_review
    look_id = 7
    lookv = identity_vector(look_id, dim)
    l_entry, l_dwell = base_ts + 400.0, 500.0
    expected[look_id] = l_dwell
    events.append({"kind": "entry", "ts": l_entry, "emb": lookv, "identity": look_id,
                   "ref": {"cam": "entry", "track": 7}})
    events.append({"kind": "exit", "ts": l_entry + l_dwell, "emb": at_cosine(lookv, 0.70, rng),
                   "identity": look_id, "ref": {"cam": "exit", "track": 1007}})

    # unmatched exit — random embedding, matches nothing in window
    events.append({"kind": "exit", "ts": base_ts + 800.0, "emb": identity_vector(9999, dim),
                   "identity": -1, "ref": {"cam": "exit", "track": 1999}})

    # still-inside customer — entry with no exit
    si_id = 8
    events.append({"kind": "entry", "ts": base_ts + 900.0, "emb": identity_vector(si_id, dim),
                   "identity": si_id, "ref": {"cam": "entry", "track": 8}})

    events.sort(key=lambda e: e["ts"])
    return events, expected, employee_gallery
