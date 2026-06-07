"""Cross-camera fusion (MTMC) — co-temporal clustering into global appearance IDs.

A person crossing the entrance is seen by the entry cam AND (overlapping) common cam at
~the same instant. This merges those simultaneous cross-camera sightings into ONE
`Appearance` (a global id) so we don't double-count. It does NOT link entry to exit — that's
the session matcher's job (a much longer time window).

Rules:
- Two tracklets merge only if: |Δt| ≤ time_window, DIFFERENT cameras, and cosine ≥ sim_threshold.
- A camera sees a person as one tracklet, so same-camera tracklets never merge into one appearance.
- The fused embedding is the mean of members; crossing direction is inherited from any member.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from fusion.tracklet import cosine, l2_normalize


@dataclass
class Tracklet:
    camera_id: str
    role: str  # entry | exit | common
    local_id: int
    ts: float
    embedding: np.ndarray
    crossing: Optional[str] = None  # "in" | "out" | None
    attrs: dict = field(default_factory=dict)  # carried demographics / metadata
    identity: Optional[int] = None  # ground-truth identity (synthetic/eval only)


@dataclass
class Appearance:
    global_id: int
    ts: float
    embedding: np.ndarray
    members: list  # [(camera_id, local_id), ...]
    crossing: Optional[str]
    roles: list
    attrs: dict
    identities: list  # ground-truth identities of members (synthetic/eval only)

    @property
    def n_cameras(self) -> int:
        return len({m[0] for m in self.members})


def fuse(tracklets, time_window: float = 5.0, sim_threshold: float = 0.6) -> list[Appearance]:
    """Greedy co-temporal cross-camera clustering → list of Appearances (global ids)."""
    items = sorted(tracklets, key=lambda t: (t.ts, t.camera_id, t.local_id))
    clusters: list[dict] = []

    for t in items:
        placed = False
        for c in clusters:
            if (
                abs(t.ts - c["ts"]) <= time_window
                and t.camera_id not in c["cameras"]
                and cosine(t.embedding, c["proto"]) >= sim_threshold
            ):
                c["embs"].append(np.asarray(t.embedding, float))
                c["proto"] = l2_normalize(np.mean(c["embs"], axis=0))
                c["members"].append((t.camera_id, t.local_id))
                c["cameras"].add(t.camera_id)
                c["roles"].append(t.role)
                c["ts"] = min(c["ts"], t.ts)
                if t.crossing and not c["crossing"]:
                    c["crossing"] = t.crossing
                if t.attrs:
                    c["attrs"].update(t.attrs)
                if t.identity is not None:
                    c["identities"].append(t.identity)
                placed = True
                break
        if not placed:
            clusters.append(
                {
                    "ts": t.ts,
                    "embs": [np.asarray(t.embedding, float)],
                    "proto": l2_normalize(t.embedding),
                    "members": [(t.camera_id, t.local_id)],
                    "cameras": {t.camera_id},
                    "roles": [t.role],
                    "crossing": t.crossing,
                    "attrs": dict(t.attrs),
                    "identities": [t.identity] if t.identity is not None else [],
                }
            )

    appearances = []
    for i, c in enumerate(clusters, start=1):
        appearances.append(
            Appearance(
                global_id=i,
                ts=c["ts"],
                embedding=c["proto"],
                members=c["members"],
                crossing=c["crossing"],
                roles=c["roles"],
                attrs=c["attrs"],
                identities=c["identities"],
            )
        )
    appearances.sort(key=lambda a: a.ts)
    return appearances
