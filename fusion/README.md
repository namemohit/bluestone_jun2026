# fusion/ — Cross-camera multi-target tracking (MTMC) — Phase 1

Associates per-camera tracks into **one global ID per physical person** so the same visitor seen by
multiple cameras (e.g. entry + a common cam) is not double-counted.

Planned modules:
- `reid.py` — body appearance embeddings (OSNet / CLIP-ReID via boxmot).
- `gallery.py` — time-indexed embedding gallery with the windowed matching described in the plan
  (coarse spatio-temporal/topology gate → ReID similarity → face verify → constrained assignment →
  hysteresis, mid-confidence routed to HITL).
- `topology.py` — camera adjacency + plausible transition times (config-driven per camera role).

See the "Identity matching & temporal gating" section of the plan for the design.
