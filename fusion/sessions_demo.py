"""GPU-free demo of the entry<->exit session + dwell matcher.

  python -m fusion.sessions_demo

Runs the synthetic scenario through SessionManager and prints the customer/employee split,
dwell times vs expected, max occupancy, and flagged (review/unmatched) cases.
"""
from __future__ import annotations

import json

from fusion.sessions import SessionManager, cosine_gallery_classifier
from fusion.synthetic_visits import make_scenario


def run(seed: int = 0) -> SessionManager:
    events, expected, gallery = make_scenario(seed=seed)
    mgr = SessionManager(
        window_sec=7200,
        match_threshold=0.80,
        review_threshold=0.60,
        employee_classifier=cosine_gallery_classifier(gallery, threshold=0.70),
    )
    last_ts = 0.0
    for e in events:
        last_ts = max(last_ts, e["ts"])
        if e["kind"] == "entry":
            mgr.on_entry(e["emb"], e["ts"], ref={**e["ref"], "identity": e["identity"]})
        else:
            mgr.on_exit(e["emb"], e["ts"], ref=e["ref"])
    mgr.expire(last_ts + 1)

    print("=== Summary (customers vs employees) ===")
    print(json.dumps(mgr.summary(), indent=2))
    print("\n=== Matched sessions: dwell vs expected ===")
    for s in mgr.closed_sessions:
        if not s.matched:
            continue
        ident = s.entry_ref.get("identity")
        exp = expected.get(ident)
        exp_s = f"{exp:.0f}s" if exp is not None else "n/a"
        tag = "EMP " if s.is_employee else "cust"
        flag = "  [NEEDS REVIEW]" if s.needs_review else ""
        print(f"  sess {s.session_id:>2} {tag} id={ident:>4}  dwell={s.dwell_sec:>5.0f}s  "
              f"exp={exp_s:>6}  conf={s.match_confidence:.2f}{flag}")

    summ = mgr.summary()
    print(f"\nunmatched exits: {summ['unmatched_exits']}  |  "
          f"still inside/timed-out: {summ['still_inside_or_timed_out']}")
    return mgr


def main() -> None:
    run()


if __name__ == "__main__":
    main()
