"""Run the self-improving HITL loop over dummy data.

  python run_train.py

Shows the cold-start day scoring poorly, the human correction, the retrain, the eval-gate
decision, and the improved score.
"""
from __future__ import annotations

from hitl.loop import run_improvement_cycle


def main() -> None:
    result = run_improvement_cycle(seed=0, target=0.95, rounds=1)
    print("================ SELF-IMPROVING LOOP ================")
    for h in result["history"]:
        line = (f"round {h['round']}:  score={h['score']:.2f}  "
                f"promoted={h['promoted']}  human_labels={h['labels']}")
        print(line)
        print(f"            checks: {h['checks']}")
        if "reason" in h:
            print(f"            gate: {h['reason']}")
    print("-----------------------------------------------------")
    print(f"active model version: {result['active_version']}  |  "
          f"final score: {result['final_score']:.2f}  |  versions: {result['n_versions']}")


if __name__ == "__main__":
    main()
