# training/ — Retraining + eval gate — Phase 4

- Fine-tune detector / ReID / demographic heads on accumulated HITL-corrected data.
- Dataset + experiment versioning (DVC + MLflow/W&B).
- **Eval gate**: a frozen, human-labeled golden benchmark set. A new model is promoted only if it
  beats the current one AND clears targets. Shadow/canary before full rollout.
- Drift monitoring triggers retraining on metric degradation.
