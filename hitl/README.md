# hitl/ — Human-in-the-loop review + active learning — Phase 4

The "data engine" that drives self-improvement toward 95%+:
- Daily reconciliation dashboard (counts, groups, demographics, employee timesheets, video scrub).
- Review queue of low-confidence / conflicting decisions (mid-confidence ReID matches, ID switches,
  borderline employee-vs-visitor, group merge/split, uncertain age/gender).
- Reviewer corrections saved as ground truth → feed retraining (`training/`).
- Active-learning sampling: uncertainty + diversity + hard-example mining.
