"""Accuracy harness: compare pipeline counts vs hand-counted ground truth.

Ground-truth file (JSON):
{
  "name": "cam_entry_2026-06-07_morning",
  "segments": [
    {"id": "full", "start_ts": null, "end_ts": null, "truth": {"in": 30, "out": 28}}
  ]
}
start_ts/end_ts are epoch seconds (null = unbounded). Use multiple segments to localize error.

Usage:
  python -m evaluation.harness --truth data/ground_truth/sample.json \
                               --events outputs/synthetic_demo.events.json [--gate 5.0]
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def load_json(path: str) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def sum_events(events: list[dict], start_ts, end_ts) -> dict:
    counts = {"in": 0, "out": 0}
    for e in events:
        ts = e.get("ts")
        if start_ts is not None and ts < start_ts:
            continue
        if end_ts is not None and ts > end_ts:
            continue
        d = e.get("direction")
        if d in counts:
            counts[d] += 1
    return counts


def _pct_err(pred: int, truth: int):
    if truth == 0:
        return None  # undefined; reported as abs error only
    return abs(pred - truth) / truth * 100.0


def evaluate(truth_doc: dict, events_doc: dict, gate: float = 5.0) -> dict:
    events = events_doc.get("events", [])
    seg_reports = []
    pct_errors = []
    for seg in truth_doc.get("segments", []):
        pred = sum_events(events, seg.get("start_ts"), seg.get("end_ts"))
        truth = seg.get("truth", {})
        metrics = {}
        for d in ("in", "out"):
            t = int(truth.get(d, 0))
            p = int(pred.get(d, 0))
            pe = _pct_err(p, t)
            if pe is not None:
                pct_errors.append(pe)
            metrics[d] = {"pred": p, "truth": t, "abs_err": abs(p - t), "pct_err": pe}
        seg_reports.append({"id": seg.get("id", "?"), "metrics": metrics})

    mape = sum(pct_errors) / len(pct_errors) if pct_errors else None
    return {
        "name": truth_doc.get("name", "?"),
        "segments": seg_reports,
        "mape": mape,
        "gate": gate,
        "passed": (mape is not None and mape <= gate),
    }


def format_report(report: dict) -> str:
    lines = [f"Accuracy report: {report['name']}", "-" * 60]
    for seg in report["segments"]:
        lines.append(f"segment [{seg['id']}]")
        for d, m in seg["metrics"].items():
            pe = "n/a" if m["pct_err"] is None else f"{m['pct_err']:.1f}%"
            lines.append(
                f"   {d:>3}: pred={m['pred']:>4}  truth={m['truth']:>4}  "
                f"abs_err={m['abs_err']:>3}  pct_err={pe}"
            )
    lines.append("-" * 60)
    mape = report["mape"]
    mape_s = "n/a" if mape is None else f"{mape:.2f}%"
    verdict = "PASS" if report["passed"] else "FAIL"
    lines.append(f"MAPE={mape_s}  gate<={report['gate']:.1f}%  ->  {verdict}")
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser(description="Count-accuracy harness (MAPE vs ground truth)")
    ap.add_argument("--truth", required=True, help="ground-truth JSON")
    ap.add_argument("--events", required=True, help="pipeline events JSON")
    ap.add_argument("--gate", type=float, default=5.0, help="max MAPE %% to pass (default 5)")
    args = ap.parse_args()

    report = evaluate(load_json(args.truth), load_json(args.events), gate=args.gate)
    print(format_report(report))
    raise SystemExit(0 if report["passed"] else 1)


if __name__ == "__main__":
    main()
