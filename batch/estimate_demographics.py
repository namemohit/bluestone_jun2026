"""Estimate age + gender for a day's CUSTOMER crops and cache them (face-based, an ESTIMATE only).

Runs InsightFace buffalo_l (face-detect + genderage) once per displayed customer crop and writes a
path-keyed cache the dashboard reads at render time. No L4 re-run, no pipeline change. ~half of the
full-body CCTV crops have a usable face; the rest cache as face=false and show no label.

  python -m batch.estimate_demographics --date 2026-06-03            # CPU (default)
  python -m batch.estimate_demographics --date 2026-06-03 --gpu      # needs onnxruntime-gpu

The dashboard server must be running (this reads the displayed crops from its API).
"""
from __future__ import annotations

import argparse
import json
import os
import pathlib
import urllib.request

from vision.demographics import build_demographics

CACHE = pathlib.Path("outputs/demographics_cache.json")


def _get(base: str, path: str):
    with urllib.request.urlopen(base + path, timeout=120) as r:
        return json.load(r)


def main() -> None:
    ap = argparse.ArgumentParser(description="Cache age/gender estimates for a day's customer crops")
    ap.add_argument("--date", required=True)
    ap.add_argument("--base", default="http://127.0.0.1:8000")
    ap.add_argument("--gpu", action="store_true", help="use CUDA (needs onnxruntime-gpu)")
    ap.add_argument("--staff", action="store_true", help="also estimate the staff bucket crops")
    args = ap.parse_args()

    crops: set[str] = set()
    d = _get(args.base, f"/api/hitl/detections-day/{args.date}")
    buckets = ["customer"] + (["staff"] if args.staff else [])
    for b in buckets:
        for x in d.get("buckets", {}).get(b, []):
            if x.get("crop"):
                crops.add(x["crop"])
    rep = _get(args.base, f"/api/hitl/report/{args.date}")
    for g in rep.get("customer_groups", []):
        for m in g.get("members", []):
            if m.get("crop"):
                crops.add(m["crop"])
    for h in rep.get("hot_leads", []):
        if h.get("crop"):
            crops.add(h["crop"])

    cache = json.loads(CACHE.read_text(encoding="utf-8")) if CACHE.exists() else {}
    todo = sorted(c for c in crops if c not in cache)
    print(f"{len(crops)} customer crops on {args.date}; {len(todo)} to estimate "
          f"({len(crops) - len(todo)} already cached)")
    if todo:
        est = build_demographics({"backend": "insightface", "gpu": args.gpu})
        for i, c in enumerate(todo):
            cache[c] = est.estimate(os.path.normpath(c))
            if (i + 1) % 15 == 0 or i + 1 == len(todo):
                print(f"  {i + 1}/{len(todo)}")
        CACHE.write_text(json.dumps(cache), encoding="utf-8")
    faces = sum(1 for c in crops if cache.get(c, {}).get("face"))
    print(f"done -> {CACHE}  |  faces found: {faces}/{len(crops)} ({round(100*faces/max(1,len(crops)))}%)")


if __name__ == "__main__":
    main()
