"""Export the human allocations (the `annotations` table) into a portable training dataset you own.

Each closed-day allocation is a labelled person crop. This rolls them into a self-contained folder so
the data accrues independently of the dashboard and is ready for future model work:
  - reid/id<eid>_<name>/    : identity-grouped staff crops -> ReID / TransReID fine-tuning
  - person/ , not_person/   : crops split by "is this a person?" -> detector hard-negatives
  - manifest.csv / .json    : every annotation (crop, window, camera, track, category, employee_id)
  - embeddings.jsonl        : the cached 512-d OSNet vector per crop (skip re-embedding)

  python -m training.export_dataset --date 2026-06-03           # one day
  python -m training.export_dataset                              # everything

Note: only person CROPS are kept by L1 (not full frames), so this directly serves ReID/identity work;
full detector (YOLO) bbox training would need frames — re-extract from the on-disk day clips using
tracks.json, or add an L1 --save-frames pass (future).
"""
from __future__ import annotations

import argparse
import csv
import json
import re
import shutil
from collections import Counter
from pathlib import Path

from hitl.store_supabase import SupabaseStore


def _safe(s: str) -> str:
    return re.sub(r"[^A-Za-z0-9_]+", "_", str(s)).strip("_")


def main() -> None:
    ap = argparse.ArgumentParser(description="Export human annotations -> portable training dataset")
    ap.add_argument("--date", default=None, help="limit to one date's windows (YYYY-MM-DD); else all")
    ap.add_argument("--out", default="training_data")
    args = ap.parse_args()

    s = SupabaseStore()
    with s._cx() as cx, cx.cursor() as cur:
        base = ("select window_id,camera,track,crop_url,category,employee_id,embedding "
                "from latest_annotations")
        if args.date:
            cur.execute(base + " where window_id like %s", (f"{args.date}_%",))
        else:
            cur.execute(base)
        anns = [dict(r) for r in cur.fetchall()]
    emps = {e["id"]: (e.get("name") or e.get("code")) for e in s.list_employees()}

    out = Path(args.out) / (args.date or "all")
    for sub in ("reid", "person", "not_person"):
        (out / sub).mkdir(parents=True, exist_ok=True)

    rows, emb_rows, placed = [], [], 0
    for an in anns:
        crop, cat = an.get("crop_url"), an["category"]
        name = f"{an['window_id']}_{an['camera']}_{an['track']}.jpg"
        cp = Path(crop) if crop else None
        if cp and cp.exists():
            if cat == "staff" and an.get("employee_id"):
                d = out / "reid" / f"id{an['employee_id']}_{_safe(emps.get(an['employee_id'], ''))}"
                d.mkdir(parents=True, exist_ok=True)
                shutil.copy2(cp, d / name)
            shutil.copy2(cp, out / ("not_person" if cat == "not_person" else "person") / name)
            placed += 1
        rows.append({"crop": crop, "window": an["window_id"], "camera": an["camera"],
                     "track": an["track"], "category": cat, "employee_id": an.get("employee_id")})
        if an.get("embedding"):
            emb_rows.append({"crop": crop, "category": cat,
                             "employee_id": an.get("employee_id"), "embedding": an["embedding"]})

    with open(out / "manifest.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["crop", "window", "camera", "track", "category", "employee_id"])
        w.writeheader()
        w.writerows(rows)
    by_cat = dict(Counter(r["category"] for r in rows))
    (out / "manifest.json").write_text(json.dumps(
        {"date": args.date, "total": len(rows), "crops_placed": placed, "by_category": by_cat},
        indent=2), encoding="utf-8")
    with open(out / "embeddings.jsonl", "w", encoding="utf-8") as f:
        for r in emb_rows:
            f.write(json.dumps(r) + "\n")

    print(f"exported {len(rows)} annotations ({placed} crops copied) -> {out}")
    print(f"  by category: {by_cat}")
    print(f"  reid identities: {sorted(p.name for p in (out / 'reid').glob('id*'))}")
    print(f"  embeddings captured: {len(emb_rows)}")


if __name__ == "__main__":
    main()
