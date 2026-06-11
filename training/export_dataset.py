"""Export the human allocations (the `annotations` table) into a portable training dataset you own.

Each closed-day allocation is a labelled person crop. This rolls them into a self-contained folder so
the data accrues independently of the dashboard and is ready for future model work — usable by ANY
future ReID backbone (CLIP-ReID, TransReID, …), not just OSNet:
  - reid/id<eid>_<name>/    : identity-grouped staff crops -> ReID / TransReID fine-tuning
  - pairs/ + pairs.jsonl    : human-confirmed ✓same / ✗different crop PAIRS (the ReID trainer's effort),
                              copied in self-contained, model-agnostic (no embeddings — re-embed the JPEGs)
  - person/ , not_person/   : crops split by "is this a person?" -> detector hard-negatives
  - manifest.csv / .json    : every annotation (crop, window, camera, track, category, employee_id)
  - embeddings.jsonl        : the cached 512-d OSNet vector per crop (OSNet-only/OPTIONAL; a new model ignores it)
  - DATASET.md              : the layout + how a new backbone consumes it

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


_DATASET_MD = """# ReID training dataset ({date})

Self-contained, **model-agnostic** snapshot of the human labelling effort. Every label references real
crop JPEGs copied into this folder — NO model embeddings are required. A future ReID backbone (CLIP-ReID,
TransReID, a new OSNet, …) trains from this directly: load the JPEGs, embed with the new model, train.

## Layout
- `reid/id<employee_id>_<name>/*.jpg` — identity-labelled staff crops. Crops in the same folder are the
  SAME person (positives); crops in different folders are different people (negatives).
- `pairs/*.jpg` + `pairs.jsonl` — human-confirmed look-alike verdicts. One JSON object per line:
  `{{"label": "same"|"different", "crop_a": "pairs/..jpg", "crop_b": "pairs/..jpg",
     "window_a","track_a","window_b","track_b","reviewer","reason"}}`
  `same` = a person reviewed two crops and confirmed they are the SAME individual; `different` = confirmed
  DIFFERENT. Use for pairwise / contrastive / triplet objectives. Revoked verdicts are already excluded.
- `person/`, `not_person/` — crops split by "is this a person?" (detector hard-negatives).
- `manifest.csv` / `manifest.json` — every annotation (crop, window, camera, track, category, employee_id).
- `embeddings.jsonl` — **OSNet-only, OPTIONAL**. Cached 512-d OSNet vectors. A different backbone IGNORES
  this file and re-embeds the JPEGs.

## Contents
- annotations: {total} ({placed} crops copied) — by category: {by_cat}
- reid identities: {ids}
- human pairs: {pairs}  ({same} same / {diff} different)
- cached OSNet embeddings: {embeds}

## Reuse for a new backbone (no re-labelling)
1. Identity training: read `reid/id*/` folders -> (crop, identity) samples.
2. Pairwise training: read `pairs.jsonl` -> (crop_a, crop_b, same/different) -> contrastive/triplet loss.
3. Re-embed all crops with the new model; ignore `embeddings.jsonl`.
The labels never embed a model, so this dataset stays valid across model generations.
"""


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

    # ---- human-confirmed ReID PAIRS (✓ same / ✗ different): self-contained, model-agnostic ----------
    from training.reid_labels import human_pairs
    pdir = out / "pairs"
    pdir.mkdir(parents=True, exist_ok=True)

    def _copy_named(src, window, track):                       # copy the JPEG into pairs/, dedup by (window,track)
        sp = Path(str(src or "").replace("\\", "/"))
        if not src or not sp.exists():
            return None
        nm = _safe(f"{window}_{track}") + ".jpg"
        if not (pdir / nm).exists():
            shutil.copy2(sp, pdir / nm)
        return f"pairs/{nm}"

    pair_rows, pair_miss = [], 0
    for p in human_pairs(s, args.date):
        na = _copy_named(p["crop_a"], p["window_a"], p["track_a"])
        nb = _copy_named(p["crop_b"], p["window_b"], p["track_b"])
        if not na or not nb:                                   # crop not on disk locally -> can't ship it self-contained
            pair_miss += 1
            continue
        pair_rows.append({**p, "crop_a": na, "crop_b": nb})
    with open(out / "pairs.jsonl", "w", encoding="utf-8") as f:
        for r in pair_rows:
            f.write(json.dumps(r) + "\n")
    n_same = sum(1 for r in pair_rows if r["label"] == "same")
    n_diff = sum(1 for r in pair_rows if r["label"] == "different")

    (out / "DATASET.md").write_text(_DATASET_MD.format(
        date=args.date or "all-time", total=len(rows), placed=placed, by_cat=by_cat,
        ids=len(list((out / "reid").glob("id*"))), pairs=len(pair_rows),
        same=n_same, diff=n_diff, embeds=len(emb_rows)), encoding="utf-8")

    print(f"exported {len(rows)} annotations ({placed} crops copied) -> {out}")
    print(f"  by category: {by_cat}")
    print(f"  reid identities: {sorted(p.name for p in (out / 'reid').glob('id*'))}")
    print(f"  human pairs: {len(pair_rows)} ({n_same} same / {n_diff} different)"
          + (f", {pair_miss} skipped (crop not on disk)" if pair_miss else ""))
    print(f"  embeddings captured: {len(emb_rows)} (OSNet-only, optional)")


if __name__ == "__main__":
    main()
