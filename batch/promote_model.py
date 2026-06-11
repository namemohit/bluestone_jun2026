"""Promote a contrastive fine-tune (training/finetune_triplet.py output) into the LIVE embedding space.

  python -m batch.promote_model --weights outputs/models/osnet_triplet_YYYYMMDD_HHMMSS.pt --date 2026-06-03

Idempotent + non-destructive. Steps (all behind backups for batch.rollback_model):
  1. back up the cache + gallery embeddings + active-model params (cache.OLD_*, gallery_backup_*, reid_rollback_*)
  2. point the active-weights sentinel at the new weights -> stack.reid embeds in the new space everywhere
  3. re-embed the WHOLE crop cache into the new space   (python -m batch.reembed_all)
  4. re-embed the staff gallery in place from the new cache (UPDATE by id; no DELETE)
  5. re-tune staff_sim (gallery) + reid_threshold (calibration) + reid_exit_floor (precision curve)
  6. verify #C4 stays presumed + print the day's matched/presumed split

Then restart the dashboard so it drops in-process memos. Roll back with: python -m batch.rollback_model --yes
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time

import numpy as np

STORE = "s14"


def _backup(H):
    ts = time.strftime("%Y%m%d_%H%M%S")
    import shutil
    if os.path.exists("outputs/osnet_emb_cache.pkl"):
        shutil.copyfile("outputs/osnet_emb_cache.pkl", f"outputs/osnet_emb_cache.OLD_{ts}.pkl")
    am = H.store.active_model() or {}
    gal = H.store.get_gallery_with_id(STORE)
    json.dump({"active_model": {"version": am.get("version"), "kind": am.get("kind"),
                                "params": am.get("params")}, "gallery_count": len(gal)},
              open(f"outputs/reid_rollback_{ts}.json", "w"), indent=2, default=str)
    json.dump([{"id": r["id"], "employee_id": r["employee_id"], "embedding": r["embedding"]} for r in gal],
              open(f"outputs/gallery_backup_{ts}.json", "w"))
    print(f"[promote] backed up cache + gallery({len(gal)}) + active model v{am.get('version')} (stamp {ts})")


def main():
    ap = argparse.ArgumentParser(description="Promote a contrastive fine-tune into the live space")
    ap.add_argument("--weights", required=True, help="path to the osnet_triplet_*.pt weights to promote")
    ap.add_argument("--date", default="2026-06-03")
    ap.add_argument("--min-same-keep", type=float, default=0.85, help="exit floor must keep >= this fraction of same pairs")
    args = ap.parse_args()
    if not os.path.exists(args.weights):
        sys.exit(f"weights not found: {args.weights}")

    import dashboard.hitl_api as H
    from stack import reid as R

    # 1) backup
    _backup(H)

    # 2) sentinel -> new space everywhere
    open("outputs/reid_active_weights.txt", "w", encoding="utf-8").write(os.path.abspath(args.weights))
    R.reset_osnet()
    print(f"[promote] sentinel -> {args.weights}")

    # 3) re-embed the whole cache (separate process so torch/cuda starts clean)
    print("[promote] re-embedding the crop cache (this is the GPU pass)...")
    subprocess.run([sys.executable, "-m", "batch.reembed_all"], check=True)
    H._EMB_CACHE["mtime"] = None                    # force the in-process cache to reload the new pickle
    cache = {str(k).replace("\\", "/"): v for k, v in H._cache_norm().items()}

    # 4) re-embed the gallery in place from the new cache
    rows = H.store.get_gallery_with_id(STORE)
    fixed = 0
    for r in rows:
        v = cache.get((r.get("crop_url") or "").replace("\\", "/"))
        if v is None:
            continue
        cur = np.asarray(r["embedding"], "float32"); cur = cur / (np.linalg.norm(cur) + 1e-9)
        if float(np.dot(cur, np.asarray(v, "float32"))) > 0.999:
            continue
        H.store.update_gallery_embedding(r["id"], np.asarray(v, "float32"))
        fixed += 1
    print(f"[promote] gallery: re-embedded {fixed}/{len(rows)} rows into the new space")

    # 5) re-tune: gallery staff_sim -> reid_threshold -> reid_exit_floor
    gr = H._train_rebuild(None)
    print("[promote] " + gr["summary"])
    rt = H.reid_train({"date": args.date})
    print(f"[promote] reid_threshold -> {rt.get('best_threshold')} (sep {rt.get('separation')}, acc {rt.get('best_acc')})")
    same, diff = H._reid_labeled_pairs(args.date)
    dr = lambda t: sum(1 for d in diff if d < t) / len(diff) if diff else 1.0
    sk = lambda t: sum(1 for s in same if s >= t) / len(same) if same else 0.0
    grid = [0.45, 0.50, 0.55, 0.60, 0.65, 0.70]
    print("[promote] exit-floor curve (%diff-rej / %same-keep): " +
          "  ".join(f"{t:.2f}:{dr(t)*100:.0f}/{sk(t)*100:.0f}" for t in grid))
    cands = [t for t in grid if sk(t) >= args.min_same_keep]
    floor = max(cands) if cands else 0.55
    cur = (H.store.active_model() or {}).get("params", {}) or {}
    merged = {**cur, "reid_exit_floor": round(float(floor), 3)}
    ver = H.store.register_model_version(
        "thresholds", merged, trained_on=cur.get("reid_calibrated_pairs", 0),
        notes=f"promote: new contrastive space; staff_sim {cur.get('staff_sim')}, "
              f"reid_threshold {cur.get('reid_threshold')}, exit_floor {floor:.2f}", active=True)
    fin = (H.store.active_model() or {}).get("params", {})
    print(f"[promote] ACTIVE v{ver}: staff_sim={fin.get('staff_sim')} "
          f"reid_threshold={fin.get('reid_threshold')} reid_exit_floor={fin.get('reid_exit_floor')}")

    # 6) verify #C4 + day split
    H._STATE_VERSION += 1
    ex = H._exit_resolve(args.date)
    from collections import Counter
    src = Counter(v.get("source") for v in ex.values())
    c4 = next((v for (w, t), v in ex.items() if "1422" in w and t == 656), None)
    print(f"[promote] exits {dict(src)} | #C4(1422/656): "
          + (f"{c4.get('source')} dwell {c4.get('dwell_s')}s" if c4 else "matched(!) — investigate"))
    print("[promote] DONE — restart the dashboard (uvicorn) to clear its memos. Rollback: python -m batch.rollback_model --yes")


if __name__ == "__main__":
    main()
