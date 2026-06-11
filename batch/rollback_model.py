"""ONE-COMMAND ROLLBACK of a promoted ReID model back to the previous (stock) embedding space.

  python -m batch.rollback_model            # roll back to the newest pre-promote backup
  python -m batch.rollback_model --yes      # skip the confirmation prompt

Non-destructive — restores from the backups the promote wrote, never deletes rows:
  1. restore the newest outputs/osnet_emb_cache.OLD_*.pkl  -> osnet_emb_cache.pkl
  2. remove outputs/reid_active_weights.txt                -> stack.reid falls back to stock OSNet
  3. re-embed EVERY gallery row in place from the restored cache (UPDATE by id; the rows the promote
     added stay, just re-embedded to the old space — no DELETE)
  4. re-activate the pre-promote model version recorded in outputs/reid_rollback_*.json
Then restart the dashboard (uvicorn) so it drops its in-process memos.
"""
from __future__ import annotations

import glob
import json
import os
import sys

import numpy as np


def _newest(pattern):
    xs = sorted(glob.glob(pattern))
    return xs[-1] if xs else None


def main():
    yes = "--yes" in sys.argv
    old_cache = _newest("outputs/osnet_emb_cache.OLD_*.pkl")
    snap = _newest("outputs/reid_rollback_*.json")
    if not old_cache:
        print("[rollback] no outputs/osnet_emb_cache.OLD_*.pkl backup found — cannot roll back.")
        return
    prev_ver = None
    if snap:
        try:
            prev_ver = json.loads(open(snap, encoding="utf-8").read()).get("active_model", {}).get("version")
        except Exception:
            pass
    print(f"[rollback] restore cache : {old_cache}")
    print(f"[rollback] reactivate    : model version {prev_ver} (from {snap})")
    print(f"[rollback] remove sentinel: outputs/reid_active_weights.txt")
    if not yes:
        print("\n[rollback] re-run with --yes to execute.")
        return

    import shutil
    shutil.copyfile(old_cache, "outputs/osnet_emb_cache.pkl")
    if os.path.exists("outputs/reid_active_weights.txt"):
        os.remove("outputs/reid_active_weights.txt")
    if os.path.exists("outputs/osnet_emb_cache.space.txt"):
        os.remove("outputs/osnet_emb_cache.space.txt")

    import dashboard.hitl_api as H
    cache = {str(k).replace("\\", "/"): v for k, v in H._cache_norm().items()}
    rows = H.store.get_gallery_with_id("s14")
    fixed = miss = 0
    for r in rows:
        v = cache.get((r.get("crop_url") or "").replace("\\", "/"))
        if v is None:
            miss += 1
            continue
        H.store.update_gallery_embedding(r["id"], np.asarray(v, "float32"))
        fixed += 1
    print(f"[rollback] gallery re-embedded to old space: {fixed} rows ({miss} unresolved)")

    if prev_ver is not None:
        with H.store._cx() as cx, cx.cursor() as cur:
            cur.execute("update model_versions set active=false where active=true")
            cur.execute("update model_versions set active=true where version=%s", (prev_ver,))
        am = H.store.active_model() or {}
        print(f"[rollback] active model now v{am.get('version')} ({am.get('kind')})")
    print("[rollback] DONE — restart the dashboard (uvicorn) to clear its memos.")


if __name__ == "__main__":
    main()
