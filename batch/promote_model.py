"""Promote a ReID model into the LIVE embedding space — gated, backup-first, one-command rollback.

  python -m batch.promote_model --weights boxmot:osnet_ain_x1_0_msmt17.pt \
         --from-pickle outputs/bakeoff/osnet_ain.pkl --date 2026-06-03
  python -m batch.promote_model --weights outputs/models/osnet_triplet_X.pt --date 2026-06-03

Weights forms: `boxmot:<name>` (boxmot factory weights, auto-downloaded) or a path to a local
contrastive checkpoint (training/finetune_triplet.py). `--from-pickle` reuses a full embedding
pickle (e.g. produced by training.bakeoff) instead of re-embedding — same vectors, zero GPU.

HARD GATE (the 2026-06-11 overfit lesson — runs BEFORE the live cache is touched):
  vs the current live space, on ALL human ✓/✗ pairs (held-out for zero-shot models):
    1. held-out best-threshold accuracy must IMPROVE, and
    2. the stranger FLOOD (random cross-track pairs scoring above the candidate's own operating
       threshold = Review-Queue false-"same person" noise) must stay <= 2x the current model's.
  A model that fails is NOT promoted and the live system is untouched.

Steps: backup (cache/gallery/params) -> candidate embeddings (side pickle, live untouched) ->
GATE -> sentinel -> atomic cache swap -> gallery re-embed IN PLACE -> re-tune staff_sim /
reid_threshold / reid_exit_floor -> verify #C4 + day split + top queue pairs for human eyeball.
Rollback: python -m batch.rollback_model --yes
"""
from __future__ import annotations

import argparse
import json
import os
import pickle
import sys
import time

import numpy as np

STORE = "s14"
CACHE = "outputs/osnet_emb_cache.pkl"


def _backup(H):
    ts = time.strftime("%Y%m%d_%H%M%S")
    import shutil
    if os.path.exists(CACHE):
        shutil.copyfile(CACHE, f"outputs/osnet_emb_cache.OLD_{ts}.pkl")
    am = H.store.active_model() or {}
    gal = H.store.get_gallery_with_id(STORE)
    json.dump({"active_model": {"version": am.get("version"), "kind": am.get("kind"),
                                "params": am.get("params")}, "gallery_count": len(gal)},
              open(f"outputs/reid_rollback_{ts}.json", "w"), indent=2, default=str)
    json.dump([{"id": r["id"], "employee_id": r["employee_id"], "embedding": r["embedding"]} for r in gal],
              open(f"outputs/gallery_backup_{ts}.json", "w"))
    print(f"[promote] backed up cache + gallery({len(gal)}) + active model v{am.get('version')} (stamp {ts})")


def main():
    ap = argparse.ArgumentParser(description="Gated promote of a ReID model into the live space")
    ap.add_argument("--weights", required=True,
                    help="boxmot:<weights_name> or path to a local contrastive .pt")
    ap.add_argument("--from-pickle", default=None,
                    help="reuse a full embedding pickle (training.bakeoff output) instead of re-embedding")
    ap.add_argument("--date", default="2026-06-03")
    ap.add_argument("--min-same-keep", type=float, default=0.85,
                    help="exit floor must keep >= this fraction of same pairs")
    args = ap.parse_args()

    is_boxmot = args.weights.startswith("boxmot:")
    if not is_boxmot and not os.path.exists(args.weights):
        sys.exit(f"weights not found: {args.weights}")

    import dashboard.hitl_api as H
    from stack import reid as R
    from training.bakeoff import _embed_model, _flood_at_threshold, _score_pairs, _stranger_probe
    from training.reid_labels import human_pairs

    pairs = human_pairs(H.store, None)
    print(f"[promote] human pairs for the gate: {len(pairs)}")

    # ---- 1) candidate embeddings on the side (live cache untouched) ----
    keys = sorted({str(k).replace("\\", "/") for k in pickle.load(open(CACHE, "rb"))})
    if args.from_pickle:
        cand = {str(k).replace("\\", "/"): np.asarray(v, "float32")
                for k, v in pickle.load(open(args.from_pickle, "rb")).items()}
        print(f"[promote] candidate embeddings from {args.from_pickle} ({len(cand)} vectors)")
    else:
        name = "promote_" + os.path.basename(args.weights).replace("boxmot:", "").replace(".pt", "")
        kind = "boxmot" if is_boxmot else "triplet"
        w = args.weights[len("boxmot:"):] if is_boxmot else args.weights
        cand = _embed_model(name, w, kind, keys)
        if not cand:
            sys.exit("[promote] candidate embedding failed")
    missing = sum(1 for k in keys if k not in cand)
    if missing > len(keys) * 0.02:
        sys.exit(f"[promote] candidate pickle misses {missing}/{len(keys)} live-cache crops — refusing")

    # ---- 2) HARD GATE vs the current live space ----
    live = {str(k).replace("\\", "/"): np.asarray(v, "float32")
            for k, v in pickle.load(open(CACHE, "rb")).items()}
    acc_old = (_score_pairs(live, pairs)["all"] or {}).get("best_acc") or 0
    acc_new = (_score_pairs(cand, pairs)["all"] or {}).get("best_acc") or 0
    fl_old = _flood_at_threshold(live, pairs)
    fl_new = _flood_at_threshold(cand, pairs)
    pr_new = _stranger_probe(cand)
    print(f"[promote] GATE  held-out acc {acc_old} -> {acc_new} | "
          f"flood {fl_old['flood_pct']}% -> {fl_new['flood_pct']}% (limit {2 * fl_old['flood_pct']:.1f}%) | "
          f"strangers p99 {pr_new['p99']}")
    if not (acc_new > acc_old and fl_new["flood_pct"] is not None
            and fl_new["flood_pct"] <= 2.0 * fl_old["flood_pct"]):
        sys.exit("[promote] ** GATE FAIL — not promoted; live system untouched **")
    print("[promote] gate PASS")

    # ---- 3) backup, then go live ----
    _backup(H)
    R._ACTIVE_WEIGHTS_FILE.parent.mkdir(parents=True, exist_ok=True)
    sentinel = args.weights if is_boxmot else os.path.abspath(args.weights)
    R._ACTIVE_WEIGHTS_FILE.write_text(sentinel, encoding="utf-8")
    R.reset_osnet()
    print(f"[promote] sentinel -> {sentinel}")

    tmp = CACHE + ".tmp"
    pickle.dump(cand, open(tmp, "wb"))
    os.replace(tmp, CACHE)                                     # atomic swap
    open("outputs/osnet_emb_cache.space.txt", "w", encoding="utf-8").write(
        f"weights={sentinel}\nentries={len(cand)}\nstamped={time.strftime('%Y-%m-%d %H:%M:%S')}\n")
    H._EMB_CACHE["mtime"] = None
    cache = {str(k).replace("\\", "/"): v for k, v in H._cache_norm().items()}
    print(f"[promote] live cache swapped ({len(cand)} vectors)")

    # ---- 4) gallery re-embed IN PLACE (UPDATE by id, no DELETE) ----
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

    # ---- 5) re-tune: staff_sim -> reid_threshold -> reid_exit_floor ----
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
        notes=f"promote {sentinel}: staff_sim {cur.get('staff_sim')}, "
              f"reid_threshold {cur.get('reid_threshold')}, exit_floor {floor:.2f}; "
              f"gate acc {acc_old}->{acc_new}, flood {fl_old['flood_pct']}->{fl_new['flood_pct']}%",
        active=True)
    fin = (H.store.active_model() or {}).get("params", {})
    print(f"[promote] ACTIVE v{ver}: staff_sim={fin.get('staff_sim')} "
          f"reid_threshold={fin.get('reid_threshold')} reid_exit_floor={fin.get('reid_exit_floor')}")

    # ---- 6) verify: #C4 + day split + the queue pairs a human will actually see ----
    H._STATE_VERSION += 1
    ex = H._exit_resolve(args.date)
    from collections import Counter
    src = Counter(v.get("source") for v in ex.values())
    c4 = next((v for (w, t), v in ex.items() if "1422" in w and t == 656), None)
    print(f"[promote] exits {dict(src)} | #C4(1422/656): "
          + (f"{c4.get('source')} dwell {c4.get('dwell_s')}s" if c4 else "matched(!) — investigate"))
    thr = float(fin.get("reid_threshold", 0.62) or 0.62)
    top = H.reid_candidates(args.date, threshold=min(thr, 0.62))["candidates"][:8]
    print("[promote] EYEBALL these top queue pairs before declaring success:")
    for c in top:
        print(f"   {c['a_pid']} vs {c['b_pid']}  sim {c['sim']}  verdict={c['verdict']}")
    print("[promote] DONE — restart the dashboard. Rollback: python -m batch.rollback_model --yes")


if __name__ == "__main__":
    main()
