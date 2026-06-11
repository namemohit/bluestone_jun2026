"""Zero-shot ReID backbone BAKE-OFF on this store's own footage — pick the model by numbers.

Embeds every cached crop with each candidate backbone (one pickle per model under
outputs/bakeoff/, resumable) and scores each on:

  (a) ALL human ✓same/✗different pairs (zero-shot candidates never trained on them, so the
      whole labeled set is held-out), split same-camera vs cross-camera. NOTE the ✗ pairs
      were hard-mined against STOCK OSNet, which slightly favours the challengers — that is
      the operationally correct test (these are the pairs the Review Queue actually shows).
  (b) the unlabeled STRANGER PROBE: random cross-track cosine (mean/p90/p99). A model that
      inflates strangers floods the queue with false "same person" — the 2026-06-11 overfit
      failure this harness exists to catch.

Read-only w.r.t. the live cache; NO promotion happens here. The winner goes live only via
batch/promote_model.py (with its gate) after an explicit human go.

  python -m training.bakeoff                 # all candidates
  python -m training.bakeoff --probe-n 5000  # bigger stranger probe
"""
from __future__ import annotations

import argparse
import json
import os
import pickle
import random
import time
from pathlib import Path

import cv2
import numpy as np

OUT = Path("outputs/bakeoff")
LIVE_CACHE = "outputs/osnet_emb_cache.pkl"

# (name, weights, kind):  boxmot = ReID factory one-string swap; triplet = local contrastive ckpt
CANDIDATES = [
    ("stock_osnet",     "osnet_x1_0_msmt17.pt",                          "live_cache"),
    ("osnet_ain",       "osnet_ain_x1_0_msmt17.pt",                      "boxmot"),
    ("clip_reid",       "clip_market1501.pt",                            "boxmot"),
    ("triplet_overfit", "outputs/models/osnet_triplet_20260611_170104.pt", "triplet"),  # reference: the gate must fail it
]


def _crop_keys() -> list[str]:
    """The evaluation universe = every crop in the live cache (one best crop per L1 track)."""
    raw = pickle.load(open(LIVE_CACHE, "rb"))
    return sorted({str(k).replace("\\", "/") for k in raw})


def _embed_model(name: str, weights: str, kind: str, keys: list[str]) -> dict | None:
    """crop path -> L2-normalised vector for one candidate; cached to outputs/bakeoff/<name>.pkl."""
    OUT.mkdir(parents=True, exist_ok=True)
    pkl = OUT / f"{name}.pkl"
    if pkl.exists():
        d = pickle.load(open(pkl, "rb"))
        print(f"[bakeoff] {name}: reusing {pkl} ({len(d)} vectors)", flush=True)
        return d

    if kind == "live_cache":                                   # the live cache IS stock boxmot OSNet right now
        raw = pickle.load(open(LIVE_CACHE, "rb"))
        d = {str(k).replace("\\", "/"): np.asarray(v, "float32") for k, v in raw.items()}
        pickle.dump(d, open(pkl, "wb"))
        print(f"[bakeoff] {name}: copied live cache ({len(d)} vectors)", flush=True)
        return d

    if kind == "triplet":
        from stack.reid import _triplet_embed
        if not os.path.exists(weights):
            print(f"[bakeoff] {name}: weights missing ({weights}) — skipped", flush=True)
            return None
        embed_one = lambda im: _triplet_embed(im, weights)
    else:                                                      # boxmot factory: auto-downloads TRAINED_URLS weights
        try:
            import torch
            from boxmot.reid.core.reid import ReID
            dev = "0" if torch.cuda.is_available() else "cpu"
            reid = ReID(weights=Path(weights), device=dev, half=False)
        except Exception as e:
            print(f"[bakeoff] {name}: backbone unavailable ({type(e).__name__}: {e}) — skipped", flush=True)
            return None

        def embed_one(im):
            h, w = im.shape[:2]
            f = reid(im, boxes=np.array([[0, 0, w, h]], dtype=float))
            v = np.asarray(f, dtype="float32").ravel()
            return v / (np.linalg.norm(v) + 1e-9)

    d, t0 = {}, time.time()
    for i, k in enumerate(keys):
        im = cv2.imread(k)
        if im is None:
            continue
        try:
            d[k] = embed_one(im)
        except Exception:
            continue
        if (i + 1) % 1000 == 0:
            rate = (i + 1) / max(1e-6, time.time() - t0)
            print(f"[bakeoff] {name}: {i + 1}/{len(keys)} ({rate:.1f}/s)", flush=True)
            pickle.dump(d, open(pkl, "wb"))
    pickle.dump(d, open(pkl, "wb"))
    print(f"[bakeoff] {name}: embedded {len(d)}/{len(keys)} in {time.time() - t0:.0f}s", flush=True)
    return d


def _score_pairs(d: dict, pairs: list[dict]) -> dict:
    """Separation + best-threshold accuracy on the human pairs, overall + per samecam/xcam tier."""
    def bucket(rows):
        same = [s for s, p in rows if p["label"] == "same"]
        diff = [s for s, p in rows if p["label"] == "different"]
        if not same or not diff:
            return {"n_same": len(same), "n_diff": len(diff), "gap": None, "best_acc": None}
        n = len(same) + len(diff)
        best = max((sum(1 for s in same if s >= t) + sum(1 for x in diff if x < t)) / n
                   for t in sorted(set(same + diff)))
        return {"n_same": len(same), "n_diff": len(diff),
                "same_mean": round(float(np.mean(same)), 3), "diff_mean": round(float(np.mean(diff)), 3),
                "gap": round(float(np.mean(same) - np.mean(diff)), 3), "best_acc": round(best, 3)}

    rows = []
    for p in pairs:
        va, vb = d.get(p["crop_a"]), d.get(p["crop_b"])
        if va is None or vb is None:
            continue
        rows.append((float(np.dot(va, vb)), p))
    return {"all": bucket(rows),
            "samecam": bucket([r for r in rows if not r[1]["cross_camera"]]),
            "xcam": bucket([r for r in rows if r[1]["cross_camera"]])}


def _stranger_probe(d: dict, n: int = 3000) -> dict:
    """Random cross-track cosine — the population the overfit model collapsed. Fixed seed."""
    rng = random.Random(0)
    keys = list(d)
    sims = [float(np.dot(d[a], d[b])) for a, b in (rng.sample(keys, 2) for _ in range(n))]
    return {"mean": round(float(np.mean(sims)), 3),
            "p90": round(float(np.percentile(sims, 90)), 3),
            "p99": round(float(np.percentile(sims, 99)), 3)}


def _flood_at_threshold(d: dict, pairs: list[dict], n: int = 5000) -> dict:
    """The decisive gate metric: at the model's OWN best operating threshold (thresholds are always
    recalibrated per space), what %% of random stranger pairs would score 'same person'? This is the
    Review-Queue noise a human would actually see — the 2026-06-11 overfit floods 25%%, stock ~3%%."""
    rows = []
    for p in pairs:
        va, vb = d.get(p["crop_a"]), d.get(p["crop_b"])
        if va is not None and vb is not None:
            rows.append((float(np.dot(va, vb)), p["label"]))
    same = [s for s, l in rows if l == "same"]
    diff = [s for s, l in rows if l == "different"]
    if not same or not diff:
        return {"threshold": None, "flood_pct": None, "same_recall": None}
    best_t, best_a = 0.5, 0.0
    for c in sorted(set(same + diff)):
        a = (sum(1 for s in same if s >= c - 1e-4) + sum(1 for x in diff if x < c - 1e-4)) / len(rows)
        if a > best_a:
            best_a, best_t = a, c
    keys = sorted(d)
    rng = random.Random(0)
    idx = [(rng.randrange(len(keys)), rng.randrange(len(keys))) for _ in range(n)]
    flood = sum(1 for i, j in idx if i != j and float(np.dot(d[keys[i]], d[keys[j]])) >= best_t)
    return {"threshold": round(best_t, 3),
            "flood_pct": round(100.0 * flood / n, 2),
            "same_recall": round(sum(1 for s in same if s >= best_t) / len(same), 3)}


def main() -> None:
    ap = argparse.ArgumentParser(description="Zero-shot ReID backbone bake-off")
    ap.add_argument("--probe-n", type=int, default=3000)
    ap.add_argument("--models", nargs="*", help="subset of candidate names to run")
    args = ap.parse_args()

    keys = _crop_keys()
    print(f"[bakeoff] evaluation universe: {len(keys)} crops", flush=True)

    from hitl.store_supabase import SupabaseStore
    from training.reid_labels import human_pairs
    pairs = human_pairs(SupabaseStore(), None)
    n_same = sum(1 for p in pairs if p["label"] == "same")
    print(f"[bakeoff] human pairs: {len(pairs)} ({n_same} same / {len(pairs) - n_same} diff)", flush=True)

    results = {}
    for name, weights, kind in CANDIDATES:
        if args.models and name not in args.models:
            continue
        print(f"\n[bakeoff] ===== {name} ({weights}) =====", flush=True)
        d = _embed_model(name, weights, kind, keys)
        if not d:
            results[name] = {"error": "embed failed/skipped"}
            continue
        sc = _score_pairs(d, pairs)
        probe = _stranger_probe(d, args.probe_n)
        fl = _flood_at_threshold(d, pairs)
        sc["stranger_probe"] = probe
        sc["flood"] = fl
        results[name] = sc
        a = sc["all"]
        print(f"[bakeoff] {name}: gap {a.get('gap')} acc {a.get('best_acc')} "
              f"(samecam gap {sc['samecam'].get('gap')} / xcam gap {sc['xcam'].get('gap')}) | "
              f"strangers mean {probe['mean']} p99 {probe['p99']} | "
              f"thr {fl['threshold']} same-recall {fl['same_recall']} FLOOD {fl['flood_pct']}%", flush=True)

    # the gate this harness exists for: vs the stock baseline, held-out accuracy must IMPROVE and
    # the stranger flood at the operating threshold must not blow past 2x stock (the overfit = 25%)
    base = results.get("stock_osnet", {})
    base_acc = ((base.get("all") or {}).get("best_acc")) or 0
    base_flood = ((base.get("flood") or {}).get("flood_pct"))
    for name, sc in results.items():
        if "error" in sc or name == "stock_osnet":
            continue
        acc = ((sc.get("all") or {}).get("best_acc")) or 0
        fp = ((sc.get("flood") or {}).get("flood_pct"))
        sc["gate_pass"] = bool(acc > base_acc and fp is not None and base_flood is not None
                               and fp <= 2.0 * base_flood)

    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "results.json").write_text(json.dumps(
        {"stamped": time.strftime("%Y-%m-%d %H:%M:%S"), "crops": len(keys),
         "pairs": {"total": len(pairs), "same": n_same}, "models": results}, indent=2), encoding="utf-8")
    print(f"\n[bakeoff] results -> {OUT / 'results.json'}", flush=True)

    print("\n[bakeoff] ============ SUMMARY (pick by numbers) ============", flush=True)
    print(f"{'model':<16} {'gap':>6} {'acc':>6} {'xcam gap':>9} {'same-recall':>12} {'FLOOD%':>7}  gate", flush=True)
    for name, sc in results.items():
        if "error" in sc:
            print(f"{name:<16} {sc['error']}", flush=True)
            continue
        a, fl = sc["all"], sc["flood"]
        gate = "baseline" if name == "stock_osnet" else ("pass" if sc.get("gate_pass") else "FAIL")
        print(f"{name:<16} {str(a.get('gap')):>6} {str(a.get('best_acc')):>6} "
              f"{str(sc['xcam'].get('gap')):>9} {str(fl.get('same_recall')):>12} "
              f"{str(fl.get('flood_pct')):>7}  {gate}", flush=True)


if __name__ == "__main__":
    main()
