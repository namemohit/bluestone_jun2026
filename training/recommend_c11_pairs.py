"""Active-learning pair recommender for C11 ReID fine-tuning.

Embeds every C11 tracklet crop with the CURRENT TransReID model, then surfaces the pairs whose human
✓/✗ verdict would teach the model the MOST (max confusion = max gradient = max learning):

  • HARD NEGATIVES — crops from DIFFERENT tracklets the model wrongly thinks look alike (highest
    cross-tracklet similarity). Your ✗ sharpens the decision boundary; your ✓ (it's the same person
    returning) is a rare, gold cross-tracklet positive. Either verdict is high-information.
  • HARD POSITIVES — the SAME person the model fails on: the lowest-similarity crop pair INSIDE one
    tracklet. Your ✓ teaches pose/lighting invariance (your ✗ catches a tracking merge error).

Output → outputs/c11_train_candidates.json (consumed by the ReID tab to surface for marking).

Re-run with --model <finetuned.pt> after each training round so the recommendations track the
improving model (classic active-learning loop).

    python -m training.recommend_c11_pairs                 # base TransReID
    python -m training.recommend_c11_pairs --limit 60      # quick smoke test
    python -m training.recommend_c11_pairs --model outputs/models/transreid_xxx.pt
"""
import os
import sys
import json
import argparse
import numpy as np

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))   # repo root
if HERE not in sys.path:
    sys.path.insert(0, HERE)
import cv2
from stack import transreid

MANIFEST = os.path.join(HERE, "training_data", "c11_tracklets", "manifest.jsonl")
OUT = os.path.join(HERE, "outputs", "c11_train_candidates.json")


def _closest_crops(ta, tb):
    """The single most-confusing view pair between two tracklets (argmax crop-to-crop sim)."""
    s = ta["embs"] @ tb["embs"].T
    i, j = np.unravel_index(int(np.argmax(s)), s.shape)
    return ta["crops"][i], tb["crops"][j], float(s[i, j])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="base", help="'base' or a path to a finetuned .pt")
    ap.add_argument("--neg", type=int, default=150, help="hard-negative candidate pairs to surface")
    ap.add_argument("--pos", type=int, default=50, help="hard-positive candidate pairs to surface")
    ap.add_argument("--per-tracklet-cap", type=int, default=3, help="max neg pairs touching one tracklet (diversity)")
    ap.add_argument("--limit", type=int, default=0, help="embed only the first N tracklets (smoke test)")
    args = ap.parse_args()

    if args.model == "base":
        model = transreid._load()
        use_sie = False
        model_name = "transreid-base"
    else:
        model, use_sie = transreid.load_finetuned(args.model)
        model_name = os.path.basename(args.model)
    print(f"[recommend] model={model_name}  use_sie={use_sie}", flush=True)

    rows = [json.loads(l) for l in open(MANIFEST, encoding="utf-8")]
    if args.limit:
        rows = rows[: args.limit]
    print(f"[recommend] {len(rows)} tracklets to embed", flush=True)

    trk, n_crops = [], 0
    for i, r in enumerate(rows):
        embs, paths = [], []
        for c in r.get("crops", []):
            img = cv2.imread(os.path.join(HERE, c["path"]))
            if img is None:
                continue
            v = np.asarray(transreid.embed_with(model, img, use_sie), dtype="float32")
            nrm = np.linalg.norm(v)
            if nrm > 0:
                v = v / nrm
            embs.append(v)
            paths.append(c["path"])
            n_crops += 1
        if not embs:
            continue
        E = np.vstack(embs)
        mean = E.mean(0)
        mn = np.linalg.norm(mean)
        trk.append({"window": r["window"], "track": r["track"], "crops": paths,
                    "embs": E, "mean": (mean / mn) if mn > 0 else mean})
        if (i + 1) % 100 == 0:
            print(f"  embedded {i + 1}/{len(rows)}  ({n_crops} crops)", flush=True)
    print(f"[recommend] embedded {len(trk)} tracklets / {n_crops} crops", flush=True)

    M = np.vstack([t["mean"] for t in trk])                    # [T, D]
    S = M @ M.T
    np.fill_diagonal(S, -1.0)
    T = len(trk)

    # ---- HARD NEGATIVES: the highest cross-tracklet similarities (the model's worst confusions) ----
    pairs = [(float(S[a, b]), a, b) for a in range(T) for b in range(a + 1, T)]
    pairs.sort(reverse=True)
    neg, per = [], {}
    for sim, a, b in pairs:
        if len(neg) >= args.neg:
            break
        if per.get(a, 0) >= args.per_tracklet_cap or per.get(b, 0) >= args.per_tracklet_cap:
            continue
        ta, tb = trk[a], trk[b]
        ca, cb, csim = _closest_crops(ta, tb)
        neg.append({"kind": "hard_neg", "sim": round(csim, 3), "track_sim": round(sim, 3),
                    "window_a": ta["window"], "track_a": ta["track"], "crop_a": ca,
                    "window_b": tb["window"], "track_b": tb["track"], "crop_b": cb,
                    "guess": "different"})
        per[a] = per.get(a, 0) + 1
        per[b] = per.get(b, 0) + 1

    # ---- HARD POSITIVES: the lowest within-tracklet crop-pair sim (same person the model fails on) ----
    hp = []
    for t in trk:
        E = t["embs"]
        if len(E) < 2:
            continue
        s = E @ E.T
        iu = np.triu_indices(len(E), k=1)
        k = int(np.argmin(s[iu]))
        i, j = iu[0][k], iu[1][k]
        hp.append((float(s[i, j]), t, int(i), int(j)))
    hp.sort()
    pos = [{"kind": "hard_pos", "sim": round(sim, 3),
            "window_a": t["window"], "track_a": t["track"], "crop_a": t["crops"][i],
            "window_b": t["window"], "track_b": t["track"], "crop_b": t["crops"][j],
            "guess": "same"} for sim, t, i, j in hp[: args.pos]]

    allsims = [p[0] for p in pairs]
    out = {"model": model_name, "n_tracklets": T, "n_crops": n_crops,
           "hard_negatives": neg, "hard_positives": pos,
           "stranger_sim": {"mean": round(float(np.mean(allsims)), 3) if allsims else None,
                            "p90": round(float(np.percentile(allsims, 90)), 3) if allsims else None,
                            "p99": round(float(np.percentile(allsims, 99)), 3) if allsims else None,
                            "max": round(float(np.max(allsims)), 3) if allsims else None}}
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    json.dump(out, open(OUT, "w"), indent=1)
    print(f"[recommend] wrote {OUT}", flush=True)
    print(f"[recommend] {len(neg)} hard-neg + {len(pos)} hard-pos candidates", flush=True)
    print(f"[recommend] cross-tracklet sim: mean={out['stranger_sim']['mean']} "
          f"p99={out['stranger_sim']['p99']} max={out['stranger_sim']['max']}", flush=True)
    if neg:
        print(f"[recommend] hardest negative: sim={neg[0]['sim']}  "
              f"{neg[0]['window_a']}#{neg[0]['track_a']} vs {neg[0]['window_b']}#{neg[0]['track_b']}", flush=True)


if __name__ == "__main__":
    main()
