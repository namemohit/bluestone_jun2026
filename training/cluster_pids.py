"""Quick ReID-clustering PID prototype (L3-as-master).

Takes the All-Detections staff + customer bucket CARDS for a day and clusters them into UNIQUE
persons by ReID cosine similarity (greedy union-find at a threshold). Each cluster = one PID;
unique-cluster count = the de-duplicated footfall. Uses the ACTIVE model's cached embeddings
(read-only — does not touch the live model), so it's the BASELINE we re-run after C11 training
to measure the improvement.

  python -m training.cluster_pids --date 2026-06-03-c
  python -m training.cluster_pids --date 2026-06-03-c --montage 0.62   # eyeball clusters at a threshold
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

import dashboard.hitl_api as H


def card_emb(card, cache):
    embs = []
    for c in card.get("crops", []):
        v = cache.get((c.get("crop") or "").replace("\\", "/"))
        if v is not None:
            embs.append(np.asarray(v, dtype=np.float32))
    if not embs:
        return None
    m = np.mean(embs, axis=0)
    return m / (np.linalg.norm(m) + 1e-9)


def cluster(embs, thr):
    """Greedy union-find: merge cards with cosine >= thr. Returns list of index-groups."""
    n = len(embs)
    parent = list(range(n))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    M = np.vstack(embs) if embs else np.zeros((0, 512), np.float32)
    sims = M @ M.T if n else np.zeros((0, 0))
    for i in range(n):
        for j in range(i + 1, n):
            if sims[i, j] >= thr:
                parent[find(i)] = find(j)
    comps: dict = {}
    for i in range(n):
        comps.setdefault(find(i), []).append(i)
    return list(comps.values())


def main() -> None:
    ap = argparse.ArgumentParser(description="ReID-clustering PID prototype")
    ap.add_argument("--date", default="2026-06-03-c")
    ap.add_argument("--thresholds", default="0.5,0.55,0.6,0.65,0.7,0.75")
    ap.add_argument("--montage", type=float, default=0.0, help="if set, montage clusters at this threshold")
    args = ap.parse_args()

    H._STATE_VERSION += 1
    cache = H._load_emb_cache()
    cards = H._cards_from_dets(args.date)
    print(f"date={args.date}  active embeddings in cache: {len(cache)}")

    out = {}
    for kind in ("customer_cards", "staff_cards"):
        cs = cards.get(kind, [])
        embs, pids = [], []
        for c in cs:
            e = card_emb(c, cache)
            if e is not None:
                embs.append(e)
                pids.append(c.get("pid"))
        out[kind] = (cs, embs, pids)
        print(f"\n=== {kind}: {len(cs)} cards, {len(embs)} with embeddings ===")
        print(f"   raw cards (current L2-minted PIDs): {len(embs)}")
        for thr in [float(t) for t in args.thresholds.split(",")]:
            comps = cluster(embs, thr)
            sizes = sorted((len(c) for c in comps), reverse=True)
            merged = sum(1 for c in comps if len(c) > 1)
            print(f"   thr={thr:.2f} -> {len(comps):3} unique persons  "
                  f"({merged} clusters merged >1 card; biggest={sizes[0] if sizes else 0})")

    if args.montage:
        import cv2
        cs, embs, pids = out["customer_cards"]
        comps = sorted(cluster(embs, args.montage), key=len, reverse=True)
        rows = []
        for comp in [c for c in comps if len(c) > 1][:8]:           # show the merges (de-dups)
            tiles = []
            for idx in comp[:6]:
                cr = (cs[idx].get("crops") or [{}])[0].get("crop")
                im = cv2.imread(cr) if cr and Path(cr).exists() else None
                if im is None:
                    im = np.full((150, 70, 3), 40, np.uint8)
                t = cv2.resize(im, (70, 150))
                cv2.putText(t, str(pids[idx] or ""), (2, 14), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)
                tiles.append(t)
            while len(tiles) < 6:
                tiles.append(np.full((150, 70, 3), 20, np.uint8))
            rows.append(np.hstack(tiles))
        if rows:
            Path("_xcheck").mkdir(exist_ok=True)
            cv2.imwrite("_xcheck/pid_clusters.jpg", np.vstack(rows))
            print(f"\n   montage of merged clusters @thr={args.montage} -> _xcheck/pid_clusters.jpg "
                  f"(each row = cards the model thinks are ONE person)")


if __name__ == "__main__":
    main()
