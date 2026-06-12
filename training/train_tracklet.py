"""Train ReID on C11 tracklets, self-supervised (the "from scratch on C11 only" run).

Each tracklet (from `training.build_c11_tracklets`) = one pseudo-identity. POSITIVES = two views of
the same tracklet; NEGATIVES = two tracklets whose time-spans OVERLAP in the same window (they cannot be
the same person, so these are SAFE hard negatives — we deliberately avoid cross-time different-tracklet
negatives, where a person returning would be a false negative). Held-out VAL tracklets (disjoint
identities) drive model selection — the explicit fix for the prior in-sample overfit.

Reuses the OSNet backbone + contrastive loss + 128x256 ImageNet preprocessing + checkpoint format from
`training.finetune_triplet`, so `stack/reid.py:_triplet_embed`, `training/bakeoff.py`, and
`batch/promote_model.py` all consume the output unchanged.

  python -m training.train_tracklet --data training_data/c11_tracklets --epochs 20
  python -m training.train_tracklet --data training_data/c11_tracklets --epochs 4 --steps 30   # smoke test
"""
from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path

from training.finetune_triplet import _load_image    # identical preprocessing -> same embedding space

MARGIN = 0.5


def _feat(o):
    o = o[-1] if isinstance(o, (tuple, list)) else o   # OSNet(loss='triplet') returns (logits, feats) in train mode
    return o / (o.norm(dim=1, keepdim=True) + 1e-9)


def overlaps(a, b) -> bool:
    return (a["window"] == b["window"]
            and a["first_ts"] <= b["last_ts"] and b["first_ts"] <= a["last_ts"])


def build_val_pairs(val, rng, n_pos=400, n_neg=400):
    """Fixed held-out eval set: positives = 2 views of one val tracklet; negatives = 2 time-overlapping
    val tracklets (guaranteed different people)."""
    by_win: dict = {}
    for t in val:
        by_win.setdefault(t["window"], []).append(t)
    pos = []
    for _ in range(n_pos):
        t = rng.choice(val)
        a, b = rng.sample(t["crops"], 2)
        pos.append((a["path"], b["path"]))
    neg, tries = [], 0
    while len(neg) < n_neg and tries < n_neg * 30:
        tries += 1
        t = rng.choice(val)
        cands = [u for u in by_win[t["window"]] if u["tracklet"] != t["tracklet"] and overlaps(t, u)]
        if not cands:
            continue
        u = rng.choice(cands)
        neg.append((rng.choice(t["crops"])["path"], rng.choice(u["crops"])["path"]))
    return pos, neg


def val_separation(model, pos, neg, dev, batch=64):
    """Embed every unique val crop ONCE in batches (not per-pair) -> fast; then look up pair cosines."""
    import numpy as np
    import torch
    paths = sorted({p for pr in (pos + neg) for p in pr})
    emb: dict = {}
    model.eval()
    with torch.no_grad():
        for i in range(0, len(paths), batch):
            ims, keep = [], []
            for p in paths[i:i + batch]:
                im = _load_image(p)
                if im is not None:
                    ims.append(im); keep.append(p)
            if not ims:
                continue
            x = torch.tensor(np.stack(ims)).permute(0, 3, 1, 2).float().to(dev)
            f = _feat(model(x)).cpu().numpy()
            for k, v in zip(keep, f):
                emb[k] = v

    def sims(pairs):
        return [float(np.dot(emb[a], emb[b])) for a, b in pairs if a in emb and b in emb]

    s, d = sims(pos), sims(neg)
    sm = float(np.mean(s)) if s else 0.0
    dm = float(np.mean(d)) if d else 0.0
    return sm, dm, sm - dm


def main() -> None:
    ap = argparse.ArgumentParser(description="Self-supervised C11 tracklet ReID training")
    ap.add_argument("--data", default="training_data/c11_tracklets")
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--steps", type=int, default=150, help="batches per epoch")
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--margin", type=float, default=MARGIN)
    ap.add_argument("--scratch", action="store_true",
                    help="random init (default: fine-tune the pretrained OSNet — recommended on one day)")
    ap.add_argument("--out", default="outputs/models")
    args = ap.parse_args()
    rng = random.Random(0)

    man = [json.loads(l) for l in (Path(args.data) / "manifest.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
    train = [t for t in man if t["split"] == "train" and len(t["crops"]) >= 2]
    val = [t for t in man if t["split"] == "val" and len(t["crops"]) >= 2]
    print(f"[train] {len(train)} train / {len(val)} val tracklets "
          f"({sum(len(t['crops']) for t in train)} train crops)", flush=True)

    by_win: dict = {}
    for t in train:
        by_win.setdefault(t["window"], []).append(t)
    overlap = {t["tracklet"]: [u for u in by_win[t["window"]]
                               if u["tracklet"] != t["tracklet"] and overlaps(t, u)] for t in train}

    import numpy as np
    import torch
    from boxmot.reid.backbones.osnet import osnet_x1_0
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    model = osnet_x1_0(num_classes=0, pretrained=not args.scratch, loss="triplet").to(dev)
    print(f"[train] device={dev}  init={'random (scratch)' if args.scratch else 'pretrained OSNet'}", flush=True)

    vpos, vneg = build_val_pairs(val, rng)
    sm0, dm0, gap0 = val_separation(model, vpos, vneg, dev)
    print(f"[train] VAL before: same={sm0:.4f} diff={dm0:.4f} sep={gap0:+.4f}  "
          f"({len(vpos)} pos / {len(vneg)} neg)", flush=True)

    opt = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)

    def sample_pair(positive):
        t = rng.choice(train)
        if positive:
            a, b = rng.sample(t["crops"], 2)
            return a["path"], b["path"], 1.0
        cands = overlap[t["tracklet"]] or [u for u in train if u["window"] != t["window"]]
        u = rng.choice(cands)
        return rng.choice(t["crops"])["path"], rng.choice(u["crops"])["path"], 0.0

    ts = time.strftime("%Y%m%d_%H%M%S")
    wp = Path(args.out) / f"osnet_c11_tracklet_{ts}.pt"
    wp.parent.mkdir(parents=True, exist_ok=True)

    def save_ckpt(b):                                              # incremental: write the best on EVERY improvement
        torch.save({"state_dict": b["state"], "arch": "osnet_x1_0", "loss": "contrastive",
                    "margin": args.margin, "before": {"same_sim": sm0, "diff_sim": dm0, "gap": gap0},
                    "after": {"same_sim": b["sm"], "diff_sim": b["dm"], "gap": b["sep"]},
                    "trained": "c11_tracklet_selfsup", "val_best_epoch": b["epoch"],
                    "train_tracklets": len(train), "val_tracklets": len(val), "epochs": args.epochs}, wp)

    best = {"sep": gap0, "state": None, "sm": sm0, "dm": dm0, "epoch": 0}
    for ep in range(args.epochs):
        model.train()
        ep_loss, nb = 0.0, 0
        for _ in range(args.steps):
            ia_l, ib_l, ys = [], [], []
            for _ in range(args.batch):
                pa, pb, y = sample_pair(rng.random() < 0.5)
                ia, ib = _load_image(pa), _load_image(pb)
                if ia is None or ib is None:
                    continue
                ia_l.append(ia); ib_l.append(ib); ys.append(y)
            if not ia_l:
                continue
            xa = torch.tensor(np.stack(ia_l)).permute(0, 3, 1, 2).float().to(dev)
            xb = torch.tensor(np.stack(ib_l)).permute(0, 3, 1, 2).float().to(dev)
            y = torch.tensor(ys, dtype=torch.float32).to(dev)
            opt.zero_grad()
            fa, fb = _feat(model(xa)), _feat(model(xb))
            dist = 1.0 - (fa * fb).sum(dim=1)                       # cosine distance [0,2]
            loss = (y * dist.pow(2) + (1.0 - y) * torch.clamp(args.margin - dist, min=0.0).pow(2)).mean()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            ep_loss += float(loss.item()); nb += 1
        sched.step()
        sm, dm, sep = val_separation(model, vpos, vneg, dev)
        improved = sep > best["sep"]
        print(f"[train] epoch {ep+1}/{args.epochs} loss={ep_loss/max(1,nb):.4f}  "
              f"VAL sep={sep:+.4f} (same={sm:.3f} diff={dm:.3f}){'  <- best (saved)' if improved else ''}", flush=True)
        if improved:
            best = {"sep": sep, "sm": sm, "dm": dm, "epoch": ep + 1,
                    "state": {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}}
            save_ckpt(best)                                        # <- never lose the best; grab it any time

    if best["state"] is None:                                      # never beat baseline -> save final, flag it
        print("[train] WARNING: VAL never improved over baseline", flush=True)
        best.update({"state": {k: v.detach().cpu().clone() for k, v in model.state_dict().items()},
                     "epoch": args.epochs, "sm": sm, "dm": dm})
        save_ckpt(best)
    print(f"[train] BEST VAL sep {gap0:+.4f} -> {best['sep']:+.4f} @epoch {best['epoch']}  saved -> {wp}", flush=True)

    try:
        from hitl.store_supabase import SupabaseStore
        SupabaseStore().register_model_version(
            "osnet_c11_tracklet",
            {"weights": str(wp), "loss": "contrastive", "val_sep_before": round(gap0, 4),
             "val_sep_after": round(best["sep"], 4), "train_tracklets": len(train),
             "val_tracklets": len(val), "epochs": args.epochs, "margin": args.margin},
            score=round((best["sep"] + 1) / 2, 4), trained_on=len(train),
            notes=f"C11 tracklet self-sup: held-out VAL sep {gap0:+.4f} -> {best['sep']:+.4f}", active=False)
    except Exception as e:
        print(f"[train] (model_version not registered: {type(e).__name__}: {e})", flush=True)


if __name__ == "__main__":
    main()
