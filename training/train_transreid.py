"""Fine-tune TransReID (ViT-base, MSMT) on C11 tracklets — same self-supervised setup as
`training.train_tracklet` (OSNet), but on the stronger ViT backbone.

POSITIVES = two views of one tracklet; NEGATIVES = two time-overlapping tracklets (guaranteed different
people). Held-out VAL tracklets (disjoint identities) drive model selection — the explicit fix for the
prior in-sample overfit.

Anti-overfit posture for a ViT-base on ONE day (~2k tracklets):
  * The MSMT backbone is FROZEN by default; only the `b1` global head + `bottleneck` train (one block,
    runs the backbone under no_grad → tiny memory, near-zero overfit capacity). `--unfreeze N` opens the
    last N used backbone blocks if head-only adaptation is too weak.
  * SIE off (matched the better off-the-shelf separation), select on held-out VAL, save a small delta.

  python -m training.train_transreid --epochs 12
  python -m training.train_transreid --epochs 12 --unfreeze 2     # also train last 2 backbone blocks
"""
from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path

import cv2
import numpy as np
import torch

from stack.transreid import CKPT, TransReID, _MEAN, _STD
from training.train_tracklet import build_val_pairs, overlaps   # reuse pair sampling + held-out eval set

N_USED = 11   # JPM inference uses base.blocks[:-1] = blocks 0..10; block 11 + b1 are the trained head


def load_chw(path):
    im = cv2.imread(path)
    if im is None:
        return None
    im = cv2.cvtColor(cv2.resize(im, (128, 256)), cv2.COLOR_BGR2RGB).astype("float32") / 255.0
    im = (im - _MEAN) / _STD
    return im.transpose(2, 0, 1)


def _norm(f):
    return f / (f.norm(dim=1, keepdim=True) + 1e-9)


def train_feat(model, x, use_sie, n_frozen):
    """Forward with the leading `n_frozen` used-blocks under no_grad (frozen), the rest + b1 + bottleneck
    with grad. n_frozen=N_USED → only the head trains."""
    B = x.shape[0]
    b = model.base
    with torch.no_grad():
        h = b.patch_embed(x)
        h = torch.cat((b.cls_token.expand(B, -1, -1), h), dim=1)
        h = h + b.pos_embed + (b.sie_xishu * b.sie_embed[0] if use_sie else 0.0)
        for blk in b.blocks[:n_frozen]:
            h = blk(h)
    h = h.detach()
    for blk in b.blocks[n_frozen:N_USED]:
        h = blk(h)
    g = model.b1(h)[:, 0]
    return _norm(model.bottleneck(g))


def val_sep(model, pos, neg, dev, use_sie, batch=32):
    paths = sorted({p for pr in (pos + neg) for p in pr})
    emb: dict = {}
    model.eval()
    with torch.no_grad():
        for i in range(0, len(paths), batch):
            ims, keep = [], []
            for p in paths[i:i + batch]:
                im = load_chw(p)
                if im is not None:
                    ims.append(im); keep.append(p)
            if not ims:
                continue
            x = torch.tensor(np.stack(ims)).float().to(dev)
            f = _norm(model(x, cam=0, use_sie=use_sie)).cpu().numpy()
            for k, v in zip(keep, f):
                emb[k] = v

    def sims(prs):
        return [float(np.dot(emb[a], emb[b])) for a, b in prs if a in emb and b in emb]

    s, d = sims(pos), sims(neg)
    sm = float(np.mean(s)) if s else 0.0
    dm = float(np.mean(d)) if d else 0.0
    return sm, dm, sm - dm


def main() -> None:
    ap = argparse.ArgumentParser(description="Fine-tune TransReID on C11 tracklets (self-supervised)")
    ap.add_argument("--data", default="training_data/c11_tracklets")
    ap.add_argument("--epochs", type=int, default=12)
    ap.add_argument("--steps", type=int, default=120, help="batches per epoch")
    ap.add_argument("--batch", type=int, default=24)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--margin", type=float, default=0.5)
    ap.add_argument("--unfreeze", type=int, default=0,
                    help="train the last N used backbone blocks too (0 = head-only)")
    ap.add_argument("--sie", action="store_true", help="use SIE cam0 (default off: better off-the-shelf)")
    ap.add_argument("--out", default="outputs/models")
    ap.add_argument("--pairs", default="", help="JSONL of HUMAN-marked pairs {crop_a,crop_b,verdict} mixed in as supervision")
    ap.add_argument("--human-frac", type=float, default=0.35, help="fraction of each batch drawn from human pairs (when --pairs given)")
    ap.add_argument("--job", default="", help="job status file to write progress to (for the dashboard)")
    args = ap.parse_args()
    use_sie = args.sie
    n_frozen = max(0, N_USED - args.unfreeze)
    rng = random.Random(0)

    jobf = Path(args.job) if args.job else None
    def _job(**kw):
        if not jobf:
            return
        try:
            cur = json.loads(jobf.read_text(encoding="utf-8")) if jobf.exists() else {}
        except Exception:
            cur = {}
        cur.update(kw)
        jobf.parent.mkdir(parents=True, exist_ok=True)
        jobf.write_text(json.dumps(cur), encoding="utf-8")
    _job(status="running", stage="loading data")

    # HUMAN pairs (active-learning marks) — folded into the self-supervised loss as gold supervision.
    human_pos, human_neg = [], []
    if args.pairs and Path(args.pairs).exists():
        for ln in Path(args.pairs).read_text(encoding="utf-8").splitlines():
            ln = ln.strip()
            if not ln:
                continue
            try:
                r = json.loads(ln)
            except Exception:
                continue
            a, b, v = r.get("crop_a"), r.get("crop_b"), r.get("verdict")
            if not a or not b or v not in ("same", "different"):
                continue
            (human_pos if v == "same" else human_neg).append((a, b))
        print(f"[transreid] human pairs: {len(human_pos)} same / {len(human_neg)} different "
              f"(mixed in at {args.human_frac:.0%})", flush=True)
    has_human = bool(human_pos or human_neg)

    man = [json.loads(l) for l in (Path(args.data) / "manifest.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
    train = [t for t in man if t["split"] == "train" and len(t["crops"]) >= 2]
    val = [t for t in man if t["split"] == "val" and len(t["crops"]) >= 2]
    print(f"[transreid] {len(train)} train / {len(val)} val tracklets "
          f"({sum(len(t['crops']) for t in train)} train crops)", flush=True)

    by_win: dict = {}
    for t in train:
        by_win.setdefault(t["window"], []).append(t)
    overlap = {t["tracklet"]: [u for u in by_win[t["window"]]
                               if u["tracklet"] != t["tracklet"] and overlaps(t, u)] for t in train}

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    model = TransReID().to(dev)
    sd = torch.load(CKPT, map_location=dev, weights_only=False)
    model.load_state_dict(sd, strict=False)

    # freeze all, then open the head (+ optional trailing blocks)
    for p in model.parameters():
        p.requires_grad_(False)
    trainable = list(model.b1.parameters()) + list(model.bottleneck.parameters())
    for blk in model.base.blocks[n_frozen:N_USED]:
        trainable += list(blk.parameters())
    for p in trainable:
        p.requires_grad_(True)
    ntr = sum(p.numel() for p in trainable)
    print(f"[transreid] device={dev}  SIE={'on' if use_sie else 'off'}  "
          f"unfreeze={args.unfreeze} blocks  trainable={ntr/1e6:.1f}M params", flush=True)

    vpos, vneg = build_val_pairs(val, rng)
    sm0, dm0, gap0 = val_sep(model, vpos, vneg, dev, use_sie)
    print(f"[transreid] VAL before: same={sm0:.4f} diff={dm0:.4f} sep={gap0:+.4f}  "
          f"({len(vpos)} pos / {len(vneg)} neg)", flush=True)

    opt = torch.optim.Adam([p for p in trainable], lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)

    def sample_pair(positive):
        # with probability human_frac, draw a GOLD human-marked pair instead of a self-supervised one
        if has_human and rng.random() < args.human_frac:
            if positive and human_pos:
                a, b = rng.choice(human_pos); return a, b, 1.0
            if (not positive) and human_neg:
                a, b = rng.choice(human_neg); return a, b, 0.0
            # fall through to self-supervised if the requested polarity has no human pairs
        t = rng.choice(train)
        if positive:
            a, b = rng.sample(t["crops"], 2)
            return a["path"], b["path"], 1.0
        cands = overlap[t["tracklet"]] or [u for u in train if u["window"] != t["window"]]
        u = rng.choice(cands)
        return rng.choice(t["crops"])["path"], rng.choice(u["crops"])["path"], 0.0

    ts = time.strftime("%Y%m%d_%H%M%S")
    wp = Path(args.out) / f"transreid_c11_tracklet_{ts}.pt"
    wp.parent.mkdir(parents=True, exist_ok=True)
    trk_keys = {id(p): n for n, p in model.named_parameters()}
    trainable_names = {trk_keys[id(p)] for p in trainable}

    def save_ckpt(b):
        # delta only: the trained tensors (head + any unfrozen blocks). base loads from CKPT at inference.
        full = model.state_dict()
        delta = {k: v.detach().cpu().clone() for k, v in full.items()
                 if any(k == n or k.startswith(n + ".") for n in trainable_names) or "bottleneck" in k}
        torch.save({"delta": delta, "arch": "transreid", "use_sie": use_sie, "base_ckpt": CKPT,
                    "margin": args.margin, "unfreeze": args.unfreeze,
                    "before": {"same_sim": sm0, "diff_sim": dm0, "gap": gap0},
                    "after": {"same_sim": b["sm"], "diff_sim": b["dm"], "gap": b["sep"]},
                    "trained": "c11_tracklet_selfsup", "val_best_epoch": b["epoch"],
                    "train_tracklets": len(train), "val_tracklets": len(val), "epochs": args.epochs}, wp)

    best = {"sep": gap0, "sm": sm0, "dm": dm0, "epoch": 0}
    for ep in range(args.epochs):
        model.train()
        ep_loss, nb = 0.0, 0
        for _ in range(args.steps):
            ia_l, ib_l, ys = [], [], []
            for _ in range(args.batch):
                pa, pb, y = sample_pair(rng.random() < 0.5)
                ia, ib = load_chw(pa), load_chw(pb)
                if ia is None or ib is None:
                    continue
                ia_l.append(ia); ib_l.append(ib); ys.append(y)
            if not ia_l:
                continue
            xa = torch.tensor(np.stack(ia_l)).float().to(dev)
            xb = torch.tensor(np.stack(ib_l)).float().to(dev)
            y = torch.tensor(ys, dtype=torch.float32).to(dev)
            opt.zero_grad()
            fa = train_feat(model, xa, use_sie, n_frozen)
            fb = train_feat(model, xb, use_sie, n_frozen)
            dist = 1.0 - (fa * fb).sum(dim=1)
            loss = (y * dist.pow(2) + (1.0 - y) * torch.clamp(args.margin - dist, min=0.0).pow(2)).mean()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable, 5.0)
            opt.step()
            ep_loss += float(loss.item()); nb += 1
        sched.step()
        sm, dm, sep = val_sep(model, vpos, vneg, dev, use_sie)
        improved = sep > best["sep"]
        print(f"[transreid] epoch {ep+1}/{args.epochs} loss={ep_loss/max(1,nb):.4f}  "
              f"VAL sep={sep:+.4f} (same={sm:.3f} diff={dm:.3f}){'  <- best (saved)' if improved else ''}", flush=True)
        if improved:
            best = {"sep": sep, "sm": sm, "dm": dm, "epoch": ep + 1}
            save_ckpt(best)
        _job(status="running", stage=f"epoch {ep+1}/{args.epochs}",
             epoch=ep + 1, epochs=args.epochs, sep=round(sep, 4), best_sep=round(best["sep"], 4))

    if best["epoch"] == 0:
        print("[transreid] WARNING: VAL never improved over baseline", flush=True)
        best.update({"sm": sm, "dm": dm, "epoch": args.epochs})
        save_ckpt(best)
    print(f"[transreid] BEST VAL sep {gap0:+.4f} -> {best['sep']:+.4f} @epoch {best['epoch']}  saved -> {wp}", flush=True)
    _job(status="done", stage="finished", checkpoint=str(wp), epochs=args.epochs,
         before_sep=round(gap0, 4), best_sep=round(best["sep"], 4), best_epoch=best["epoch"],
         human_same=len(human_pos), human_diff=len(human_neg))
    if jobf:                                   # append to the run HISTORY the dashboard's "past training" section reads
        hist = jobf.parent / "c11_history.jsonl"
        with hist.open("a", encoding="utf-8") as f:
            f.write(json.dumps({"ts": time.strftime("%Y-%m-%d %H:%M"), "checkpoint": wp.name,
                                "epochs": args.epochs, "before_sep": round(gap0, 4),
                                "best_sep": round(best["sep"], 4), "best_epoch": best["epoch"],
                                "human_same": len(human_pos), "human_diff": len(human_neg)}) + "\n")


if __name__ == "__main__":
    main()
