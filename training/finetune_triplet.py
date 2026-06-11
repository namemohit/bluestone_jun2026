"""Fine-tune the OSNet backbone with contrastive loss on human-confirmed same/different pairs.

Reads pairs.jsonl from a portable export (crop paths + labels, no embeddings) and optimises
the embedding space directly: same-pairs pulled closer, different-pairs pushed apart.
This targets the 0.088 cosine-gap between same/different seen on this store's CCTV.

Produces a candidate weights file + benchmark (cosine gap before vs after) + model_version row.
Complement to finetune_osnet.py (identity classifier): that trains "who is this person?",
this trains "are these two crops the same person?" — the harder question for cross-camera.

  python -m training.finetune_triplet --data training_data/2026-06-03 --epochs 30
  python -m training.finetune_triplet --data training_data/all --epochs 50
  python -m training.finetune_triplet --data training_data/2026-06-03 --job outputs/train_jobs/x.json
"""
from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path

MIN_PAIRS = 20      # need at least this many valid pairs (on-disk crops) to fine-tune
MARGIN = 0.5        # contrastive loss margin — push different-pairs at least this far apart
SAME_WEIGHT = 5.0   # up-weight same-pairs to balance the 39 same : 230 different imbalance


def _write(job: Path | None, **kw) -> None:
    if not job:
        return
    job.parent.mkdir(parents=True, exist_ok=True)
    cur = {}
    if job.exists():
        try:
            cur = json.loads(job.read_text(encoding="utf-8"))
        except Exception:
            cur = {}
    cur.update(kw)
    job.write_text(json.dumps(cur, indent=2), encoding="utf-8")


_IMG_CACHE: dict = {}             # decode+preprocess each crop ONCE, reuse across all epochs (training is I/O-bound, ~700 crops × N epochs)


def _load_image(path: str, size=(128, 256)):
    key = (str(path), size)
    if key in _IMG_CACHE:
        return _IMG_CACHE[key]
    import cv2
    im = cv2.imread(str(path))
    if im is None:
        _IMG_CACHE[key] = None
        return None
    im = cv2.cvtColor(cv2.resize(im, size), cv2.COLOR_BGR2RGB).astype("float32") / 255.0
    im = (im - [0.485, 0.456, 0.406]) / [0.229, 0.224, 0.225]
    out = im.astype("float32")    # the list arithmetic above promotes to float64; the model is float32
    _IMG_CACHE[key] = out
    return out


def _embed_batch(model, paths, dev, size=(128, 256)):
    """Embed a list of crop paths -> (N, 512) tensor. Skips None loads (returns None in that slot)."""
    import torch
    imgs, valid_idx = [], []
    for i, p in enumerate(paths):
        im = _load_image(p, size)
        if im is not None:
            imgs.append(im)
            valid_idx.append(i)
    if not imgs:
        return None, []
    import numpy as np
    x = torch.tensor(np.stack(imgs)).permute(0, 3, 1, 2).to(dev)
    with torch.no_grad():
        feats = model(x)
    feats = feats / (feats.norm(dim=1, keepdim=True) + 1e-9)
    return feats, valid_idx


def _cosine_gap(model, pairs, dev) -> tuple[float, float, float]:
    """Compute mean cosine sim for same-pairs and diff-pairs. Returns (same_mean, diff_mean, gap)."""
    import torch
    import numpy as np
    same_sims, diff_sims = [], []
    model.eval()
    with torch.no_grad():
        for p in pairs:
            ia = _load_image(p["crop_a_abs"])
            ib = _load_image(p["crop_b_abs"])
            if ia is None or ib is None:
                continue
            xa = torch.tensor(ia).permute(2, 0, 1).unsqueeze(0).to(dev)
            xb = torch.tensor(ib).permute(2, 0, 1).unsqueeze(0).to(dev)
            fa = model(xa); fa = fa / (fa.norm() + 1e-9)
            fb = model(xb); fb = fb / (fb.norm() + 1e-9)
            s = float(torch.dot(fa.ravel(), fb.ravel()))
            (same_sims if p["label"] == "same" else diff_sims).append(s)
    sm = float(np.mean(same_sims)) if same_sims else 0.0
    dm = float(np.mean(diff_sims)) if diff_sims else 0.0
    return sm, dm, sm - dm


def main() -> None:
    ap = argparse.ArgumentParser(description="Contrastive fine-tune OSNet on human pairs")
    ap.add_argument("--data", default="training_data/2026-06-03",
                    help="path to exported dataset dir (contains pairs.jsonl + pairs/)")
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--margin", type=float, default=MARGIN)
    ap.add_argument("--out", default="outputs/models")
    ap.add_argument("--job", default=None, help="progress JSON the dashboard polls")
    args = ap.parse_args()

    job = Path(args.job) if args.job else None
    data_dir = Path(args.data)
    pairs_file = data_dir / "pairs.jsonl"

    _write(job, status="running", stage="loading", progress=2,
           message="loading pairs from export")

    if not pairs_file.exists():
        msg = f"pairs.jsonl not found at {pairs_file} — run training.export_dataset first"
        _write(job, status="skipped", reason="no_pairs", message=msg, progress=100,
               finished_at=time.strftime("%Y-%m-%d %H:%M:%S"))
        print("[triplet] " + msg)
        return

    raw_pairs = []
    for line in pairs_file.read_text(encoding="utf-8").splitlines():
        try:
            raw_pairs.append(json.loads(line))
        except Exception:
            continue

    # Resolve crop paths relative to data_dir
    pairs = []
    for p in raw_pairs:
        ca = data_dir / p["crop_a"]
        cb = data_dir / p["crop_b"]
        if ca.exists() and cb.exists():
            pairs.append({**p, "crop_a_abs": str(ca), "crop_b_abs": str(cb)})

    n_same = sum(1 for p in pairs if p["label"] == "same")
    n_diff = sum(1 for p in pairs if p["label"] == "different")
    print(f"[triplet] loaded {len(pairs)} valid pairs ({n_same} same / {n_diff} diff) "
          f"from {pairs_file} ({len(raw_pairs)-len(pairs)} skipped — crop not on disk)")

    if len(pairs) < MIN_PAIRS:
        msg = (f"insufficient pairs: need >= {MIN_PAIRS} with crops on disk; "
               f"have {len(pairs)}. Run export_dataset first.")
        _write(job, status="skipped", reason="insufficient_pairs", message=msg, progress=100,
               finished_at=time.strftime("%Y-%m-%d %H:%M:%S"))
        print("[triplet] " + msg)
        return

    try:
        import torch
        from boxmot.reid.backbones.osnet import osnet_x1_0
    except Exception as e:
        msg = f"OSNet backbone unavailable ({type(e).__name__}) — boxmot must be installed."
        _write(job, status="skipped", reason="no_backbone", message=msg, progress=100,
               finished_at=time.strftime("%Y-%m-%d %H:%M:%S"))
        print("[triplet] " + msg)
        return

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    _write(job, stage="baseline", message=f"measuring baseline gap on {dev}", progress=8,
           n_same=n_same, n_diff=n_diff)

    # Use OSNet as feature extractor (num_classes=0 returns raw 512-d features before classifier)
    model = osnet_x1_0(num_classes=0, pretrained=True, loss="triplet")
    model = model.to(dev)

    # ---- baseline metrics BEFORE training ----
    sm0, dm0, gap0 = _cosine_gap(model, pairs, dev)
    print(f"[triplet] BEFORE: same={sm0:.4f}  diff={dm0:.4f}  gap={gap0:+.4f}")
    _write(job, before={"same_sim": round(sm0, 4), "diff_sim": round(dm0, 4), "gap": round(gap0, 4)},
           stage="training", message=f"baseline gap={gap0:+.4f}; starting training", progress=12)

    opt = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)

    model.train()
    for ep in range(args.epochs):
        random.shuffle(pairs)
        ep_loss, nb = 0.0, 0

        for i in range(0, len(pairs), args.batch):
            batch = pairs[i:i + args.batch]
            import numpy as np
            imgs_a, imgs_b, labels, weights = [], [], [], []
            for p in batch:
                ia = _load_image(p["crop_a_abs"])
                ib = _load_image(p["crop_b_abs"])
                if ia is None or ib is None:
                    continue
                imgs_a.append(ia)
                imgs_b.append(ib)
                same = p["label"] == "same"
                labels.append(1.0 if same else 0.0)
                weights.append(SAME_WEIGHT if same else 1.0)

            if not imgs_a:
                continue

            xa = torch.tensor(np.stack(imgs_a)).permute(0, 3, 1, 2).float().to(dev)
            xb = torch.tensor(np.stack(imgs_b)).permute(0, 3, 1, 2).float().to(dev)
            y  = torch.tensor(labels, dtype=torch.float32).to(dev)
            w  = torch.tensor(weights, dtype=torch.float32).to(dev)

            opt.zero_grad()
            def _feat(o):                              # OSNet(loss='triplet') returns (logits, features) in train mode, a tensor in eval
                o = o[-1] if isinstance(o, (tuple, list)) else o
                return o / (o.norm(dim=1, keepdim=True) + 1e-9)
            fa = _feat(model(xa))
            fb = _feat(model(xb))

            # Contrastive loss: same -> minimize distance; different -> push beyond margin
            dist = 1.0 - (fa * fb).sum(dim=1)        # cosine distance in [0, 2]
            loss_same = y * dist.pow(2)
            loss_diff = (1.0 - y) * torch.clamp(args.margin - dist, min=0.0).pow(2)
            loss = (w * (loss_same + loss_diff)).mean()

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            ep_loss += float(loss.item())
            nb += 1

        scheduler.step()
        last_loss = round(ep_loss / max(1, nb), 5)
        prog = 12 + int(83 * (ep + 1) / args.epochs)
        _write(job, stage="training", epoch=ep + 1, loss=last_loss, progress=prog,
               message=f"epoch {ep + 1}/{args.epochs} · loss {last_loss}")
        print(f"[triplet] epoch {ep + 1}/{args.epochs}  loss={last_loss}")

    # ---- metrics AFTER training ----
    sm1, dm1, gap1 = _cosine_gap(model, pairs, dev)
    improvement = round(gap1 - gap0, 4)
    print(f"[triplet] AFTER:  same={sm1:.4f}  diff={dm1:.4f}  gap={gap1:+.4f}  "
          f"improvement={improvement:+.4f}")

    # Save weights
    ts = time.strftime("%Y%m%d_%H%M%S")
    weights_path = Path(args.out) / f"osnet_triplet_{ts}.pt"
    weights_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": model.state_dict(), "arch": "osnet_x1_0",
                "loss": "contrastive", "margin": args.margin,
                "before": {"same_sim": sm0, "diff_sim": dm0, "gap": gap0},
                "after":  {"same_sim": sm1, "diff_sim": dm1, "gap": gap1},
                "pairs": len(pairs), "epochs": args.epochs},
               weights_path)

    try:
        from hitl.store_supabase import SupabaseStore
        params = {"weights": str(weights_path), "loss": "contrastive",
                  "pairs": len(pairs), "n_same": n_same, "n_diff": n_diff,
                  "margin": args.margin, "epochs": args.epochs, "lr": args.lr,
                  "gap_before": round(gap0, 4), "gap_after": round(gap1, 4),
                  "gap_improvement": improvement, "data": str(data_dir)}
        ver = SupabaseStore().register_model_version(
            "osnet_triplet", params, score=round((gap1 + 1) / 2, 4),
            trained_on=len(pairs),
            notes=f"Contrastive fine-tune: gap {gap0:+.4f} -> {gap1:+.4f} ({improvement:+.4f})",
            active=False)   # candidate — promote after eval; applying needs re-embed + gallery rebuild
        ver_str = str(ver)
    except Exception as e:
        ver_str = f"(not registered: {e})"

    _write(job, status="done", stage="done", progress=100,
           version=ver_str, weights=str(weights_path),
           before={"same_sim": round(sm0, 4), "diff_sim": round(dm0, 4), "gap": round(gap0, 4)},
           after={"same_sim": round(sm1, 4), "diff_sim": round(dm1, 4), "gap": round(gap1, 4)},
           gap_improvement=improvement,
           message=f"done — gap {gap0:+.4f} -> {gap1:+.4f} ({improvement:+.4f}). "
                   f"Model v{ver_str} (candidate). Applying needs re-embed + gallery rebuild.",
           finished_at=time.strftime("%Y-%m-%d %H:%M:%S"))
    print(f"[triplet] saved -> {weights_path} | model_version {ver_str}")


if __name__ == "__main__":
    main()
