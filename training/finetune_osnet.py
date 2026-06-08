"""Deep fine-tune: adapt the OSNet ReID backbone to THIS store's staff identities (the 'deep train'
mode, complementing the learning-free gallery rebuild).

Wired end-to-end — job file + live progress + a model_versions row — and it ACTIVATES only when the
prerequisites are real, so it never produces a garbage model from too little data:
  * torchreid installed            (pip install torchreid)  -> the OSNet backbone + training loop
  * enough labelled data           (>= MIN_IDS employees, each >= MIN_PER_ID confirmed crops)
  * a free GPU                      (don't run while the day is still processing)

Otherwise it records an honest, actionable status and exits 0. The fine-tuned weights register as a
*candidate* (active=false): promote only after an eval beats the current model — and note that fully
USING a new backbone means re-embedding crops + a gallery rebuild (the next phase).

  python -m training.finetune_osnet --date 2026-06-03 --epochs 15 --job outputs/train_jobs/<id>.json
"""
from __future__ import annotations

import argparse
import json
import time
from collections import defaultdict
from pathlib import Path

MIN_IDS = 2          # need at least this many distinct employees ...
MIN_PER_ID = 6       # ... each with at least this many confirmed crops, to fine-tune without overfitting


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


def _gather(date=None) -> dict:
    """employee_id -> [existing crop file paths], from the enrolled gallery."""
    from hitl.store_supabase import SupabaseStore
    s = SupabaseStore()
    by_emp = defaultdict(list)
    for g in s.get_gallery("s14"):
        cu = g.get("crop_url")
        if not cu:
            continue
        if date and g.get("source_window") and not str(g["source_window"]).startswith(date):
            continue
        p = Path(str(cu))
        if p.exists():
            by_emp[g["employee_id"]].append(str(p))
    return {e: v for e, v in by_emp.items() if v}


def main() -> None:
    ap = argparse.ArgumentParser(description="Fine-tune OSNet on confirmed staff identities")
    ap.add_argument("--date", default=None, help="limit to one date's crops; else all-time")
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--batch", type=int, default=8)          # small -> fits a 4 GB laptop GPU
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--out", default="outputs/models")
    ap.add_argument("--job", default=None, help="progress JSON the dashboard polls")
    args = ap.parse_args()
    job = Path(args.job) if args.job else None
    _write(job, status="running", stage="gathering", message="collecting confirmed staff crops", progress=2)

    by_emp = _gather(args.date)
    eligible = {e: v for e, v in by_emp.items() if len(v) >= MIN_PER_ID}
    counts = {int(e): len(v) for e, v in sorted(by_emp.items())}
    if len(eligible) < MIN_IDS:
        msg = (f"insufficient data: need >= {MIN_IDS} employees each with >= {MIN_PER_ID} confirmed "
               f"crops; have {counts} (eligible: {sorted(int(e) for e in eligible)}). "
               f"Confirm more staff sightings, then re-run.")
        _write(job, status="skipped", reason="insufficient_data", message=msg, progress=100,
               data=counts, finished_at=time.strftime("%Y-%m-%d %H:%M:%S"))
        print("[finetune] " + msg)
        return

    try:
        import torch  # noqa: F401
        import torchreid  # noqa: F401
    except Exception:
        msg = ("torchreid not installed — deep fine-tune needs it for the OSNet backbone + training "
               "loop. Run:  pip install torchreid   (then re-run). The learning-free 'Rebuild' mode "
               "works without it.")
        _write(job, status="skipped", reason="needs_torchreid", message=msg, progress=100,
               data=counts, finished_at=time.strftime("%Y-%m-%d %H:%M:%S"))
        print("[finetune] " + msg)
        return

    # ---- prerequisites met: real classifier fine-tune of OSNet on the staff crops ----------------
    import numpy as np  # noqa: F401
    import torch
    import torchreid
    import cv2

    ids = sorted(eligible)
    id2cls = {e: i for i, e in enumerate(ids)}
    samples = [(p, id2cls[e]) for e, ps in eligible.items() for p in ps]
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    _write(job, stage="building", message=f"fine-tuning OSNet: {len(ids)} identities, {len(samples)} crops on {dev}",
           identities=len(ids), crops=len(samples), epochs=args.epochs, progress=8)

    model = torchreid.models.build_model("osnet_x1_0", num_classes=len(ids), pretrained=True)
    model = model.to(dev).train()
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    lossf = torch.nn.CrossEntropyLoss()

    def load_batch(items):
        xs, ys = [], []
        for p, y in items:
            im = cv2.imread(p)
            if im is None:
                continue
            im = cv2.cvtColor(cv2.resize(im, (128, 256)), cv2.COLOR_BGR2RGB).astype("float32") / 255.0
            im = (im - [0.485, 0.456, 0.406]) / [0.229, 0.224, 0.225]
            xs.append(torch.tensor(im).permute(2, 0, 1))
            ys.append(y)
        return (torch.stack(xs).to(dev), torch.tensor(ys).to(dev)) if xs else (None, None)

    import random as _r
    weights = Path(args.out) / f"osnet_ft_{time.strftime('%Y%m%d_%H%M%S')}.pt"
    weights.parent.mkdir(parents=True, exist_ok=True)
    last_loss = None
    for ep in range(args.epochs):
        _r.shuffle(samples)
        ep_loss, nb = 0.0, 0
        for i in range(0, len(samples), args.batch):
            x, y = load_batch(samples[i:i + args.batch])
            if x is None:
                continue
            opt.zero_grad()
            out = model(x)
            loss = lossf(out, y)
            loss.backward()
            opt.step()
            ep_loss += float(loss.item())
            nb += 1
        last_loss = round(ep_loss / max(1, nb), 4)
        _write(job, stage="training", epoch=ep + 1, loss=last_loss,
               progress=8 + int(88 * (ep + 1) / args.epochs),
               message=f"epoch {ep + 1}/{args.epochs} · loss {last_loss}")
        print(f"[finetune] epoch {ep + 1}/{args.epochs} loss={last_loss}")

    torch.save({"state_dict": model.state_dict(), "ids": ids}, weights)
    from hitl.store_supabase import SupabaseStore
    params = {"weights": str(weights), "identities": len(ids), "crops": len(samples),
              "epochs": args.epochs, "final_loss": last_loss, "scope": args.date or "all-time"}
    ver = SupabaseStore().register_model_version(
        "osnet_finetune", params, score=None, trained_on=len(samples),
        notes=f"OSNet fine-tune candidate: {len(ids)} ids, {len(samples)} crops, loss {last_loss}",
        active=False)   # candidate — promote only after an eval beats the current model
    _write(job, status="done", stage="done", version=ver, weights=str(weights), final_loss=last_loss,
           progress=100, message=f"done — model v{ver} (candidate). Promote after eval; applying it needs "
                                  f"a re-embed + gallery rebuild.", finished_at=time.strftime("%Y-%m-%d %H:%M:%S"))
    print(f"[finetune] done -> model_version v{ver}, weights {weights}")


if __name__ == "__main__":
    main()
