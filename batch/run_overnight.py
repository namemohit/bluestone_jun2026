"""Overnight driver: tile the OPEN store-day into back-to-back 60-min windows from a start IST,
each processed through run_hour and pushed to the HITL console as it finishes.

Tiling at the store-open offset (default 11:22) means windows abut without overlap
(11:22-12:22, 12:22-13:22, ...), so summing windows never double-counts. Resumable: a window
whose outputs/<key>/visits.json already exists is skipped, so a crash (or a deliberate restart)
picks up exactly where it stopped — already-done hours are free.

  python -m batch.run_overnight --date 2026-06-03 --start 11:22:00 --end 22:00:00 \
      --interior C11 --fps 3
"""
from __future__ import annotations

import argparse
import subprocess
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path

PY = [sys.executable, "-m"]


def main() -> None:
    ap = argparse.ArgumentParser(description="Tile the open store-day into 60-min windows -> HITL console")
    ap.add_argument("--date", required=True)
    ap.add_argument("--start", default="11:22:00", help="first window start IST (store open)")
    ap.add_argument("--end", default="22:00:00", help="stop once a window would start at/after this IST")
    ap.add_argument("--minutes", type=float, default=60)
    ap.add_argument("--clips-dir", default="data/footage_day/2026-06-03")
    ap.add_argument("--day-start", default="09:00:00", help="IST the <cam>.ts clips begin at")
    ap.add_argument("--entry", default="C05")
    ap.add_argument("--interior", nargs="*", default=["C11"])
    ap.add_argument("--zones", default="configs/c05_zones.json")
    ap.add_argument("--fps", type=float, default=3.0)
    ap.add_argument("--store", default="s14")
    args = ap.parse_args()

    cd = Path(args.clips_dir)
    entry_clip = cd / f"{args.entry}.ts"
    if not entry_clip.exists():
        raise SystemExit(f"entry clip missing: {entry_clip}")
    interior = [(n, cd / f"{n}.ts") for n in args.interior]
    present = [(n, p) for n, p in interior if p.exists()]
    missing = [n for n, p in interior if not p.exists()]
    if missing:
        print(f"[overnight] interior not available, skipping: {missing} (fold in later)")
    clip_start = f"{args.date} {args.day_start}"

    t = datetime.strptime(f"{args.date} {args.start}", "%Y-%m-%d %H:%M:%S")
    t_end = datetime.strptime(f"{args.date} {args.end}", "%Y-%m-%d %H:%M:%S")
    print(f"=== OVERNIGHT {args.date}: {args.start} -> {args.end}  entry={args.entry}  "
          f"interior={[n for n, _ in present]}  fps={args.fps:g}  step={args.minutes:g}m ===", flush=True)
    t0 = time.monotonic()
    ok, failed = [], []
    while t < t_end:
        ws = t.strftime("%Y-%m-%d %H:%M:%S")
        key = t.strftime("%Y-%m-%d_%H%M")
        if (Path("outputs") / key / "visits.json").exists():
            print(f"\n########## {key} already processed — skipping (resume) ##########", flush=True)
            ok.append(key)
            t += timedelta(minutes=args.minutes)
            continue
        print(f"\n########## WINDOW {key}  ({ws})  elapsed {int((time.monotonic()-t0)//60)}m ##########", flush=True)
        cmd = [*PY, "batch.run_hour", "--date", args.date, "--window-start", ws,
               "--minutes", str(args.minutes), "--store", args.store,
               "--entry-clip", str(entry_clip), "--clip-start", clip_start,
               "--zones", args.zones, "--fps", str(args.fps), "--push"]
        if present:
            cmd += ["--interior-clips", *[str(p) for _, p in present],
                    "--interior-names", *[n for n, _ in present]]
        rc = subprocess.run(cmd).returncode
        (ok if rc == 0 else failed).append(key)
        t += timedelta(minutes=args.minutes)

    mins = int((time.monotonic() - t0) // 60)
    print(f"\n=== OVERNIGHT COMPLETE in {mins}m — ok: {ok}  failed: {failed} ===", flush=True)


if __name__ == "__main__":
    main()
