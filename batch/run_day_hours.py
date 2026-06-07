"""Drive the whole store day hour by hour: 9:00 → 22:00, each hour through run_hour, each pushed
to the HITL console as it finishes.

Sequential (one local GPU). Hour 9 lands in /review while hour 10 is still processing, so you
review as you go. Independent per hour, so a crash mid-day just skips that hour; re-run later.
A labelled hour re-runs for free (cached embeddings).

  python -m batch.run_day_hours --date 2026-06-03 --clips-dir data/footage_day/2026-06-03 \
      --interior C11 C14 --fps 3
"""
from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

PY = [sys.executable, "-m"]


def main() -> None:
    ap = argparse.ArgumentParser(description="Process a whole store-day hour by hour -> HITL console")
    ap.add_argument("--date", required=True)
    ap.add_argument("--clips-dir", default="data/footage_day/2026-06-03",
                    help="dir holding the per-camera day clips <cam>.ts (all starting at --day-start)")
    ap.add_argument("--day-start", default="09:00:00", help="IST that the <cam>.ts clips begin at")
    ap.add_argument("--from-hour", type=int, default=9)
    ap.add_argument("--to-hour", type=int, default=21, help="last hour processed = HH:00–(HH+1):00")
    ap.add_argument("--entry", default="C05")
    ap.add_argument("--interior", nargs="*", default=["C11", "C14"])
    ap.add_argument("--zones", default="configs/c05_zones.json")
    ap.add_argument("--fps", type=float, default=3.0)
    ap.add_argument("--minutes", type=float, default=60)
    ap.add_argument("--store", default="s14")
    ap.add_argument("--resume", action="store_true",
                    help="skip hours whose outputs/<date>_<HH>/visits.json already exists")
    args = ap.parse_args()

    cd = Path(args.clips_dir)
    entry_clip = cd / f"{args.entry}.ts"
    if not entry_clip.exists():
        raise SystemExit(f"entry clip missing: {entry_clip} (concat the segments first)")
    interior = [(n, cd / f"{n}.ts") for n in args.interior]
    present = [(n, p) for n, p in interior if p.exists()]
    missing = [n for n, p in interior if not p.exists()]
    if missing:
        print(f"[day] interior NOT yet available, skipping for now: {missing} "
              f"(re-run those hours later to fold them in)")
    clip_start = f"{args.date} {args.day_start}"

    hours = list(range(args.from_hour, args.to_hour + 1))
    print(f"=== DAY {args.date}: hours {hours[0]:02d}:00–{hours[-1]+1:02d}:00  "
          f"entry={args.entry}  interior={[n for n,_ in present]}  fps={args.fps:g} ===")
    t0 = time.monotonic()
    ok, failed = [], []
    for hour in hours:
        if args.resume and Path(f"outputs/{args.date}_{hour:02d}/visits.json").exists():
            print(f"\n########## HOUR {hour:02d}:00 — already processed, skipping (--resume) ##########", flush=True)
            ok.append(hour)
            continue
        print(f"\n########## HOUR {hour:02d}:00–{hour+1:02d}:00  "
              f"(elapsed {int((time.monotonic()-t0)//60)}m) ##########", flush=True)
        cmd = [*PY, "batch.run_hour", "--date", args.date, "--hour", str(hour),
               "--minutes", str(args.minutes), "--store", args.store,
               "--entry-clip", str(entry_clip), "--clip-start", clip_start,
               "--zones", args.zones, "--fps", str(args.fps), "--push"]
        if present:
            cmd += ["--interior-clips", *[str(p) for _, p in present],
                    "--interior-names", *[n for n, _ in present]]
        rc = subprocess.run(cmd).returncode
        (ok if rc == 0 else failed).append(hour)

    mins = int((time.monotonic() - t0) // 60)
    print(f"\n=== DAY COMPLETE in {mins}m — ok: {ok}  failed: {failed} ===")


if __name__ == "__main__":
    main()
