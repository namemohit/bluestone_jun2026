"""Hour-runner — process ONE hour of the store day and push it into the HITL console.

The unit of the hour-by-hour loop. For each camera it cuts the hour slice (ffmpeg stream-copy),
runs L1 (GPU, once), then L4 with the OSNet interior bridge + any human feedback already on file,
and (optionally) pushes the result to Supabase so the hour shows up in /review. Re-running a
labelled hour is free (embeddings cached) — that's the learning loop.

Clips are per-camera files that begin at --clip-start (e.g. the concatenated <cam>.ts, or a single
segment). The runner seeks (window_start − clip_start) into each.

  python -m batch.run_hour --date 2026-06-03 --hour 9 --minutes 60 --store s14 \
      --entry-clip data/footage_day/2026-06-03/C05.ts \
      --interior-clips data/footage_day/2026-06-03/C11.ts data/footage_day/2026-06-03/C14.ts \
      --interior-names C11 C14 --clip-start "2026-06-03 09:00:00" --push
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path

PY = [sys.executable, "-m"]


def sh(*args) -> None:
    print("  $", " ".join(str(a) for a in args), flush=True)
    subprocess.run(list(args), check=True)


def ffmpeg_slice(clip: str, seek_s: float, dur_s: float, out: Path) -> Path:
    """Cut [seek, seek+dur] out of a clip. Stream-copy first (fast); re-encode only if that fails."""
    base = ["ffmpeg", "-y", "-ss", str(seek_s), "-i", str(clip), "-t", str(dur_s)]
    for tail in (["-c", "copy", str(out)],
                 ["-c:v", "libx264", "-preset", "veryfast", "-an", str(out)]):
        r = subprocess.run(base + tail, capture_output=True, text=True)
        if r.returncode == 0 and out.exists() and out.stat().st_size > 0:
            return out
    raise RuntimeError(f"ffmpeg slice failed for {clip}: {r.stderr[-300:]}")


def main() -> None:
    ap = argparse.ArgumentParser(description="Process one store-hour -> L1..L4 -> HITL store")
    ap.add_argument("--date", required=True, help="YYYY-MM-DD")
    ap.add_argument("--hour", type=int, default=None, help="IST hour (9..21)")
    ap.add_argument("--window-start", default=None,
                    help="explicit IST start 'YYYY-MM-DD HH:MM:SS' (overrides --hour; lets the "
                         "window begin at the real store-opening moment, not an hour boundary)")
    ap.add_argument("--minutes", type=float, default=60)
    ap.add_argument("--store", default="s14")
    ap.add_argument("--entry-clip", required=True)
    ap.add_argument("--interior-clips", nargs="*", default=[])
    ap.add_argument("--interior-names", nargs="*", default=[])
    ap.add_argument("--clip-start", required=True, help="IST of each clip's frame 0, 'YYYY-MM-DD HH:MM:SS'")
    ap.add_argument("--zones", default="configs/c05_zones.json")
    ap.add_argument("--fps", type=float, default=5.0)
    ap.add_argument("--out-root", default="outputs")
    ap.add_argument("--push", action="store_true", help="push result to the HITL store (Supabase)")
    args = ap.parse_args()

    if args.window_start:
        win_start = datetime.strptime(args.window_start, "%Y-%m-%d %H:%M:%S")
        window = f"{args.date}_{win_start.strftime('%H%M')}"
    elif args.hour is not None:
        win_start = datetime.strptime(f"{args.date} {args.hour:02d}:00:00", "%Y-%m-%d %H:%M:%S")
        window = f"{args.date}_{args.hour:02d}"
    else:
        raise SystemExit("need --hour or --window-start")
    clip_start = datetime.strptime(args.clip_start, "%Y-%m-%d %H:%M:%S")
    seek = (win_start - clip_start).total_seconds()
    if seek < 0:
        raise SystemExit(f"window {win_start} precedes clip start {clip_start}")
    win_start_str = win_start.strftime("%Y-%m-%d %H:%M:%S")
    dur = args.minutes * 60
    out = Path(args.out_root) / window
    (out / "slices").mkdir(parents=True, exist_ok=True)

    print(f"=== hour-runner {window}  IST {win_start_str} +{args.minutes:g}min  seek={seek:g}s  fps={args.fps:g} ===")

    def l1(clip: str, name: str) -> str:
        sl = ffmpeg_slice(clip, seek, dur, out / "slices" / f"{name}.ts")
        d = out / f"L1_{name}"
        sh(*PY, "stack.l1_humans", "--video", str(sl), "--start", win_start_str,
           "--fps", str(args.fps), "--out", str(d))
        return str(d)

    l1_entry = l1(args.entry_clip, "entry")
    names = args.interior_names or [f"int{i+1}" for i in range(len(args.interior_clips))]
    interior_dirs = [l1(c, n) for c, n in zip(args.interior_clips, names)]

    fb = out / "feedback.json"
    gal = out / "gallery.json"
    try:  # enrolled-staff gallery -> L4 auto-recognises known staff (needs DB; skip if local-only)
        from hitl.store_supabase import SupabaseStore as _S
        _g = _S(root=args.out_root).get_gallery(args.store)
        if _g:
            gal.write_text(json.dumps(_g), encoding="utf-8")
            print(f"  gallery: {len(_g)} staff embeddings -> auto-recognition ON")
    except Exception:
        pass
    l4 = [*PY, "stack.l4_visits", "--l1", l1_entry, "--config", args.zones, "--out", str(out)]
    if interior_dirs:
        l4 += ["--interior", *interior_dirs]
    if fb.exists():
        l4 += ["--feedback", str(fb)]
    if interior_dirs and gal.exists():
        l4 += ["--gallery", str(gal)]
    sh(*l4)

    # window.json lets the dashboard re-run L4 on this hour when a human labels it
    (out / "window.json").write_text(json.dumps(
        {"window": window, "store": args.store, "label": f"{win_start_str} IST +{args.minutes:g}min",
         "l1": l1_entry, "config": args.zones, "interior": interior_dirs}, indent=2), encoding="utf-8")

    if args.push:
        from hitl.store_supabase import SupabaseStore
        s = SupabaseStore(root=args.out_root)
        s.sync(window, args.store)
        n = s.upload_crops(window)  # crops -> Storage so the cloud dashboard can show them
        print(f"[push] {window} -> Supabase  crops={n}  metrics={s.metrics(window)}")
    print(f"=== done: {window} -> {out} ===")


if __name__ == "__main__":
    main()
