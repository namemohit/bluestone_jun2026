"""Fold camera C14 into ALREADY-processed windows WITHOUT re-running C05/C11 detection.

The overnight run skipped C14 (its clip finished prepping after the runner had moved on). Re-running
the whole pipeline would re-number C05/C11 tracks and break every human annotation, which is keyed to
(camera, track). So this adds C14 ONLY: slice C14 for the window, run L1 on it, then re-run the L4
bridge with C11 + C14, and re-push. Existing C05/C11 L1 + all human labels stay intact; C14 shows up
as new interior detections to review.

  python -m batch.foldin_c14 --date 2026-06-03 --windows 1122            # one window (test)
  python -m batch.foldin_c14 --date 2026-06-03                           # every processed window
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path

PY = [sys.executable, "-m"]


def sh(*args) -> None:
    print("  $", " ".join(str(a) for a in args), flush=True)
    subprocess.run(list(args), check=True)


def ffmpeg_slice(clip: str, seek_s: float, dur_s: float, out: Path) -> Path:
    base = ["ffmpeg", "-y", "-loglevel", "error", "-ss", str(seek_s), "-i", str(clip), "-t", str(dur_s)]
    for tail in (["-c", "copy", str(out)], ["-c:v", "libx264", "-preset", "veryfast", "-an", str(out)]):
        r = subprocess.run(base + tail, capture_output=True, text=True)
        if r.returncode == 0 and out.exists() and out.stat().st_size > 0:
            return out
    raise RuntimeError(f"ffmpeg slice failed for {clip}: {r.stderr[-300:]}")


def foldin(window: str, clip: str, clip_start: datetime, fps: float, store: str, out_root: str) -> None:
    out = Path(out_root) / window
    wj = out / "window.json"
    if not wj.exists():
        print(f"[skip] {window}: no window.json"); return
    cfg = json.loads(wj.read_text(encoding="utf-8"))
    start_str = cfg["label"].split(" IST")[0].strip()                  # "2026-06-03 11:22:00" or seconds-less "2026-06-03 09:00"
    win_start = None
    for fmt in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%d %H:%M"):
        try:
            win_start = datetime.strptime(start_str, fmt); break
        except ValueError:
            pass
    if win_start is None:
        print(f"[skip] {window}: unparseable label {start_str!r}"); return
    start_str = win_start.strftime("%Y-%m-%d %H:%M:%S")               # canonical -> l1_humans --start needs seconds
    seek = (win_start - clip_start).total_seconds()
    if seek < 0:
        print(f"[skip] {window}: precedes clip start"); return

    l1c14 = out / "L1_C14"
    if (l1c14 / "tracks.json").exists():
        print(f"[skip-l1] {window}: L1_C14 already present")
    else:
        (out / "slices").mkdir(parents=True, exist_ok=True)
        sl = ffmpeg_slice(clip, seek, 3600, out / "slices" / "C14.ts")
        sh(*PY, "stack.l1_humans", "--video", str(sl), "--start", start_str, "--fps", str(fps), "--out", str(l1c14))

    # keep existing interior dirs, add C14 (dedup), drop any whose tracks.json is missing
    interior = [d for d in list(cfg.get("interior", [])) if (Path(d) / "tracks.json").exists()]
    if not any(Path(d).name == "L1_C14" for d in interior):
        interior.append(str(l1c14))

    gal = out / "gallery.json"                                         # current enrolled-staff gallery
    sparams = {}
    try:
        from hitl.store_supabase import SupabaseStore as _S
        s = _S(root=out_root)
        g = s.get_gallery(store)
        if g:
            gal.write_text(json.dumps(g), encoding="utf-8")
        sparams = (s.active_model() or {}).get("params") or {}         # trained thresholds (match the API /rerun, not the L4 defaults)
    except Exception:
        pass

    l4 = [*PY, "stack.l4_visits", "--l1", cfg["l1"], "--config", cfg["config"], "--out", str(out),
          "--interior", *interior]
    fb = out / "feedback.json"
    if fb.exists():                                                    # respect the human labels already on file
        l4 += ["--feedback", str(fb)]
    if gal.exists():
        l4 += ["--gallery", str(gal)]
        for flag, key in (("--staff-sim", "staff_sim"), ("--staff-auto-sim", "staff_auto_sim"),
                          ("--staff-margin", "staff_margin")):         # trained thresholds; else L4 defaults
            if sparams.get(key):
                l4 += [flag, str(sparams[key])]
    sh(*l4)

    cfg["interior"] = interior
    wj.write_text(json.dumps(cfg, indent=2), encoding="utf-8")
    if store:
        from hitl.store_supabase import SupabaseStore
        s = SupabaseStore(root=out_root)
        s.sync(window, store)
        print(f"[push] {window} synced; metrics={s.metrics(window)}")
    print(f"=== folded C14 into {window} ===")


def main() -> None:
    ap = argparse.ArgumentParser(description="Fold C14 into already-processed windows (C14-only)")
    ap.add_argument("--date", required=True)
    ap.add_argument("--windows", nargs="*", default=[], help="HHMM list, e.g. 1122 1222; default = all")
    ap.add_argument("--clip", default=None)
    ap.add_argument("--clip-start", default=None, help="IST of C14 frame 0 (default <date> 09:00:00)")
    ap.add_argument("--fps", type=float, default=3.0)
    ap.add_argument("--store", default="s14")
    ap.add_argument("--out-root", default="outputs")
    args = ap.parse_args()
    clip = args.clip or f"data/footage_day/{args.date}/C14.ts"
    clip_start = datetime.strptime(args.clip_start or f"{args.date} 09:00:00", "%Y-%m-%d %H:%M:%S")
    wins = ([f"{args.date}_{w}" for w in args.windows] if args.windows
            else sorted(p.parent.name for p in Path(args.out_root).glob(f"{args.date}_*/window.json")))
    print(f"folding C14 into {len(wins)} window(s): {wins}")
    failed = []
    for w in wins:                                  # isolate failures: one bad window must not abort an overnight run
        try:
            foldin(w, clip, clip_start, args.fps, args.store, args.out_root)
        except Exception as e:
            print(f"[FAIL] {w}: {e!r}", flush=True)
            failed.append(w)
    print(f"[done] {len(wins)-len(failed)}/{len(wins)} folded"
          + (f"; FAILED: {failed} (re-run just these)" if failed else "; all OK"))


if __name__ == "__main__":
    main()
