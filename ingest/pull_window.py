"""Robust RTSP-playback window pull with auto-resume on stalls.

Realtime RTSP playback from the NVR stalls occasionally (CSeq glitches, blips), leaving
ffmpeg hung. This pulls a [start,end] window in SHORT chunks: a healthy chunk finishes in
~chunk seconds; a stalled one is killed by a watchdog timeout, and we resume from exactly
where we got to (skipping past persistently-bad points). Chunks are concatenated at the end
(mpegts joins cleanly). Reusable for the full-day pull where stalls are inevitable.

  python -m ingest.pull_window --track 1101 --start 20260603T122100Z --end 20260603T125200Z \
      --out data/footage_rtsp/c11_6pm_ist.ts
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
from datetime import datetime, timedelta
from urllib.parse import quote


def parse_z(z):
    return datetime.strptime(z, "%Y%m%dT%H%M%SZ")


def fmt_z(dt):
    return dt.strftime("%Y%m%dT%H%M%SZ")


def probe_dur(path):
    if not os.path.exists(path):
        return 0.0
    try:
        r = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration",
                            "-of", "csv=p=0", path], capture_output=True, text=True, timeout=30)
        return float((r.stdout or "0").strip() or 0)
    except Exception:
        return 0.0


def creds():
    cfg = json.load(open("configs/nvr.json"))
    dev = next(d for d in cfg["devices"] if d["name"] == "NVR 2")
    return cfg["nvr_public_ip"], dev["username"], quote(dev["password"], safe="")


def pull_chunk(track, start_z, end_z, out, host, user, pw, t_secs, watchdog):
    url = f"rtsp://{user}:{pw}@{host}:554/Streaming/tracks/{track}/?starttime={start_z}&endtime={end_z}"
    cmd = ["ffmpeg", "-y", "-rtsp_transport", "tcp", "-i", url, "-an", "-c:v", "copy",
           "-t", str(t_secs), "-f", "mpegts", out, "-hide_banner", "-loglevel", "error"]
    try:
        subprocess.run(cmd, timeout=watchdog)
    except subprocess.TimeoutExpired:
        pass  # hung chunk -> killed; we keep whatever was written and resume


def main():
    ap = argparse.ArgumentParser(description="Robust auto-resume RTSP window pull")
    ap.add_argument("--track", required=True)
    ap.add_argument("--start", required=True, help="YYYYMMDDThhmmssZ (NVR local field)")
    ap.add_argument("--end", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--chunk", type=int, default=180, help="chunk seconds")
    ap.add_argument("--max-retries", type=int, default=25)
    ap.add_argument("--skip-on-stall", type=int, default=4)
    args = ap.parse_args()

    host, user, pw = creds()
    start, end = parse_z(args.start), parse_z(args.end)
    total = (end - start).total_seconds()
    tmp = args.out + ".chunks"
    os.makedirs(tmp, exist_ok=True)
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)

    chunks, got, retries, i = [], 0.0, 0, 0
    while got < total - 2 and retries < args.max_retries:
        cur = start + timedelta(seconds=got)
        this_len = min(args.chunk, total - got)
        ch = os.path.join(tmp, f"part_{i:03d}.ts")
        print(f"[pull {args.track}] chunk {i}: {fmt_z(cur)} (+{this_len/60:.1f}m)  "
              f"have {got/60:.1f}/{total/60:.0f}m", flush=True)
        # endtime = window end so the NVR caps; -t ends the healthy chunk while frames flow
        pull_chunk(args.track, fmt_z(cur), args.end, ch, host, user, pw,
                   int(this_len) + 3, watchdog=int(this_len) + 75)
        d = probe_dur(ch)
        if d > 1.0:
            chunks.append(ch)
            got += d
            i += 1
            retries = 0
            print(f"[pull {args.track}]   +{d/60:.1f}m -> {got/60:.1f}m", flush=True)
        else:
            retries += 1
            got += args.skip_on_stall
            print(f"[pull {args.track}]   stalled; skip {args.skip_on_stall}s (retry {retries})", flush=True)

    if chunks:
        lst = os.path.join(tmp, "list.txt")
        with open(lst, "w", encoding="utf-8") as f:
            for c in chunks:
                f.write(f"file '{os.path.abspath(c)}'\n")
        subprocess.run(["ffmpeg", "-y", "-f", "concat", "-safe", "0", "-i", lst,
                        "-c", "copy", "-f", "mpegts", args.out, "-hide_banner", "-loglevel", "error"])
    print(f"[pull {args.track}] DONE: {got/60:.1f}m across {len(chunks)} chunks -> {args.out}", flush=True)


if __name__ == "__main__":
    main()
