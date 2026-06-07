"""Robust full-day bulk download of recorded NVR footage over ISAPI.

Secures a whole store-day of recorded video off the NVR (which ring-buffers and will eventually
overwrite it) onto local disk and, optionally, GCS. Thinks in REAL IST and converts to the NVR's
OSD clock -- which runs `true_ist_minus_osd_seconds` BEHIND real IST (configs/clock_offset.json) --
only for the segment search.

Per camera, for every segment overlapping the window:
  - resume  : a segment file already present at its exact expected size is left untouched
  - verify  : the downloaded byte count is checked against the size Hikvision encodes in the
              playbackURI (...&size=NNN); a short/failed file is re-fetched up to --retries times
  - manifest: <out>/<date>/download_manifest.json records each segment's real-IST span + bytes
              (+ gs:// path when uploaded), so the processing stage knows each clip's IST start
  - concat  : --concat stitches a camera's segments into one <cam>.ts (ffmpeg stream-copy)
  - gcs     : --gcs-bucket uploads the verified clip(s) to gs://<bucket>/<store>/<date>/<cam>...

Credentials are read from the NVR config and never printed. Downloads run SEQUENTIALLY by design:
this NVR serves recorded media slowly (~9 Mbps) and rejects parallel pulls.

CLI:
  # recon only -- list segments + sizes for every camera, no download
  python -m ingest.bulk_day --config configs/nvr.json --nvr NVR02 --date 2026-06-03 --list

  # the real thing
  python -m ingest.bulk_day --config configs/nvr.json --nvr NVR02 --store s14 \
      --cameras C05 C11 C14 --date 2026-06-03 --ist-start 09:00:00 --ist-end 22:00:00 \
      --out data/footage_day --concat
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import time
from datetime import datetime, timedelta
from pathlib import Path

from ingest.nvr_isapi import (
    IsapiFootageClient, channel_from_camera, load_config, to_search_time, uri_size,
)

CLOCK_OFFSET_PATH = "configs/clock_offset.json"
DEFAULT_OFFSET_S = 20315  # measured 2026-06-07: OSD clock runs 5h38m35s behind real IST


def load_offset(path: str = CLOCK_OFFSET_PATH) -> int:
    try:
        return int(json.loads(Path(path).read_text(encoding="utf-8"))["true_ist_minus_osd_seconds"])
    except Exception:
        return DEFAULT_OFFSET_S


def ist_to_osd(date: str, hms: str, offset_s: int) -> str:
    """'2026-06-03' + '09:00:00' (real IST) -> OSD wall-clock 'YYYY-MM-DDTHH:MM:SS' for the search."""
    ist = datetime.fromisoformat(f"{date}T{hms}")
    return (ist - timedelta(seconds=offset_s)).strftime("%Y-%m-%dT%H:%M:%S")


def osd_to_ist(osd_z: str, offset_s: int) -> str:
    """OSD segment time 'YYYY-MM-DDTHH:MM:SSZ' -> real-IST 'YYYY-MM-DD HH:MM:SS'."""
    raw = osd_z[:-1] if osd_z.endswith("Z") else osd_z
    try:
        dt = datetime.fromisoformat(raw)
    except ValueError:
        return osd_z
    return (dt + timedelta(seconds=offset_s)).strftime("%Y-%m-%d %H:%M:%S")


def human_mb(n) -> str:
    return f"{(n or 0) / 1024 / 1024:.0f} MB"


def fetch_segment(client, seg, out_file: Path, retries: int) -> dict:
    """Download one segment with resume + completeness-verify + retry. Returns a result dict.

    The NVR's advertised &size= is APPROXIMATE -- the delivered byte count differs from it a
    little, and our stream stops at >= size so it can overshoot. So completeness is judged with
    a tolerance (within 2% of advertised), never exact equality: otherwise a perfectly valid
    file fails the check and gets re-downloaded forever. A real truncation (network cut) lands
    far below the floor and still retries.
    """
    expected = uri_size(seg["playback_uri"])
    floor = int(expected * 0.98) if expected else 0  # >= this many bytes == complete

    if out_file.exists():
        have = out_file.stat().st_size
        if have > 0 and have >= floor:
            return {"size_bytes": have, "expected": expected,
                    "status": "skip(resume)" if expected else "skip(exists)"}

    last_err = None
    for _ in range(max(1, retries)):
        try:
            t0 = time.monotonic()
            size = client.download_segment(seg["playback_uri"], out_file)
            dt = max(1e-3, time.monotonic() - t0)
            if expected and size < floor:
                last_err = f"short read {size} < floor {floor} (advertised {expected})"
                continue
            return {"size_bytes": size, "expected": expected, "status": "ok",
                    "mbps": round(size * 8 / dt / 1e6, 1)}
        except Exception as e:  # transient NVR stall / reset -> retry
            last_err = str(e)
    return {"size_bytes": 0, "expected": expected, "status": "failed", "error": last_err}


def concat_segments(seg_files: list[str], out_ts: Path) -> bool:
    """ffmpeg concat (stream-copy, no re-encode) the segment files into one .ts. Returns success."""
    if not seg_files:
        return False
    listing = out_ts.with_suffix(".concat.txt")
    # ffmpeg's concat demuxer resolves list paths relative to the LIST FILE's dir, so write
    # ABSOLUTE posix paths or they double up (data/.../data/...).
    listing.write_text("".join(f"file '{Path(f).resolve().as_posix()}'\n" for f in seg_files),
                       encoding="utf-8")
    try:
        r = subprocess.run(
            ["ffmpeg", "-y", "-f", "concat", "-safe", "0", "-i", str(listing),
             "-c", "copy", str(out_ts)],
            capture_output=True, text=True,
        )
        ok = r.returncode == 0 and out_ts.exists() and out_ts.stat().st_size > 0
        if not ok:
            print(f"    [concat] ffmpeg rc={r.returncode}: {r.stderr[-300:]}")
        return ok
    except FileNotFoundError:
        print("    [concat] ffmpeg not on PATH; leaving segments unmerged")
        return False
    finally:
        listing.unlink(missing_ok=True)


def gcs_upload(bucket: str, local: str, blob_name: str) -> str:
    from google.cloud import storage  # lazy: local-only runs need no GCS lib
    storage.Client().bucket(bucket).blob(blob_name).upload_from_filename(local)
    return f"gs://{bucket}/{blob_name}"


def main() -> None:
    ap = argparse.ArgumentParser(description="Robust full-day NVR bulk download (ISAPI)")
    ap.add_argument("--config", required=True)
    ap.add_argument("--nvr", default="NVR02")
    ap.add_argument("--store", default="s14")
    ap.add_argument("--cameras", nargs="+", default=["C05", "C11", "C14"])
    ap.add_argument("--date", required=True, help="YYYY-MM-DD (real IST)")
    ap.add_argument("--ist-start", default="09:00:00")
    ap.add_argument("--ist-end", default="22:00:00")
    ap.add_argument("--out", default="data/footage_day")
    ap.add_argument("--retries", type=int, default=4)
    ap.add_argument("--segments", nargs="*", type=int, default=None,
                    help="download only these segment numbers (1-based); default = all")
    ap.add_argument("--list", action="store_true", help="search + print segments only (no download)")
    ap.add_argument("--concat", action="store_true", help="stitch each camera's segments into one <cam>.ts")
    ap.add_argument("--gcs-bucket", default=None, help="upload verified clips to gs://bucket/store/date/")
    ap.add_argument("--stream", default="main", choices=["main", "sub"])
    args = ap.parse_args()

    offset = load_offset()
    osd_start = ist_to_osd(args.date, args.ist_start, offset)
    osd_end = ist_to_osd(args.date, args.ist_end, offset)
    osd_start_z, osd_end_z = to_search_time(osd_start), to_search_time(osd_end)

    devices = load_config(args.config)
    dev = devices.get(args.nvr)
    if dev is None:
        raise SystemExit(f"NVR {args.nvr} not in config. Known: {', '.join(devices)}")
    client = IsapiFootageClient(dev, timeout=180.0)

    out_root = Path(args.out) / args.date
    out_root.mkdir(parents=True, exist_ok=True)

    print(f"=== bulk day download  store={args.store}  date={args.date}{'  [LIST ONLY]' if args.list else ''} ===")
    print(f"  real IST {args.ist_start}-{args.ist_end}  ->  OSD {osd_start[11:]}-{osd_end[11:]}  (offset {offset}s)")
    print(f"  cameras: {', '.join(args.cameras)}   out: {out_root}")

    manifest = {"store": args.store, "date": args.date, "nvr": args.nvr,
                "ist_window": [args.ist_start, args.ist_end], "osd_window": [osd_start, osd_end],
                "clock_offset_s": offset, "cameras": {}}
    grand_bytes = 0
    for cam in args.cameras:
        ch = channel_from_camera(cam)
        cam_dir = out_root / cam
        cam_dir.mkdir(parents=True, exist_ok=True)
        print(f"\n--- {cam} (channel {ch}) ---")
        try:
            segs = client.search_segments(ch, osd_start_z, osd_end_z, stream=args.stream)
        except Exception as e:
            print(f"  [search] FAILED: {e}")
            manifest["cameras"][cam] = {"error": str(e)}
            continue
        exp_total = sum(uri_size(s["playback_uri"]) or 0 for s in segs)
        print(f"  {len(segs)} segment(s) overlap the window  (~{human_mb(exp_total)} total)")

        seg_records, ok_files = [], []
        for seg in segs:
            n = seg["segment_number"]
            if args.segments and n not in args.segments:
                continue
            out_file = cam_dir / f"seg_{n:04d}.mp4"
            exp = uri_size(seg["playback_uri"])
            ist_s, ist_e = osd_to_ist(seg["start_time"], offset), osd_to_ist(seg["end_time"], offset)
            head = f"  seg {n}: IST {ist_s[11:]}-{ist_e[11:]}  ~{human_mb(exp) if exp else '??'}"
            if args.list:
                print(head)
                seg_records.append({"segment_number": n, "ist_start": ist_s, "ist_end": ist_e,
                                    "expected": exp})
                continue
            print(head + " ...", end=" ", flush=True)
            res = fetch_segment(client, seg, out_file, args.retries)
            res.update({"segment_number": n, "file": str(out_file), "osd_start": seg["start_time"],
                        "osd_end": seg["end_time"], "ist_start": ist_s, "ist_end": ist_e})
            seg_records.append(res)
            print(f"{res['status']}  {human_mb(res['size_bytes'])}"
                  + (f"  @{res['mbps']} Mbps" if res.get("mbps") else "")
                  + (f"  ERR {res['error']}" if res["status"] == "failed" else ""))
            if res["status"].startswith(("ok", "skip")):
                ok_files.append(str(out_file))
                grand_bytes += res["size_bytes"]

        cam_entry = {"channel": ch, "segments": seg_records}
        if args.concat and ok_files:
            ts_path = out_root / f"{cam}.ts"
            print(f"  [concat] {len(ok_files)} seg(s) -> {ts_path.name} ...", end=" ", flush=True)
            if concat_segments(ok_files, ts_path):
                cam_entry["clip"] = str(ts_path)
                cam_entry["clip_ist_start"] = seg_records[0]["ist_start"]
                print(f"ok  {human_mb(ts_path.stat().st_size)}")
        if args.gcs_bucket and not args.list:
            clip = cam_entry.get("clip")
            try:
                if clip:
                    blob = f"{args.store}/{args.date}/{cam}.ts"
                    print(f"  [gcs] {Path(clip).name} -> gs://{args.gcs_bucket}/{blob} ...", end=" ", flush=True)
                    cam_entry["gcs"] = gcs_upload(args.gcs_bucket, clip, blob)
                    print("ok")
                else:
                    for f in ok_files:
                        gcs_upload(args.gcs_bucket, f, f"{args.store}/{args.date}/{cam}/{Path(f).name}")
                    cam_entry["gcs_prefix"] = f"gs://{args.gcs_bucket}/{args.store}/{args.date}/{cam}/"
            except Exception as e:
                print(f"FAILED: {e}")
        manifest["cameras"][cam] = cam_entry

    if not args.list:
        man_path = out_root / "download_manifest.json"
        man_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        print(f"\n=== done  total {human_mb(grand_bytes)} on disk  manifest -> {man_path} ===")
    else:
        print("\n=== list complete (no files downloaded) ===")


if __name__ == "__main__":
    main()
