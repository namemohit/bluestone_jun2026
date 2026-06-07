"""Cloud Run Job entrypoint: process a store's day footage through the layered stack.

Pulls per-camera day clips from GCS, runs L1 -> L2/L3 (entry) + L1 (interior) -> L4 (visits
with the OSNet interior bridge), and writes the day report + crops back to GCS. Built for a
Cloud Run GPU Job (NVIDIA L4): scale-to-zero, pay only while it runs.

  python -m batch.run_stack --bucket bluestone-clips --store s14 --date 2026-06-03 \
      --manifest configs/store_manifest.json

Local dry-run (clips already on disk, no GCS):
  python -m batch.run_stack --local data/footage_rtsp --store s14 --date 2026-06-03 \
      --manifest configs/store_manifest.json --out outputs/day_report.json
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile

PY = [sys.executable, "-m"]


def sh(*args):
    print("  $", " ".join(str(a) for a in args), flush=True)
    subprocess.run(list(args), check=True)


# ---- GCS helpers (lazy import so local runs need no google-cloud-storage) ----
def gcs_client():
    from google.cloud import storage  # noqa
    return storage.Client()


def gcs_download(bucket, prefix, dest_dir):
    cl = gcs_client()
    os.makedirs(dest_dir, exist_ok=True)
    got = []
    for blob in cl.list_blobs(bucket, prefix=prefix):
        if blob.name.endswith("/"):
            continue
        local = os.path.join(dest_dir, os.path.basename(blob.name))
        blob.download_to_filename(local)
        got.append(local)
    return got


def gcs_upload_dir(bucket, src_dir, prefix):
    cl = gcs_client()
    b = cl.bucket(bucket)
    for root, _, files in os.walk(src_dir):
        for f in files:
            lp = os.path.join(root, f)
            rel = os.path.relpath(lp, src_dir).replace("\\", "/")
            b.blob(f"{prefix}/{rel}").upload_from_filename(lp)


def main():
    ap = argparse.ArgumentParser(description="Day footage -> stack -> day report (Cloud Run Job)")
    ap.add_argument("--manifest", required=True, help="store manifest JSON (cameras, zones, fps)")
    ap.add_argument("--store", required=True)
    ap.add_argument("--date", required=True, help="YYYY-MM-DD")
    ap.add_argument("--bucket", help="GCS bucket; clips read from gs://bucket/store/date/<cam>.ts")
    ap.add_argument("--local", help="local footage dir instead of GCS (dry-run)")
    ap.add_argument("--workdir", default=None)
    ap.add_argument("--out", default="/tmp/day_report.json")
    args = ap.parse_args()

    man = json.load(open(args.manifest))
    fps = str(man.get("fps", 5))
    ist_start = man["ist_start"]
    entry, interiors = man["entry"], man["interior"]
    work = args.workdir or tempfile.mkdtemp(prefix="day_")
    clips, outs = os.path.join(work, "clips"), os.path.join(work, "out")
    os.makedirs(clips, exist_ok=True)
    os.makedirs(outs, exist_ok=True)

    # 1) fetch the day's per-camera clips
    cams = [entry["cam"]] + interiors
    if args.bucket:
        print(f"[run_stack] GCS gs://{args.bucket}/{args.store}/{args.date}/ -> {clips}")
        gcs_download(args.bucket, f"{args.store}/{args.date}/", clips)
    else:
        for c in cams:  # local dry-run: symlink/copy expected names
            src = os.path.join(args.local, f"{c}_6pm_ist.ts")
            if os.path.exists(src):
                import shutil
                shutil.copy(src, os.path.join(clips, f"{c}.ts"))

    def clip(c):
        return os.path.join(clips, f"{c}.ts")

    def l1(cam):
        d = os.path.join(outs, f"L1_{cam}")
        sh(*PY, "stack.l1_humans", "--video", clip(cam), "--start", ist_start, "--fps", fps, "--out", d)
        return d

    # 2) entry camera: L1 -> entries (L2) + exits (L3)
    l1_entry = l1(entry["cam"])
    sh(*PY, "stack.l2_entries", "--l1", l1_entry, "--config", entry["zones"],
       "--direction", "in", "--out", os.path.join(outs, "L2_entries"))
    sh(*PY, "stack.l2_entries", "--l1", l1_entry, "--config", entry["zones"],
       "--direction", "out", "--out", os.path.join(outs, "L3_exits"))

    # 3) interior cameras: L1 (for the bridge)
    interior_dirs = [l1(c) for c in interiors]

    # 4) L4 visits with the OSNet interior bridge
    l4_out = os.path.join(outs, "L4_visits")
    sh(*PY, "stack.l4_visits", "--l1", l1_entry, "--config", entry["zones"],
       "--interior", *interior_dirs, "--out", l4_out)

    # 5) assemble report + ship results to GCS
    report = {"store": args.store, "date": args.date, "cameras": cams,
              "entries_csv": os.path.join(outs, "L2_entries", "in_events.csv"),
              "exits_csv": os.path.join(outs, "L3_exits", "out_events.csv"),
              "visits_csv": os.path.join(l4_out, "visits.csv"),
              "occupancy_csv": os.path.join(l4_out, "occupancy.csv")}
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    print(f"[run_stack] report -> {args.out}")

    if args.bucket:
        gcs_upload_dir(args.bucket, outs, f"{args.store}/{args.date}/outputs")
        print(f"[run_stack] outputs -> gs://{args.bucket}/{args.store}/{args.date}/outputs/")


if __name__ == "__main__":
    main()
