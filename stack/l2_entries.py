"""L2 - filter L1's raw human-tracks down to real ENTRIES (or EXITS, with --direction out).

Pure post-filter on L1's tracks.json (no detector re-run). Three stages:
  1. street-mask : drop tracks whose feet are mostly in the through-glass street region
  2. line-cross  : keep tracks that cross the door line in the wanted direction (in/out)
  3. de-dup      : merge crossings within a time window -> one person split into many tracks = 1

Outputs (under --out): entries.csv, crops/, entries_contact_sheet.jpg, and the count.
Mirror usage for L4 (exits on C14):
  python -m stack.l2_entries --l1 outputs/L3_c14 --config configs/c14_zones.json \
      --direction out --out outputs/L4_c14
"""
from __future__ import annotations

import argparse
import csv
import json
import os
from datetime import datetime, timezone, timedelta

import cv2

from vision.geometry import side, segments_intersect, inside_sign_from_label
from stack.l1_humans import build_contact_sheet

IST = timezone(timedelta(hours=5, minutes=30))


def ist(ts: float) -> str:
    return datetime.fromtimestamp(ts, IST).strftime("%H:%M:%S")


def point_in_poly(x, y, poly) -> bool:
    inside = False
    n = len(poly)
    j = n - 1
    for i in range(n):
        xi, yi = poly[i]
        xj, yj = poly[j]
        if ((yi > y) != (yj > y)) and (x < (xj - xi) * (y - yi) / (yj - yi + 1e-12) + xi):
            inside = not inside
        j = i
    return inside


def main() -> None:
    ap = argparse.ArgumentParser(description="L2: filter L1 tracks -> real entries/exits")
    ap.add_argument("--l1", default="outputs/L1_3min", help="L1 output dir (with tracks.json)")
    ap.add_argument("--config", default="configs/c05_zones.json")
    ap.add_argument("--direction", default="in", choices=["in", "out"])
    ap.add_argument("--out", default="outputs/L2_3min")
    args = ap.parse_args()

    L1 = json.load(open(os.path.join(args.l1, "tracks.json")))
    cfg = json.load(open(args.config))
    p1, p2 = tuple(cfg["entry_line"][0]), tuple(cfg["entry_line"][1])
    inside_sign = inside_sign_from_label(cfg.get("inside", "right"))
    street = cfg.get("street_mask", [])
    street_frac = cfg.get("street_drop_frac", 0.5)
    dedup_w = cfg.get("dedup_window_s", 18)
    min_bbox_h = cfg.get("min_bbox_h", 0.0)

    os.makedirs(args.out, exist_ok=True)
    cropdir = os.path.join(args.out, "crops")
    os.makedirs(cropdir, exist_ok=True)

    crossings = []
    dropped_street = no_cross = 0
    for t in L1["tracks"]:
        traj = t["traj"]
        if not traj:
            continue
        # 1. street-mask
        if street and sum(point_in_poly(x, y, street) for _, x, y, _ in traj) / len(traj) > street_frac:
            dropped_street += 1
            continue
        # optional size gate (must get reasonably close to the camera)
        if min_bbox_h and max(p[3] for p in traj) < min_bbox_h:
            no_cross += 1
            continue
        # 2. line-cross in the requested direction
        cross_ts = None
        prev = prev_sign = None
        for ts, x, y, h in traj:
            cur = side(p1, p2, (x, y))
            if (prev is not None and prev_sign not in (None, 0) and cur != 0
                    and cur != prev_sign and segments_intersect(prev, (x, y), p1, p2)):
                if ("in" if cur == inside_sign else "out") == args.direction:
                    cross_ts = ts
                    break
            if cur != 0:
                prev_sign = cur
            prev = (x, y)
        if cross_ts is None:
            no_cross += 1
            continue
        crossings.append({"track": t["track"], "cross_ts": cross_ts, "crop": t["crop"],
                          "conf": t["peak_conf"]})

    # 3. de-dup by time window (one fragmented person -> one event)
    crossings.sort(key=lambda e: e["cross_ts"])
    deduped = []
    for e in crossings:
        if deduped and e["cross_ts"] - deduped[-1]["cross_ts"] <= dedup_w:
            deduped[-1]["merged"].append(e["track"])
            continue
        e["merged"] = [e["track"]]
        deduped.append(e)

    rows, thumbs = [], []
    for e in deduped:
        crop = cv2.imread(e["crop"])
        if crop is not None:
            cv2.imwrite(os.path.join(cropdir, f"{args.direction}_{e['track']:04d}.jpg"), crop)
            thumbs.append((crop, f"{args.direction.upper()} {ist(e['cross_ts'])} (#{e['track']})"))
        rows.append({"track": e["track"], f"{args.direction}_ist": ist(e["cross_ts"]),
                     "merged_tracks": "+".join(map(str, e["merged"])), "conf": e["conf"]})

    with open(os.path.join(args.out, f"{args.direction}_events.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["track", f"{args.direction}_ist", "merged_tracks", "conf"])
        w.writeheader()
        w.writerows(rows)
    build_contact_sheet(thumbs, os.path.join(args.out, f"{args.direction}_contact_sheet.jpg"))

    print(f"[L2] L1 had {len(L1['tracks'])} raw human-tracks")
    print(f"[L2]   - {dropped_street} dropped as street-through-glass")
    print(f"[L2]   - {no_cross} never crossed the line ({args.direction})")
    print(f"[L2]   - {len(crossings)} raw crossings -> de-dup -> {len(deduped)} real {args.direction.upper()}")
    for e in deduped:
        print(f"       {ist(e['cross_ts'])}  {args.direction.upper()}  track#{e['track']}  merged={e['merged']}")


if __name__ == "__main__":
    main()
