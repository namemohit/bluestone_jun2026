"""L5 (v2) - interior bridge: link C05 door events to interior presence (C11/C14).

For each C05 entry/exit, find interior tracks that appear (after an entry) or were last
seen (before an exit) within the bridge window. Yields, per event:
  - PRESENCE   : real visit (seen inside) vs pass-through (entered, never reached the floor)
  - clean crop : the interior shot (multi-angle, no identical-doorway background) for ReID
  - measured lag: the C05->interior delay (the "bridge delta", straight from the data)

  python -m stack.l5_bridge --c05 outputs/L1_c05_full --config configs/c05_zones.json \
      --interior outputs/L1_c11_partial outputs/L1_c14_partial --out outputs/L5_bridge
"""
from __future__ import annotations

import argparse
import json
import os

import cv2
import numpy as np

from stack.l4_visits import detect_events, ist, hsv_hist, hist_sim


def cam_of(d):
    n = os.path.basename(d.rstrip("/\\")).lower()
    return "C11" if "c11" in n else "C14" if "c14" in n else n


def load_interior(dirs):
    out = []
    for d in dirs:
        p = os.path.join(d, "tracks.json")
        if not os.path.exists(p):
            continue
        j = json.load(open(p))
        cam = cam_of(d)
        for t in j["tracks"]:
            out.append({"cam": cam, "track": t["track"], "first": t["first_ts"],
                        "last": t["last_ts"], "crop": t["crop"], "frames": t["frames"]})
    return out


def bridge_sheet(results, path, cw=130, ch=190, pad=10):
    rows = [r for r in results if r["matches"]]
    if not rows:
        cv2.imwrite(path, np.full((120, 520, 3), 30, np.uint8))
        return
    F = cv2.FONT_HERSHEY_SIMPLEX
    rowh = ch + 30
    W = 3 * cw + 4 * pad + 300
    H = len(rows) * (rowh + pad) + pad
    sheet = np.full((H, W, 3), 26, np.uint8)
    for i, r in enumerate(rows):
        y0 = pad + i * (rowh + pad)
        e = r["event"]
        crops = [("C05 door", e["crop"])]
        for k, m in enumerate(r["matches"][:2]):
            it = m["it"]
            tag = "PICK" if k == 0 else "alt"
            crops.append((f"{it['cam']} {tag} s{m['sim']:.2f}", it["crop"]))
        for j, (lab, cp) in enumerate(crops[:3]):
            img = cv2.imread(cp)
            x0 = pad + j * (cw + pad)
            if img is not None:
                s = min(cw / img.shape[1], ch / img.shape[0])
                rs = cv2.resize(img, (max(1, int(img.shape[1] * s)), max(1, int(img.shape[0] * s))))
                sheet[y0:y0 + rs.shape[0], x0:x0 + rs.shape[1]] = rs
            cv2.putText(sheet, lab, (x0, y0 + ch + 18), F, 0.42, (180, 220, 180), 1)
        tx = 3 * cw + 4 * pad
        cv2.putText(sheet, f"{ist(e['ts'])} {e['dir'].upper()}", (tx, y0 + 34), F, 0.6, (230, 230, 120), 2)
        cv2.putText(sheet, "PRESENT inside" if r["matches"] else "no presence", (tx, y0 + 70),
                    F, 0.55, (120, 240, 120), 2)
    cv2.imwrite(path, sheet)


def main():
    ap = argparse.ArgumentParser(description="L5 interior bridge: C05 events -> interior presence")
    ap.add_argument("--c05", default="outputs/L1_c05_full")
    ap.add_argument("--config", default="configs/c05_zones.json")
    ap.add_argument("--interior", nargs="+", required=True)
    ap.add_argument("--bridge-window", type=float, default=2.0,
                    help="transit allowance (s). Start tight at 2s; widen from MEASURED lag.")
    ap.add_argument("--skew", type=float, default=2.0, help="camera-clock skew tolerance (s)")
    ap.add_argument("--min-frames", type=int, default=2)
    ap.add_argument("--embedder", default="osnet", choices=["osnet", "colorhist"],
                    help="appearance model for the pick: osnet ReID (good) or colorhist (weak)")
    ap.add_argument("--out", default="outputs/L5_bridge")
    args = ap.parse_args()

    C05 = json.load(open(os.path.join(args.c05, "tracks.json")))
    cfg = json.load(open(args.config))
    events = sorted(detect_events(C05, cfg), key=lambda e: e["ts"])
    dedup = cfg.get("dedup_window_s", 18)
    kept = []
    for e in events:
        if kept and e["dir"] == kept[-1]["dir"] and e["ts"] - kept[-1]["ts"] <= dedup:
            continue
        kept.append(e)
    interior = load_interior(args.interior)
    os.makedirs(args.out, exist_ok=True)

    if args.embedder == "osnet":
        from stack import reid
        embed_fn, sim_fn = reid.osnet_embed, reid.osnet_sim
    else:
        embed_fn, sim_fn = hsv_hist, hist_sim
    for it in interior:  # embed each interior crop once
        it["emb"] = embed_fn(cv2.imread(it["crop"]))

    results = []
    for e in kept:
        t, d = e["ts"], e["dir"]
        door_emb = embed_fn(cv2.imread(e["crop"]))
        matches = []
        for it in interior:
            if it["frames"] < args.min_frames:
                continue
            lag = (it["first"] - t) if d == "in" else (t - it["last"])
            if -args.skew <= lag <= args.bridge_window:
                sim = sim_fn(door_emb, it["emb"])
                matches.append({"lag": lag, "it": it, "sim": sim})
        # PICK by appearance (door<->interior colour) WITHIN the time window -- not time alone,
        # so co-present staff don't win just by being nearest in time.
        matches.sort(key=lambda m: -m["sim"])
        results.append({"event": e, "matches": matches})

    print("=" * 60)
    print(f"[L5 interior bridge]  window=[{-args.skew:.0f}, +{args.bridge_window:.0f}]s  "
          f"interior tracks loaded: {len(interior)}")
    for r in results:
        e, ms = r["event"], r["matches"]
        tag = "PRESENT inside" if ms else "NO interior presence (pass-through?)"
        print(f"  {ist(e['ts'])} {e['dir'].upper():3s} -> {tag}")
        for k, m in enumerate(ms[:3]):
            it = m["it"]
            mark = "  <== PICK (best appearance)" if k == 0 else ""
            print(f"        {it['cam']} track#{it['track']:<4d} appr {m['sim']:+.2f}  lag {m['lag']:+5.1f}s{mark}")
    lags = [r["matches"][0]["lag"] for r in results if r["matches"]]
    if lags:
        print(f"  MEASURED bridge delta: mean {np.mean(lags):+.1f}s  range [{min(lags):+.1f}, {max(lags):+.1f}]s")
    print("=" * 60)
    bridge_sheet(results, os.path.join(args.out, "bridge_sheet.jpg"))
    print(f"  -> {args.out}/bridge_sheet.jpg")


if __name__ == "__main__":
    main()
