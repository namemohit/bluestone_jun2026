"""L4 - Strategy 1: pair entries (IN) with exits (OUT) into VISITS (dwell + occupancy).

Single customer door, one camera (C05) -- see memory in-out-matching-strategy. Matching,
most-reliable first:
  1. occupancy/timing  : an OUT matches an open entry-session within [0, max_dwell]
  2. body-appearance    : when >1 session is open, disambiguate by HSV colour histogram of
     the crop -- aspect-tolerant, so a red dress reads "red" from front (entry) or back (exit)
The interior-camera (C11/C14) bridge is the v2 enhancement; this v1 runs on C05 alone.

Reads L1 tracks.json + zone config. Writes visits.csv, occupancy.csv, visits_sheet.jpg.
  python -m stack.l4_visits --l1 outputs/L1_c05_full --config configs/c05_zones.json
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import pickle
import sys
from datetime import datetime, timezone, timedelta

import cv2
import numpy as np

from vision.geometry import side, segments_intersect, inside_sign_from_label
from stack.l2_entries import point_in_poly

IST = timezone(timedelta(hours=5, minutes=30))


def ist(ts: float) -> str:
    return datetime.fromtimestamp(ts, IST).strftime("%H:%M:%S")


def hsv_hist(crop):
    if crop is None or crop.size == 0:
        return None
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    h = cv2.calcHist([hsv], [0, 1], None, [32, 32], [0, 180, 0, 256])
    cv2.normalize(h, h)
    return h.flatten().astype("float32")


def hist_sim(a, b) -> float:
    if a is None or b is None:
        return 0.0
    return float(cv2.compareHist(a, b, cv2.HISTCMP_CORREL))


# --- OSNet embedding cache: GPU-embed each crop ONCE, keyed by path. Re-running L4 with new
#     --feedback then hits the cache (no GPU) -> the HITL loop is instant + free. ---
_EMB_CACHE = "outputs/osnet_emb_cache.pkl"


def _load_emb_cache() -> dict:
    try:
        with open(_EMB_CACHE, "rb") as f:
            return pickle.load(f)
    except Exception:
        return {}


def _save_emb_cache(cache: dict) -> None:
    os.makedirs(os.path.dirname(_EMB_CACHE), exist_ok=True)
    with open(_EMB_CACHE, "wb") as f:
        pickle.dump(cache, f)


def _embed_paths(reid, paths, cache) -> tuple[dict, int]:
    out, n_new = {}, 0
    for p in dict.fromkeys(paths):  # de-dup, preserve order
        if p not in cache:
            cache[p] = reid.osnet_embed(cv2.imread(p))
            n_new += 1
        out[p] = cache[p]
    return out, n_new


def detect_events(L1, cfg):
    """One IN/OUT crossing event per track (after street-mask)."""
    p1, p2 = tuple(cfg["entry_line"][0]), tuple(cfg["entry_line"][1])
    inside_sign = inside_sign_from_label(cfg.get("inside", "right"))
    street = cfg.get("street_mask", [])
    sfrac = cfg.get("street_drop_frac", 0.5)
    events = []
    for t in L1["tracks"]:
        traj = t["traj"]
        if not traj:
            continue
        if street and sum(point_in_poly(x, y, street) for _, x, y, _ in traj) / len(traj) > sfrac:
            continue
        prev = prev_sign = None
        for ts, x, y, h in traj:
            cur = side(p1, p2, (x, y))
            if (prev is not None and prev_sign not in (None, 0) and cur != 0
                    and cur != prev_sign and segments_intersect(prev, (x, y), p1, p2)):
                events.append({"track": t["track"], "ts": ts,
                               "dir": "in" if cur == inside_sign else "out", "crop": t["crop"]})
                break
            if cur != 0:
                prev_sign = cur
            prev = (x, y)
    return events


def visits_sheet(visits, path, cw=150, ch=210, pad=10):
    if not visits:
        cv2.imwrite(path, np.full((120, 520, 3), 30, np.uint8))
        return
    rowh = ch + 34
    W = 2 * cw + 3 * pad + 260
    H = len(visits) * (rowh + pad) + pad
    sheet = np.full((H, W, 3), 26, np.uint8)
    F = cv2.FONT_HERSHEY_SIMPLEX
    for i, v in enumerate(visits):
        y0 = pad + i * (rowh + pad)
        for j, key in enumerate(("in_crop", "out_crop")):
            crop = cv2.imread(v[key])
            x0 = pad + j * (cw + pad)
            if crop is not None:
                s = min(cw / crop.shape[1], ch / crop.shape[0])
                rs = cv2.resize(crop, (max(1, int(crop.shape[1] * s)), max(1, int(crop.shape[0] * s))))
                sheet[y0:y0 + rs.shape[0], x0:x0 + rs.shape[1]] = rs
            cv2.putText(sheet, "IN" if j == 0 else "OUT", (x0, y0 + ch + 20), F, 0.5, (180, 220, 180), 1)
        tx = 2 * cw + 3 * pad
        cv2.putText(sheet, f"{ist(v['in'])} -> {ist(v['out'])}", (tx, y0 + 40), F, 0.6, (230, 230, 120), 2)
        cv2.putText(sheet, f"dwell {int(v['dwell']//60)}m{int(v['dwell']%60):02d}s", (tx, y0 + 78), F, 0.6, (200, 255, 200), 2)
        cv2.putText(sheet, v["how"], (tx, y0 + 112), F, 0.45, (160, 160, 160), 1)
    cv2.imwrite(path, sheet)


def main() -> None:
    ap = argparse.ArgumentParser(description="L4 Strategy 1: IN<->OUT -> visits + dwell + occupancy")
    ap.add_argument("--l1", default="outputs/L1_c05_full")
    ap.add_argument("--config", default="configs/c05_zones.json")
    ap.add_argument("--max-dwell-min", type=float, default=180)
    ap.add_argument("--embedder", default="colorhist", choices=["colorhist", "cnn"],
                    help="door-crop appearance cue used only when NOT bridging interior")
    ap.add_argument("--interior", nargs="*", default=None,
                    help="interior L1 dirs (C11/C14): bridge each event to its clean interior "
                         "crop and match with OSNet ReID (v2)")
    ap.add_argument("--bridge-window", type=float, default=2.0)
    ap.add_argument("--skew", type=float, default=2.0)
    ap.add_argument("--min-sim", type=float, default=0.55,
                    help="appearance floor: when >1 session is open, refuse to pair an OUT to the "
                         "best open IN if their sim is below this (avoids force-pairing two "
                         "different people); the OUT becomes a pre-window exit, the IN stays open")
    ap.add_argument("--feedback", default=None,
                    help="HITL feedback JSON (human ticks/crosses): {cannot_link:[[in,out]..], "
                         "must_link:[[in,out]..], employees:[track..], min_sim:float}. Re-running "
                         "L4 with this re-routes matches for FREE (no GPU) -- the learning loop.")
    ap.add_argument("--occ-floor", type=float, default=0.55,
                    help="appearance floor applied even when only ONE session is open (mirrors "
                         "--min-sim): reject a lone-candidate occupancy match if the exiting person "
                         "clearly isn't them (e.g. a pre-window exit coinciding with one inside). "
                         "Appearance vetoes occupancy whenever a crop is available.")
    ap.add_argument("--out", default="outputs/L4_visits")
    args = ap.parse_args()
    try:
        sys.stdout.reconfigure(encoding="utf-8")  # console may default to cp1252 on Windows
    except Exception:
        pass

    if args.embedder == "cnn":
        from stack import reid
        embed_fn, sim_fn = reid.embed, reid.sim
    else:
        embed_fn, sim_fn = hsv_hist, hist_sim

    L1 = json.load(open(os.path.join(args.l1, "tracks.json")))
    cfg = json.load(open(args.config))
    max_dwell = args.max_dwell_min * 60
    dedup_w = cfg.get("dedup_window_s", 18)
    os.makedirs(args.out, exist_ok=True)

    feedback = json.load(open(args.feedback)) if (args.feedback and os.path.exists(args.feedback)) else {}
    min_sim = float(feedback.get("min_sim", args.min_sim))
    cannot = {tuple(p) for p in feedback.get("cannot_link", [])}   # (in_track, out_track) forbidden
    must = {tuple(p) for p in feedback.get("must_link", [])}       # (in_track, out_track) forced
    employees = set(feedback.get("employees", []))                  # entry-cam tracks that are staff

    events = sorted(detect_events(L1, cfg), key=lambda e: e["ts"])
    # de-dup same-direction crossings within window (one fragmented person)
    kept = []
    for e in events:
        if e["track"] in employees:  # confirmed staff -> not a customer visit
            continue
        if kept and e["dir"] == kept[-1]["dir"] and e["ts"] - kept[-1]["ts"] <= dedup_w:
            continue
        kept.append(e)
    if args.interior:
        # v2: bridge each door event to its clean interior crop (OSNet-picked), match on those
        from stack.l5_bridge import load_interior
        from stack import reid
        sim_fn = reid.osnet_sim
        interior = load_interior(args.interior)
        cache = _load_emb_cache()
        want = [it["crop"] for it in interior] + [e["crop"] for e in kept]
        embs, n_new = _embed_paths(reid, want, cache)
        if n_new:
            _save_emb_cache(cache)
        print(f"  OSNet embeddings: {len(set(want)) - n_new} cached, {n_new} new "
              f"({'instant re-run' if n_new == 0 else 'GPU pass'})")
        for it in interior:
            it["emb"] = embs.get(it["crop"])
        for e in kept:
            door_emb = embs.get(e["crop"])
            best, best_sim = None, -1.0
            for it in interior:
                if it["emb"] is None or door_emb is None:
                    continue
                lag = (it["first"] - e["ts"]) if e["dir"] == "in" else (e["ts"] - it["last"])
                if -args.skew <= lag <= args.bridge_window:
                    s = reid.osnet_sim(door_emb, it["emb"])
                    if s > best_sim:
                        best_sim, best = s, it
            e["present"] = best is not None
            e["emb"] = best["emb"] if best else None
            e["int_crop"] = best["crop"] if best else None
    else:
        for e in kept:
            e["emb"] = embed_fn(cv2.imread(e["crop"]))
            e["present"], e["int_crop"] = None, None

    # --- Strategy 1 matching (feedback-aware: human ticks/crosses re-route pairings) ---
    open_sessions, visits, unmatched_out = [], [], []
    occ_curve, peak = [], 0
    for e in kept:
        if e["dir"] == "in":
            open_sessions.append(e)
        else:
            in_win = [s for s in open_sessions if 0 <= e["ts"] - s["ts"] <= max_dwell]
            cands = [s for s in in_win if (s["track"], e["track"]) not in cannot]  # crosses removed
            s, how, conf, uncert = None, None, 0.0, 0.0
            forced = [c for c in cands if (c["track"], e["track"]) in must]
            if forced:  # human-confirmed pairing -> lock it
                s, how, conf, uncert = forced[0], "confirmed (human tick)", 1.0, 0.0
            elif len(cands) == 1:
                c = cands[0]
                sim1 = sim_fn(c["emb"], e["emb"]) if (c["emb"] is not None and e["emb"] is not None) else None
                if sim1 is not None and sim1 < args.occ_floor:
                    how = f"rejected occupancy sim={sim1:.2f}<{args.occ_floor:.2f} (lone open is a mismatch)"
                elif sim1 is not None:
                    s, how, conf, uncert = c, f"occupancy+appearance sim={sim1:.2f}", max(0.7, sim1), 0.15
                else:
                    s, how, conf, uncert = c, "occupancy (1 inside, no appearance)", 0.7, 0.25
            elif len(cands) > 1:
                scored = sorted(((sim_fn(c["emb"], e["emb"]), c) for c in cands),
                                key=lambda z: z[0], reverse=True)
                best_sim, cand = scored[0]
                second = scored[1][0] if len(scored) > 1 else 0.0
                tag = "OSNet-interior" if args.interior else f"appearance-{args.embedder}"
                if best_sim >= min_sim:
                    s, how = cand, f"{tag} sim={best_sim:.2f} ({len(cands)} open)"
                    conf, uncert = best_sim, 1.0 - (best_sim - second)  # tight margin => uncertain
                else:
                    how = f"rejected {tag} sim={best_sim:.2f}<{min_sim:.2f} ({len(cands)} open)"
            if s is not None:
                open_sessions.remove(s)
                visits.append({"id": f"{s['track']}-{e['track']}",
                               "in_track": s["track"], "out_track": e["track"],
                               "in": s["ts"], "out": e["ts"], "dwell": e["ts"] - s["ts"],
                               "in_crop": s.get("int_crop") or s["crop"],
                               "out_crop": e.get("int_crop") or e["crop"], "how": how,
                               "confidence": round(float(conf), 3),
                               "uncertainty": round(float(max(0.0, min(1.0, uncert))), 3),
                               "in_present": s.get("present"), "out_present": e.get("present")})
            else:
                unmatched_out.append(e)
        peak = max(peak, len(open_sessions))
        occ_curve.append((e["ts"], len(open_sessions)))
    still_inside = open_sessions

    visits.sort(key=lambda v: v["in"])
    with open(os.path.join(args.out, "visits.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["in_ist", "out_ist", "dwell_min", "matched_by"])
        for v in visits:
            w.writerow([ist(v["in"]), ist(v["out"]), round(v["dwell"] / 60, 1), v["how"]])
    with open(os.path.join(args.out, "occupancy.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["ist", "occupancy"])
        for ts, o in occ_curve:
            w.writerow([ist(ts), o])
    visits_sheet(visits, os.path.join(args.out, "visits_sheet.jpg"))

    # rich JSON for the dashboard / Supabase: stable id, crops, confidence, review status.
    # needs_review = uncertain (tight appearance margin) -> bubbles to the top of the queue.
    review_band = 0.4
    visits_json = {
        "window_start_ist": L1.get("start"),
        "params": {"min_sim": min_sim, "interior": bool(args.interior)},
        "feedback_applied": {"cannot_link": len(cannot), "must_link": len(must),
                             "employees": len(employees)},
        "counts": {"visits": len(visits), "still_inside": len(still_inside),
                   "pre_window_exits": len(unmatched_out), "peak_occupancy": peak},
        "visits": [{"id": v["id"], "in_track": v["in_track"], "out_track": v["out_track"],
                    "in_ist": ist(v["in"]), "out_ist": ist(v["out"]),
                    "dwell_s": round(v["dwell"], 1), "how": v["how"],
                    "confidence": v["confidence"], "uncertainty": v["uncertainty"],
                    "status": "needs_review" if v["uncertainty"] >= review_band else "auto",
                    "in_crop": v["in_crop"].replace("\\", "/"),
                    "out_crop": v["out_crop"].replace("\\", "/")}
                   for v in sorted(visits, key=lambda z: -z["uncertainty"])],  # uncertain first
        # unmatched events the human can also act on: link an open IN to a pre-exit OUT
        # (must_link -> recovers a missed visit), or mark either as staff (employee).
        "open_sessions": [{"track": s["track"], "ist": ist(s["ts"]),
                           "crop": (s.get("int_crop") or s["crop"]).replace("\\", "/")}
                          for s in sorted(still_inside, key=lambda z: z["ts"])],
        "pre_window_exits": [{"track": e["track"], "ist": ist(e["ts"]),
                              "crop": (e.get("int_crop") or e["crop"]).replace("\\", "/")}
                             for e in sorted(unmatched_out, key=lambda z: z["ts"])],
    }
    with open(os.path.join(args.out, "visits.json"), "w", encoding="utf-8") as f:
        json.dump(visits_json, f, indent=2)

    dwell_list = [v["dwell"] for v in visits]
    print("=" * 48)
    print(f"[L4 Strategy 1]  window from L1: {L1.get('start')}")
    fb_n = len(cannot) + len(must) + len(employees)
    if fb_n:
        print(f"  feedback applied: {len(cannot)} crosses, {len(must)} confirms, {len(employees)} staff")
    print(f"  completed visits (in+out paired): {len(visits)}")
    if dwell_list:
        print(f"  dwell  mean {np.mean(dwell_list)/60:.1f} min  |  median {np.median(dwell_list)/60:.1f} min "
              f"|  range {min(dwell_list)/60:.1f}-{max(dwell_list)/60:.1f}")
    print(f"  peak occupancy: {peak}")
    print(f"  still inside at window end (entered, no exit yet): {len(still_inside)}")
    print(f"  exits with no in-window entry (entered before 18:00): {len(unmatched_out)}")
    print("-" * 48)
    for v in visits:
        warn = "  [!] weak interior presence" if (v.get("in_present") is False or v.get("out_present") is False) else ""
        print(f"  VISIT  {ist(v['in'])} -> {ist(v['out'])}   dwell {int(v['dwell']//60)}m{int(v['dwell']%60):02d}s   [{v['how']}]{warn}")
    for s in still_inside:
        print(f"  OPEN   {ist(s['ts'])} -> (still inside)")
    for e in unmatched_out:
        print(f"  PRE    (entered <18:00) -> {ist(e['ts'])} OUT")
    print("=" * 48)


if __name__ == "__main__":
    main()
