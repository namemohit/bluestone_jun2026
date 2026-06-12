"""Parallel PUREPLAY reprocess of a day into an isolated namespace — no GPU, non-destructive.

For each existing `<date>_<HHMM>` window, create a parallel `<date>-<tag>_<HHMM>` window that REUSES the
original window's L1 dirs (tracks.json + crops — provably pristine, the detector is their only writer) and
re-runs ONLY L2→L4 (`stack.l4_visits`) with an EMPTY human-feedback file. The result is the raw machine
output with ZERO per-day HITL — for certifying L1→L2 and measuring L3 capability against the reviewed day.

  python -m batch.reprocess_pure --date 2026-06-03           # -> outputs/2026-06-03-pure_*/visits.json

Disk-only: writes NEW `outputs/<date>-<tag>_*` dirs, touches nothing else (no Supabase writes, the reviewed
day untouched). Undo = delete those dirs. Because L2 is pure geometry over the SHARED L1, the pure day's door
crossings are byte-identical to the reviewed day's — so all divergence is L3 (matching/identity).

The only human-derived input retained is the persistent staff GALLERY (the deployed model's roster, not
per-day HITL) so staff auto-recognition stays measurable — flagged as such in the scorecard.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import re
import subprocess
import sys
from pathlib import Path

OUTPUTS = Path("outputs")
EMPTY_FB = {"cannot_link": [], "must_link": [], "employees": [], "not_staff": [], "false": []}


def src_windows(date: str) -> list[str]:
    return sorted({Path(p).parent.name for p in glob.glob(f"outputs/{date}_*/visits.json")
                   if re.search(r"_\d{4}$", Path(p).parent.name)})


def main() -> None:
    ap = argparse.ArgumentParser(description="Parallel pureplay reprocess (L2->L4, empty feedback)")
    ap.add_argument("--date", default="2026-06-03")
    ap.add_argument("--tag", default="pure", help="parallel namespace suffix -> <date>-<tag>_<HHMM>")
    ap.add_argument("--register-only", action="store_true",
                    help="skip L4; just register existing pure windows in the store so the dropdown lists them")
    ap.add_argument("--count-cam", default=None,
                    help="count crossings on this camera instead of the C05 door (e.g. C11 -> 'Variant C'). "
                         "Promotes that window's L1_<cam> dir into the door slot; needs --count-config.")
    ap.add_argument("--count-config", default=None,
                    help="zones config (entry_line/inside) for --count-cam, e.g. configs/c11_zones.json")
    args = ap.parse_args()

    wins = src_windows(args.date)
    if not wins:
        sys.exit(f"[pure] no processed windows for {args.date}")

    st = None
    gal, thr = [], {}                                          # deployed model roster + thresholds (read-only)
    try:
        from hitl.store_supabase import SupabaseStore
        st = SupabaseStore()
        gal = st.get_gallery() or []
        thr = (st.active_model() or {}).get("params", {}) or {}
        print(f"[pure] deployed gallery: {len(gal)} embeddings; staff thresholds: "
              f"sim={thr.get('staff_sim')} auto={thr.get('staff_auto_sim')} margin={thr.get('staff_margin')}")
    except Exception as e:
        print(f"[pure] WARN gallery/model unavailable ({type(e).__name__}: {e}) — staff matching will be off")

    def register(pure_window: str):
        """Register the pure window so the header dropdown lists it (window row + visits; NO detections push —
        the pure report/cards read from disk). Additive: new window_ids, the reviewed day untouched."""
        if st is None:
            return
        pj = OUTPUTS / pure_window / "visits.json"
        if not pj.exists():
            return
        try:
            st.push_window(pure_window, "s14", json.loads(pj.read_text(encoding="utf-8")), with_detections=False)
        except Exception as e:
            print(f"[pure]   register WARN {pure_window}: {type(e).__name__}: {e}")

    if args.register_only:
        n = 0
        for w in wins:
            pure = f"{args.date}-{args.tag}_{w.split('_')[1]}"
            if (OUTPUTS / pure / "visits.json").exists():
                register(pure); n += 1
        print(f"[pure] registered {n} existing pure windows -> dropdown shows '{args.date}-{args.tag}'")
        return

    ok = 0
    for w in wins:
        hhmm = w.split("_")[1]
        pure = f"{args.date}-{args.tag}_{hhmm}"
        pdir = OUTPUTS / pure
        pdir.mkdir(parents=True, exist_ok=True)
        src_cfg = json.loads((OUTPUTS / w / "window.json").read_text(encoding="utf-8"))

        # default: count at the original C05 door. --count-cam swaps the counting camera (Variant C):
        # promote that window's L1_<cam> dir into the door slot, the rest stay interior, point the
        # frame/clip overlay (hitl_api hardcodes entry.ts) at the real <cam>.ts.
        door_l1, door_cfg = src_cfg["l1"], src_cfg["config"]
        interior = list(src_cfg.get("interior") or [])
        if args.count_cam:
            cam = args.count_cam.upper()
            promoted = next((d for d in interior if Path(d).name == f"L1_{cam}"), None)
            if promoted is None:
                print(f"[pure]   SKIP {w}: no L1_{cam} interior dir to count on"); continue
            door_l1 = promoted
            door_cfg = args.count_config or src_cfg["config"]
            interior = [d for d in interior if d != promoted]    # the rest stay interior (e.g. C14)
            (pdir / "slices").mkdir(exist_ok=True)
            real_ts, link = OUTPUTS / w / "slices" / f"{cam}.ts", pdir / "slices" / "entry.ts"
            if real_ts.exists() and not link.exists():
                try:
                    os.link(real_ts, link)                       # hardlink (same volume, no admin needed)
                except Exception:
                    try:
                        link.symlink_to(real_ts.resolve())
                    except Exception as e:
                        print(f"[pure]   (frame slice link skipped: {type(e).__name__})")

        pcfg = dict(src_cfg)                                   # reuse zones + label; swap door/interior for the variant
        pcfg["l1"], pcfg["config"], pcfg["interior"] = door_l1, door_cfg, interior
        pcfg["source_window"] = w                             # provenance: which reviewed window this mirrors
        pcfg["count_cam"] = (args.count_cam or "C05").upper()
        (pdir / "window.json").write_text(json.dumps(pcfg, indent=2), encoding="utf-8")
        (pdir / "feedback.json").write_text(json.dumps(EMPTY_FB, indent=2), encoding="utf-8")

        cmd = [sys.executable, "-m", "stack.l4_visits",
               "--l1", door_l1, "--config", door_cfg,
               "--out", str(pdir), "--feedback", str(pdir / "feedback.json")]
        if interior:
            cmd += ["--interior", *interior]
            if gal:
                galp = pdir / "gallery.json"
                galp.write_text(json.dumps(gal), encoding="utf-8")
                cmd += ["--gallery", str(galp)]
                for flag, key in (("--staff-sim", "staff_sim"), ("--staff-auto-sim", "staff_auto_sim"),
                                  ("--staff-margin", "staff_margin")):
                    if thr.get(key):
                        cmd += [flag, str(thr[key])]
        print(f"[pure] {pure}  <- L1 {door_l1}  (count_cam={pcfg['count_cam']})", flush=True)
        r = subprocess.run(cmd, capture_output=True, text=True)
        if r.returncode != 0:
            print(f"[pure]   FAIL: {r.stderr[-300:]}", flush=True)
            continue
        register(pure)                                         # list it in the header dropdown (window row + visits, no detections)
        ok += 1

    print(f"[pure] DONE: {ok}/{len(wins)} windows -> outputs/{args.date}-{args.tag}_*  (reviewed day untouched)")
    print(f"[pure] compare: reviewed='{args.date}'  vs  pure='{args.date}-{args.tag}'")


if __name__ == "__main__":
    main()
