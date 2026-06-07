"""Manual tally tool — fast, reliable hand-count of a recorded camera clip (no GPU/models).

Play the entry clip at speed and tap keys as people cross. Produces a timestamped CSV + a
summary you can put in front of a customer. Run it on the machine that has the footage.

  python -m tools.tally --video C05_entry_jun3.mp4 --start "2026-06-03 10:00:00" --out jun3_entry.csv

Keys:
  i = entry (+1 IN)      o = exit (+1 OUT)      g = mark a group arrival
  u = undo last event    space = pause/play     [ / ] = slower / faster
  s = save now           q = save & quit
"""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timedelta

import cv2


def main() -> None:
    ap = argparse.ArgumentParser(description="Manual tally for a recorded clip")
    ap.add_argument("--video", required=True)
    ap.add_argument("--start", default="", help='wall-clock of frame 0, e.g. "2026-06-03 10:00:00"')
    ap.add_argument("--out", default="tally.csv")
    ap.add_argument("--speed", type=int, default=6, help="initial frame-skip (review speed)")
    args = ap.parse_args()

    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        raise SystemExit(f"Could not open {args.video}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    start_dt = None
    if args.start:
        try:
            start_dt = datetime.strptime(args.start, "%Y-%m-%d %H:%M:%S")
        except ValueError:
            print("start time ignored (use 'YYYY-MM-DD HH:MM:SS')")

    events: list[dict] = []
    counts = {"in": 0, "out": 0, "group": 0}
    skip = max(1, args.speed)
    paused = False
    idx = -1

    def stamp(i):
        secs = i / fps
        if start_dt:
            return (start_dt + timedelta(seconds=secs)).strftime("%H:%M:%S")
        return f"{int(secs // 3600):02d}:{int(secs % 3600 // 60):02d}:{int(secs % 60):02d}"

    def log(kind, i):
        key = {"i": "in", "o": "out", "g": "group"}[kind]
        counts[key] += 1
        events.append({"kind": key, "video_idx": i, "time": stamp(i)})

    def save():
        with open(args.out, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=["kind", "video_idx", "time"])
            w.writeheader()
            w.writerows(events)
        print(f"saved {len(events)} events -> {args.out}  |  IN={counts['in']} OUT={counts['out']} GROUPS={counts['group']}")

    win = "tally  [i]n [o]ut [g]roup [u]ndo [space]pause [ ]slow []]fast [s]ave [q]uit"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    while True:
        if not paused:
            for _ in range(skip):
                ok, frame = cap.read()
                idx += 1
                if not ok:
                    frame = None
                    break
            if frame is None:
                print("end of video")
                break
        hud = frame.copy()
        cv2.rectangle(hud, (0, 0), (hud.shape[1], 34), (0, 0, 0), -1)
        cv2.putText(hud, f"{stamp(idx)}   IN {counts['in']}   OUT {counts['out']}   GROUPS {counts['group']}"
                    f"   x{skip}{'  PAUSED' if paused else ''}",
                    (10, 23), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        cv2.imshow(win, hud)

        k = cv2.waitKey(0 if paused else 15) & 0xFF
        if k in (ord("i"), ord("o"), ord("g")):
            log(chr(k), idx)
        elif k == ord("u") and events:
            e = events.pop()
            counts[e["kind"]] -= 1
        elif k == ord(" "):
            paused = not paused
        elif k == ord("]"):
            skip = min(60, skip + 2)
        elif k == ord("["):
            skip = max(1, skip - 2)
        elif k == ord("s"):
            save()
        elif k == ord("q"):
            break

    save()
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
