"""Presence scan: over a clip, sample ~1 frame/sec, detect persons, and classify each
person's feet as INSIDE (below the counting line) or OUTSIDE. Reveals whether anyone
actually approached/entered the door (vs the counter silently missing crossings)."""
import sys
from datetime import datetime, timedelta, timezone

import cv2
from ultralytics import YOLO

IST = timezone(timedelta(hours=5, minutes=30))
video = sys.argv[1] if len(sys.argv) > 1 else "data/footage_rtsp/c05_6pm_ist.ts"
start_ist = datetime(2026, 6, 3, 18, 0, 0, tzinfo=IST)
x1, y1, x2, y2 = 0.25, 0.77, 0.63, 0.65  # counting line P1->P2 (normalized)


def line_side(px, py):
    return (x2 - x1) * (py - y1) - (y2 - y1) * (px - x1)  # >0 => below/inside


m = YOLO("yolo11m.pt")
cap = cv2.VideoCapture(video)
fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
step = max(1, int(round(fps)))  # ~1 sec
idx = -1
sampled = frames_person = frames_inside = saved = 0
rows = []
while True:
    ok, fr = cap.read()
    if not ok:
        break
    idx += 1
    if idx % step != 0:
        continue
    sampled += 1
    h, w = fr.shape[:2]
    r = m.predict(fr, classes=[0], conf=0.30, imgsz=736, device=0, verbose=False)[0]
    n = nin = 0
    tags = []
    if r.boxes is not None and len(r.boxes):
        for (bx1, by1, bx2, by2), cf in zip(r.boxes.xyxy.cpu().numpy(), r.boxes.conf.cpu().numpy()):
            n += 1
            cx = ((bx1 + bx2) / 2) / w
            cy = by2 / h  # feet (bottom-center), normalized
            inside = line_side(cx, cy) > 0
            nin += 1 if inside else 0
            tags.append(("IN " if inside else "out") + f"({cx:.2f},{cy:.2f},{cf:.2f})")
    if n:
        frames_person += 1
        if nin:
            frames_inside += 1
        t = start_ist + timedelta(seconds=idx / fps)
        rows.append((t.strftime("%H:%M:%S"), n, nin, tags))
        if nin and saved < 12:
            cv2.imwrite(f"outputs/presence_{saved:02d}.jpg", r.plot())
            saved += 1
cap.release()
print(f"sampled {sampled} frames (~1/sec over the clip)")
print(f"frames with >=1 person detected   : {frames_person}")
print(f"frames with a person INSIDE the line (feet below): {frames_inside}")
print(f"saved {saved} proof frames (outputs/presence_*.jpg)")
print("\nTimeline of every second with a detected person (IST | nPersons | nInside | tags):")
for t, n, nin, tags in rows:
    print(f"  {t} | {n} | {nin} | {', '.join(tags)}")
