"""Traffic map: plot every detected foot-point over the clip onto one frame, with the
current counting line. Shows where people actually walk so the line can be re-placed."""
import sys
import cv2
import numpy as np
from ultralytics import YOLO

video = sys.argv[1] if len(sys.argv) > 1 else "data/footage_rtsp/c05_6pm_ist.ts"
x1, y1, x2, y2 = 0.25, 0.77, 0.63, 0.65  # current line
m = YOLO("yolo11m.pt")
cap = cv2.VideoCapture(video)
fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
N = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
step = max(1, int(round(fps * 0.5)))  # every ~0.5s
LIMIT = 4500  # first ~3 min (the busy stretch) — fast

cap.set(cv2.CAP_PROP_POS_FRAMES, 1500)
ok, bg = cap.read()
if not ok:
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0); ok, bg = cap.read()
h, w = bg.shape[:2]
canvas = (bg * 0.5).astype(np.uint8)  # dim background so dots pop

cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
idx = -1
pts = []
while True:
    ok, fr = cap.read()
    if not ok:
        break
    idx += 1
    if idx > LIMIT:
        break
    if idx % step:
        continue
    r = m.predict(fr, classes=[0], conf=0.30, imgsz=736, device=0, verbose=False)[0]
    if r.boxes is not None and len(r.boxes):
        for (bx1, by1, bx2, by2) in r.boxes.xyxy.cpu().numpy():
            pts.append((int((bx1 + bx2) / 2), int(by2)))
cap.release()

for (cx, cy) in pts:
    cv2.circle(canvas, (cx, cy), 5, (0, 200, 255), -1)   # amber feet
cv2.line(canvas, (int(x1 * w), int(y1 * h)), (int(x2 * w), int(y2 * h)), (0, 255, 0), 3)
cv2.putText(canvas, f"{len(pts)} footsteps | green=current line", (16, 36),
            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
cv2.imwrite("outputs/trafficmap.jpg", canvas)
print(f"plotted {len(pts)} foot-points sampled every ~2s over {idx} frames -> outputs/trafficmap.jpg")
