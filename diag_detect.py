"""Sanity-check detection + speed on a clip: run YOLO on a few spread-out frames,
report person counts + per-frame GPU time, and save annotated samples."""
import sys, time
import cv2
from ultralytics import YOLO

video = sys.argv[1] if len(sys.argv) > 1 else "data/footage_rtsp/c05_5min.mp4"
m = YOLO("yolo11m.pt")
cap = cv2.VideoCapture(video)
N = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
print(f"video={video} frames={N} fps={fps:.1f}")

# warm up (first GPU call includes CUDA init / cudnn autotune)
ok, fr = cap.read()
if ok:
    m.predict(fr, classes=[0], conf=0.25, imgsz=960, device=0, verbose=False)

samples = [0.02, 0.2, 0.4, 0.6, 0.8, 0.98]
total = 0
for k, f in enumerate(samples):
    i = int(N * f)
    cap.set(cv2.CAP_PROP_POS_FRAMES, i)
    ok, fr = cap.read()
    if not ok:
        continue
    t = time.time()
    r = m.predict(fr, classes=[0], conf=0.25, imgsz=960, device=0, verbose=False)[0]
    dt = (time.time() - t) * 1000
    n = 0 if r.boxes is None else len(r.boxes)
    total += n
    secs = i / fps
    stamp = f"{int(secs//60):02d}:{int(secs%60):02d}"
    print(f"  t~10:{stamp}  frame {i:5d}: {n} person(s)  {dt:.0f} ms")
    cv2.imwrite(f"outputs/diag_{k}.jpg", r.plot())
cap.release()
print(f"total person detections across {len(samples)} sampled frames: {total}")
