"""Mockup of the proposed C05 entry-counting solution (multi-signal) on a real frame:
  - STREET MASK (red): detections here ignored (street through glass).
  - ENTRY LINE (yellow): steep, at the door opening, perpendicular to entry (user-specified).
  - INSIDE ZONE (green): mat + interior; entries confirmed + de-duplicated here.
  - FACE ZONE (cyan): a face whose crop exceeds a size threshold here = definitely entering.
Illustrative overlay only — no detection runs."""
import sys
import cv2
import numpy as np

F = cv2.FONT_HERSHEY_SIMPLEX
base = sys.argv[1] if len(sys.argv) > 1 else "outputs/c05_1230osd.jpg"
img = cv2.imread(base)
h, w = img.shape[:2]


def Pt(x, y):
    return (int(x * w), int(y * h))


def poly(pts):
    return np.array([Pt(x, y) for x, y in pts], np.int32)


over = img.copy()
# STREET region (ignore) — through-glass parking/sidewalk
cv2.fillPoly(over, [poly([(0.34, 0.05), (1.0, 0.05), (1.0, 0.60), (0.66, 0.66),
                          (0.40, 0.52), (0.34, 0.22)])], (40, 40, 200))
# INSIDE zone (count) — interior + mat, LEFT of the entry line
cv2.fillPoly(over, [poly([(0.0, 0.90), (0.30, 0.83), (0.45, 0.79), (0.55, 1.0),
                          (0.0, 1.0)])], (40, 150, 40))
img = cv2.addWeighted(over, 0.28, img, 0.72, 0)

# ENTRY LINE — steep diagonal at the door opening (your red-line position)
cv2.line(img, Pt(0.45, 0.79), Pt(0.64, 1.0), (0, 255, 255), 6, cv2.LINE_AA)
cv2.putText(img, "ENTRY LINE (at door, perp. to entry)", Pt(0.06, 0.77), F, 0.6, (0, 255, 255), 2)

# FACE ZONE — outline around the door where entering faces appear/grow
cv2.polylines(img, [poly([(0.36, 0.40), (0.66, 0.40), (0.66, 0.80), (0.40, 0.80)])],
              True, (255, 255, 0), 2)
cv2.putText(img, "FACE ZONE", Pt(0.40, 0.44), F, 0.6, (255, 255, 0), 2)

# illustrative faces: tiny/far (ignore) vs large/near (entering)
cv2.rectangle(img, Pt(0.605, 0.28), Pt(0.625, 0.31), (255, 255, 255), 2)
cv2.putText(img, "far face -> ignore", Pt(0.52, 0.26), F, 0.45, (255, 255, 255), 1)
cv2.rectangle(img, Pt(0.46, 0.50), Pt(0.55, 0.67), (0, 255, 0), 3)
cv2.putText(img, "LARGE face in zone = ENTERING", Pt(0.30, 0.49), F, 0.5, (0, 255, 0), 2)

cv2.putText(img, "IGNORE: street through glass", Pt(0.46, 0.31), F, 0.8, (60, 60, 255), 2)
cv2.putText(img, "INSIDE zone -> confirm + de-dup", Pt(0.02, 0.965), F, 0.6, (120, 240, 120), 2)
cv2.imwrite("outputs/mockup_solution.jpg", img)
print("wrote outputs/mockup_solution.jpg")
