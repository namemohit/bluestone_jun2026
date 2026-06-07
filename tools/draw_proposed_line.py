"""Overlay a PROPOSED counting line (perpendicular to entry flow) + flow arrows on the
traffic map, alongside the OLD mis-placed line, for visual comparison."""
import cv2

img = cv2.imread("outputs/trafficmap.jpg")
h, w = img.shape[:2]


def P(x, y):
    return (int(x * w), int(y * h))


# OLD line (mis-placed, horizontal-ish across the glass-door zone) — red, for contrast
cv2.line(img, P(0.25, 0.77), P(0.63, 0.65), (60, 60, 255), 2)
cv2.putText(img, "OLD (misses entries)", P(0.26, 0.74),
            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (60, 60, 255), 2)

# PROPOSED line: steep / vertical, perpendicular to the down-left entry flow — cyan
cv2.line(img, P(0.34, 0.55), P(0.52, 1.0), (255, 255, 0), 4)
cv2.putText(img, "PROPOSED (perp. to flow)", P(0.36, 0.52),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)

# entry-flow arrows (magenta): people move down-and-left into the store
for sx, sy in [(0.70, 0.70), (0.62, 0.86), (0.55, 0.95)]:
    cv2.arrowedLine(img, P(sx, sy), P(sx - 0.20, sy + 0.07), (255, 0, 255), 3, tipLength=0.35)
cv2.putText(img, "entry flow (IN: right->left)", P(0.40, 0.92),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 255), 2)

cv2.imwrite("outputs/proposed_line.jpg", img)
print("wrote outputs/proposed_line.jpg")
