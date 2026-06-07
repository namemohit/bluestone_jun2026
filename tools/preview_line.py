"""Draw a candidate counting line on a still frame (non-interactive) for human confirmation.

  python -m tools.preview_line --image outputs/c05_1000.jpg \
      --line 0.25,0.77,0.63,0.65 --inside right --out outputs/c05_line_preview.jpg

Green line = the counting boundary. The shaded side is "inside" (counts as IN when a
person's feet cross onto it). Arrow shows the IN direction.
"""
from __future__ import annotations

import argparse

import cv2
import numpy as np

from vision.geometry import cross


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", required=True)
    ap.add_argument("--line", required=True, help="x1,y1,x2,y2 normalized, P1=left -> P2=right")
    ap.add_argument("--inside", default="right", help="left|right (right = below the line)")
    ap.add_argument("--out", default="outputs/line_preview.jpg")
    args = ap.parse_args()

    img = cv2.imread(args.image)
    if img is None:
        raise SystemExit(f"could not read {args.image}")
    h, w = img.shape[:2]
    x1, y1, x2, y2 = (float(v) for v in args.line.split(","))
    p1 = (int(x1 * w), int(y1 * h))
    p2 = (int(x2 * w), int(y2 * h))
    want = 1 if args.inside.strip().lower() == "right" else -1

    # shade the inside region (every pixel on the chosen side of the directed line)
    overlay = img.copy()
    ys, xs = np.mgrid[0:h, 0:w]
    cr = (p2[0] - p1[0]) * (ys - p1[1]) - (p2[1] - p1[1]) * (xs - p1[0])
    mask = (cr > 0) if want == 1 else (cr < 0)
    overlay[mask] = (0, 140, 0)
    img = cv2.addWeighted(overlay, 0.22, img, 0.78, 0)

    # the line itself
    cv2.line(img, p1, p2, (0, 230, 0), 4, cv2.LINE_AA)
    for p in (p1, p2):
        cv2.circle(img, p, 7, (0, 230, 0), -1)

    # IN-direction arrow (from line midpoint toward the inside)
    mid = ((p1[0] + p2[0]) // 2, (p1[1] + p2[1]) // 2)
    nrm = np.array([-(p2[1] - p1[1]), (p2[0] - p1[0])], float)  # right-normal
    if want == -1:
        nrm = -nrm
    nrm = nrm / (np.linalg.norm(nrm) + 1e-9)
    tip = (int(mid[0] + nrm[0] * 90), int(mid[1] + nrm[1] * 90))
    cv2.arrowedLine(img, mid, tip, (0, 230, 0), 4, cv2.LINE_AA, tipLength=0.3)
    cv2.putText(img, "IN", (tip[0] + 6, tip[1] + 6), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 230, 0), 2)

    cv2.putText(img, "ENTRY COUNTING LINE (C05)", (16, h - 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 230, 0), 2)

    cv2.imwrite(args.out, img)
    print(f"wrote {args.out}  ({w}x{h})  line px {p1}->{p2}  inside={args.inside}")


if __name__ == "__main__":
    main()
