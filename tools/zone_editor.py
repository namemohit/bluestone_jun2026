"""Preview (and optionally draw) counting lines for a camera.

Grabs a sample frame and renders the configured counting lines so you can confirm
placement + the inside (IN) direction before running the pipeline.

Preview from a still frame:
  python -m tools.zone_editor --config configs/cameras.yaml --camera cam_entry \
                              --frame outputs/_preview.jpg --out outputs/cam_entry_lines.jpg

Preview by grabbing a frame from the live source / NVR:
  python -m tools.zone_editor --config configs/cameras.yaml --camera cam_entry

Interactive draw (needs a desktop/GUI): click two points per line; prints normalized
coords to paste into the config. Press 'n' = next line, 's' = save, 'q' = quit.
  python -m tools.zone_editor --config configs/cameras.yaml --camera cam_entry --interactive
"""
from __future__ import annotations

import argparse
from pathlib import Path

import cv2

from common.config import get_camera, load_config
from vision.counter import build_lines
from vision.draw import draw_counting_lines


def grab_frame(source):
    from ingest.video_source import VideoStream

    vs = VideoStream(source)
    for _, _, frame in vs.frames():
        vs.release()
        return frame
    raise RuntimeError(f"Could not read a frame from {source!r}")


def _resolve_frame(args, cfg, cam):
    if args.frame:
        frame = cv2.imread(args.frame)
        if frame is None:
            raise RuntimeError(f"Could not read image: {args.frame}")
        return frame
    source = args.source
    if not source:
        from ingest.nvr import build_rtsp_url

        source = build_rtsp_url(cfg.get("nvr", {}), cam)
    return grab_frame(source)


def _interactive(frame, camera_id):
    """Click pairs of points; print normalized coords for the config."""
    h, w = frame.shape[:2]
    pts: list[tuple[int, int]] = []
    lines: list[list[tuple[int, int]]] = []

    def on_mouse(event, x, y, flags, _):
        if event == cv2.EVENT_LBUTTONDOWN:
            pts.append((x, y))
            if len(pts) == 2:
                lines.append([pts[0], pts[1]])
                pts.clear()

    win = f"zone_editor:{camera_id}  [n]ext  [s]ave  [q]uit"
    cv2.namedWindow(win)
    cv2.setMouseCallback(win, on_mouse)
    while True:
        vis = frame.copy()
        for a, b in lines:
            cv2.line(vis, a, b, (0, 215, 255), 2)
        for p in pts:
            cv2.circle(vis, p, 4, (0, 0, 255), -1)
        cv2.imshow(win, vis)
        key = cv2.waitKey(20) & 0xFF
        if key in (ord("q"), 27):
            break
        if key == ord("s"):
            print("\n# paste into the camera's counting_lines:")
            for i, (a, b) in enumerate(lines):
                print(
                    f"  - id: line_{i}\n"
                    f"    points: [[{a[0]/w:.4f}, {a[1]/h:.4f}], "
                    f"[{b[0]/w:.4f}, {b[1]/h:.4f}]]\n"
                    f"    inside: left   # flip to 'right' if the IN arrow points the wrong way"
                )
    cv2.destroyAllWindows()


def main() -> None:
    ap = argparse.ArgumentParser(description="Preview/draw counting lines for a camera")
    ap.add_argument("--config", required=True)
    ap.add_argument("--camera", required=True)
    ap.add_argument("--frame", help="use this still image instead of grabbing from the source")
    ap.add_argument("--source", help="override source (file path or rtsp:// URL)")
    ap.add_argument("--out", help="output overlay image path")
    ap.add_argument("--interactive", action="store_true")
    args = ap.parse_args()

    cfg = load_config(args.config)
    cam = get_camera(cfg, args.camera)
    frame = _resolve_frame(args, cfg, cam)

    if args.interactive:
        _interactive(frame, args.camera)
        return

    h, w = frame.shape[:2]
    lines = build_lines(cam, w, h)
    vis = draw_counting_lines(frame.copy(), lines)
    out = args.out or f"outputs/{args.camera}_lines.jpg"
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(out, vis)

    print(f"camera   : {args.camera} (role={cam.get('role')})")
    print(f"frame    : {w}x{h}")
    if not lines:
        print("lines    : (none configured for this camera)")
    for ln in lines:
        print(f"line[{ln.id}]: p1={ln.p1} p2={ln.p2} inside_sign={ln.inside_sign}")
    print(f"overlay  : {out}")


if __name__ == "__main__":
    main()
