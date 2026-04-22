"""Draw hole/cup rectangles on a reference frame; save JSON for track_balls_video overlay."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2

CALIBRATION_VERSION = 1


def _order_xyxy(x1: int, y1: int, x2: int, y2: int) -> tuple[int, int, int, int]:
    if x1 > x2:
        x1, x2 = x2, x1
    if y1 > y2:
        y1, y2 = y2, y1
    return x1, y1, x2, y2


def _to_norm(x1: int, y1: int, x2: int, y2: int, fw: int, fh: int) -> tuple[float, float, float, float]:
    return x1 / fw, y1 / fh, x2 / fw, y2 / fh


def main() -> None:
    p = argparse.ArgumentParser(
        description="Click-drag rectangles on a still frame to calibrate hole regions."
    )
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--video", type=str, help="Video path; uses one frame (see --frame-index)")
    g.add_argument("--image", type=str, help="Image path (jpg/png)")
    p.add_argument(
        "--frame-index",
        type=int,
        default=0,
        help="Frame index to load from video (default first frame)",
    )
    p.add_argument(
        "--out",
        type=Path,
        default=Path("hole_calibration.json"),
        help="Output JSON path",
    )
    args = p.parse_args()

    if args.video:
        cap = cv2.VideoCapture(args.video)
        if not cap.isOpened():
            raise SystemExit(f"Could not open video: {args.video}")
        cap.set(cv2.CAP_PROP_POS_FRAMES, max(0, args.frame_index))
        ok, frame = cap.read()
        cap.release()
        if not ok or frame is None:
            raise SystemExit(f"Could not read frame {args.frame_index}")
        source = {"type": "video", "path": str(Path(args.video).resolve()), "frame_index": args.frame_index}
    else:
        frame = cv2.imread(args.image, cv2.IMREAD_COLOR)
        if frame is None:
            raise SystemExit(f"Could not read image: {args.image}")
        source = {"type": "image", "path": str(Path(args.image).resolve())}

    fh, fw = frame.shape[:2]
    base = frame.copy()

    state: dict = {
        "holes": [],
        "dragging": False,
        "x0": 0,
        "y0": 0,
        "x1": 0,
        "y1": 0,
    }

    def redraw() -> None:
        vis = base.copy()
        for i, h in enumerate(state["holes"]):
            x1, y1, x2, y2 = h["rect_pixels"]
            cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 200, 255), 2)
            cv2.putText(
                vis,
                h["id"],
                (x1, max(20, y1 - 8)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 200, 255),
                2,
                cv2.LINE_AA,
            )
        if state["dragging"]:
            x1, y1, x2, y2 = _order_xyxy(state["x0"], state["y0"], state["x1"], state["y1"])
            cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 255, 0), 1)
        cv2.putText(
            vis,
            "Drag=new rect  s=save rect  d=undo last  q=quit & write",
            (10, fh - 16),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (220, 220, 220),
            1,
            cv2.LINE_AA,
        )
        cv2.imshow(win, vis)

    def on_mouse(event: int, mx: int, my: int, flags: int, param: object) -> None:
        if event == cv2.EVENT_LBUTTONDOWN:
            state["dragging"] = True
            state["x0"], state["y0"] = mx, my
            state["x1"], state["y1"] = mx, my
        elif event == cv2.EVENT_MOUSEMOVE and state["dragging"]:
            state["x1"], state["y1"] = mx, my
        elif event == cv2.EVENT_LBUTTONUP and state["dragging"]:
            state["dragging"] = False
            state["x1"], state["y1"] = mx, my
        redraw()

    win = "Hole calibration (drag rectangle)"
    cv2.namedWindow(win)
    cv2.setMouseCallback(win, on_mouse)
    redraw()

    while True:
        key = cv2.waitKey(20) & 0xFF
        if key == ord("q"):
            break
        if key == ord("s"):
            if state["dragging"]:
                continue
            x1, y1, x2, y2 = _order_xyxy(state["x0"], state["y0"], state["x1"], state["y1"])
            if abs(x2 - x1) < 4 or abs(y2 - y1) < 4:
                print("Rectangle too small; drag again.", file=sys.stderr)
                continue
            x1 = max(0, min(fw - 1, x1))
            x2 = max(0, min(fw - 1, x2))
            y1 = max(0, min(fh - 1, y1))
            y2 = max(0, min(fh - 1, y2))
            hid = f"hole_{len(state['holes']) + 1}"
            rn = _to_norm(x1, y1, x2, y2, fw, fh)
            state["holes"].append(
                {
                    "id": hid,
                    "rect_pixels": [x1, y1, x2, y2],
                    "rect_norm": [round(rn[0], 6), round(rn[1], 6), round(rn[2], 6), round(rn[3], 6)],
                }
            )
            print(f"Saved {hid} pixels {state['holes'][-1]['rect_pixels']} norm {state['holes'][-1]['rect_norm']}")
            redraw()
        if key == ord("d") and state["holes"]:
            removed = state["holes"].pop()
            print(f"Removed {removed['id']}")
            redraw()

    cv2.destroyAllWindows()

    if not state["holes"]:
        raise SystemExit("No holes saved. Run again: drag a box around the cup, press s, then q.")

    payload = {
        "version": CALIBRATION_VERSION,
        "frame_width": fw,
        "frame_height": fh,
        "source": source,
        "holes": state["holes"],
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"Wrote {args.out.resolve()} with {len(state['holes'])} hole(s).")
    print("Use with track_balls_video.py: --calibration", args.out)


if __name__ == "__main__":
    main()
    sys.exit(0)
