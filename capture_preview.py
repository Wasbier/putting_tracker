"""
Preview webcam or a video file. Use this to verify angle, lighting, and FPS
before adding ball detection and counting logic.

Examples:
  python capture_preview.py --camera 0
  python capture_preview.py --video "C:/path/to/putts.mp4"
"""

from __future__ import annotations

import argparse
import sys

import cv2

from capture_utils import open_capture


def main() -> None:
    p = argparse.ArgumentParser(description="Webcam or file preview for putting tracker dev")
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--camera", type=int, metavar="N", help="Webcam device index (often 0)")
    g.add_argument("--video", type=str, metavar="PATH", help="Path to a video file (e.g. phone recording)")
    p.add_argument(
        "--scale",
        type=float,
        default=1.0,
        help="Display scale (e.g. 0.5 for smaller window)",
    )
    args = p.parse_args()

    cap = open_capture(args.camera, args.video)
    window = "Putting tracker — preview (q to quit)"

    try:
        while True:
            ok, frame = cap.read()
            if not ok or frame is None:
                if args.video:
                    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    continue
                break

            if args.scale != 1.0:
                frame = cv2.resize(
                    frame,
                    None,
                    fx=args.scale,
                    fy=args.scale,
                    interpolation=cv2.INTER_AREA,
                )

            cv2.imshow(window, frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
    sys.exit(0)
