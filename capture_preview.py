"""
Preview webcam or a video file. Use this to verify angle, lighting, and FPS
before adding ball detection and counting logic.

Examples:
  python capture_preview.py --camera 0
  python capture_preview.py --video "C:/path/to/putts.mp4"
  python capture_preview.py --stream "rtsp://user:pass@cam/stream1" --record
"""

from __future__ import annotations

import argparse
import datetime as dt
import sys
from pathlib import Path

import cv2

from capture_utils import open_capture, open_stream


def main() -> None:
    p = argparse.ArgumentParser(description="Webcam or file preview for putting tracker dev")
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--camera", type=int, metavar="N", help="Webcam device index (often 0)")
    g.add_argument("--video", type=str, metavar="PATH", help="Path to a video file (e.g. phone recording)")
    g.add_argument("--stream", type=str, metavar="URL", help="RTSP/HTTP stream URL")
    p.add_argument(
        "--scale",
        type=float,
        default=1.0,
        help="Display scale (e.g. 0.5 for smaller window)",
    )
    p.add_argument(
        "--record",
        action="store_true",
        help="Save the preview to a timestamped MP4 under recordings/.",
    )
    p.add_argument(
        "--record-to",
        type=Path,
        default=None,
        help="Explicit output MP4 path. If omitted with --record, a timestamped file is created.",
    )
    args = p.parse_args()

    if args.stream:
        cap = open_stream(args.stream)
    else:
        cap = open_capture(args.camera, args.video)
    window = "Putting tracker — preview (q to quit)"
    writer: cv2.VideoWriter | None = None
    first_frame = None

    if args.record or args.record_to is not None:
        ok, frame0 = cap.read()
        if not ok or frame0 is None:
            cap.release()
            raise SystemExit("Could not read first frame for recording.")
        fh, fw = frame0.shape[:2]
        first_frame = frame0
        out_path = args.record_to
        if out_path is None:
            stamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
            out_path = Path("recordings") / f"capture_{stamp}.mp4"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fps = float(cap.get(cv2.CAP_PROP_FPS)) or 25.0
        if fps <= 0 or fps > 120:
            fps = 25.0
        writer = cv2.VideoWriter(
            str(out_path),
            cv2.VideoWriter_fourcc(*"mp4v"),
            fps,
            (fw, fh),
        )
        if not writer.isOpened():
            cap.release()
            raise SystemExit(f"Could not open VideoWriter for {out_path}")
        print(f"Recording to {out_path.resolve()} @ {fps:.2f} fps")
        if args.video:
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            first_frame = None

    try:
        while True:
            if first_frame is not None:
                frame = first_frame
                first_frame = None
                ok = True
            else:
                ok, frame = cap.read()
            if not ok or frame is None:
                if args.video:
                    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    continue
                break

            if writer is not None:
                writer.write(frame)

            disp = frame
            if args.scale != 1.0:
                disp = cv2.resize(
                    frame,
                    None,
                    fx=args.scale,
                    fy=args.scale,
                    interpolation=cv2.INTER_AREA,
                )

            cv2.imshow(window, disp)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
    finally:
        cap.release()
        if writer is not None:
            writer.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
    sys.exit(0)
