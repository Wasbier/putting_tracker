"""Run trained YOLO model on a video and display detections in real-time."""
from __future__ import annotations

import argparse
import sys

import cv2

from yolo_ball_detector import YoloBallDetector, draw_yolo_boxes


def main() -> None:
    p = argparse.ArgumentParser(description="Test YOLO ball detection on a video")
    p.add_argument("--video", type=str, required=True, help="Path to video file")
    p.add_argument(
        "--model",
        type=str,
        default="runs/detect/runs/yolo_ball/golf_ball/weights/best.pt",
        help="Path to trained YOLO weights",
    )
    p.add_argument("--conf", type=float, default=0.25, help="Confidence threshold")
    p.add_argument("--imgsz", type=int, default=640, help="Inference image size")
    p.add_argument("--scale", type=float, default=1.0, help="Display scale")
    p.add_argument("--device", type=str, default=None, help="Device (cpu, 0, cuda:0)")
    args = p.parse_args()

    detector = YoloBallDetector(
        model_path=args.model,
        conf=args.conf,
        imgsz=args.imgsz,
        device=args.device,
    )

    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        raise SystemExit(f"Could not open video: {args.video}")

    window = "YOLO Ball Detection (q=quit, +/-=conf)"
    conf = args.conf

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        detector.conf = conf
        detections = detector.detect(frame)
        vis = draw_yolo_boxes(frame, detections)

        cv2.putText(
            vis,
            f"Balls: {len(detections)}  Conf: {conf:.2f}",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 0),
            2,
            cv2.LINE_AA,
        )

        if args.scale != 1.0:
            vis = cv2.resize(vis, None, fx=args.scale, fy=args.scale, interpolation=cv2.INTER_AREA)

        cv2.imshow(window, vis)
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        elif key == ord("+") or key == ord("="):
            conf = min(0.95, conf + 0.05)
            print(f"Conf threshold: {conf:.2f}")
        elif key == ord("-"):
            conf = max(0.05, conf - 0.05)
            print(f"Conf threshold: {conf:.2f}")

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
    sys.exit(0)
