"""Shared helpers for opening a webcam or video file."""

from __future__ import annotations

import cv2


def open_capture(camera: int | None, video_path: str | None) -> cv2.VideoCapture:
    if video_path:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise SystemExit(f"Could not open video file: {video_path}")
        return cap
    if camera is None:
        raise SystemExit("Specify --camera N or --video path")
    cap = cv2.VideoCapture(camera)
    if not cap.isOpened():
        raise SystemExit(f"Could not open camera index {camera}")
    return cap
