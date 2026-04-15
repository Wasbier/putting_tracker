from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

try:
    from ultralytics import YOLO
except Exception:  # pragma: no cover - optional dependency at runtime
    YOLO = None


class YoloBallDetector:
    def __init__(
        self,
        model_path: str | Path,
        conf: float = 0.20,
        imgsz: int = 1280,
        device: str | None = None,
    ) -> None:
        if YOLO is None:
            raise RuntimeError(
                "Ultralytics YOLO is not available. Install dependencies with `pip install -r requirements.txt`."
            )
        self.model_path = str(model_path)
        self.conf = float(conf)
        self.imgsz = int(imgsz)
        self.device = device
        self.model = YOLO(self.model_path)

    def detect(
        self,
        frame_bgr: np.ndarray,
        scene_roi: tuple[int, int, int, int] | None = None,
    ) -> list[tuple[int, int, int, float]]:
        frame_in = frame_bgr
        ox = 0
        oy = 0
        if scene_roi is not None:
            rx, ry, rw, rh = scene_roi
            frame_in = frame_bgr[ry : ry + rh, rx : rx + rw]
            ox, oy = rx, ry
            if frame_in.size == 0:
                return []

        results = self.model.predict(
            source=frame_in,
            conf=self.conf,
            imgsz=self.imgsz,
            device=self.device,
            verbose=False,
        )

        detections: list[tuple[int, int, int, float]] = []
        if not results:
            return detections
        boxes = results[0].boxes
        if boxes is None:
            return detections

        xyxy = boxes.xyxy.cpu().numpy()
        confs = boxes.conf.cpu().numpy() if boxes.conf is not None else np.ones(len(xyxy))
        for (x1, y1, x2, y2), conf in zip(xyxy, confs):
            cx = int(round((x1 + x2) * 0.5)) + ox
            cy = int(round((y1 + y2) * 0.5)) + oy
            r = int(round(max(x2 - x1, y2 - y1) * 0.5))
            detections.append((cx, cy, max(1, r), float(conf)))
        detections.sort(key=lambda item: (item[1], item[0]))
        return detections


def draw_yolo_boxes(
    frame_bgr: np.ndarray,
    detections: list[tuple[int, int, int, float]],
) -> np.ndarray:
    vis = frame_bgr.copy()
    for x, y, r, conf in detections:
        cv2.circle(vis, (x, y), r, (255, 0, 255), 2)
        cv2.putText(
            vis,
            f"{conf:.2f}",
            (x + r + 2, max(18, y - r - 6)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (255, 0, 255),
            1,
            cv2.LINE_AA,
        )
    return vis
