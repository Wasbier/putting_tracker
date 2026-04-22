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

    def _predict_xyxy(
        self,
        frame_bgr: np.ndarray,
        scene_roi: tuple[int, int, int, int] | None = None,
    ) -> np.ndarray:
        """Run YOLO; return ``(N, 5)`` float32 ``[x1, y1, x2, y2, conf]`` in full-frame coords."""
        frame_in = frame_bgr
        ox = 0
        oy = 0
        if scene_roi is not None:
            rx, ry, rw, rh = scene_roi
            frame_in = frame_bgr[ry : ry + rh, rx : rx + rw]
            ox, oy = rx, ry
            if frame_in.size == 0:
                return np.zeros((0, 5), dtype=np.float32)

        results = self.model.predict(
            source=frame_in,
            conf=self.conf,
            imgsz=self.imgsz,
            device=self.device,
            verbose=False,
        )
        if not results:
            return np.zeros((0, 5), dtype=np.float32)
        boxes = results[0].boxes
        if boxes is None or len(boxes) == 0:
            return np.zeros((0, 5), dtype=np.float32)

        xyxy = boxes.xyxy.cpu().numpy().astype(np.float32)
        confs = boxes.conf.cpu().numpy().astype(np.float32)
        if confs.ndim == 0:
            confs = np.array([float(confs)], dtype=np.float32)
        xyxy[:, [0, 2]] += float(ox)
        xyxy[:, [1, 3]] += float(oy)
        return np.column_stack([xyxy, confs.reshape(-1, 1)]).astype(np.float32)

    def detect_xyxy(
        self,
        frame_bgr: np.ndarray,
        scene_roi: tuple[int, int, int, int] | None = None,
    ) -> np.ndarray:
        """Detections as ``(N, 5)`` rows: ``x1, y1, x2, y2, confidence`` (full image)."""
        return self._predict_xyxy(frame_bgr, scene_roi=scene_roi)

    def detect(
        self,
        frame_bgr: np.ndarray,
        scene_roi: tuple[int, int, int, int] | None = None,
    ) -> list[tuple[int, int, int, float]]:
        arr = self._predict_xyxy(frame_bgr, scene_roi=scene_roi)
        detections: list[tuple[int, int, int, float]] = []
        for row in arr:
            x1, y1, x2, y2 = row[:4]
            conf = float(row[4])
            cx = int(round((x1 + x2) * 0.5))
            cy = int(round((y1 + y2) * 0.5))
            r = int(round(max(x2 - x1, y2 - y1) * 0.5))
            detections.append((cx, cy, max(1, r), conf))
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
