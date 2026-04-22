"""
Lightweight multi-object tracker for golf balls (YOLO boxes).

Greedy nearest-neighbour matching, no external MOT libraries.
Uses NumPy for distances; OpenCV for drawing helpers only.
"""
from __future__ import annotations

import heapq
import json
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
import cv2
import numpy as np


def circularity_from_bbox_crop(
    frame_bgr: np.ndarray,
    x1: float,
    y1: float,
    x2: float,
    y2: float,
    *,
    pad: int = 2,
) -> float:
    """
    Rough circularity in ``[0, 1]`` from the largest contour in a grayscale Otsu crop.
    Tries normal and inverted binary (ball can be brighter or darker than mat).
    """
    h, w = frame_bgr.shape[:2]
    xi1 = max(0, int(np.floor(x1)) - pad)
    yi1 = max(0, int(np.floor(y1)) - pad)
    xi2 = min(w, int(np.ceil(x2)) + pad)
    yi2 = min(h, int(np.ceil(y2)) + pad)
    if xi2 <= xi1 or yi2 <= yi1:
        return 0.0
    crop = frame_bgr[yi1:yi2, xi1:xi2]
    if crop.size < 30:
        return 0.0
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (3, 3), 0)
    _, bw = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    best = 0.0
    for mask in (bw, 255 - bw):
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for c in contours:
            area = float(cv2.contourArea(c))
            if area < 12.0:
                continue
            peri = float(cv2.arcLength(c, True))
            if peri < 1e-3:
                continue
            circ = (4.0 * np.pi * area) / (peri * peri)
            best = max(best, float(min(circ, 1.0)))
    return best


def aspect_ratio_xyxy(x1: float, y1: float, x2: float, y2: float) -> float:
    w = max(float(x2 - x1), 1e-3)
    h = max(float(y2 - y1), 1e-3)
    return w / h


def iou_xyxy(
    ax1: float,
    ay1: float,
    ax2: float,
    ay2: float,
    bx1: float,
    by1: float,
    bx2: float,
    by2: float,
) -> float:
    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)
    iw = max(0.0, ix2 - ix1)
    ih = max(0.0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0.0:
        return 0.0
    aa = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    ba = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = aa + ba - inter
    return float(inter / union) if union > 1e-9 else 0.0


@dataclass
class BallTrack:
    """Internal mutable track state."""

    track_id: int
    x1: float
    y1: float
    x2: float
    y2: float
    confidence: float
    cx: float
    cy: float
    prev_cx: float
    prev_cy: float
    vel_x: float
    vel_y: float
    frames_since_seen: int
    position_history: deque[tuple[float, float]] = field(repr=False)
    confirmed: bool = False
    good_window: deque[bool] = field(default_factory=lambda: deque(maxlen=5), repr=False)
    last_circularity: float = 0.0
    last_aspect_ratio: float = 1.0
    good_streak: int = 0

    @staticmethod
    def from_detection(
        track_id: int,
        row: np.ndarray,
        history_len: int,
        gate_m_window: int,
    ) -> BallTrack:
        x1, y1, x2, y2, conf = float(row[0]), float(row[1]), float(row[2]), float(row[3]), float(row[4])
        cx = 0.5 * (x1 + x2)
        cy = 0.5 * (y1 + y2)
        hist: deque[tuple[float, float]] = deque(maxlen=history_len)
        hist.append((cx, cy))
        gw: deque[bool] = deque(maxlen=max(1, int(gate_m_window)))
        return BallTrack(
            track_id=track_id,
            x1=x1,
            y1=y1,
            x2=x2,
            y2=y2,
            confidence=conf,
            cx=cx,
            cy=cy,
            prev_cx=cx,
            prev_cy=cy,
            vel_x=0.0,
            vel_y=0.0,
            frames_since_seen=0,
            position_history=hist,
            confirmed=False,
            good_window=gw,
            last_circularity=0.0,
            last_aspect_ratio=aspect_ratio_xyxy(x1, y1, x2, y2),
        )


@dataclass(frozen=True)
class TrackedBall:
    """Per-frame snapshot for one ball (output of :meth:`MultiObjectBallTracker.update`)."""

    # Public id for overlay: 0 while the FP gate has not yet confirmed the track
    # (treat as "no id"). ``association_id`` is always the internal stable id.
    track_id: int
    association_id: int
    x1: float
    y1: float
    x2: float
    y2: float
    confidence: float
    cx: float
    cy: float
    moving: bool
    vel_x: float
    vel_y: float
    position_history: tuple[tuple[float, float], ...]
    confirmed: bool
    circularity: float
    aspect_ratio: float
    # 0 = YOLO matched this frame; >0 = no detection this many frames (stale holdover)
    coast_frames: int


class MultiObjectBallTracker:
    """
    Track 1–10 golf balls with stable IDs: greedy minimum-distance matching.

    * ``update`` ingests detections ``(N, 5)`` as ``x1, y1, x2, y2, conf``.
    * Unmatched detections spawn new tracks; unmatched tracks increment
      ``frames_since_seen`` and are dropped after ``max_frames_lost``.
    * Association uses **predicted centroid**, **velocity-scaled** distance cap, optional
      **IoU** gate, greedy pairing by (-IoU, dist), and **recycled** track IDs (lowest free).
    * Optional **FP gate**: a new track must pass **K consecutive** "good" frames
      (see ``min_confirm_streak``: stricter conf+aspect+circ) before
      :attr:`TrackedBall.confirmed` is True. **Until then**, association uses the
      base match distance only, **shorter** ``max_frames_lost_unconfirmed`` (drop
      FPs fast), and no relaxed thresholds. **After** confirmation, the tracker is
      more forgiving: ``conf_low`` + circularity factor, ``match_distance_bonus``,
      and ``max_frames_lost_confirmed`` (longer) so a real ball is not lost to brief
      occlusion or weak YOLO frames.
    """

    def __init__(
        self,
        max_match_distance: float = 90.0,
        max_frames_lost: int = 22,
        motion_pixel_threshold: float = 5.0,
        history_len: int = 10,
        *,
        match_use_velocity: bool = True,
        match_velocity_scale: float = 2.0,
        min_match_iou: float = 0.02,
        match_effective_dist_cap: float = 220.0,
        fp_gate: bool = True,
        gate_m_window: int = 5,
        conf_high: float = 0.30,
        conf_low: float = 0.20,
        max_aspect_ratio: float = 1.45,
        min_circularity: float = 0.42,
        confirmed_circularity_factor: float = 0.85,
        # Consecutive good frames before confirming (rejects short YOLO FPs).
        min_confirm_streak: int = 8,
        # If unseen, drop **confirmed** tracks after this many frames (or auto from
        # *max_frames_lost*). This is the generous (post-confirm) coast budget.
        max_frames_lost_confirmed: int | None = None,
        # Stricter: drop **unconfirmed** (still proving) tracks after this many
        # missed frames. None = min(12, max(4, ~0.35 * max_frames_lost)) to shed FPs
        # quickly while a real ball is expected to keep matching until it confirms.
        max_frames_lost_unconfirmed: int | None = None,
        # Extra max match distance (px) for **confirmed** tracks (association only).
        match_distance_bonus_confirmed: float = 20.0,
    ) -> None:
        self.max_match_distance = float(max_match_distance)
        self.motion_pixel_threshold = float(motion_pixel_threshold)
        self.history_len = int(history_len)
        self.match_use_velocity = bool(match_use_velocity)
        self.match_velocity_scale = float(match_velocity_scale)
        self.min_match_iou = float(min_match_iou)
        self.match_effective_dist_cap = float(match_effective_dist_cap)
        self.fp_gate = bool(fp_gate)
        mcs = max(1, int(min_confirm_streak))
        self.min_confirm_streak = mcs
        self.gate_m_window = max(mcs, int(gate_m_window))
        self.conf_high = float(conf_high)
        self.conf_low = float(conf_low)
        self.max_aspect_ratio = float(max_aspect_ratio)
        self.min_circularity = float(min_circularity)
        self.confirmed_circularity_factor = float(confirmed_circularity_factor)
        self.match_distance_bonus_confirmed = max(
            0.0, float(match_distance_bonus_confirmed)
        )
        mfl = int(max_frames_lost)
        self.max_frames_lost = mfl
        if max_frames_lost_confirmed is None:
            self._max_frames_lost_confirmed = max(mfl, int(round(mfl * 1.5)))
        else:
            self._max_frames_lost_confirmed = max(1, int(max_frames_lost_confirmed))
        if max_frames_lost_unconfirmed is None:
            # Tight: candidates that miss often are FPs, not a ball yet
            self._max_frames_lost_unconfirmed = min(12, max(4, int(round(mfl * 0.35))))
        else:
            self._max_frames_lost_unconfirmed = max(1, int(max_frames_lost_unconfirmed))
        self._tracks: list[BallTrack] = []
        self._next_id = 1
        self._free_ids: list[int] = []

    def _alloc_track_id(self) -> int:
        if self._free_ids:
            return int(heapq.heappop(self._free_ids))
        tid = self._next_id
        self._next_id += 1
        return tid

    @property
    def tracks(self) -> list[BallTrack]:
        """Read-only view of internal tracks (mutable objects; treat as internal)."""
        return self._tracks

    def match_detections_to_tracks(
        self,
        det_centroids: np.ndarray,
        track_centroids: np.ndarray,
    ) -> list[tuple[int, int]]:
        """
        Legacy centroid-only greedy match (distance ascending).

        Prefer :meth:`match_detections_assoc` from :meth:`update`; kept for tests.
        """
        if det_centroids.size == 0 or track_centroids.size == 0:
            return []
        diff = track_centroids[:, None, :] - det_centroids[None, :, :]
        dists = np.sqrt(np.sum(diff * diff, axis=2))
        candidates: list[tuple[float, int, int]] = []
        t_count, d_count = dists.shape
        for ti in range(t_count):
            for dj in range(d_count):
                d = float(dists[ti, dj])
                if d <= self.max_match_distance:
                    candidates.append((d, ti, dj))
        candidates.sort(key=lambda item: item[0])
        used_t: set[int] = set()
        used_d: set[int] = set()
        pairs: list[tuple[int, int]] = []
        for _, ti, dj in candidates:
            if ti in used_t or dj in used_d:
                continue
            used_t.add(ti)
            used_d.add(dj)
            pairs.append((ti, dj))
        return pairs

    def match_detections_assoc(self, dets: np.ndarray) -> list[tuple[int, int]]:
        """
        Greedy association: allow pair if dist to predicted centroid <= effective max
        (base + speed * scale, capped) OR IoU >= min_match_iou.

        Sort by (-IoU, dist) so overlapping same-ball pairs win over distant ones.
        """
        if dets.size == 0 or not self._tracks:
            return []
        d_count = int(dets.shape[0])
        candidates: list[tuple[tuple[float, float], int, int]] = []
        for ti, t in enumerate(self._tracks):
            if self.match_use_velocity:
                tcx = float(t.cx + t.vel_x)
                tcy = float(t.cy + t.vel_y)
            else:
                tcx, tcy = float(t.cx), float(t.cy)
            spd = float(np.hypot(t.vel_x, t.vel_y))
            base = self.max_match_distance + self.match_velocity_scale * spd
            if t.confirmed:
                base += self.match_distance_bonus_confirmed
            eff_max = min(self.match_effective_dist_cap, base)
            tx1, ty1, tx2, ty2 = float(t.x1), float(t.y1), float(t.x2), float(t.y2)
            for dj in range(d_count):
                dx1, dy1, dx2, dy2 = (
                    float(dets[dj, 0]),
                    float(dets[dj, 1]),
                    float(dets[dj, 2]),
                    float(dets[dj, 3]),
                )
                dcx = 0.5 * (dx1 + dx2)
                dcy = 0.5 * (dy1 + dy2)
                dist = float(np.hypot(tcx - dcx, tcy - dcy))
                iou = iou_xyxy(tx1, ty1, tx2, ty2, dx1, dy1, dx2, dy2)
                ok = dist <= eff_max
                if self.min_match_iou > 0.0:
                    ok = ok or iou >= self.min_match_iou
                if ok:
                    candidates.append(((-iou, dist), ti, dj))
        candidates.sort(key=lambda item: (item[0][0], item[0][1]))
        used_t: set[int] = set()
        used_d: set[int] = set()
        pairs: list[tuple[int, int]] = []
        for _, ti, dj in candidates:
            if ti in used_t or dj in used_d:
                continue
            used_t.add(ti)
            used_d.add(dj)
            pairs.append((ti, dj))
        return pairs

    def remove_lost_tracks(self) -> None:
        """
        Unconfirmed: ``_max_frames_lost_unconfirmed`` (strict). Confirmed:
        ``_max_frames_lost_confirmed`` (lenient). Recycle dropped IDs.
        """
        kept: list[BallTrack] = []
        for t in self._tracks:
            limit = (
                self._max_frames_lost_confirmed
                if t.confirmed
                else self._max_frames_lost_unconfirmed
            )
            if t.frames_since_seen <= limit:
                kept.append(t)
            else:
                heapq.heappush(self._free_ids, int(t.track_id))
        self._tracks = kept

    def _append_fp_gate(self, t: BallTrack, row: np.ndarray, frame_bgr: np.ndarray | None) -> None:
        """One sample for FP gate. Stricter (conf_high, full circ) until *confirmed*."""
        if not self.fp_gate:
            t.confirmed = True
            return
        x1, y1, x2, y2, conf = float(row[0]), float(row[1]), float(row[2]), float(row[3]), float(row[4])
        t.last_aspect_ratio = aspect_ratio_xyxy(x1, y1, x2, y2)
        mar = self.max_aspect_ratio
        ar_ok = (1.0 / mar) <= t.last_aspect_ratio <= mar
        if frame_bgr is not None:
            t.last_circularity = circularity_from_bbox_crop(frame_bgr, x1, y1, x2, y2)
        else:
            t.last_circularity = 1.0
        if self.min_circularity <= 0.0:
            circ_ok = True
        else:
            thr = self.min_circularity * (
                self.confirmed_circularity_factor if t.confirmed else 1.0
            )
            circ_ok = t.last_circularity >= thr
        conf_thr = self.conf_low if t.confirmed else self.conf_high
        good = (conf >= conf_thr) and ar_ok and circ_ok
        if good:
            t.good_streak += 1
        else:
            t.good_streak = 0
        t.good_window.append(good)
        if not t.confirmed:
            if t.good_streak >= self.min_confirm_streak:
                t.confirmed = True

    def update(
        self,
        detections: np.ndarray,
        frame_bgr: np.ndarray | None = None,
    ) -> list[TrackedBall]:
        """
        ``detections`` shape ``(N, 5)``: ``x1, y1, x2, y2, confidence``.

        Pass ``frame_bgr`` (same image YOLO ran on) to enable circularity in the FP gate.

        Returns a list of :class:`TrackedBall` for this frame (after matching and removal).
        """
        dets = np.asarray(detections, dtype=np.float64)
        if dets.ndim == 1 and dets.size == 5:
            dets = dets.reshape(1, 5)
        if dets.ndim != 2 or dets.shape[1] != 5:
            raise ValueError("detections must be (N, 5) with rows [x1,y1,x2,y2,conf]")

        if dets.size == 0:
            for t in self._tracks:
                t.frames_since_seen += 1
                if self.fp_gate:
                    t.good_window.append(False)
                    t.good_streak = 0
            self.remove_lost_tracks()
            return []

        if not self._tracks:
            for j in range(len(dets)):
                tr = BallTrack.from_detection(
                    self._alloc_track_id(), dets[j], self.history_len, self.gate_m_window
                )
                self._append_fp_gate(tr, dets[j], frame_bgr)
                self._tracks.append(tr)
            self.remove_lost_tracks()
            return [self._snapshot(t) for t in self._tracks]

        pairs = self.match_detections_assoc(dets)
        matched_t = {ti for ti, _ in pairs}
        matched_d = {dj for _, dj in pairs}

        for ti, dj in pairs:
            self._apply_match(self._tracks[ti], dets[dj])
            self._append_fp_gate(self._tracks[ti], dets[dj], frame_bgr)

        for ti, t in enumerate(self._tracks):
            if ti not in matched_t:
                t.frames_since_seen += 1
                if self.fp_gate:
                    t.good_window.append(False)
                    t.good_streak = 0

        for dj in range(len(dets)):
            if dj not in matched_d:
                tr = BallTrack.from_detection(
                    self._alloc_track_id(), dets[dj], self.history_len, self.gate_m_window
                )
                self._append_fp_gate(tr, dets[dj], frame_bgr)
                self._tracks.append(tr)

        self.remove_lost_tracks()
        return [self._snapshot(t) for t in self._tracks]

    def _apply_match(self, t: BallTrack, row: np.ndarray) -> None:
        x1, y1, x2, y2, conf = float(row[0]), float(row[1]), float(row[2]), float(row[3]), float(row[4])
        ncx = 0.5 * (x1 + x2)
        ncy = 0.5 * (y1 + y2)
        t.prev_cx, t.prev_cy = t.cx, t.cy
        t.vel_x = ncx - t.prev_cx
        t.vel_y = ncy - t.prev_cy
        t.x1, t.y1, t.x2, t.y2 = x1, y1, x2, y2
        t.confidence = conf
        t.cx, t.cy = ncx, ncy
        t.frames_since_seen = 0
        t.position_history.append((ncx, ncy))

    def _snapshot(self, t: BallTrack) -> TrackedBall:
        disp = float(np.hypot(t.cx - t.prev_cx, t.cy - t.prev_cy))
        moving = disp > self.motion_pixel_threshold
        hist = tuple(t.position_history)
        confirmed = True if not self.fp_gate else t.confirmed
        disp_id = 0 if (self.fp_gate and not t.confirmed) else int(t.track_id)
        return TrackedBall(
            track_id=disp_id,
            association_id=int(t.track_id),
            x1=t.x1,
            y1=t.y1,
            x2=t.x2,
            y2=t.y2,
            confidence=t.confidence,
            cx=t.cx,
            cy=t.cy,
            moving=moving,
            vel_x=t.vel_x,
            vel_y=t.vel_y,
            position_history=hist,
            confirmed=confirmed,
            circularity=t.last_circularity,
            aspect_ratio=t.last_aspect_ratio,
            coast_frames=int(t.frames_since_seen),
        )


def draw_tracked_balls(
    frame_bgr: np.ndarray,
    tracked: list[TrackedBall],
    *,
    hole_rects_xyxy: list[tuple[float, float, float, float]] | None = None,
    show_unconfirmed: bool = False,
    show_candidate_ids: bool = False,
) -> np.ndarray:
    """
    Draw boxes, IDs, centroids. Moving balls: green boxes; stationary: orange.

    By default, **only confirmed** tracks are drawn (after
    :attr:`MultiObjectBallTracker.min_confirm_streak` consecutive good frames;
    default 8). Set *show_unconfirmed* True to also draw pre-confirm tracks in
    gray (debug).

    Tracks with ``coast_frames > 0`` had no YOLO match this frame: we do not
    draw a frozen last box (avoids a 1f flash looking like 30+ frames in video).

    Uses BGR colors consistent with OpenCV.
    """
    vis = frame_bgr.copy()
    if hole_rects_xyxy:
        for hx1, hy1, hx2, hy2 in hole_rects_xyxy:
            p1 = (int(round(hx1)), int(round(hy1)))
            p2 = (int(round(hx2)), int(round(hy2)))
            cv2.rectangle(vis, p1, p2, (255, 200, 0), 2)
    move_color = (0, 220, 0)
    rest_color = (0, 140, 255)
    cand_color = (160, 160, 160)
    for b in tracked:
        if b.coast_frames > 0:
            continue
        if not b.confirmed and not show_unconfirmed:
            continue
        if b.confirmed:
            col = move_color if b.moving else rest_color
            thick = 2
            tag = f"id{b.track_id} {'mov' if b.moving else 'stop'}"
        else:
            col = cand_color
            thick = 1
            if show_candidate_ids:
                tag = f"id{b.association_id} cand c={b.confidence:.2f} circ={b.circularity:.2f}"
            else:
                tag = f"cand c={b.confidence:.2f} circ={b.circularity:.2f}"
        p1 = (int(round(b.x1)), int(round(b.y1)))
        p2 = (int(round(b.x2)), int(round(b.y2)))
        cv2.rectangle(vis, p1, p2, col, thick)
        cc = (int(round(b.cx)), int(round(b.cy)))
        cv2.circle(vis, cc, 3, col, -1, lineType=cv2.LINE_AA)
        cv2.putText(
            vis,
            tag,
            (p1[0], max(18, p1[1] - 6)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            col,
            1 if not b.confirmed else 2,
            cv2.LINE_AA,
        )
    return vis


def load_calibration_hole_rects_pixels(
    path: str | Path,
    frame_w: int,
    frame_h: int,
) -> list[tuple[float, float, float, float]]:
    """Load ``rect_norm`` entries from ``calibrate_holes.py`` JSON as pixel ``xyxy``."""
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    holes = data.get("holes")
    if not isinstance(holes, list):
        return []
    out: list[tuple[float, float, float, float]] = []
    for h in holes:
        rn = h.get("rect_norm")
        if not isinstance(rn, list) or len(rn) != 4:
            continue
        x1, y1, x2, y2 = float(rn[0]) * frame_w, float(rn[1]) * frame_h, float(rn[2]) * frame_w, float(rn[3]) * frame_h
        if x1 > x2:
            x1, x2 = x2, x1
        if y1 > y2:
            y1, y2 = y2, y1
        out.append((x1, y1, x2, y2))
    return out
