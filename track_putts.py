"""
Ball tracking + simple putt counting (attempts when crossing start line toward cup;
made when the ball settles slowly near the cup center (not just inside the big cup box — avoids lip-outs / flyovers).

1) Calibrate (first time or with --calibrate):
   - Drag a rectangle around the cup / hole (ENTER to confirm).
   - Click two points along your START LINE (where the ball crosses when a putt begins).
     The script uses the cup position to learn which side is "toward the hole"; draw the
     line roughly perpendicular to the putt path.
   - Optional: drag a box around the SPARE BALL pile, ENTER (tiny = skip). Masked on re-acquire.
   - If extras stay in frame: draw step 4 TIGHT on YOUR ball. Re-acquire ANDs the mask to that box
     so blur cannot merge the pile in; tracking also drops any centroid inside the pile ROI.
   - Optional step 5: two clicks along the mat edge toward the floor/rocks — same side as the cup
     stays valid; the far side is ignored for ball detection.

2) Run without --calibrate to use tracking_config.json. Press r to reset on-screen counts.

Examples:
  python track_putts.py --video training_videos/first_test_night.mp4 --scale 0.75 --calibrate
  python track_putts.py --video training_videos/first_test_night.mp4 --scale 0.75
  python track_putts.py --video clip.mp4 --scale 0.75 --no-loop   # stay on last frame; default is to loop the file
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from capture_utils import open_capture

CONFIG_NAME = "tracking_config.json"


# --- geometry ---


def line_side(p: tuple[float, float], a: tuple[float, float], b: tuple[float, float]) -> float:
    """> 0: p is to the left of ray a->b (in image coords, y down)."""
    return (b[0] - a[0]) * (p[1] - a[1]) - (b[1] - a[1]) * (p[0] - a[0])


def point_in_rect(
    p: tuple[float, float],
    rect: tuple[float, float, float, float],
) -> bool:
    x, y, w, h = rect
    return x <= p[0] <= x + w and y <= p[1] <= y + h


def dist_sq(a: tuple[float, float], b: tuple[float, float]) -> float:
    dx = a[0] - b[0]
    dy = a[1] - b[1]
    return dx * dx + dy * dy


def dist_point_to_segment_sq(
    px: float,
    py: float,
    x1: float,
    y1: float,
    x2: float,
    y2: float,
) -> float:
    dx, dy = x2 - x1, y2 - y1
    lensq = dx * dx + dy * dy
    if lensq < 1e-6:
        return dist_sq((px, py), (x1, y1))
    t = max(0.0, min(1.0, ((px - x1) * dx + (py - y1) * dy) / lensq))
    qx, qy = x1 + t * dx, y1 + t * dy
    qdx, qdy = px - qx, py - qy
    return qdx * qdx + qdy * qdy


# --- config ---


def _norm_to_px(cfg: dict[str, Any], w: int, h: int) -> dict[str, Any]:
    cup = cfg["cup"]
    line = cfg["line"]
    return {
        "cup": (
            int(cup["x"] * w),
            int(cup["y"] * h),
            int(cup["w"] * w),
            int(cup["h"] * h),
        ),
        "line": (
            (int(line["x1"] * w), int(line["y1"] * h)),
            (int(line["x2"] * w), int(line["y2"] * h)),
        ),
    }


def scene_roi_from_calibration(
    cup_px: tuple[int, int, int, int],
    line_px: tuple[tuple[int, int], tuple[int, int]],
    fw: int,
    fh: int,
    pad_frac: float = 0.14,
    min_span_frac: float = 0.30,
) -> tuple[int, int, int, int]:
    """Axis-aligned ROI around cup + start line so we ignore sky/lights outside the green."""
    x, y, cw, ch = cup_px
    (lx1, ly1), (lx2, ly2) = line_px
    xs = [x, x + cw, lx1, lx2]
    ys = [y, y + ch, ly1, ly2]
    pad_x = int(fw * pad_frac)
    pad_y = int(fh * pad_frac)
    x0 = max(0, min(xs) - pad_x)
    x1 = min(fw, max(xs) + pad_x)
    y0 = max(0, min(ys) - pad_y)
    y1 = min(fh, max(ys) + pad_y)
    bw, bh = x1 - x0, y1 - y0
    min_w = int(fw * min_span_frac)
    min_h = int(fh * min_span_frac)
    if bw < min_w:
        cx = (x0 + x1) // 2
        x0 = max(0, cx - min_w // 2)
        x1 = min(fw, x0 + min_w)
        x0 = max(0, x1 - min_w)
    if bh < min_h:
        cy = (y0 + y1) // 2
        y0 = max(0, cy - min_h // 2)
        y1 = min(fh, y0 + min_h)
        y0 = max(0, y1 - min_h)
    return (x0, y0, x1 - x0, y1 - y0)


def load_config(path: Path, frame_w: int, frame_h: int) -> dict[str, Any]:
    data = json.loads(path.read_text(encoding="utf-8"))
    px = _norm_to_px(data, frame_w, frame_h)
    c = px["cup"]
    cup_center = (c[0] + c[2] / 2.0, c[1] + c[3] / 2.0)
    p1, p2 = px["line"]
    s = line_side(cup_center, p1, p2)
    if abs(s) < 1e-3:
        raise SystemExit("Cup center lies too close to the start line; redraw calibration.")
    cup_left_sign = 1.0 if s > 0 else -1.0
    scene = scene_roi_from_calibration(c, (p1, p2), frame_w, frame_h)
    ignore_pile: tuple[int, int, int, int] | None = None
    ip = data.get("ignore_pile")
    if isinstance(ip, dict) and {"x", "y", "w", "h"}.issubset(ip.keys()):
        ignore_pile = (
            int(ip["x"] * frame_w),
            int(ip["y"] * frame_h),
            int(ip["w"] * frame_w),
            int(ip["h"] * frame_h),
        )
    active_ball: tuple[int, int, int, int] | None = None
    ab = data.get("active_ball")
    if isinstance(ab, dict) and {"x", "y", "w", "h"}.issubset(ab.keys()):
        active_ball = (
            int(ab["x"] * frame_w),
            int(ab["y"] * frame_h),
            int(ab["w"] * frame_w),
            int(ab["h"] * frame_h),
        )
    green_edge_line: tuple[tuple[int, int], tuple[int, int]] | None = None
    green_valid_sign: float | None = None
    ge = data.get("green_edge")
    if isinstance(ge, dict) and {"x1", "y1", "x2", "y2"}.issubset(ge.keys()):
        e1 = (int(ge["x1"] * frame_w), int(ge["y1"] * frame_h))
        e2 = (int(ge["x2"] * frame_w), int(ge["y2"] * frame_h))
        sge = line_side(cup_center, e1, e2)
        if abs(sge) >= 3.0:
            green_edge_line = (e1, e2)
            green_valid_sign = 1.0 if sge > 0 else -1.0
    out: dict[str, Any] = {
        "cup_px": c,
        "line_px": (p1, p2),
        "cup_left_sign": cup_left_sign,
        "scene_roi": scene,
        "ignore_pile_px": ignore_pile,
        "active_ball_px": active_ball,
        "green_edge_line": green_edge_line,
        "green_valid_sign": green_valid_sign,
    }
    return out


def save_config_norm(
    path: Path,
    cup_roi: tuple[int, int, int, int],
    line_pts: tuple[tuple[int, int], tuple[int, int]],
    w: int,
    h: int,
    ignore_pile_roi: tuple[int, int, int, int] | None = None,
    active_ball_roi: tuple[int, int, int, int] | None = None,
    green_edge_pts: tuple[tuple[int, int], tuple[int, int]] | None = None,
) -> None:
    x, y, rw, rh = cup_roi
    (x1, y1), (x2, y2) = line_pts
    data: dict[str, Any] = {
        "cup": {"x": x / w, "y": y / h, "w": rw / w, "h": rh / h},
        "line": {"x1": x1 / w, "y1": y1 / h, "x2": x2 / w, "y2": y2 / h},
    }
    if ignore_pile_roi is not None:
        ix, iy, iw, ih = ignore_pile_roi
        if iw >= 8 and ih >= 8:
            data["ignore_pile"] = {"x": ix / w, "y": iy / h, "w": iw / w, "h": ih / h}
    if active_ball_roi is not None:
        ax, ay, aw, ah = active_ball_roi
        if aw >= 8 and ah >= 8:
            data["active_ball"] = {"x": ax / w, "y": ay / h, "w": aw / w, "h": ah / h}
    if green_edge_pts is not None:
        (gx1, gy1), (gx2, gy2) = green_edge_pts
        data["green_edge"] = {
            "x1": gx1 / w,
            "y1": gy1 / h,
            "x2": gx2 / w,
            "y2": gy2 / h,
        }
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")


# --- calibration UI ---


def calibrate_interactive(frame: np.ndarray, config_path: Path) -> None:
    h, w = frame.shape[:2]
    disp = frame.copy()
    cv2.putText(
        disp,
        "1) Drag rectangle for CUP region, press ENTER",
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 255, 255),
        2,
        cv2.LINE_AA,
    )
    cv2.imshow("Calibrate", disp)
    cup = cv2.selectROI("Calibrate", frame, showCrosshair=True, fromCenter=False)
    cup = (int(cup[0]), int(cup[1]), int(cup[2]), int(cup[3]))
    if cup[2] <= 0 or cup[3] <= 0:
        raise SystemExit("Cup ROI empty; try again.")

    points: list[tuple[int, int]] = []

    def on_mouse(event: int, x: int, y: int, _flags: int, _param: Any) -> None:
        if event == cv2.EVENT_LBUTTONDOWN and len(points) < 2:
            points.append((x, y))

    cv2.setMouseCallback("Calibrate", on_mouse)
    while len(points) < 2:
        img = frame.copy()
        cv2.putText(
            img,
            "2) Click TWO points on START LINE | SPACE confirm | c clear",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (0, 255, 0),
            2,
            cv2.LINE_AA,
        )
        cv2.rectangle(img, (cup[0], cup[1]), (cup[0] + cup[2], cup[1] + cup[3]), (0, 165, 255), 2)
        for p in points:
            cv2.circle(img, p, 6, (0, 255, 0), -1)
        if len(points) == 2:
            cv2.line(img, points[0], points[1], (0, 255, 0), 2)
        cv2.imshow("Calibrate", img)
        key = cv2.waitKey(30) & 0xFF
        if key == ord("c"):
            points.clear()
        if key == ord(" ") and len(points) == 2:
            break

    cv2.setMouseCallback("Calibrate", lambda *a, **k: None)
    img3 = frame.copy()
    cv2.rectangle(img3, (cup[0], cup[1]), (cup[0] + cup[2], cup[1] + cup[3]), (0, 165, 255), 2)
    cv2.line(img3, points[0], points[1], (0, 255, 0), 2)
    cv2.putText(
        img3,
        "3) OPTIONAL: drag box around SPARE BALL PILE only, ENTER (very small = skip)",
        (10, 28),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.48,
        (200, 200, 255),
        2,
        cv2.LINE_AA,
    )
    cv2.imshow("Calibrate", img3)
    pile = cv2.selectROI("Calibrate", frame, showCrosshair=True, fromCenter=False)
    pile_t = (int(pile[0]), int(pile[1]), int(pile[2]), int(pile[3]))
    ignore_pile = pile_t if pile_t[2] >= 8 and pile_t[3] >= 8 else None

    img4 = frame.copy()
    cv2.rectangle(img4, (cup[0], cup[1]), (cup[0] + cup[2], cup[1] + cup[3]), (0, 165, 255), 2)
    cv2.line(img4, points[0], points[1], (0, 255, 0), 2)
    if ignore_pile is not None:
        cv2.rectangle(
            img4,
            (ignore_pile[0], ignore_pile[1]),
            (ignore_pile[0] + ignore_pile[2], ignore_pile[1] + ignore_pile[3]),
            (180, 105, 255),
            2,
        )
    cv2.putText(
        img4,
        "4) Putt ball: drag TIGHT box where YOUR ball is at address, ENTER (small = skip)",
        (10, 26),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.45,
        (80, 255, 80),
        2,
        cv2.LINE_AA,
    )
    cv2.imshow("Calibrate", img4)
    ab = cv2.selectROI("Calibrate", frame, showCrosshair=True, fromCenter=False)
    ab_t = (int(ab[0]), int(ab[1]), int(ab[2]), int(ab[3]))
    active_ball = ab_t if ab_t[2] >= 8 and ab_t[3] >= 8 else None

    edge_pts: list[tuple[int, int]] = []

    def on_edge(event: int, x: int, y: int, _flags: int, _param: Any) -> None:
        if event == cv2.EVENT_LBUTTONDOWN and len(edge_pts) < 2:
            edge_pts.append((x, y))

    cv2.setMouseCallback("Calibrate", on_edge)
    skip_edge = False
    while not skip_edge:
        img5 = frame.copy()
        cv2.rectangle(img5, (cup[0], cup[1]), (cup[0] + cup[2], cup[1] + cup[3]), (0, 165, 255), 2)
        cv2.line(img5, points[0], points[1], (0, 255, 0), 2)
        if ignore_pile is not None:
            cv2.rectangle(
                img5,
                (ignore_pile[0], ignore_pile[1]),
                (ignore_pile[0] + ignore_pile[2], ignore_pile[1] + ignore_pile[3]),
                (180, 105, 255),
                2,
            )
        if active_ball is not None:
            cv2.rectangle(
                img5,
                (active_ball[0], active_ball[1]),
                (active_ball[0] + active_ball[2], active_ball[1] + active_ball[3]),
                (80, 255, 80),
                2,
            )
        cv2.putText(
            img5,
            "5) OPTIONAL green/floor edge: TWO clicks (e.g. along mat edge) | n skip | c clear | SPACE",
            (10, 24),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.42,
            (255, 200, 100),
            2,
            cv2.LINE_AA,
        )
        for p in edge_pts:
            cv2.circle(img5, p, 6, (255, 200, 100), -1)
        if len(edge_pts) == 2:
            cv2.line(img5, edge_pts[0], edge_pts[1], (255, 200, 100), 2)
        cv2.imshow("Calibrate", img5)
        key = cv2.waitKey(30) & 0xFF
        if key == ord("n"):
            skip_edge = True
            edge_pts.clear()
            break
        if key == ord("c"):
            edge_pts.clear()
        if key == ord(" ") and len(edge_pts) == 2:
            break
    cv2.setMouseCallback("Calibrate", lambda *a, **k: None)
    green_edge = (edge_pts[0], edge_pts[1]) if len(edge_pts) == 2 else None
    cv2.destroyWindow("Calibrate")
    save_config_norm(
        config_path, cup, (points[0], points[1]), w, h, ignore_pile, active_ball, green_edge
    )
    print(f"Saved {config_path.resolve()}")
    if ignore_pile and not active_ball:
        print("Tip: with spare balls in frame, draw step 4 around YOUR ball only — otherwise the pile often wins.")


# --- ball detection ---


@dataclass
class DetectorParams:
    v_min: int = 90
    s_max: int = 120
    min_area: int = 40
    min_circularity: float = 0.52
    # max ball area scales with resolution (full-frame huge blobs = lights, not ball)
    max_area_frac: float = 1.0 / 350.0


def _max_ball_area(fw: int, fh: int, frac: float) -> int:
    return max(800, int(fw * fh * frac))


def _zero_mask_rectangle(
    mask: np.ndarray,
    rect: tuple[int, int, int, int],
    pad: int,
) -> None:
    """Clear mask inside rect expanded by pad (clamped to image). Removes pile blobs even if centroid falls outside a tight box."""
    fh, fw = mask.shape[:2]
    ix, iy, iw, ih = rect
    if iw < 1 or ih < 1:
        return
    x0 = max(0, ix - pad)
    y0 = max(0, iy - pad)
    x1 = min(fw, ix + iw + pad)
    y1 = min(fh, iy + ih + pad)
    if x1 > x0 and y1 > y0:
        mask[y0:y1, x0:x1] = 0


def expand_roi_xywh(
    rect: tuple[int, int, int, int],
    pad: int,
    fw: int,
    fh: int,
) -> tuple[float, float, float, float]:
    x, y, w, h = rect
    x0 = max(0, x - pad)
    y0 = max(0, y - pad)
    x1 = min(fw, x + w + pad)
    y1 = min(fh, y + h + pad)
    return (float(x0), float(y0), float(x1 - x0), float(y1 - y0))


def cup_ignore_zone(cup_px: tuple[int, int, int, int], scale: float = 1.38) -> tuple[int, int, int, int]:
    """Expanded cup box: ignore bright blobs here right after a make (hole rim / glare)."""
    x, y, w, h = cup_px
    cx = x + w * 0.5
    cy = y + h * 0.5
    nw, nh = w * scale, h * scale
    nx = int(cx - nw * 0.5)
    ny = int(cy - nh * 0.5)
    return (nx, ny, max(1, int(nw)), max(1, int(nh)))


def find_ball(
    frame_bgr: np.ndarray,
    last_xy: tuple[float, float] | None,
    params: DetectorParams,
    scene_roi: tuple[int, int, int, int] | None,
    line_px: tuple[tuple[int, int], tuple[int, int]],
    cup_px: tuple[int, int, int, int],
    max_jump_px: float,
    cup_left_sign: float,
    ignore_near_cup: tuple[int, int, int, int] | None = None,
    ignore_pile_px: tuple[int, int, int, int] | None = None,
    active_ball_px: tuple[int, int, int, int] | None = None,
    green_edge_line: tuple[tuple[int, int], tuple[int, int]] | None = None,
    green_valid_sign: float | None = None,
) -> tuple[float, float] | None:
    fh, fw = frame_bgr.shape[:2]
    max_area = _max_ball_area(fw, fh, params.max_area_frac)
    pile_pad = max(48, int(min(fw, fh) * 0.055))

    hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)
    lower = np.array([0, 0, params.v_min], dtype=np.uint8)
    upper = np.array([180, params.s_max, 255], dtype=np.uint8)
    mask = cv2.inRange(hsv, lower, upper)
    # Drop obvious green turf (helps white ball pop on green)
    green = cv2.inRange(hsv, (35, 40, 40), (90, 255, 255))
    mask = cv2.bitwise_and(mask, cv2.bitwise_not(green))
    if last_xy is None and ignore_pile_px is not None:
        _zero_mask_rectangle(mask, ignore_pile_px, pile_pad)
    mask = cv2.medianBlur(mask, 5)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))

    if scene_roi is not None:
        rx, ry, rw, rh = scene_roi
        roi_mask = np.zeros((fh, fw), dtype=np.uint8)
        roi_mask[ry : ry + rh, rx : rx + rw] = 255
        mask = cv2.bitwise_and(mask, roi_mask)

    if last_xy is None and ignore_pile_px is not None:
        _zero_mask_rectangle(mask, ignore_pile_px, pile_pad)

    # Re-acquire: only pixels inside address box (tight pad). Stops pile blobs merging in via blur.
    if last_xy is None and active_ball_px is not None:
        strict_pad = max(6, int(min(fw, fh) * 0.008))
        zx, zy, zw, zh = expand_roi_xywh(active_ball_px, strict_pad, fw, fh)
        xi, yi, wi, hi = int(zx), int(zy), int(zw), int(zh)
        if wi >= 4 and hi >= 4:
            zone_m = np.zeros((fh, fw), dtype=np.uint8)
            zone_m[yi : yi + hi, xi : xi + wi] = 255
            mask = cv2.bitwise_and(mask, zone_m)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    (lx1, ly1), (lx2, ly2) = line_px
    cx_cup = cup_px[0] + cup_px[2] * 0.5
    cy_cup = cup_px[1] + cup_px[3] * 0.5
    lmx, lmy = (lx1 + lx2) * 0.5, (ly1 + ly2) * 0.5
    cw, ch = cup_px[2], cup_px[3]
    hx0 = max(0, min(fw - 1, int(cx_cup - cw * 0.18)))
    hy0 = max(0, min(fh - 1, int(cy_cup - ch * 0.18)))
    hw = max(1, min(fw - hx0, int(cw * 0.36)))
    hh = max(1, min(fh - hy0, int(ch * 0.36)))
    hole_center_rect = (hx0, hy0, hw, hh)
    approach_release = min(cw, ch) * 0.62
    # Typical address is on the tee side: from cup through the line, past the line midpoint.
    vx, vy = lmx - cx_cup, lmy - cy_cup
    ln = math.hypot(vx, vy) or 1.0
    ux, uy = vx / ln, vy / ln
    tee_off = max(48.0, min(fw, fh) * 0.08)
    address_anchor = (lmx + ux * tee_off, lmy + uy * tee_off)

    candidates: list[tuple[float, tuple[float, float]]] = []
    for c in contours:
        a = cv2.contourArea(c)
        if a < params.min_area or a > max_area:
            continue
        peri = cv2.arcLength(c, True)
        if peri <= 1e-6:
            continue
        circ = 4.0 * math.pi * a / (peri * peri)
        if circ < params.min_circularity:
            continue
        m = cv2.moments(c)
        if m["m00"] == 0:
            continue
        cx = m["m10"] / m["m00"]
        cy = m["m01"] / m["m00"]
        if ignore_near_cup is not None and point_in_rect((cx, cy), ignore_near_cup):
            continue
        if green_edge_line is not None and green_valid_sign is not None:
            ge1, ge2 = green_edge_line
            if green_valid_sign * line_side((cx, cy), ge1, ge2) <= 1e-3:
                continue
        if last_xy is not None:
            d = math.sqrt(dist_sq((cx, cy), last_xy))
            if d > max_jump_px:
                continue
            d_last_cup = math.hypot(last_xy[0] - cx_cup, last_xy[1] - cy_cup)
            if d_last_cup > approach_release and point_in_rect((cx, cy), hole_center_rect):
                continue
        score = circ * math.sqrt(a)
        candidates.append((score, (cx, cy)))

    if ignore_pile_px is not None:
        pile_rect = expand_roi_xywh(ignore_pile_px, pile_pad, fw, fh)
        candidates = [t for t in candidates if not point_in_rect(t[1], pile_rect)]

    if not candidates:
        return None
    if last_xy is not None:
        best = max(candidates, key=lambda t: (t[0], -dist_sq(t[1], last_xy)))
        return best[1]

    def init_key(t: tuple[float, tuple[float, float]]) -> tuple[float, float, float, float]:
        px, py = t[1]
        d_line = math.sqrt(dist_point_to_segment_sq(px, py, float(lx1), float(ly1), float(lx2), float(ly2)))
        return (
            dist_sq(t[1], address_anchor),
            d_line,
            -py,
            -t[0],
        )

    if active_ball_px is not None:
        return min(candidates, key=init_key)[1]

    # Init: prefer tee side of line; among those, closest to address.
    p1, p2 = (lx1, ly1), (lx2, ly2)
    tee_only: list[tuple[float, tuple[float, float]]] = []
    for t in candidates:
        px, py = t[1]
        if cup_left_sign * line_side((px, py), p1, p2) >= 0:
            continue
        tee_only.append(t)
    pool = tee_only if tee_only else candidates

    d_line_cap = max(95.0, min(fw, fh) * 0.105)
    line_pool: list[tuple[float, tuple[float, float]]] = []
    for t in pool:
        px, py = t[1]
        d_line = math.sqrt(dist_point_to_segment_sq(px, py, float(lx1), float(ly1), float(lx2), float(ly2)))
        if d_line <= d_line_cap:
            line_pool.append(t)
    pool2 = line_pool if line_pool else pool

    return min(pool2, key=init_key)[1]


# --- putt logic ---

# Require ball on tee side this many frames before a line-cross counts (stops drag-in false attempts).
MIN_TEE_FRAMES_FOR_STROKE = 22
MIN_TEE_FRAMES_FIRST_STROKE = 10
# After a make, new ball at same address: clear "already made" once settled on tee side.
FRAMES_ON_TEE_TO_ARM_NEXT_MAKE = 10
# Need this many frames on tee before "made" can register (blocks false make at clip start).
MIN_ADDRESS_FRAMES_FOR_MADE = 10
# Made geometry: must be near cup center (fraction of min(cup w,h)), not only inside loose ROI.
MADE_CUP_CENTER_FRAC = 0.44
CUP_INNER_MARGIN_FRAC = 0.17
# Raw detection speed (px per frame); lip-outs stay fast, holed putts slow before counting as made.
MADE_MAX_SPEED_PPF = 8.5
# Consecutive frames ball must satisfy made geometry + speed before registering a make.
MADE_DWELL_FRAMES = 5


@dataclass
class PuttCounter:
    attempts: int = 0
    made: int = 0
    armed: bool = True
    last_side: float | None = None
    in_cup_frames: int = 0
    made_for_current_roll: bool = False
    cooldown_frames: int = 0
    # True once this stroke has an attempt (line cross or implied by make).
    counted_attempt_this_stroke: bool = False
    back_side_streak: int = 0
    tee_consecutive: int = 0
    cup_suppress_frames: int = 0
    pending_tracker_reset: bool = False
    away_cup_streak: int = 0
    addressed_ok: bool = False

    def clear_line_memory(self) -> None:
        self.last_side = None

    def tick_timers(self) -> None:
        if self.cup_suppress_frames > 0:
            self.cup_suppress_frames -= 1

    def pull_tracker_reset(self) -> bool:
        if not self.pending_tracker_reset:
            return False
        self.pending_tracker_reset = False
        return True

    def _on_make_registered(self) -> None:
        self.made_for_current_roll = True
        self.pending_tracker_reset = True
        self.cup_suppress_frames = 60
        self.addressed_ok = False

    def update(
        self,
        ball: tuple[float, float] | None,
        cup_px: tuple[int, int, int, int],
        line_px: tuple[tuple[int, int], tuple[int, int]],
        cup_left_sign: float,
        ball_speed_ppf: float | None = None,
    ) -> None:
        p1, p2 = line_px
        if self.cooldown_frames > 0:
            self.cooldown_frames -= 1

        if ball is None:
            return

        prev_tee = self.tee_consecutive
        side_raw = line_side(ball, p1, p2)
        side = cup_left_sign * side_raw

        if side < 0:
            self.back_side_streak += 1
        else:
            self.back_side_streak = 0

        # Ready for a new stroke's attempt counting after ball has sat on tee side briefly.
        if self.back_side_streak >= 5:
            self.counted_attempt_this_stroke = False

        # Same spot, new ball: you don't always cross the line "backward" after a make.
        if self.made_for_current_roll and self.back_side_streak >= FRAMES_ON_TEE_TO_ARM_NEXT_MAKE:
            self.made_for_current_roll = False
            self.in_cup_frames = 0

        x, y, w, h = cup_px
        cup_cx = x + w * 0.5
        cup_cy = y + h * 0.5
        dist_cup = math.hypot(ball[0] - cup_cx, ball[1] - cup_cy)
        rim = math.hypot(w, h) * 0.55
        if self.made_for_current_roll and dist_cup > rim * 1.9:
            self.away_cup_streak += 1
        else:
            self.away_cup_streak = 0
        if self.made_for_current_roll and self.away_cup_streak >= 8:
            self.made_for_current_roll = False
            self.in_cup_frames = 0

        if self.last_side is not None and abs(self.last_side) > 1e-3 and abs(side) > 1e-3:
            if (
                self.last_side < 0
                and side > 0
                and self.armed
                and self.cooldown_frames == 0
                and prev_tee
                >= (
                    MIN_TEE_FRAMES_FOR_STROKE
                    if (self.attempts > 0 or self.made > 0)
                    else MIN_TEE_FRAMES_FIRST_STROKE
                )
            ):
                self.attempts += 1
                self.counted_attempt_this_stroke = True
                self.made_for_current_roll = False
                self.in_cup_frames = 0
                self.armed = False
                self.cooldown_frames = 14
                self.addressed_ok = True

        self.last_side = side

        if side < 0:
            self.armed = True

        m = CUP_INNER_MARGIN_FRAC
        inner_w, inner_h = w * (1 - m), h * (1 - m)
        ox, oy = x + (w - inner_w) / 2, y + (h - inner_h) / 2
        inner_rect = (ox, oy, inner_w, inner_h)
        made_radius = min(w, h) * MADE_CUP_CENTER_FRAC
        near_cup_center = dist_cup <= made_radius
        in_inner = point_in_rect(ball, inner_rect)
        speed_ok = (
            ball_speed_ppf is not None
            and ball_speed_ppf <= MADE_MAX_SPEED_PPF
        )
        if (
            in_inner
            and near_cup_center
            and speed_ok
            and not self.made_for_current_roll
            and (self.addressed_ok or self.counted_attempt_this_stroke)
        ):
            self.in_cup_frames += 1
            if self.in_cup_frames >= MADE_DWELL_FRAMES:
                if not self.counted_attempt_this_stroke:
                    self.attempts += 1
                    self.counted_attempt_this_stroke = True
                self.made += 1
                self._on_make_registered()
        else:
            self.in_cup_frames = 0

        if side < 0:
            self.tee_consecutive += 1
        else:
            self.tee_consecutive = 0

        if self.tee_consecutive >= MIN_ADDRESS_FRAMES_FOR_MADE:
            self.addressed_ok = True


def _avg_displacement_ppf(positions: deque[tuple[float, float]]) -> float | None:
    """Mean step length between consecutive raw detections (px per frame)."""
    if len(positions) < 2:
        return None
    pts = list(positions)
    # Recent segment only so a fast approach earlier in the deque does not block a slow settle in the cup.
    if len(pts) > 6:
        pts = pts[-6:]
    total = 0.0
    for i in range(1, len(pts)):
        total += math.hypot(pts[i][0] - pts[i - 1][0], pts[i][1] - pts[i - 1][1])
    return total / (len(pts) - 1)


def main() -> None:
    p = argparse.ArgumentParser(description="Track putts from webcam or video")
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--camera", type=int, metavar="N")
    g.add_argument("--video", type=str, metavar="PATH")
    p.add_argument("--scale", type=float, default=1.0)
    p.add_argument(
        "--config",
        type=Path,
        default=Path(CONFIG_NAME),
        help=f"Calibration JSON (default: {CONFIG_NAME})",
    )
    p.add_argument("--calibrate", action="store_true", help="Draw cup ROI + start line, save config")
    p.add_argument(
        "--no-loop",
        action="store_true",
        help="When the video ends, stop on the last frame instead of restarting (counts are not reset).",
    )
    args = p.parse_args()

    cap = open_capture(args.camera, args.video)
    ok, frame0 = cap.read()
    if not ok or frame0 is None:
        raise SystemExit("Could not read first frame.")
    fh, fw = frame0.shape[:2]

    if args.calibrate:
        calibrate_interactive(frame0, args.config)

    if not args.config.is_file():
        raise SystemExit(f"Missing {args.config}; run once with --calibrate")

    cfg = load_config(args.config, fw, fh)
    cup_px = cfg["cup_px"]
    line_px = cfg["line_px"]
    cup_left_sign = cfg["cup_left_sign"]
    scene_roi = cfg["scene_roi"]
    ignore_pile_px: tuple[int, int, int, int] | None = cfg.get("ignore_pile_px")
    active_ball_px: tuple[int, int, int, int] | None = cfg.get("active_ball_px")
    green_edge_line: tuple[tuple[int, int], tuple[int, int]] | None = cfg.get("green_edge_line")
    green_valid_sign: float | None = cfg.get("green_valid_sign")
    max_jump = max(36.0, min(fw, fh) * 0.11)
    hold_frames = 12
    lose_track_frames = hold_frames * 8

    detector = DetectorParams()
    counter = PuttCounter()
    last_ball: tuple[float, float] | None = None
    smooth: tuple[float, float] | None = None
    alpha = 0.35
    miss_streak = 0
    jump_boost_frames = 0
    raw_positions: deque[tuple[float, float]] = deque(maxlen=12)

    window = "Putting tracker (q quit, r reset stats)"
    print("q = quit | r = reset attempt/made counts")
    print(f"Tracking zone from calibration; max jump {max_jump:.0f}px — cyan box")
    if ignore_pile_px is not None:
        print("ignore_pile ROI active (masked on re-acquire)")
    if active_ball_px is not None:
        print("active_ball ROI active — re-acquire only inside green box")
    elif ignore_pile_px is not None:
        print("Re-run --calibrate and complete step 4 (tight box on YOUR ball) if the pile still steals lock.")
    if green_edge_line is not None:
        print("green_edge active — blobs on the floor side of the orange line are ignored")

    try:
        while True:
            ok, frame = cap.read()
            if not ok or frame is None:
                if args.video and not args.no_loop:
                    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    counter = PuttCounter()
                    last_ball = None
                    smooth = None
                    miss_streak = 0
                    jump_boost_frames = 0
                    raw_positions.clear()
                    continue
                break

            counter.tick_timers()
            ignore_cup = cup_ignore_zone(cup_px) if counter.cup_suppress_frames > 0 else None

            effective_jump = max_jump * (2.35 if jump_boost_frames > 0 else 1.0)
            raw_ball = find_ball(
                frame,
                last_ball,
                detector,
                scene_roi,
                line_px,
                cup_px,
                max_jump_px=effective_jump,
                cup_left_sign=cup_left_sign,
                ignore_near_cup=ignore_cup,
                ignore_pile_px=ignore_pile_px,
                active_ball_px=active_ball_px,
                green_edge_line=green_edge_line,
                green_valid_sign=green_valid_sign,
            )
            if raw_ball is not None:
                raw_positions.append(raw_ball)
                if last_ball is None:
                    jump_boost_frames = 28
                miss_streak = 0
                last_ball = raw_ball
                if smooth is None:
                    smooth = raw_ball
                else:
                    smooth = (
                        alpha * raw_ball[0] + (1 - alpha) * smooth[0],
                        alpha * raw_ball[1] + (1 - alpha) * smooth[1],
                    )
            else:
                miss_streak += 1
                if miss_streak > 8:
                    raw_positions.clear()
                if miss_streak > hold_frames:
                    counter.clear_line_memory()
                if counter.made_for_current_roll and miss_streak > 14:
                    last_ball = None
                    smooth = None
                elif miss_streak > lose_track_frames:
                    last_ball = None
                    smooth = None
                    raw_positions.clear()
            if jump_boost_frames > 0:
                jump_boost_frames -= 1

            ball_for_logic = smooth if smooth is not None and miss_streak <= hold_frames else None
            ball_speed = _avg_displacement_ppf(raw_positions)
            counter.update(
                ball_for_logic,
                cup_px,
                line_px,
                cup_left_sign,
                ball_speed_ppf=ball_speed,
            )
            if counter.pull_tracker_reset():
                last_ball = None
                smooth = None
                miss_streak = 0
                jump_boost_frames = 32
                raw_positions.clear()
                counter.clear_line_memory()

            vis = frame.copy()
            sx, sy, sw, sh = scene_roi
            cv2.rectangle(vis, (sx, sy), (sx + sw, sy + sh), (255, 255, 0), 1)
            x, y, w, h = cup_px
            cv2.rectangle(vis, (x, y), (x + w, y + h), (0, 165, 255), 2)
            ccx, ccy = int(x + w * 0.5), int(y + h * 0.5)
            made_r = int(min(w, h) * MADE_CUP_CENTER_FRAC)
            cv2.circle(vis, (ccx, ccy), made_r, (180, 255, 200), 1)
            cv2.line(vis, line_px[0], line_px[1], (0, 255, 0), 2)
            if ignore_pile_px is not None:
                ix, iy, iw, ih = ignore_pile_px
                cv2.rectangle(vis, (ix, iy), (ix + iw, iy + ih), (180, 105, 255), 1)
            if active_ball_px is not None:
                ax, ay, aw, ah = active_ball_px
                cv2.rectangle(vis, (ax, ay), (ax + aw, ay + ah), (80, 255, 80), 1)
            if green_edge_line is not None:
                cv2.line(vis, green_edge_line[0], green_edge_line[1], (255, 200, 100), 2)
            if smooth is not None:
                cv2.circle(vis, (int(smooth[0]), int(smooth[1])), 10, (255, 0, 255), 2)

            label = f"Attempts: {counter.attempts}  Made: {counter.made}"
            cv2.putText(
                vis,
                label,
                (10, fh - 20),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.9,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )

            disp = vis
            if args.scale != 1.0:
                disp = cv2.resize(
                    vis,
                    None,
                    fx=args.scale,
                    fy=args.scale,
                    interpolation=cv2.INTER_AREA,
                )
            cv2.imshow(window, disp)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            if key == ord("r"):
                counter = PuttCounter()
                last_ball = None
                smooth = None
                raw_positions.clear()
    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
    sys.exit(0)
