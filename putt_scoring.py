"""
Single-ball putt counting and make/miss using hole rectangles from calibrate_holes.py.

Does not require stable track IDs: picks one primary ball per frame. With
hole geometry, the primary is the moving ball if any else the one nearest
any hole (avoids choosing a spare ball by YOLO confidence alone).
Resists brief YOLO dropouts: does not reset "armed" state on 1–2 frame gaps.
Make/miss: padded hole rects + overlap with detection box, not centroid-only.
When the ball is in the cup, YOLO box jitter often keeps the tracker in "mov"
(never "still") — if so, a make is also scored after N consecutive in-hole frames
without waiting for global settle.
Balls that drop into the cup often disappear to YOLO on the next frames: if we
saw the ball in the hole once, a much shorter "no detection" wait scores MADE;
if we did not, the last position is re-checked with extra padding in case the
last box was on the edge of the cup.

The ball tracker's N-of-M "good frames" only gates **track confirmation**, not
the putt state machine. Stroke start is debounced here (consecutive effective
move and/or an optional N-of-M window on ``ready``). An optional bottom frame
band ignores motion for arming when the centroid sits in that strip (foot/shoe
FPs). There is no putter in the ball model; use ``--putt-exclude-bottom``,
``--roi`` on the detector, or extra training data if you need stricter filtering.
"""
from __future__ import annotations

import json
import math
import sys
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from ball_mot_tracker import TrackedBall


@dataclass(frozen=True)
class HoleTarget:
    hole_id: str
    x1: float
    y1: float
    x2: float
    y2: float


def load_hole_targets(path: str | Path, frame_w: int, frame_h: int) -> list[HoleTarget]:
    """Load hole ids and rects from hole_calibration.json (uses rect_norm)."""
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    out: list[HoleTarget] = []
    for h in data.get("holes", []):
        if not isinstance(h, dict):
            continue
        rn = h.get("rect_norm")
        hid = str(h.get("id", "hole"))
        if not isinstance(rn, list) or len(rn) != 4:
            continue
        x1, y1, x2, y2 = (
            float(rn[0]) * frame_w,
            float(rn[1]) * frame_h,
            float(rn[2]) * frame_w,
            float(rn[3]) * frame_h,
        )
        if x1 > x2:
            x1, x2 = x2, x1
        if y1 > y2:
            y1, y2 = y2, y1
        out.append(HoleTarget(hid, x1, y1, x2, y2))
    return out


def _pad_hole(h: HoleTarget, pad: float) -> tuple[float, float, float, float]:
    return h.x1 - pad, h.y1 - pad, h.x2 + pad, h.y2 + pad


def _centroid_in_rect(
    cx: float, cy: float, x1: float, y1: float, x2: float, y2: float
) -> bool:
    return x1 <= cx <= x2 and y1 <= cy <= y2


def _boxes_overlap(
    ax1: float, ay1: float, ax2: float, ay2: float,
    bx1: float, by1: float, bx2: float, by2: float,
) -> bool:
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    return ix1 < ix2 and iy1 < iy2


def ball_hole_outcome(
    cx: float,
    cy: float,
    x1: float,
    y1: float,
    x2: float,
    y2: float,
    holes: list[HoleTarget],
    pad: float,
) -> str | None:
    """
    Legacy: centroid in padded hole *or* ball box overlaps padded hole.
    Prefer :func:`ball_in_hole_make` for make/miss (centroid-only; less "lip" FPs).
    """
    for h in holes:
        px1, py1, px2, py2 = _pad_hole(h, pad)
        if _centroid_in_rect(cx, cy, px1, py1, px2, py2):
            return h.hole_id
        if _boxes_overlap(x1, y1, x2, y2, px1, py1, px2, py2):
            return h.hole_id
    return None


def ball_in_hole_make(
    cx: float, cy: float, holes: list[HoleTarget], pad: float
) -> str | None:
    """
    Stricter make condition: **centroid** inside the hole rect expanded by ``pad``
    only (no box-overlap shortcut). Stops fast passes from scoring when the box
    merely grazes a large padded region.
    """
    for h in holes:
        px1, py1, px2, py2 = _pad_hole(h, pad)
        if _centroid_in_rect(cx, cy, px1, py1, px2, py2):
            return h.hole_id
    return None


def ball_in_hole_make_putt_stroke(
    cx: float,
    cy: float,
    holes: list[HoleTarget],
    pad: float,
    putt_anchor: tuple[float, float],
    past_cup_min_forward_px: float,
    in_cup_tight_pad: float,
) -> str | None:
    """
    Like :func:`ball_in_hole_make` with the same loose ``pad`, but if the
    centroid lies in the padded rect **only** because the pad is large, a ball
    that has clearly rolled **past** the cup (along the line from address to
    hole) is **not** "in" the make zone. Real makes use the *tight* box
    (``in_cup_tight_pad``) to remain eligible even if slightly past the cup on
    the image when the true cup center sits offset from calibration.

    * ``putt_forward`` = hole_center - address. If
      (centroid - hole_center) · u_hat > ``past_cup_min_forward_px`` and the
    centroid is not in the *tight* in-cup rect, the ball is out for scoring.
    * If |hole_center - address| is very small, the past-cup test is skipped
    (unstable line for tap-ins from on top of the cup).
    """
    ax, ay = putt_anchor
    for h in holes:
        px1, py1, px2, py2 = _pad_hole(h, pad)
        if not _centroid_in_rect(cx, cy, px1, py1, px2, py2):
            continue
        hx = 0.5 * (h.x1 + h.x2)
        hy = 0.5 * (h.y1 + h.y2)
        t1, t2, t3, t4 = _pad_hole(h, float(in_cup_tight_pad))
        in_tight = _centroid_in_rect(cx, cy, t1, t2, t3, t4)
        if past_cup_min_forward_px > 0.0 and not in_tight:
            ux, uy = hx - ax, hy - ay
            nu = math.hypot(ux, uy)
            if nu >= 8.0:
                vx, vy = cx - hx, cy - hy
                forward = (vx * ux + vy * uy) / nu
                if forward > past_cup_min_forward_px:
                    continue
        return h.hole_id
    return None


def _sio_zone_hole_id(
    cx: float,
    cy: float,
    holes: list[HoleTarget],
    pad: float,
    c: "PuttScoringConfig",
    putt_anchor: tuple[float, float] | None,
) -> str | None:
    """
    In/out zone for simple scoring during a live stroke. Uses past-cup logic
    when an address anchor and past margin are configured.
    """
    if putt_anchor is not None and c.putt_past_cup_min_forward_px > 0.0:
        return ball_in_hole_make_putt_stroke(
            cx,
            cy,
            holes,
            pad,
            putt_anchor,
            c.putt_past_cup_min_forward_px,
            c.in_cup_tight_pad_px,
        )
    return ball_in_hole_make(cx, cy, holes, pad)


def _nearest_hole_dist_sq(cx: float, cy: float, holes: list[HoleTarget]) -> float:
    """Squared distance from a point to the nearest hole center."""
    if not holes:
        return float("inf")
    best = float("inf")
    for h in holes:
        mx = 0.5 * (h.x1 + h.x2)
        my = 0.5 * (h.y1 + h.y2)
        dx, dy = cx - mx, cy - my
        d2 = dx * dx + dy * dy
        if d2 < best:
            best = d2
    return best


def _putt_arming_near_hole(
    c: PuttScoringConfig,
    cx: float,
    cy: float,
    holes: list[HoleTarget],
) -> bool:
    """
    If putt_arming_max_dist_from_hole_px > 0, the centroid must be within that
    distance (px) of the *nearest* hole center to count as “at the mat” for
    arming (wait_rest → ready). 0 = do not use this gate. Separates stroke/Putt#
    from in-hole *make* logic (a distant FP can’t arm a putt).
    """
    r = float(c.putt_arming_max_dist_from_hole_px)
    if r <= 0.0 or not holes:
        return True
    return _nearest_hole_dist_sq(cx, cy, holes) <= r * r


def _putt_arming_fresh(
    c: PuttScoringConfig,
    coast_frames: int,
) -> str:
    """
    When building address rest, require a real YOLO match this frame so a
    1–2f flash never stacks ``rest_streak``.

    Return ``inc`` to increment rest, ``stall`` to leave rest unchanged (brief
    dropout), ``reset`` to zero rest. If require_fresh is off, always ``inc``
    (caller only uses when not moving, etc.).
    """
    if not c.putt_arming_require_fresh_detection:
        return "inc"
    cf = int(coast_frames)
    if cf == 0:
        return "inc"
    st = int(c.putt_arming_stall_max_coast_frames)
    if st > 0 and 1 <= cf <= st:
        return "stall"
    return "reset"


def _nearest_hole_id_and_distsq(
    cx: float, cy: float, holes: list[HoleTarget],
) -> tuple[str | None, float]:
    """Nearest cup center by Euclidean distance; returns (hole_id, dist_sq)."""
    if not holes:
        return (None, float("inf"))
    best_d2: float = float("inf")
    best_id: str | None = None
    for h in holes:
        mx = 0.5 * (h.x1 + h.x2)
        my = 0.5 * (h.y1 + h.y2)
        dx, dy = cx - mx, cy - my
        d2 = dx * dx + dy * dy
        if d2 < best_d2:
            best_d2 = d2
            best_id = h.hole_id
    return (best_id, best_d2)


def pick_primary_ball(
    tracked: list[Any],
    holes: list[HoleTarget] | None = None,
    *,
    vel_pick_px: float = 0.35,
) -> Any | None:
    """
    One logical ball for scoring among **FP-gate-confirmed** tracks only.

    The pool is every **confirmed** track. Among them, a **coasting** track (no
    YOLO match this frame; last good pose) is still used so makes work when the
    ball sits in the cup and detections go quiet. With multiple tracks, we
    disambiguate with holes / motion; nearest-hole logic still picks the in-cup
    ball over a spare on the mat when the cup ball is coasting.

    With multiple balls (e.g. spare + ball in play), **highest YOLO confidence
    often picks the wrong ball** (spare near the camera / person). When
    ``holes`` is set we instead:
    - prefer any track that is moving (or fast enough vs ``vel_pick_px``);
    - if several are moving, take the one nearest any hole;
    - if none are moving, take the one nearest any hole (usually the ball on the mat).
    When ``holes`` is None, use max confidence among the chosen pool.
    """
    if not tracked:
        return None
    confirmed = [b for b in tracked if getattr(b, "confirmed", False)]
    if not confirmed:
        return None
    pool: list[Any] = list(confirmed)
    if not holes:
        return max(pool, key=lambda b: float(getattr(b, "confidence", 0.0)))

    def nearest_sq(b: Any) -> float:
        return _nearest_hole_dist_sq(float(b.cx), float(b.cy), holes)

    def speed(b: Any) -> float:
        vx = float(getattr(b, "vel_x", 0.0))
        vy = float(getattr(b, "vel_y", 0.0))
        return math.hypot(vx, vy)

    movers = [
        b
        for b in pool
        if getattr(b, "moving", False) or speed(b) > vel_pick_px
    ]
    if movers:
        if len(movers) == 1:
            return movers[0]
        return min(movers, key=nearest_sq)
    return min(pool, key=nearest_sq)


@dataclass
class PuttScoreboard:
    putts: int = 0
    makes: int = 0
    misses: int = 0
    phase: str = "wait_rest"  # wait_rest | ready | rolling | settling
    rest_streak: int = 0
    move_streak: int = 0
    still_streak: int = 0
    lost_streak: int = 0
    detection_gap: int = 0
    last_cx: float = 0.0
    last_cy: float = 0.0
    last_x1: float = 0.0
    last_y1: float = 0.0
    last_x2: float = 0.0
    last_y2: float = 0.0
    last_event: str = ""
    saw_in_hole: bool = False
    saw_hole_id: str | None = None
    # Consecutive frames (this stroke) with centroid in make zone; resets on exit
    saw_cup_streak: int = 0
    # Longest consecutive run in make zone this stroke (passes rarely reach 3+)
    saw_cup_max: int = 0
    in_hole_streak: int = 0
    # last M booleans: primary effectively moving in "ready" (for N-of-M stroke gate)
    ready_move_window: deque[bool] | None = None
    # Set when a putt is counted: max distance (px) from that anchor during stroke
    putt_anchor: tuple[float, float] | None = None
    putt_travel_max: float = 0.0
    # Frames in rolling/settling with tracker "moving"; used to gate Putt# by travel
    putt_moving_frames: int = 0
    # Frames spent in "ready" (armed); increments each frame while in ready
    ready_elapsed_frames: int = 0
    # Snapshot of ready_elapsed at stroke (how long you stood at address)
    putt_address_frames: int = 0
    # True centroid ever entered a hole make zone (geometry) this stroke, even
    # one frame; handles YOLO dropping the ball the frame after the cup. Paired
    # with a short lost_frames_resolve. Reset each stroke.
    saw_cup_once_id: str | None = None
    # If min_putt_commit_travel > 0: do not add to putts on stroke; add when
    # putt_travel_max reaches that, or on any MAke (short tap-in). Prevents
    # low-conf YOLO FPs from incrementing putts. Reset each stroke.
    putt_credited: bool = False
    # Frames in rolling/settling with an active putt; used with max_stroke_frames
    # and (simple in/out) min_stroke_frames_for_rolling_make
    stroke_frame_age: int = 0
    # Set while centroid is in the make zone and tracker is still "moving" (or
    # held after roll-in for lost resolution when YOLO vanishes the same frame).
    last_moving_cup_id: str | None = None
    # simple_inout_scoring: saw centroid in any padded make zone this stroke
    sio_ever_in: bool = False
    sio_in_streak: int = 0
    sio_out_streak: int = 0
    # This stroke: saw a frame where the centroid was in the *loose* pad but the
    # stroke-corrected zone said out (e.g. past the cup) — do not count MADE;
    # settle/lost/rolling must be MISS.
    sio_past_mismatch: bool = False
    # Consecutive frames where rolling-MAKE base conditions (in scoring zone, etc.) hold
    sio_make_delay_streak: int = 0


@dataclass
class PuttScoringConfig:
    # If >0: ball centroid must be within this many px of the *nearest* hole
    # center to build rest toward "ready" (and thus allow a stroke / Putt#). Off
    # to ignore. Cuts FPs that sit far from the mat while in-hole *make* is unchanged.
    putt_arming_max_dist_from_hole_px: float = 0.0
    # If True, address rest and stroke need a *fresh* YOLO match (coast_frames==0)
    # so a few-frame flash cannot look "still at address". Rolling/settling unchanged.
    putt_arming_require_fresh_detection: bool = True
    # If >0, while still: coast 1..stall does not add rest but also does not reset
    # (only when require_fresh). 0 = one missed frame while still resets rest.
    putt_arming_stall_max_coast_frames: int = 2
    arm_rest_frames: int = 7
    # Consecutive *effective* moving frames in "ready" to start a putt
    move_start_frames: int = 6
    # Must stay in "ready" at least this many frames before a stroke can fire
    # (jitter in the first frames after arming is ignored for stroke). 0=off.
    min_ready_frames: int = 4
    # Optional: also allow a stroke if at least N of the last M ready frames are
    # effective-moving (0 = do not use; typical 4 of 7). Current frame must be
    # effective-moving for the N-of-M path.
    move_start_window_n: int = 0
    move_start_window_m: int = 7
    # Ignore centroid motion in the bottom strip of the frame (y norm) when
    # deciding rest vs stroke for wait_rest/ready only — YOLO foot FPs, etc.
    putt_exclude_bottom_frac: float = 0.0
    settle_frames: int = 8
    # After this many *additional* still frames, resolve settle make/miss (delay
    # so a lip-out / pass past the hole can show out-of-zone before final call).
    settle_post_hang_frames: int = 5
    # "Fast" in-zone make in rolling: require stroke this many frames old (0=off).
    # Stops the ball being MADE the instant it nicks the padding while still fast.
    min_stroke_frames_for_rolling_make: int = 18
    # No ball during stroke: require this many lost frames to score make/miss
    lost_frames_resolve: int = 18
    # If we already overlapped a hole that stroke, use this (shorter) limit —
    # typical when the ball is no longer seen once it sits in the bottom of the cup
    lost_frames_resolve_saw_in_hole: int = 5
    # Added to hole_pad for last-position make check when ball is lost and never
    # scored "in" with the normal pad (last box often sits on the lip)
    lost_last_pos_pad_extra: float = 8.0
    # Centroid must be inside hole box expanded by this many pixels (tighter = stricter)
    hole_pad_px: float = 12.0
    gap_reset_frames: int = 18
    # Need this many *consecutive* centroid-in-make frames at least once this stroke
    # before "saw in cup" counts for settle / lost-ball make
    make_min_saw_cup_streak: int = 2
    # 0 = disable; else fast MADE when centroid stays in make zone this many
    # consecutive *speed-qualified* frames (jittery "mov" in the cup; see below)
    make_in_hole_frames: int = 3
    # Strict cap: only for the *fast* "in cup xN" early-make streak (in_hole_streak)
    # and the associated gate — not for saw_cup (below). 0=off.
    make_max_speed_in_cup_px: float = 1.8
    # Looser cap for "saw ball in the cup" (saw_cup_*, lost-ball make, still-settle
    # eligibility). A fast but real make often has speed > 1.8 in the make zone; using
    # the same limit for *both* used to block saw_cup and the first-putt lost make.
    # 0=off (geometry-only for that frame, riskier for lip pass FPs; pair with
    # settle-geometry check). Typical 2.2–2.6 at 15 fps.
    make_saw_cup_max_speed_in_cup_px: float = 2.4
    # Widen the centroid zone only for the *final* check when we require the ball
    # to be in the cup on settle (handles tracker jitter; does not change early
    # make or the main hole_pad_px for rolling).
    hole_pad_settle_extra_px: float = 3.0
    # Extra on top of hole_pad+settle for the saw+settle fallback (8 still frames
    # and we saw 2+ in-cup, but the strict pad misses on the last frame).
    hole_pad_settle_saw_fallback_extra_px: float = 8.0
    # If a stroke would end as a miss and travel was below this, never had
    # credible in-cup — treat as a putter nudge and revert putt. 0=off.
    min_putt_travel_px: float = 12.0
    # If >0, nudge revert (short travel, no in-cup) only when address was *shorter*
    # than this many frames in "ready" before the stroke. Tap-ins with short
    # roll but a deliberate address are kept. 0=ignore address time.
    min_address_frames: int = 5
    # If >0: on ball *lost* (no YOLO), if last centroid is within this distance
    # (px) of the *nearest* hole center and putt travel >= min travel below, score
    # MADE. Use when the model drops the ball at the cup (never shows a long rest
    # past the hole) but full centroid-in-rect is never reached. 0=off. Tune on
    # your frame size; try 32–64 at 1080p.
    lost_proximity_make_max_dist_px: float = 0.0
    # With lost_proximity_make_max_dist_px, require at least this roll (px) from
    # address so random tap-ins are not all forced to be makes.
    lost_proximity_make_min_travel_px: float = 20.0
    # If >0, Putt# is not incremented on stroke; increments once putt_travel_max
    # reaches this (or on MADE if tap-in never reaches it). Sub-threshold stroke
    # outcomes become nudge/ignored, not a miss. 0=count putt on stroke (old).
    min_putt_commit_travel_px: float = 15.0
    # With min_putt_commit_travel_px>0: also require this many frames where the
    # ball is tracker-"moving" in rolling/settling before crediting Putt# from
    # travel alone (not from MADE tap-in). 0=off. Stops one-frame distance jumps.
    putt_commit_min_moving_frames: int = 6
    # 0=off. Else force an outcome after this many frames in rolling/settling
    # (YOLO flicker can prevent consecutive lost; jitter can block 8 still).
    # ~300 at 15 fps ≈ 20s. Credited strokes become MISS; uncredited+min commit
    # fizzles (same as other incomplete strokes).
    max_stroke_frames: int = 0
    # True: centroid in the padded make zone; in for N frames => MADE in motion;
    # after ever-in, N consecutive out => MISS. Settle: MADE only if *at rest* still
    # in the make zone (not sio_ever + fallback) so a pass through the box then
    # rest down-table is a MISS.
    simple_inout_scoring: bool = True
    # Consecutive in-zone frames to allow fast rolling make (use with
    # min_stroke_frames_for_rolling_make; not a single in-only frame)
    sio_in_frames: int = 4
    # Consecutive "outside make zone" frames after sio_ever_in to count as MISS
    sio_out_frames: int = 2
    # While centroid is in the *loose* ``hole_pad`` rect, a ball that has
    # rolled *past* the cup (along line address→hole) by more than this many
    # px along that line is not "in" the make zone, unless the centroid is
    # inside the *tight* in-cup rect (``in_cup_tight_pad_px``). 0=off. Fixes
    # long hole pads that still contain the ball after it has stopped past the hole.
    putt_past_cup_min_forward_px: float = 8.0
    # Inner hole rect for past-cup: centroid in hole + this pad (px) is always a
    # candidate make; past-cup is not applied there.
    in_cup_tight_pad_px: float = 3.0
    # After kin + min_stroke would allow a rolling make, require this many
    # *consecutive* frames where that remains true (and no past_mismatch) before
    # committing MADE. 0 = no extra holdoff.
    rolling_make_min_delay_frames: int = 5


def _stroke_reset_flags(board: PuttScoreboard) -> None:
    board.saw_in_hole = False
    board.saw_hole_id = None
    board.saw_cup_streak = 0
    board.saw_cup_max = 0
    board.saw_cup_once_id = None
    board.last_moving_cup_id = None


def _primary_speed_xy(primary: Any) -> float:
    vx = float(getattr(primary, "vel_x", 0.0))
    vy = float(getattr(primary, "vel_y", 0.0))
    return math.hypot(vx, vy)


def _nudge_cancels_stroke(
    c: PuttScoringConfig, board: PuttScoreboard,
) -> bool:
    if c.min_putt_travel_px <= 0 or board.putt_anchor is None:
        return False
    if (
        c.min_address_frames > 0
        and board.putt_address_frames >= c.min_address_frames
    ):
        return False
    if board.putt_travel_max < c.min_putt_travel_px:
        if (
            board.saw_cup_max < c.make_min_saw_cup_streak
            and board.saw_cup_once_id is None
        ):
            return True
    return False


def _putt_commit_deferred_makes_tap_in(
    board: PuttScoreboard, c: PuttScoringConfig, events: list[tuple[str, ...]],
) -> None:
    """With min_putt_commit_travel, credit a putt on MADE even if travel < min."""
    if c.min_putt_commit_travel_px <= 0.0 or board.putt_credited:
        return
    board.putts += 1
    board.putt_credited = True
    events.append(("putt",))


def _ready_allows_stroke(
    c: PuttScoringConfig, board: PuttScoreboard,
) -> bool:
    if c.min_ready_frames <= 0:
        return True
    return board.ready_elapsed_frames >= c.min_ready_frames


def _try_putt_commit_by_travel(
    board: PuttScoreboard, c: PuttScoringConfig, events: list[tuple[str, ...]],
) -> None:
    if c.min_putt_commit_travel_px <= 0.0 or board.putt_credited:
        return
    nmin = int(c.putt_commit_min_moving_frames)
    if nmin > 0 and board.putt_moving_frames < nmin:
        return
    if board.putt_travel_max >= float(c.min_putt_commit_travel_px):
        board.putts += 1
        board.putt_credited = True
        events.append(("putt",))


def _should_fizzle_incomplete_stroke(
    c: PuttScoringConfig, board: PuttScoreboard,
) -> bool:
    return c.min_putt_commit_travel_px > 0.0 and not board.putt_credited


def _nudge_cancels_stroke_simple(
    c: PuttScoringConfig, board: PuttScoreboard,
) -> bool:
    """Nudge: short travel and never entered any hole make box this stroke."""
    if c.min_putt_travel_px <= 0 or board.putt_anchor is None:
        return False
    if (
        c.min_address_frames > 0
        and board.putt_address_frames >= c.min_address_frames
    ):
        return False
    if board.putt_travel_max < c.min_putt_travel_px:
        if not board.sio_ever_in:
            return True
    return False


def _sio_reset(board: PuttScoreboard) -> None:
    board.sio_ever_in = False
    board.sio_in_streak = 0
    board.sio_out_streak = 0
    board.sio_past_mismatch = False
    board.sio_make_delay_streak = 0


def _putt_print_events(events: list[tuple[str, ...]], board: PuttScoreboard) -> None:
    for ev in events:
        if ev[0] == "putt":
            print(f"Putt #{board.putts} (stroke)", file=sys.stderr)
        elif ev[0] == "make":
            print(f"  → MADE ({ev[1]})", file=sys.stderr)
        elif ev[0] == "miss":
            print("  → MISS", file=sys.stderr)


def _putt_simple_inout_scoring(
    board: PuttScoreboard,
    primary: "TrackedBall | None",
    holes: list[HoleTarget],
    c: PuttScoringConfig,
    *,
    frame_h: int | None = None,
) -> list[tuple[str, ...]]:
    events: list[tuple[str, ...]] = []
    pad = float(c.hole_pad_px)
    kin = max(1, int(c.sio_in_frames))
    kout = max(1, int(c.sio_out_frames))

    if (
        board.putt_anchor is not None
        and board.phase in ("rolling", "settling")
    ):
        board.stroke_frame_age += 1
    if (
        c.max_stroke_frames > 0
        and board.putt_anchor is not None
        and board.phase in ("rolling", "settling")
    ):
        if board.stroke_frame_age >= c.max_stroke_frames:
            if _should_fizzle_incomplete_stroke(c, board):
                board.last_event = "nudge (time cap, min travel; no putt#)"
                print(
                    "  → nudge: stroke time cap, no putt#",
                    file=sys.stderr,
                )
            else:
                board.misses += 1
                events.append(("miss",))
                board.last_event = "miss (stale putt, time cap)"
            _reset_after_outcome(board)
            _putt_print_events(events, board)
            return events

    if primary is None:
        board.detection_gap += 1
        g = board.detection_gap
        if g > c.gap_reset_frames and board.phase in ("wait_rest", "ready"):
            board.rest_streak = 0
            board.move_streak = 0
            board.ready_move_window = None
            if board.phase == "ready":
                board.phase = "wait_rest"
                board.ready_elapsed_frames = 0
        if board.phase in ("rolling", "settling"):
            board.lost_streak += 1
            if board.lost_streak > c.lost_frames_resolve:
                if _nudge_cancels_stroke_simple(c, board):
                    if board.putt_credited:
                        board.putts -= 1
                    board.last_event = "nudge (putt reverted, ball lost)"
                    print("  → nudge: putt reverted (short travel)", file=sys.stderr)
                elif board.sio_ever_in:
                    _putt_commit_deferred_makes_tap_in(board, c, events)
                    if board.sio_past_mismatch:
                        board.misses += 1
                        events.append(("miss",))
                        board.last_event = "miss (ball lost; saw past/loose not in-cup)"
                    else:
                        hid0 = _sio_zone_hole_id(
                            board.last_cx, board.last_cy, holes, pad, c, board.putt_anchor,
                        )
                        if hid0 is not None:
                            board.makes += 1
                            events.append(("make", str(hid0)))
                            board.last_event = f"make {hid0} (ball lost, was in box)"
                        else:
                            board.misses += 1
                            events.append(("miss",))
                            board.last_event = (
                                "miss (ball lost; last pos not in make zone)"
                            )
                else:
                    if _should_fizzle_incomplete_stroke(c, board):
                        board.last_event = "nudge (stroke below min travel; no putt)"
                        print(
                            "  → nudge: stroke ignored (min travel not met)",
                            file=sys.stderr,
                        )
                    else:
                        board.misses += 1
                        events.append(("miss",))
                        board.last_event = "miss (ball lost, never in box)"
                _reset_after_outcome(board)
    else:
        board.detection_gap = 0
        board.lost_streak = 0
        board.last_cx = float(primary.cx)
        board.last_cy = float(primary.cy)
        board.last_x1 = float(primary.x1)
        board.last_y1 = float(primary.y1)
        board.last_x2 = float(primary.x2)
        board.last_y2 = float(primary.y2)
        moving = bool(getattr(primary, "moving", False))
        if board.putt_anchor is not None and board.phase in (
            "rolling", "settling"
        ):
            ax, ay = board.putt_anchor
            d = math.hypot(
                float(primary.cx) - ax,
                float(primary.cy) - ay,
            )
            if d > board.putt_travel_max:
                board.putt_travel_max = d
            if moving:
                board.putt_moving_frames += 1
        _try_putt_commit_by_travel(board, c, events)

        def m_arm() -> bool:
            if not moving:
                return False
            ex = c.putt_exclude_bottom_frac
            if (
                frame_h is not None
                and frame_h > 0
                and ex > 0.0
            ):
                yn = float(primary.cy) / float(frame_h)
                if yn >= 1.0 - ex:
                    return False
            return True

        loose_h = ball_in_hole_make(
            board.last_cx, board.last_cy, holes, pad,
        )
        _hidg = _sio_zone_hole_id(
            board.last_cx, board.last_cy, holes, pad, c, board.putt_anchor,
        )
        if board.phase in ("rolling", "settling") and board.putt_anchor is not None:
            if (
                c.putt_past_cup_min_forward_px > 0.0
                and loose_h is not None
                and _hidg is None
            ):
                # Loose pad still contains the centroid, but line-of-putt rules
                # say "past the cup" — never MADE this stroke.
                board.sio_past_mismatch = True
            if _hidg is not None:
                board.sio_ever_in = True
                board.sio_in_streak += 1
                board.sio_out_streak = 0
            else:
                board.sio_in_streak = 0
                if board.sio_ever_in:
                    board.sio_out_streak += 1
            if board.sio_ever_in and board.sio_out_streak >= kout:
                if not _nudge_cancels_stroke_simple(c, board):
                    _putt_commit_deferred_makes_tap_in(board, c, events)
                    board.misses += 1
                    events.append(("miss",))
                    board.last_event = "miss (saw in box, then out)"
                _reset_after_outcome(board)
                _putt_print_events(events, board)
                return events
            can_make_base = (
                _hidg is not None
                and not board.sio_past_mismatch
                and board.sio_in_streak >= kin
                and (
                    c.min_stroke_frames_for_rolling_make <= 0
                    or board.stroke_frame_age
                    >= c.min_stroke_frames_for_rolling_make
                )
            )
            if can_make_base:
                board.sio_make_delay_streak += 1
            else:
                board.sio_make_delay_streak = 0
            delay_ok = (
                c.rolling_make_min_delay_frames <= 0
                or board.sio_make_delay_streak
                >= c.rolling_make_min_delay_frames
            )
            if can_make_base and delay_ok:
                _putt_commit_deferred_makes_tap_in(board, c, events)
                board.makes += 1
                events.append(("make", str(_hidg)))
                board.last_event = (
                    f"make {_hidg} (in box x{board.sio_in_streak}, "
                    f"delay {board.sio_make_delay_streak}f)"
                )
                _putt_print_events(events, board)
                _reset_after_outcome(board)
                return events

        if board.phase == "wait_rest":
            cf_arming = int(getattr(primary, "coast_frames", 0))
            if not m_arm():
                if _putt_arming_near_hole(
                    c, float(primary.cx), float(primary.cy), holes
                ):
                    frp = _putt_arming_fresh(c, cf_arming)
                    if frp == "inc":
                        board.rest_streak += 1
                    elif frp == "reset":
                        board.rest_streak = 0
                else:
                    board.rest_streak = 0
                board.move_streak = 0
                if board.rest_streak >= c.arm_rest_frames:
                    board.phase = "ready"
                    board.last_event = "ready"
                    board.ready_move_window = None
                    board.ready_elapsed_frames = 0
            else:
                board.rest_streak = 0
        elif board.phase == "ready":
            cf_arming = int(getattr(primary, "coast_frames", 0))
            board.ready_elapsed_frames += 1
            wn, wm = c.move_start_window_n, c.move_start_window_m
            if wn > 0 and wm > 0:
                if (
                    board.ready_move_window is None
                    or board.ready_move_window.maxlen != wm
                ):
                    board.ready_move_window = deque(maxlen=wm)
                board.ready_move_window.append(m_arm())
            n_of_m = False
            if (
                wn > 0
                and board.ready_move_window is not None
                and len(board.ready_move_window) == wm
                and sum(1 for x in board.ready_move_window if x) >= wn
                and m_arm()
            ):
                n_of_m = True
            if m_arm():
                board.move_streak += 1
            else:
                board.move_streak = 0
            stroke_fresh = (
                not c.putt_arming_require_fresh_detection
                or cf_arming == 0
            )
            stroke = (
                stroke_fresh
                and _putt_arming_near_hole(
                    c, float(primary.cx), float(primary.cy), holes
                )
                and _ready_allows_stroke(c, board)
                and (
                    (board.move_streak >= c.move_start_frames) or n_of_m
                )
            )
            if stroke:
                board.stroke_frame_age = 0
                if c.min_putt_commit_travel_px > 0.0:
                    board.putt_credited = False
                    board.last_event = (
                        f"stroke (address {board.putt_address_frames}f, "
                        f"Putt# at travel≥{c.min_putt_commit_travel_px} px or MADE)"
                    )
                else:
                    board.putts += 1
                    board.putt_credited = True
                    events.append(("putt",))
                    board.last_event = (
                        f"putt (address {board.putt_address_frames}f)"
                    )
                _stroke_reset_flags(board)
                _sio_reset(board)
                board.putt_address_frames = board.ready_elapsed_frames
                board.putt_anchor = (float(primary.cx), float(primary.cy))
                board.putt_travel_max = 0.0
                board.putt_moving_frames = 0
                board.ready_elapsed_frames = 0
                board.phase = "rolling"
                board.move_streak = 0
                board.still_streak = 0
                board.in_hole_streak = 0
                board.ready_move_window = None
        elif board.phase == "rolling":
            if moving:
                board.still_streak = 0
            else:
                board.still_streak = 1
                board.phase = "settling"
        elif board.phase == "settling":
            if moving:
                board.still_streak = 0
                board.phase = "rolling"
            else:
                board.still_streak += 1
                need_still = c.settle_frames + max(0, c.settle_post_hang_frames)
                if board.still_streak >= need_still:
                    if _nudge_cancels_stroke_simple(c, board):
                        if board.putt_credited:
                            board.putts -= 1
                        board.last_event = "nudge (putt reverted, settle)"
                        print(
                            "  → nudge: putt reverted (short travel)",
                            file=sys.stderr,
                        )
                    elif _should_fizzle_incomplete_stroke(c, board):
                        board.last_event = "nudge (stroke below min travel; no putt)"
                        print(
                            "  → nudge: stroke ignored (min travel not met)",
                            file=sys.stderr,
                        )
                    else:
                        # At rest after settle+hang: never MADE if we ever saw
                        # past-the-cup (loose pad but stroke zone out) on this stroke.
                        if board.sio_past_mismatch:
                            _putt_commit_deferred_makes_tap_in(board, c, events)
                            board.misses += 1
                            events.append(("miss",))
                            board.last_event = "miss (settle; saw past/loose not in-cup this stroke)"
                        else:
                            hid_settle = _sio_zone_hole_id(
                                float(primary.cx), float(primary.cy), holes, pad, c,
                                board.putt_anchor,
                            )
                            if hid_settle is not None:
                                _putt_commit_deferred_makes_tap_in(board, c, events)
                                board.makes += 1
                                events.append(("make", str(hid_settle)))
                                board.last_event = (
                                    f"make {hid_settle} (settle, at rest in box)"
                                )
                            else:
                                board.misses += 1
                                events.append(("miss",))
                                if board.sio_ever_in:
                                    board.last_event = (
                                        "miss (settle, at rest outside make zone; "
                                        "grazed or passed through)"
                                    )
                                else:
                                    board.last_event = "miss (settle, never in box)"
                    _reset_after_outcome(board)

    _putt_print_events(events, board)
    return events


def update_putt_scoring(
    board: PuttScoreboard,
    primary: "TrackedBall | None",
    holes: list[HoleTarget],
    cfg: PuttScoringConfig,
    *,
    frame_h: int | None = None,
) -> list[tuple[str, ...]]:
    if not holes:
        return []
    c = cfg
    if c.simple_inout_scoring:
        return _putt_simple_inout_scoring(
            board, primary, holes, c, frame_h=frame_h
        )
    events: list[tuple[str, ...]] = []
    pad = float(c.hole_pad_px)

    if (
        board.putt_anchor is not None
        and board.phase in ("rolling", "settling")
    ):
        board.stroke_frame_age += 1
    if (
        c.max_stroke_frames > 0
        and board.putt_anchor is not None
        and board.phase in ("rolling", "settling")
    ):
        if board.stroke_frame_age >= c.max_stroke_frames:
            if _should_fizzle_incomplete_stroke(c, board):
                board.last_event = "nudge (time cap, min travel; no putt#)"
                print(
                    "  → nudge: stroke time cap, no putt#",
                    file=sys.stderr,
                )
            else:
                board.misses += 1
                events.append(("miss",))
                board.last_event = "miss (stale putt, time cap)"
            _reset_after_outcome(board)
            for ev in events:
                if ev[0] == "putt":
                    print(f"Putt #{board.putts} (stroke)", file=sys.stderr)
                elif ev[0] == "make":
                    print(f"  → MADE ({ev[1]})", file=sys.stderr)
                elif ev[0] == "miss":
                    print("  → MISS", file=sys.stderr)
            return events

    if primary is None:
        board.detection_gap += 1
        g = board.detection_gap

        if g > c.gap_reset_frames and board.phase in ("wait_rest", "ready"):
            board.rest_streak = 0
            board.move_streak = 0
            board.ready_move_window = None
            if board.phase == "ready":
                board.phase = "wait_rest"
                board.ready_elapsed_frames = 0

        if board.phase in ("rolling", "settling"):
            board.lost_streak += 1
            saw_any = (
                board.saw_cup_max >= c.make_min_saw_cup_streak
            ) or (board.saw_cup_once_id is not None)
            cup_ok = saw_any
            lost_lim = (
                c.lost_frames_resolve_saw_in_hole
                if cup_ok
                else c.lost_frames_resolve
            )
            if board.lost_streak > lost_lim:
                last_in = ball_in_hole_make(
                    board.last_cx, board.last_cy, holes, pad,
                )
                if last_in is None and c.lost_last_pos_pad_extra > 0:
                    p2 = pad + float(c.lost_last_pos_pad_extra)
                    last_in = ball_in_hole_make(
                        board.last_cx, board.last_cy, holes, p2,
                    )
                # One-frame in-cup then YOLO drop: last centroid is often on the lip
                if last_in is None and board.saw_cup_once_id is not None:
                    p3 = pad + float(c.lost_last_pos_pad_extra) + 4.0
                    last_in = ball_in_hole_make(
                        board.last_cx, board.last_cy, holes, p3,
                    )
                if cup_ok and board.saw_hole_id is not None:
                    _putt_commit_deferred_makes_tap_in(board, c, events)
                    board.makes += 1
                    events.append(("make", board.saw_hole_id))
                    board.last_event = f"make {board.saw_hole_id} (ball lost)"
                elif last_in is not None:
                    _putt_commit_deferred_makes_tap_in(board, c, events)
                    board.makes += 1
                    events.append(("make", last_in))
                    board.last_event = f"make {last_in} (ball lost, last pos)"
                elif board.last_moving_cup_id is not None:
                    # Last YOLO frame: centroid in make zone and track still "moving"
                    _putt_commit_deferred_makes_tap_in(board, c, events)
                    mid = board.last_moving_cup_id
                    board.makes += 1
                    events.append(("make", mid))
                    board.last_event = (
                        f"make {mid} (ball lost, was moving in cup)"
                    )
                elif _nudge_cancels_stroke(c, board):
                    if board.putt_credited:
                        board.putts -= 1
                    board.last_event = "nudge (putt reverted, ball lost)"
                    print("  → nudge: putt reverted (short travel)", file=sys.stderr)
                else:
                    made_prox = False
                    rmax = float(c.lost_proximity_make_max_dist_px)
                    if (
                        rmax > 0.0
                        and board.putt_travel_max
                        >= float(c.lost_proximity_make_min_travel_px)
                    ):
                        hid, dsq = _nearest_hole_id_and_distsq(
                            board.last_cx, board.last_cy, holes
                        )
                        if hid is not None and dsq <= rmax * rmax:
                            _putt_commit_deferred_makes_tap_in(board, c, events)
                            board.makes += 1
                            events.append(("make", hid))
                            made_prox = True
                            board.last_event = (
                                f"make {hid} (ball lost, near cup+travel)"
                            )
                    if not made_prox:
                        if _should_fizzle_incomplete_stroke(c, board):
                            board.last_event = (
                                "nudge (stroke below min travel; no putt)"
                            )
                            print(
                                "  → nudge: stroke ignored (min travel not met)",
                                file=sys.stderr,
                            )
                        else:
                            board.misses += 1
                            events.append(("miss",))
                            board.last_event = "miss (ball lost)"
                _reset_after_outcome(board)

    else:
        board.detection_gap = 0
        board.lost_streak = 0
        board.last_cx = float(primary.cx)
        board.last_cy = float(primary.cy)
        board.last_x1 = float(primary.x1)
        board.last_y1 = float(primary.y1)
        board.last_x2 = float(primary.x2)
        board.last_y2 = float(primary.y2)
        moving = bool(getattr(primary, "moving", False))
        if board.putt_anchor is not None and board.phase in (
            "rolling", "settling"
        ):
            ax, ay = board.putt_anchor
            d = math.hypot(
                float(primary.cx) - ax,
                float(primary.cy) - ay,
            )
            if d > board.putt_travel_max:
                board.putt_travel_max = d
            if moving:
                board.putt_moving_frames += 1
        _try_putt_commit_by_travel(board, c, events)

        def m_arm() -> bool:
            """Motion for rest/ready only: YOLO foot FPs in bottom strip are ignored."""
            if not moving:
                return False
            ex = c.putt_exclude_bottom_frac
            if (
                frame_h is not None
                and frame_h > 0
                and ex > 0.0
            ):
                yn = float(primary.cy) / float(frame_h)
                if yn >= 1.0 - ex:
                    return False
            return True

        _hidg = ball_in_hole_make(
            board.last_cx, board.last_cy, holes, pad,
        )
        if (
            board.phase in ("rolling", "settling")
            and _hidg is not None
        ):
            if board.saw_cup_once_id is None:
                board.saw_cup_once_id = _hidg
        if board.phase in ("rolling", "settling"):
            if _hidg is not None and moving:
                board.last_moving_cup_id = _hidg
            elif _hidg is None:
                board.last_moving_cup_id = None
        _spc = _primary_speed_xy(primary)
        if (
            c.make_max_speed_in_cup_px > 0.0
            and _hidg is not None
            and _spc > c.make_max_speed_in_cup_px
        ):
            hid_strict = None
        else:
            hid_strict = _hidg
        _saw_max = float(c.make_saw_cup_max_speed_in_cup_px)
        if (
            _saw_max > 0.0
            and _hidg is not None
            and _spc > _saw_max
        ):
            hid_saw = None
        else:
            hid_saw = _hidg
        if board.phase in ("rolling", "settling"):
            if hid_saw is not None:
                board.saw_cup_streak += 1
                board.saw_cup_max = max(
                    board.saw_cup_max, board.saw_cup_streak
                )
                if (
                    board.saw_cup_streak == c.make_min_saw_cup_streak
                    and board.saw_hole_id is None
                ):
                    board.saw_hole_id = hid_saw
            else:
                board.saw_cup_streak = 0
            board.saw_in_hole = (
                board.saw_cup_max >= c.make_min_saw_cup_streak
            )
            if c.make_in_hole_frames > 0:
                if hid_strict is not None:
                    board.in_hole_streak += 1
                else:
                    board.in_hole_streak = 0
                if (
                    hid_strict is not None
                    and board.in_hole_streak >= c.make_in_hole_frames
                    and board.saw_cup_streak >= c.make_min_saw_cup_streak
                ):
                    _putt_commit_deferred_makes_tap_in(board, c, events)
                    board.makes += 1
                    events.append(("make", hid_strict))
                    board.last_event = (
                        f"make {hid_strict} (in cup x{board.in_hole_streak})"
                    )
                    _reset_after_outcome(board)
                    for ev in events:
                        if ev[0] == "putt":
                            print(
                                f"Putt #{board.putts} (stroke)",
                                file=sys.stderr,
                            )
                        elif ev[0] == "make":
                            print(
                                f"  → MADE ({ev[1]})",
                                file=sys.stderr,
                            )
                        elif ev[0] == "miss":
                            print("  → MISS", file=sys.stderr)
                    return events
            else:
                board.in_hole_streak = 0
        else:
            board.in_hole_streak = 0

        if board.phase == "wait_rest":
            cf_arming2 = int(getattr(primary, "coast_frames", 0))
            if not m_arm():
                if _putt_arming_near_hole(
                    c, float(primary.cx), float(primary.cy), holes
                ):
                    frl = _putt_arming_fresh(c, cf_arming2)
                    if frl == "inc":
                        board.rest_streak += 1
                    elif frl == "reset":
                        board.rest_streak = 0
                else:
                    board.rest_streak = 0
                board.move_streak = 0
                if board.rest_streak >= c.arm_rest_frames:
                    board.phase = "ready"
                    board.last_event = "ready"
                    board.ready_move_window = None
                    board.ready_elapsed_frames = 0
            else:
                board.rest_streak = 0

        elif board.phase == "ready":
            cf_arming2 = int(getattr(primary, "coast_frames", 0))
            board.ready_elapsed_frames += 1
            wn, wm = c.move_start_window_n, c.move_start_window_m
            if wn > 0 and wm > 0:
                if (
                    board.ready_move_window is None
                    or board.ready_move_window.maxlen != wm
                ):
                    board.ready_move_window = deque(maxlen=wm)
                board.ready_move_window.append(m_arm())
            n_of_m = False
            if (
                wn > 0
                and board.ready_move_window is not None
                and len(board.ready_move_window) == wm
                and sum(1 for x in board.ready_move_window if x) >= wn
                and m_arm()
            ):
                n_of_m = True
            if m_arm():
                board.move_streak += 1
            else:
                board.move_streak = 0
            stroke_fresh2 = (
                not c.putt_arming_require_fresh_detection
                or cf_arming2 == 0
            )
            stroke = (
                stroke_fresh2
                and _putt_arming_near_hole(
                    c, float(primary.cx), float(primary.cy), holes
                )
                and _ready_allows_stroke(c, board)
                and (
                    (board.move_streak >= c.move_start_frames) or n_of_m
                )
            )
            if stroke:
                board.stroke_frame_age = 0
                if c.min_putt_commit_travel_px > 0.0:
                    board.putt_credited = False
                    board.last_event = (
                        f"stroke (address {board.putt_address_frames}f, "
                        f"Putt# at travel≥{c.min_putt_commit_travel_px} px or MADE)"
                    )
                else:
                    board.putts += 1
                    board.putt_credited = True
                    events.append(("putt",))
                    board.last_event = (
                        f"putt (address {board.putt_address_frames}f)"
                    )
                _stroke_reset_flags(board)
                board.putt_address_frames = board.ready_elapsed_frames
                board.putt_anchor = (float(primary.cx), float(primary.cy))
                board.putt_travel_max = 0.0
                board.putt_moving_frames = 0
                board.ready_elapsed_frames = 0
                board.phase = "rolling"
                board.move_streak = 0
                board.still_streak = 0
                board.in_hole_streak = 0
                board.ready_move_window = None

        elif board.phase == "rolling":
            if moving:
                board.still_streak = 0
            else:
                board.still_streak = 1
                board.phase = "settling"

        elif board.phase == "settling":
            if moving:
                board.still_streak = 0
                board.phase = "rolling"
            else:
                board.still_streak += 1
                need_still_legacy = c.settle_frames + max(
                    0, c.settle_post_hang_frames
                )
                if board.still_streak >= need_still_legacy:
                    pad_s = float(c.hole_pad_px) + float(
                        c.hole_pad_settle_extra_px
                    )
                    settle_in = ball_in_hole_make(
                        float(primary.cx), float(primary.cy), holes, pad_s,
                    )
                    if settle_in is not None:
                        _putt_commit_deferred_makes_tap_in(board, c, events)
                        board.makes += 1
                        events.append(("make", settle_in))
                        board.last_event = f"make {settle_in} (settle)"
                    elif (
                        board.saw_cup_max >= c.make_min_saw_cup_streak
                        and board.saw_hole_id is not None
                    ):
                        pad_loose = (
                            float(c.hole_pad_px)
                            + float(c.hole_pad_settle_extra_px)
                            + float(c.hole_pad_settle_saw_fallback_extra_px)
                        )
                        settle_lo = ball_in_hole_make(
                            float(primary.cx),
                            float(primary.cy),
                            holes,
                            pad_loose,
                        )
                        if settle_lo is not None:
                            make_id = settle_lo
                            _putt_commit_deferred_makes_tap_in(board, c, events)
                            board.makes += 1
                            events.append(("make", make_id))
                            board.last_event = f"make {make_id} (settle, saw+pad)"
                    elif board.saw_cup_once_id is not None:
                        # One in-cup frame then weak track: 8 "still" with centroid jitter
                        pad_once = (
                            float(c.hole_pad_px)
                            + float(c.hole_pad_settle_extra_px)
                            + float(c.hole_pad_settle_saw_fallback_extra_px)
                            + 4.0
                        )
                        settle1 = ball_in_hole_make(
                            float(primary.cx),
                            float(primary.cy),
                            holes,
                            pad_once,
                        )
                        if settle1 is not None:
                            make_id = settle1
                            _putt_commit_deferred_makes_tap_in(board, c, events)
                            board.makes += 1
                            events.append(("make", make_id))
                            board.last_event = f"make {make_id} (settle, once touched cup)"
                    elif _nudge_cancels_stroke(c, board):
                        if board.putt_credited:
                            board.putts -= 1
                        board.last_event = "nudge (putt reverted, settle)"
                        print(
                            "  → nudge: putt reverted (short travel)",
                            file=sys.stderr,
                        )
                    elif _should_fizzle_incomplete_stroke(c, board):
                        board.last_event = (
                            "nudge (stroke below min travel; no putt)"
                        )
                        print(
                            "  → nudge: stroke ignored (min travel not met)",
                            file=sys.stderr,
                        )
                    else:
                        board.misses += 1
                        events.append(("miss",))
                        board.last_event = "miss"
                    _reset_after_outcome(board)

    for ev in events:
        if ev[0] == "putt":
            print(f"Putt #{board.putts} (stroke)", file=sys.stderr)
        elif ev[0] == "make":
            print(f"  → MADE ({ev[1]})", file=sys.stderr)
        elif ev[0] == "miss":
            print("  → MISS", file=sys.stderr)

    return events


def _reset_after_outcome(board: PuttScoreboard) -> None:
    board.phase = "wait_rest"
    board.rest_streak = 0
    board.move_streak = 0
    board.still_streak = 0
    board.in_hole_streak = 0
    board.putt_anchor = None
    board.putt_travel_max = 0.0
    board.putt_moving_frames = 0
    board.putt_address_frames = 0
    board.ready_elapsed_frames = 0
    board.ready_move_window = None
    board.putt_credited = False
    board.stroke_frame_age = 0
    _sio_reset(board)
    _stroke_reset_flags(board)


def draw_putt_hud(vis: Any, board: PuttScoreboard) -> None:
    """Draw scoreline at bottom-right (putText y = text baseline)."""
    import cv2

    h, w = vis.shape[:2]
    line1 = f"Putts:{board.putts}  Made:{board.makes}  Miss:{board.misses}  [{board.phase}]"
    line2 = (board.last_event[:72] + "…") if len(board.last_event) > 72 else board.last_event

    font = cv2.FONT_HERSHEY_SIMPLEX
    (tw1, _), _ = cv2.getTextSize(line1, font, 0.7, 2)
    (tw2, _), _ = cv2.getTextSize(line2, font, 0.5, 1) if line2 else ((0, 0), 0)

    margin = 10
    x1 = w - max(tw1, tw2) - margin
    if x1 < 6:
        x1 = 6
    if line2:
        cv2.putText(vis, line1, (x1, h - 32), font, 0.7, (0, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(vis, line2, (x1, h - 10), font, 0.5, (220, 220, 200), 1, cv2.LINE_AA)
    else:
        cv2.putText(vis, line1, (x1, h - 10), font, 0.7, (0, 255, 255), 2, cv2.LINE_AA)
