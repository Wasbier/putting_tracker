"""Run YOLO + lightweight multi-ball tracker on a video or stream path; optional hole overlay."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2

from ball_mot_tracker import MultiObjectBallTracker, draw_tracked_balls, load_calibration_hole_rects_pixels
from putt_scoring import (
    PuttScoreboard,
    PuttScoringConfig,
    draw_putt_hud,
    load_hole_targets,
    pick_primary_ball,
    update_putt_scoring,
)
from yolo_ball_detector import YoloBallDetector


def main() -> None:
    p = argparse.ArgumentParser(description="Multi-object golf ball tracking on video")
    p.add_argument("--video", type=str, required=True, help="Video file path")
    p.add_argument(
        "--model",
        type=str,
        default="runs/detect/runs/yolo_ball/golf_ball4/weights/best.pt",
        help="YOLO weights",
    )
    p.add_argument(
        "--conf",
        type=float,
        default=0.6,
        help="YOLO score threshold (higher = fewer false balls / flicker; try 0.45–0.55 in dim light)",
    )
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--device", type=str, default=None)
    p.add_argument(
        "--roi",
        type=str,
        default=None,
        help="Optional detector crop x,y,w,h in pixels",
    )
    p.add_argument(
        "--max-match-dist",
        type=float,
        default=90.0,
        help="NN match gate (pixels); increase if IDs swap on fast rolls",
    )
    p.add_argument(
        "--max-lost-frames",
        type=int,
        default=32,
        help="Base for the *confirmed* track budget (used with --max-lost-frames-confirmed; ~1.5x auto). Unconfirmed use --max-lost-frames-candidate (stricter, default auto)",
    )
    p.add_argument(
        "--no-vel-match",
        action="store_true",
        help="Match using centroid only (disable velocity prediction for association)",
    )
    p.add_argument(
        "--match-vel-scale",
        type=float,
        default=2.0,
        metavar="K",
        help="Add K * speed(px/frame) to max match distance for fast rolls",
    )
    p.add_argument(
        "--min-match-iou",
        type=float,
        default=0.02,
        help="Also allow pair if box IoU >= this (0 disables IoU fallback)",
    )
    p.add_argument(
        "--match-dist-cap",
        type=float,
        default=220.0,
        help="Cap on effective max match distance (pixels)",
    )
    p.add_argument("--motion-px", type=float, default=5.0, help="Centroid move > this => moving")
    p.add_argument("--history-len", type=int, default=10, help="Centroid history length per track")
    p.add_argument(
        "--no-fp-gate",
        action="store_true",
        help="Disable FP gate (all tracks show as confirmed immediately, old behavior).",
    )
    p.add_argument(
        "--min-confirm-streak",
        type=int,
        default=8,
        metavar="K",
        help="When FP gate is on, require K consecutive 'good' frames (conf+shape) before a track is shown and used for putts; default 8",
    )
    p.add_argument(
        "--gate-m",
        type=int,
        default=5,
        metavar="M",
        help="Internal sliding window length (≥ min-confirm-streak) for the FP history buffer",
    )
    p.add_argument(
        "--conf-high",
        type=float,
        default=0.30,
        help="FP gate: min YOLO conf to count a 'good' frame before track is confirmed",
    )
    p.add_argument(
        "--conf-low",
        type=float,
        default=0.12,
        help="FP gate: min YOLO conf for a 'good' frame after confirmation (looser, helps keep a real track)",
    )
    p.add_argument(
        "--max-aspect",
        type=float,
        default=1.45,
        help="FP gate: max width/height ratio for a ball-like box",
    )
    p.add_argument(
        "--min-circ",
        type=float,
        default=0.42,
        help="FP gate: min circularity on bbox crop (0 disables)",
    )
    p.add_argument(
        "--show-candidate-boxes",
        action="store_true",
        help="Draw gray boxes for not-yet-confirmed tracks (debug; default: only show after min-confirm-streak good frames)",
    )
    p.add_argument(
        "--show-candidate-ids",
        action="store_true",
        help="Show internal track id on gray candidate boxes (default: hide id until confirmed)",
    )
    p.add_argument(
        "--max-lost-frames-candidate",
        type=int,
        default=0,
        metavar="F",
        help="0=auto (strict, ~0.35x --max-lost-frames, cap 4–12). Drop a *not-yet-confirmed* track this fast when unmatched — lenient only after the ball is confirmed; set explicitly to tune",
    )
    p.add_argument(
        "--max-lost-frames-confirmed",
        type=int,
        default=0,
        metavar="F",
        help="0=auto (≥1.5x --max-lost-frames). Max frames a *confirmed* track can coast without a detection before removal (higher = more forgiving)",
    )
    p.add_argument(
        "--match-bonus-confirmed",
        type=float,
        default=20.0,
        metavar="PX",
        help="Add this many pixels to the max match distance for confirmed tracks only (re-associate after wobble or brief weak boxes)",
    )
    p.add_argument(
        "--calibration",
        type=Path,
        default=None,
        help="hole_calibration.json from calibrate_holes.py (draws holes; use with --score-putts)",
    )
    p.add_argument(
        "--score-putts",
        action="store_true",
        help="Count putts and made/miss vs hole regions (single ball; needs --calibration)",
    )
    p.add_argument(
        "--settle-frames",
        type=int,
        default=10,
        help="Consecutive still frames before the post-hang window (with --score-putts); total still needed is this + --settle-post-hang-frames for simple in/out",
    )
    p.add_argument(
        "--settle-post-hang-frames",
        type=int,
        default=5,
        help="With --score-putts + simple in/out: extra consecutive still frames after --settle-frames before final make/miss (gives the ball time to show past/lip before locking in)",
    )
    p.add_argument(
        "--min-stroke-frames-rolling-sio-make",
        type=int,
        default=18,
        metavar="F",
        help="With simple in/out: fast rolling MADE only if the stroke is at least F frames old (0=off). Stops 'made' on the first nick of the padded zone",
    )
    p.add_argument(
        "--arm-rest-frames",
        type=int,
        default=7,
        help="Frames at rest (no effective motion) before armed for 'ready' (with --score-putts); higher = stricter before a stroke can be considered",
    )
    p.add_argument(
        "--min-ready-frames",
        type=int,
        default=4,
        metavar="F",
        help="In 'ready', require this many frames at address before a stroke can fire (blocks early jitter; 0=off)",
    )
    p.add_argument(
        "--move-start-frames",
        type=int,
        default=6,
        help="Consecutive effective-moving frames in 'ready' to count a stroke (with --score-putts); use with --putt-exclude-bottom to ignore foot FPs",
    )
    p.add_argument(
        "--putt-window-n",
        type=int,
        default=0,
        metavar="N",
        help="With --score-putts, optional N for N-of-M stroke gate in 'ready' (0=off). Example: 4 needs 4 effective-moving frames in last M",
    )
    p.add_argument(
        "--putt-window-m",
        type=int,
        default=7,
        metavar="M",
        help="Sliding window size for --putt-window-N (with --score-putts)",
    )
    p.add_argument(
        "--putt-exclude-bottom",
        type=float,
        default=0.0,
        metavar="FRAC",
        help="Ignore centroid motion in the bottom FRAC of the frame for wait/ready only (0=off; try 0.08–0.12 for foot/shoe YOLO FPs)",
    )
    p.add_argument(
        "--putt-arming-max-dist-from-hole",
        type=float,
        default=420.0,
        metavar="PX",
        help="0=off. Else ball must be within PX of nearest hole *center* to build rest/ready and to fire a stroke (Putt#). Stops far-from-mat FPs; does not change in-hole *make*",
    )
    p.add_argument(
        "--no-putt-arming-fresh",
        action="store_true",
        help="With --score-putts, allow arming on coasted (no YOLO this frame) ball pose — not recommended; default requires a fresh YOLO hit to stack address rest & to start a stroke (blocks YOLO flashes from counting as address)",
    )
    p.add_argument(
        "--putt-arming-stall-coast",
        type=int,
        default=2,
        metavar="F",
        help="With fresh arming: if 1..F frames had no new det while still, hold rest (brief dropout). 0 = one missed det resets rest",
    )
    p.add_argument(
        "--hole-pad",
        type=float,
        default=12.0,
        help="Centroid make zone: expand each cal hole rect by this many pixels (~12 for 15fps fast makes; lower if lip passers false-make)",
    )
    p.add_argument(
        "--gap-reset-frames",
        type=int,
        default=18,
        help="Consecutive frames with no YOLO ball before arming rest counter resets (with --score-putts); a bit higher helps high conf + brief occlusions",
    )
    p.add_argument(
        "--make-in-hole-frames",
        type=int,
        default=3,
        help="Consecutive in-cup frames (each must pass --make-max-speed-in-cup) for early MADE; 0=only still-settle. Use 2 at 15fps if the ball is lost in-cup very fast",
    )
    p.add_argument(
        "--make-min-cup-frames",
        type=int,
        default=2,
        metavar="K",
        help="Need K consecutive in-cup (centroid) frames for 'credible' make (with --score-putts). 2 at ~15fps; 3 at 30fps if needed",
    )
    p.add_argument(
        "--make-max-speed-in-cup",
        type=float,
        default=1.8,
        metavar="V",
        help="Stricter cap for the fast early-make in-cup *only*; lost/settle *saw* uses --make-saw-cup-max-speed. 0=off",
    )
    p.add_argument(
        "--make-saw-cup-max-speed",
        type=float,
        default=2.4,
        metavar="S",
        help="Looser in-cup speed (px/frame) for saw_cup, lost make, and early-make eligibility. Use >--make-max-speed for fast real makes; 0=off (geometry in zone only, riskier)",
    )
    p.add_argument(
        "--hole-pad-settle-extra",
        type=float,
        default=3.0,
        help="Add this many pixels to the hole pad only when doing the final *settle* in-cup check (jitter; does not change rolling logic)",
    )
    p.add_argument(
        "--hole-pad-settle-saw-fallback-extra",
        type=float,
        default=8.0,
        help="Extra pad on top of hole+settle when 8 still and 2+ saw in-cup but strict settle check missed (tracker jitter)",
    )
    p.add_argument(
        "--min-putt-travel",
        type=float,
        default=12.0,
        metavar="PX",
        help="With --score-putts + --min-address-frames: nudge revert if travel < PX and no in-cup. 0=off",
    )
    p.add_argument(
        "--min-address-frames",
        type=int,
        default=5,
        metavar="F",
        help="With --score-putts, nudge revert only if address in 'ready' was < F frames (tap-ins usually longer). 0=ignore stand time",
    )
    p.add_argument(
        "--lost-near-cup-make",
        type=float,
        default=0.0,
        metavar="D",
        help="0=off. If D>0: when track is *lost* (no YOLO) with no other make, last centroid within D px of nearest hole center AND min travel (next arg) => MADE. For balls YOLO drops at the cup with no 8-still 'visible miss'",
    )
    p.add_argument(
        "--lost-near-cup-min-travel",
        type=float,
        default=20.0,
        metavar="PX",
        help="With --lost-near-cup-make D>0, require this many px putt travel from address (stops nudge tap-ins from becoming makes)",
    )
    p.add_argument(
        "--min-putt-commit-travel",
        type=float,
        default=15.0,
        metavar="PX",
        help="0=Putt# on stroke. Default>0: credit Putt# only after the ball moves PX from address, or on MADE (tap-in); damps wiggle/FP strokes",
    )
    p.add_argument(
        "--putt-commit-min-moving-frames",
        type=int,
        default=6,
        metavar="F",
        help="With --min-putt-commit-travel>0, require F frames where the track is 'moving' during rolling/settling before crediting Putt# from travel (not from MADE). 0=off",
    )
    p.add_argument(
        "--max-stroke-frames",
        type=int,
        default=0,
        help="0=off. Else force MISS (or nudge if min-putt-commit not met) after this many frames in rolling/settling so every shot gets a terminal line. ~300 at 15fps ≈ 20s",
    )
    p.add_argument(
        "--no-simple-inout",
        action="store_true",
        help="With --score-putts, use full legacy make/miss (saw_cup, settle overlap). Default is simple centroid in/out: N frames in => MADE, then N out => MISS (after ever-in).",
    )
    p.add_argument(
        "--sio-in-frames",
        type=int,
        default=4,
        metavar="N",
        help="With simple in/out, consecutive in-make-zone frames for a fast rolling MADE (see also --min-stroke-frames-rolling-sio-make)",
    )
    p.add_argument(
        "--sio-out-frames",
        type=int,
        default=2,
        metavar="N",
        help="With simple in/out, after the centroid was in the make zone, consecutive out-of-zone frames to commit MISS. Default 2",
    )
    p.add_argument(
        "--putt-past-cup-min-forward-px",
        type=float,
        default=8.0,
        metavar="PX",
        help="With simple in/out: if the stroke started far enough from the cup, a centroid in the loose pad is *not* in-zone when the ball has moved PX past the hole along the line address→hole (unless in the tight in-cup pad). 0=off",
    )
    p.add_argument(
        "--in-cup-tight-pad-px",
        type=float,
        default=3.0,
        help="Tight in-cup rect = hole + this pad (px); used with --putt-past-cup-min-forward-px to keep real makes when slightly offset from the calibration center",
    )
    p.add_argument(
        "--rolling-make-min-delay-frames",
        type=int,
        default=5,
        metavar="F",
        help="With simple in/out: after kin + min-stroke allow a rolling make, require F more consecutive in-zone frames before MADE; 0=off",
    )
    p.add_argument(
        "--lost-resolve-frames",
        type=int,
        default=18,
        help="With --score-putts, no-ball frames before make/miss when never in-cup that stroke (lower at ~15fps)",
    )
    p.add_argument(
        "--lost-resolve-saw-frames",
        type=int,
        default=5,
        help="With --score-putts, no-ball frames before MADE after credible in-cup (ball lost in cup)",
    )
    p.add_argument(
        "--lost-last-pos-pad-extra",
        type=float,
        default=8.0,
        help="Extra px (centroid-only) for last-position make when track was lost and credible in-cup was not reached; 0 disables",
    )
    p.add_argument("--stride", type=int, default=1, help="cv2.grab() between reads for speed")
    p.add_argument("--scale", type=float, default=1.0, help="Display window scale")
    p.add_argument(
        "--json-out",
        type=Path,
        default=None,
        help="Optional NDJSON of per-frame tracks (can be large)",
    )
    p.add_argument(
        "--print-every",
        type=int,
        default=0,
        metavar="N",
        help="If >0, print track summary every N frames to stderr",
    )
    args = p.parse_args()

    if args.score_putts and args.calibration is None:
        raise SystemExit("--score-putts requires --calibration (hole JSON with rect_norm for each cup)")

    scene_roi = None
    if args.roi is not None:
        parts = [int(x.strip()) for x in args.roi.replace(" ", "").split(",")]
        if len(parts) != 4:
            raise SystemExit("--roi must be x,y,w,h")
        scene_roi = (parts[0], parts[1], parts[2], parts[3])

    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        raise SystemExit(f"Could not open video: {args.video}")

    detector = YoloBallDetector(
        model_path=args.model,
        conf=args.conf,
        imgsz=args.imgsz,
        device=args.device,
    )
    mfc = int(args.max_lost_frames_confirmed)
    mfu = int(args.max_lost_frames_candidate)
    tracker = MultiObjectBallTracker(
        max_match_distance=args.max_match_dist,
        max_frames_lost=args.max_lost_frames,
        motion_pixel_threshold=args.motion_px,
        history_len=args.history_len,
        match_use_velocity=not args.no_vel_match,
        match_velocity_scale=args.match_vel_scale,
        min_match_iou=args.min_match_iou,
        match_effective_dist_cap=args.match_dist_cap,
        fp_gate=not args.no_fp_gate,
        gate_m_window=args.gate_m,
        conf_high=args.conf_high,
        conf_low=args.conf_low,
        max_aspect_ratio=args.max_aspect,
        min_circularity=args.min_circ,
        min_confirm_streak=max(1, int(args.min_confirm_streak)),
        max_frames_lost_confirmed=None if mfc <= 0 else mfc,
        max_frames_lost_unconfirmed=None if mfu <= 0 else mfu,
        match_distance_bonus_confirmed=float(args.match_bonus_confirmed),
    )

    ok, frame0 = cap.read()
    if not ok or frame0 is None:
        cap.release()
        raise SystemExit("Could not read first frame")
    fh, fw = frame0.shape[:2]
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    hole_rects = None
    if args.calibration is not None:
        cal_path = args.calibration.expanduser()
        if not cal_path.is_file():
            raise SystemExit(
                "Calibration file not found:\n"
                f"  {cal_path.resolve()}\n"
                "Create it first, e.g.:\n"
                "  python calibrate_holes.py --video recordings/capture_20260418_103202.mp4 "
                "--frame-index 0 --out hole_calibration.json\n"
                "Or run without hole overlay: omit --calibration"
            )
        hole_rects = load_calibration_hole_rects_pixels(cal_path, fw, fh)

    hole_targets = None
    scoreboard: PuttScoreboard | None = None
    score_cfg: PuttScoringConfig | None = None
    if args.score_putts and args.calibration is not None:
        hole_targets = load_hole_targets(args.calibration.expanduser(), fw, fh)
        if not hole_targets:
            raise SystemExit("No holes in calibration file; add holes with calibrate_holes.py")
        scoreboard = PuttScoreboard()
        mih = int(args.make_in_hole_frames)
        lrs = max(1, int(args.lost_resolve_saw_frames))
        lrr = max(lrs, int(args.lost_resolve_frames))
        pwn = max(0, int(args.putt_window_n))
        pwm = max(1, int(args.putt_window_m))
        if pwn > pwm:
            pwn = pwm
        mmc = max(1, int(args.make_min_cup_frames))
        msc = max(0.0, float(args.make_saw_cup_max_speed))
        pad_max = max(0.0, float(args.putt_arming_max_dist_from_hole))
        score_cfg = PuttScoringConfig(
            putt_arming_max_dist_from_hole_px=pad_max,
            putt_arming_require_fresh_detection=not bool(
                args.no_putt_arming_fresh
            ),
            putt_arming_stall_max_coast_frames=max(
                0, int(args.putt_arming_stall_coast)
            ),
            arm_rest_frames=max(1, int(args.arm_rest_frames)),
            min_ready_frames=max(0, int(args.min_ready_frames)),
            move_start_frames=max(1, int(args.move_start_frames)),
            settle_frames=max(1, int(args.settle_frames)),
            settle_post_hang_frames=max(0, int(args.settle_post_hang_frames)),
            min_stroke_frames_for_rolling_make=max(
                0, int(args.min_stroke_frames_rolling_sio_make)
            ),
            hole_pad_px=float(args.hole_pad),
            gap_reset_frames=max(1, int(args.gap_reset_frames)),
            make_in_hole_frames=0 if mih < 0 else mih,
            make_min_saw_cup_streak=mmc,
            lost_frames_resolve=lrr,
            lost_frames_resolve_saw_in_hole=lrs,
            lost_last_pos_pad_extra=max(0.0, float(args.lost_last_pos_pad_extra)),
            move_start_window_n=pwn,
            move_start_window_m=pwm,
            putt_exclude_bottom_frac=max(0.0, min(0.5, float(args.putt_exclude_bottom))),
            make_max_speed_in_cup_px=max(0.0, float(args.make_max_speed_in_cup)),
            make_saw_cup_max_speed_in_cup_px=msc,
            hole_pad_settle_extra_px=max(0.0, float(args.hole_pad_settle_extra)),
            hole_pad_settle_saw_fallback_extra_px=max(
                0.0, float(args.hole_pad_settle_saw_fallback_extra)
            ),
            min_putt_travel_px=max(0.0, float(args.min_putt_travel)),
            min_address_frames=max(0, int(args.min_address_frames)),
            lost_proximity_make_max_dist_px=max(0.0, float(args.lost_near_cup_make)),
            lost_proximity_make_min_travel_px=max(
                0.0, float(args.lost_near_cup_min_travel)
            ),
            min_putt_commit_travel_px=max(0.0, float(args.min_putt_commit_travel)),
            putt_commit_min_moving_frames=max(
                0, int(args.putt_commit_min_moving_frames)
            ),
            max_stroke_frames=max(0, int(args.max_stroke_frames)),
            simple_inout_scoring=not bool(args.no_simple_inout),
            sio_in_frames=max(1, int(args.sio_in_frames)),
            sio_out_frames=max(1, int(args.sio_out_frames)),
            putt_past_cup_min_forward_px=max(
                0.0, float(args.putt_past_cup_min_forward_px)
            ),
            in_cup_tight_pad_px=max(0.0, float(args.in_cup_tight_pad_px)),
            rolling_make_min_delay_frames=max(
                0, int(args.rolling_make_min_delay_frames)
            ),
        )

    stride = max(1, int(args.stride))
    window = "Ball MOT (q=quit)"
    frame_i = 0
    json_f = args.json_out.open("w", encoding="utf-8") if args.json_out else None

    try:
        while True:
            ok_grab = True
            for _ in range(stride - 1):
                if not cap.grab():
                    ok_grab = False
                    break
            if not ok_grab:
                break
            ok, frame = cap.read()
            if not ok or frame is None:
                break

            dets = detector.detect_xyxy(frame, scene_roi=scene_roi)
            tracked = tracker.update(dets, frame_bgr=frame)
            vis = draw_tracked_balls(
                frame,
                tracked,
                hole_rects_xyxy=hole_rects,
                show_unconfirmed=args.show_candidate_boxes,
                show_candidate_ids=args.show_candidate_ids,
            )

            if scoreboard is not None and score_cfg is not None and hole_targets is not None:
                primary = pick_primary_ball(tracked, hole_targets)
                update_putt_scoring(
                    scoreboard, primary, hole_targets, score_cfg, frame_h=fh
                )
                draw_putt_hud(vis, scoreboard)

            if args.print_every > 0 and frame_i % args.print_every == 0:
                parts = []
                for b in tracked:
                    if b.confirmed or b.track_id:
                        tid = str(b.track_id)
                    elif args.show_candidate_ids:
                        tid = str(b.association_id)
                    else:
                        tid = "?"
                    parts.append(
                        f"id{tid}:{'M' if b.moving else 'S'}:{'ok' if b.confirmed else '?'}"
                    )
                print(f"frame {frame_i} n={len(tracked)} " + " ".join(parts), file=sys.stderr)

            if json_f is not None:
                row = {
                    "frame": frame_i,
                    "tracks": [
                        {
                            "id": b.track_id,
                            "association_id": b.association_id,
                            "xyxy": [b.x1, b.y1, b.x2, b.y2],
                            "cx": b.cx,
                            "cy": b.cy,
                            "conf": b.confidence,
                            "confirmed": b.confirmed,
                            "circularity": b.circularity,
                            "aspect_ratio": b.aspect_ratio,
                            "moving": b.moving,
                            "vel": [b.vel_x, b.vel_y],
                            "history": [list(p) for p in b.position_history],
                        }
                        for b in tracked
                    ],
                }
                json_f.write(json.dumps(row) + "\n")

            disp = vis
            if args.scale != 1.0:
                disp = cv2.resize(vis, None, fx=args.scale, fy=args.scale, interpolation=cv2.INTER_AREA)
            cv2.imshow(window, disp)
            if (cv2.waitKey(1) & 0xFF) == ord("q"):
                break
            frame_i += stride
        if scoreboard is not None:
            print(
                f"Session: putts={scoreboard.putts} made={scoreboard.makes} miss={scoreboard.misses}",
                file=sys.stderr,
            )
    finally:
        cap.release()
        if json_f is not None:
            json_f.close()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
    sys.exit(0)
