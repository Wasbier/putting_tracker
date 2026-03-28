"""
Random-search detector + logic settings from a profile JSON against regression cases.

For each case in regression_cases.json (or a filtered subset), samples many configs,
runs track_putts.py headless, and writes full profile JSON files that match
expected_attempts, expected_made, and expected_sequence.

Usage (from repo root):
  python regression_tests/search_configs.py --trials 400
  python regression_tests/search_configs.py --case night_profile_test1_baseline --trials 800
  python regression_tests/search_configs.py --max-matches 5 --trials 2000
  python regression_tests/search_configs.py --no-progress   # plain log lines (no tqdm bar)

Output: regression_tests/search_results/<case_name>/match_<n>.json
Each file is a complete calibration JSON (geometry preserved from the base profile).

Trial 0 for each case is always the unmodified base profile. Random trials also include
the base value for every tuned key in each parameter's choice pool (so e.g. max_area_frac
1/350 is reachable even if not in the fixed grid).
"""

from __future__ import annotations

import argparse
import json
import random
import subprocess
import sys
import tempfile
from copy import deepcopy
from pathlib import Path

from tqdm import tqdm

# Discrete search space (widen/tighten as needed).
DETECTOR_CHOICES: dict[str, list[float | int]] = {
    "v_min": [65, 75, 85, 95, 105, 115],
    "s_max": [85, 100, 115, 130, 145, 160],
    "min_area": [22, 30, 40, 50, 62],
    "min_circularity": [0.42, 0.48, 0.52, 0.56, 0.62],
    "max_area_frac": [1.0 / 450.0, 1.0 / 380.0, 1.0 / 320.0, 1.0 / 280.0, 1.0 / 240.0],
}

LOGIC_INT_KEYS = frozenset(
    {
        "min_tee_frames_for_stroke",
        "min_tee_frames_first_stroke",
        "frames_on_tee_to_arm_next_make",
        "min_address_frames_for_made",
        "made_dwell_frames",
        "made_confirm_frames",
        "made_confirm_max_none_frames",
        "post_attempt_cup_stall_frames",
    }
)

LOGIC_CHOICES: dict[str, list[float | int]] = {
    "min_tee_frames_for_stroke": [16, 20, 22, 26, 30],
    "min_tee_frames_first_stroke": [6, 8, 10, 12, 14],
    "frames_on_tee_to_arm_next_make": [6, 8, 10, 12, 16],
    "min_address_frames_for_made": [6, 8, 10, 12, 14],
    "made_axis_rx_frac": [0.34, 0.38, 0.42, 0.46, 0.50],
    "made_axis_ry_frac": [0.20, 0.24, 0.28, 0.32, 0.36],
    "cup_inner_margin_frac": [0.12, 0.15, 0.17, 0.19, 0.22],
    "made_max_speed_ppf": [6.0, 7.2, 8.5, 9.5, 11.0],
    "made_dwell_frames": [3, 4, 5, 6, 7],
    "made_confirm_frames": [5, 7, 8, 10, 12],
    "made_confirm_max_none_frames": [3, 4, 5, 6, 8],
    "post_attempt_cup_stall_frames": [36, 44, 52, 60, 72],
    "post_attempt_stall_max_speed_ppf": [3.5, 4.2, 4.8, 5.5, 6.2],
}


def _repo_root() -> Path:
    return Path(__file__).resolve().parent.parent


def _normalize_logic_value(key: str, v: float | int) -> float | int:
    if key in LOGIC_INT_KEYS:
        return int(round(float(v)))
    return float(v)


def _normalize_detector_value(key: str, v: float | int) -> float | int:
    if key in ("v_min", "s_max", "min_area"):
        return int(round(float(v)))
    return float(v)


def _dedupe_normalized_detector(key: str, pool: list[float | int]) -> list[float | int]:
    seen: set[float | int] = set()
    out: list[float | int] = []
    for v in pool:
        n = _normalize_detector_value(key, v)
        sig: float | int = round(float(n), 12) if key == "max_area_frac" else n
        if sig in seen:
            continue
        seen.add(sig)
        out.append(n)
    return out


def _dedupe_normalized_logic(key: str, pool: list[float | int]) -> list[float | int]:
    seen: set[float | int] = set()
    out: list[float | int] = []
    for v in pool:
        n = _normalize_logic_value(key, v)
        sig: float | int = round(float(n), 9) if isinstance(n, float) else n
        if sig in seen:
            continue
        seen.add(sig)
        out.append(n)
    return out


def sample_profile_settings(rng: random.Random, base: dict[str, object]) -> dict[str, object]:
    """Return a deep copy of base with randomized detector/logic blocks."""
    cfg = deepcopy(base)
    det = dict(cfg.get("detector", {})) if isinstance(cfg.get("detector"), dict) else {}
    for k, choices in DETECTOR_CHOICES.items():
        pool = list(choices)
        if k in det:
            pool.append(det[k])
        pool = _dedupe_normalized_detector(k, pool)
        det[k] = rng.choice(pool)
    cfg["detector"] = det

    log = dict(cfg.get("logic", {})) if isinstance(cfg.get("logic"), dict) else {}
    for k, choices in LOGIC_CHOICES.items():
        pool = list(choices)
        if k in log:
            pool.append(log[k])
        pool = _dedupe_normalized_logic(k, pool)
        log[k] = rng.choice(pool)
    cfg["logic"] = log
    return cfg


def run_tracker(
    repo_root: Path,
    config_path: Path,
    video_rel: str,
    scale: float,
) -> dict[str, object] | None:
    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as tf:
        report_path = Path(tf.name)
    try:
        cmd = [
            sys.executable,
            str(repo_root / "track_putts.py"),
            "--video",
            video_rel,
            "--config",
            str(config_path),
            "--scale",
            str(scale),
            "--headless",
            "--no-loop",
            "--report-json",
            str(report_path),
        ]
        proc = subprocess.run(cmd, cwd=str(repo_root), capture_output=True, text=True)
        if proc.returncode != 0:
            return None
        return json.loads(report_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    finally:
        report_path.unlink(missing_ok=True)


def report_matches_case(report: dict[str, object], case: dict[str, object]) -> bool:
    if int(report.get("attempts", -1)) != int(case["expected_attempts"]):
        return False
    if int(report.get("made", -1)) != int(case["expected_made"]):
        return False
    exp_seq = case.get("expected_sequence")
    if exp_seq is None:
        return True
    got = report.get("putt_sequence")
    if not isinstance(got, list):
        return False
    return [str(x) for x in got] == [str(x) for x in exp_seq]


def main() -> int:
    ap = argparse.ArgumentParser(description="Search detector/logic configs for regression cases.")
    ap.add_argument("--trials", type=int, default=500, help="Random samples per case")
    ap.add_argument("--seed", type=int, default=None, help="RNG seed (default: nondeterministic)")
    ap.add_argument(
        "--case",
        action="append",
        dest="cases",
        metavar="NAME",
        help="Regression case name (repeatable). Default: all cases.",
    )
    ap.add_argument(
        "--profiles-dir",
        type=Path,
        default=None,
        help="Profile directory (default: config/profiles under repo root)",
    )
    ap.add_argument("--scale", type=float, default=0.75)
    ap.add_argument(
        "--max-matches",
        type=int,
        default=25,
        help="Stop saving after this many hits per case (0 = unlimited)",
    )
    ap.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="Output root (default: regression_tests/search_results)",
    )
    ap.add_argument(
        "--no-progress",
        action="store_true",
        help="Disable the tqdm progress bar",
    )
    args = ap.parse_args()

    repo_root = _repo_root()
    regression_dir = Path(__file__).resolve().parent
    cases_path = regression_dir / "regression_cases.json"
    data = json.loads(cases_path.read_text(encoding="utf-8"))
    all_cases: list[dict[str, object]] = data.get("cases", [])
    if not all_cases:
        print("No cases in regression_cases.json")
        return 1

    if args.cases:
        want = set(args.cases)
        cases = [c for c in all_cases if str(c.get("name")) in want]
        missing = want - {str(c.get("name")) for c in cases}
        if missing:
            print(f"Unknown case name(s): {sorted(missing)}")
            return 1
    else:
        cases = all_cases

    profiles_dir = args.profiles_dir or (repo_root / "config" / "profiles")
    out_root = args.out_dir or (regression_dir / "search_results")
    rng = random.Random(args.seed)

    for case in cases:
        name = str(case["name"])
        profile_tag = case.get("profile")
        camera_id = str(case.get("camera_id", "camera1"))
        video = str(case["video"])
        if profile_tag is None:
            print(f"Skip {name}: no profile (search uses profile JSON as geometry base)")
            continue

        base_path = profiles_dir / f"{camera_id}_{profile_tag}.json"
        if not base_path.is_file():
            print(f"Skip {name}: missing base profile {base_path}")
            continue

        base_data = json.loads(base_path.read_text(encoding="utf-8"))
        out_dir = out_root / name
        out_dir.mkdir(parents=True, exist_ok=True)
        matches = 0
        print(f"=== {name} (base {base_path.name}) — {args.trials} trials ===")

        trial_iter = range(args.trials)
        if not args.no_progress:
            trial_iter = tqdm(
                trial_iter,
                total=args.trials,
                desc=name[:48],
                unit="trial",
                file=sys.stdout,
                leave=True,
            )

        for t in trial_iter:
            try:
                # Trial 0 = exact base profile (often not in the discrete grids, e.g. max_area_frac 1/350).
                sampled = (
                    deepcopy(base_data) if t == 0 else sample_profile_settings(rng, base_data)
                )
                with tempfile.NamedTemporaryFile(
                    mode="w",
                    suffix=".json",
                    delete=False,
                    encoding="utf-8",
                ) as tf:
                    cfg_path = Path(tf.name)
                    json.dump(sampled, tf, indent=2)
                try:
                    report = run_tracker(repo_root, cfg_path, video, args.scale)
                finally:
                    cfg_path.unlink(missing_ok=True)

                if report is None or not report_matches_case(report, case):
                    continue

                matches += 1
                out_path = out_dir / f"match_{matches:04d}.json"
                try:
                    base_rel = str(base_path.resolve().relative_to(repo_root.resolve()))
                except ValueError:
                    base_rel = str(base_path)
                payload = {
                    "meta": {
                        "case": name,
                        "trial_index": t,
                        "video": video,
                        "profile_base": base_rel,
                        "report": report,
                    },
                    "config": sampled,
                }
                out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
                msg = f"  match {matches}: trial {t} -> {out_path.relative_to(repo_root)}"
                tqdm.write(msg) if not args.no_progress else print(msg)
                if args.max_matches > 0 and matches >= args.max_matches:
                    break
            finally:
                if isinstance(trial_iter, tqdm):
                    trial_iter.set_postfix(matches=matches)

        if matches == 0:
            print(f"  No matches in {args.trials} trials (try more trials or widen search space).")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
