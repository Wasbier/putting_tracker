from __future__ import annotations

import json
import subprocess
import sys
import tempfile
from pathlib import Path


def run_case(repo_root: Path, case: dict[str, object]) -> tuple[bool, str]:
    name = str(case["name"])
    video = str(case["video"])
    expected_attempts = int(case["expected_attempts"])
    expected_made = int(case["expected_made"])

    cmd: list[str] = [
        sys.executable,
        str(repo_root / "track_putts.py"),
        "--video",
        video,
        "--scale",
        "0.75",
        "--headless",
        "--no-loop",
    ]

    profile = case.get("profile")
    if profile is not None:
        cmd.extend(["--profile", str(profile)])
        camera_id = case.get("camera_id", "camera1")
        cmd.extend(["--camera-id", str(camera_id)])
    else:
        config = case.get("config")
        if config is None:
            raise ValueError(f"Case {name} must have profile or config")
        cmd.extend(["--config", str(config)])

    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as tf:
        report_path = Path(tf.name)
    cmd.extend(["--report-json", str(report_path)])

    proc = subprocess.run(cmd, cwd=str(repo_root), capture_output=True, text=True)
    if proc.returncode != 0:
        return False, f"{name}: runner failed\n{proc.stdout}\n{proc.stderr}"

    try:
        report = json.loads(report_path.read_text(encoding="utf-8"))
    finally:
        report_path.unlink(missing_ok=True)

    got_attempts = int(report["attempts"])
    got_made = int(report["made"])
    ok = got_attempts == expected_attempts and got_made == expected_made
    exp_seq = case.get("expected_sequence")
    if exp_seq is not None:
        got_seq = report.get("putt_sequence")
        if not isinstance(got_seq, list) or [str(x) for x in got_seq] != [
            str(x) for x in exp_seq
        ]:
            ok = False
    status = "PASS" if ok else "FAIL"
    msg = (
        f"[{status}] {name}: attempts {got_attempts}/{expected_attempts}, "
        f"made {got_made}/{expected_made}"
    )
    if exp_seq is not None:
        got_seq = report.get("putt_sequence")
        msg += f", sequence {got_seq!r} vs {exp_seq!r}"
    return ok, msg


def main() -> int:
    # Script lives in regression_tests/; project root is one level up.
    regression_dir = Path(__file__).resolve().parent
    repo_root = regression_dir.parent
    cases_path = regression_dir / "regression_cases.json"
    data = json.loads(cases_path.read_text(encoding="utf-8"))
    cases = data.get("cases", [])
    if not cases:
        print("No cases in regression_cases.json")
        return 1

    all_ok = True
    for case in cases:
        ok, msg = run_case(repo_root, case)
        print(msg)
        all_ok = all_ok and ok

    return 0 if all_ok else 2


if __name__ == "__main__":
    raise SystemExit(main())
