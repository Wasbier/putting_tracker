from __future__ import annotations

import argparse
from pathlib import Path

import cv2


def extract_frames(
    video_path: Path,
    out_dir: Path,
    every_n_frames: int,
    max_frames: int | None,
) -> int:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise SystemExit(f"Could not open video file: {video_path}")

    out_dir.mkdir(parents=True, exist_ok=True)
    saved = 0
    frame_idx = 0
    stem = video_path.stem
    try:
        while True:
            ok, frame = cap.read()
            if not ok or frame is None:
                break
            if frame_idx % every_n_frames == 0:
                out_path = out_dir / f"{stem}_{frame_idx:06d}.jpg"
                cv2.imwrite(str(out_path), frame)
                saved += 1
                if max_frames is not None and saved >= max_frames:
                    break
            frame_idx += 1
    finally:
        cap.release()
    return saved


def main() -> None:
    p = argparse.ArgumentParser(description="Extract frames from recordings for YOLO labeling/training")
    p.add_argument(
        "videos",
        nargs="+",
        help="One or more input video files.",
    )
    p.add_argument(
        "--out-dir",
        type=Path,
        default=Path("yolo_dataset/images/train"),
        help="Output directory for extracted JPG frames.",
    )
    p.add_argument(
        "--every-n-frames",
        type=int,
        default=15,
        help="Extract one frame every N frames (default: 15).",
    )
    p.add_argument(
        "--max-frames-per-video",
        type=int,
        default=None,
        help="Optional cap on extracted frames per video.",
    )
    args = p.parse_args()

    total = 0
    for raw in args.videos:
        video_path = Path(raw)
        count = extract_frames(
            video_path,
            args.out_dir,
            every_n_frames=max(1, args.every_n_frames),
            max_frames=args.max_frames_per_video,
        )
        total += count
        print(f"{video_path}: extracted {count} frames")
    print(f"Done. Total extracted frames: {total}")


if __name__ == "__main__":
    main()
