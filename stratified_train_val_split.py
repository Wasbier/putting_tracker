"""
Rebuild train/val split with stratification by capture session (day + time of recording).

Filenames like capture_20260421_082028_003735.jpg -> session 20260421_082028.
Pools images from images/train and images/val, assigns ~VAL_RATIO to val per session,
then moves files and labels to match.

Usage (from repo root): python stratified_train_val_split.py
"""
from __future__ import annotations

import random
import re
import shutil
import sys
from collections import Counter, defaultdict
from pathlib import Path

# Target fraction in validation (Ultralytics often uses ~10–20%)
VAL_RATIO = 0.17
SEED = 42

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}


def session_key(stem: str) -> str:
    m = re.match(r"capture_(\d{8})_(\d{6})_", stem)
    if m:
        return f"{m.group(1)}_{m.group(2)}"
    return "other"


def val_count_for_session(n: int, ratio: float) -> int:
    """How many of n images go to val for this session."""
    if n <= 0:
        return 0
    if n == 1:
        return 0  # singleton stays train (avoid val metrics on a single box)
    if n == 2:
        return 1  # one train, one val for time-of-day variety
    k = int(round(n * ratio))
    k = max(0, min(n, k))
    if k == 0 and n >= 5:
        k = 1  # mid-size session: at least one val frame
    if k == n and n >= 2:
        k = n - 1  # keep at least one train frame
    return k


def collect_pool(root: Path) -> dict[str, tuple[Path, Path]]:
    """stem -> (image path, label path)."""
    pool: dict[str, tuple[Path, Path]] = {}
    for split in ("train", "val"):
        img_dir = root / "images" / split
        lbl_dir = root / "labels" / split
        if not img_dir.is_dir() or not lbl_dir.is_dir():
            continue
        for p in img_dir.iterdir():
            if not p.is_file() or p.suffix.lower() not in IMAGE_SUFFIXES:
                continue
            stem = p.stem
            lbl = lbl_dir / f"{stem}.txt"
            if not lbl.is_file():
                print(f"WARN: no label for {p}", file=sys.stderr)
                continue
            if stem in pool:
                print(f"WARN: duplicate stem {stem}", file=sys.stderr)
            pool[stem] = (p, lbl)
    return pool


def assign_splits(stems: list[str]) -> dict[str, str]:
    """stem -> 'train' or 'val'."""
    by_sess: dict[str, list[str]] = defaultdict(list)
    for s in stems:
        by_sess[session_key(s)].append(s)

    rng = random.Random(SEED)
    out: dict[str, str] = {}
    for sess, group in sorted(by_sess.items()):
        rng.shuffle(group)
        n = len(group)
        nv = val_count_for_session(n, VAL_RATIO)
        val_set = set(group[:nv])
        for s in group:
            out[s] = "val" if s in val_set else "train"
    return out


def move_to(img: Path, lbl: Path, split: str, root: Path) -> None:
    dest_img = root / "images" / split / img.name
    dest_lbl = root / "labels" / split / lbl.name
    dest_img.parent.mkdir(parents=True, exist_ok=True)
    dest_lbl.parent.mkdir(parents=True, exist_ok=True)
    if img.resolve() == dest_img.resolve():
        return
    if dest_img.exists() and img.resolve() != dest_img.resolve():
        raise FileExistsError(dest_img)
    if dest_lbl.exists() and lbl.resolve() != dest_lbl.resolve():
        raise FileExistsError(dest_lbl)
    shutil.move(str(img), str(dest_img))
    shutil.move(str(lbl), str(dest_lbl))


def main() -> None:
    root = Path(__file__).resolve().parent / "yolo_dataset"
    if not root.is_dir():
        print(f"Missing {root}", file=sys.stderr)
        sys.exit(1)

    pool = collect_pool(root)
    if not pool:
        print("No images found.", file=sys.stderr)
        sys.exit(1)

    assignment = assign_splits(sorted(pool.keys()))

    train_n = sum(1 for s in assignment if assignment[s] == "train")
    val_n = sum(1 for s in assignment if assignment[s] == "val")
    print(
        f"Pool: {len(pool)} images, target ~{VAL_RATIO:.0%} val -> "
        f"train={train_n} val={val_n} ({val_n / len(pool):.1%} val)"
    )

    # Move into place (order: val first avoids some edge cases)
    for stem in sorted(assignment.keys()):
        split = assignment[stem]
        img, lbl = pool[stem]
        cur = "val" if img.parent.name == "val" else "train"
        if cur != split:
            move_to(img, lbl, split, root)

    # Drop stale caches
    for cache in (root / "labels" / "train.cache", root / "labels" / "val.cache"):
        if cache.is_file():
            cache.unlink()
            print(f"Removed {cache.name}")

    # Summary: val frames per capture session (day + time)
    val_dir = root / "images" / "val"
    vc = Counter()
    if val_dir.is_dir():
        for p in val_dir.iterdir():
            if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES:
                vc[session_key(p.stem)] += 1
    print("Val images per session (YYYYMMDD_HHMMSS):")
    for sess in sorted(vc.keys()):
        print(f"  {sess}: {vc[sess]}")
    print("Done.")


if __name__ == "__main__":
    main()
