"""
Keep images/train and images/val aligned with labels/train and labels/val.

- Images in train/val with no matching .txt -> move to images/unlabelled
- .txt in labels/train or labels/val with no image in that split -> move image
  from images/unlabelled if present

Run from repo root: python sync_yolo_image_labels.py
"""
from __future__ import annotations

import shutil
import sys
from pathlib import Path

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}


def collect_images(folder: Path) -> dict[str, Path]:
    out: dict[str, Path] = {}
    if not folder.is_dir():
        return out
    for p in folder.iterdir():
        if not p.is_file() or p.suffix.lower() not in IMAGE_SUFFIXES:
            continue
        stem = p.stem
        if stem in out:
            print(f"WARN duplicate stem in {folder.name}: {stem}", file=sys.stderr)
        out[stem] = p
    return out


def collect_label_stems(folder: Path) -> set[str]:
    if not folder.is_dir():
        return set()
    skip = {"classes"}  # YOLO class names file, not per-image labels
    return {
        p.stem
        for p in folder.iterdir()
        if p.is_file() and p.suffix.lower() == ".txt" and p.stem not in skip
    }


def find_image_by_stem(folder: Path, stem: str) -> Path | None:
    if not folder.is_dir():
        return None
    for suf in IMAGE_SUFFIXES:
        p = folder / f"{stem}{suf}"
        if p.is_file():
            return p
    return None


def move_safe(src: Path, dest_dir: Path) -> bool:
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest = dest_dir / src.name
    if dest.exists():
        print(f"SKIP dest exists: {dest}", file=sys.stderr)
        return False
    shutil.move(str(src), str(dest))
    return True


def sync_split(
    name: str,
    img_dir: Path,
    lbl_dir: Path,
    unlab_dir: Path,
) -> None:
    images = collect_images(img_dir)
    label_stems = collect_label_stems(lbl_dir)

    # 1) Images without label -> unlabelled
    for stem, path in list(images.items()):
        if stem not in label_stems:
            if move_safe(path, unlab_dir):
                print(f"{name}: no label -> unlabelled: {path.name}")
                del images[stem]

    # 2) Labels without image in split -> pull from unlabelled
    image_stems = set(images.keys())
    for stem in sorted(label_stems - image_stems):
        src = find_image_by_stem(unlab_dir, stem)
        if src is None:
            print(f"{name}: label but no image anywhere: {stem}.txt", file=sys.stderr)
            continue
        if move_safe(src, img_dir):
            print(f"{name}: unlabelled -> {name}: {src.name}")
            images[stem] = img_dir / src.name


def main() -> None:
    root = Path(__file__).resolve().parent / "yolo_dataset"
    if not root.is_dir():
        print(f"Missing {root}", file=sys.stderr)
        sys.exit(1)

    img_train = root / "images" / "train"
    img_val = root / "images" / "val"
    unlab = root / "images" / "unlabelled"
    lbl_train = root / "labels" / "train"
    lbl_val = root / "labels" / "val"

    sync_split("train", img_train, lbl_train, unlab)
    sync_split("val", img_val, lbl_val, unlab)

    # Drop stale ultralytics caches
    for cache in (lbl_train.parent / "train.cache", lbl_train.parent / "val.cache"):
        if cache.is_file():
            cache.unlink()
            print(f"Removed {cache.name}")

    print("Done.")


if __name__ == "__main__":
    main()
