from __future__ import annotations

import argparse
from pathlib import Path

try:
    from ultralytics import YOLO
except Exception as exc:  # pragma: no cover - runtime dependency
    raise SystemExit(
        "Ultralytics is required. Install dependencies with `pip install -r requirements.txt`."
    ) from exc


def ensure_dataset_yaml(dataset_root: Path) -> Path:
    yaml_path = dataset_root / "dataset.yaml"
    if yaml_path.is_file():
        return yaml_path

    content = "\n".join(
        [
            f"path: {dataset_root.resolve()}",
            "train: images/train",
            "val: images/val",
            "",
            "names:",
            "  0: golf_ball",
            "",
        ]
    )
    yaml_path.write_text(content, encoding="utf-8")
    return yaml_path


def main() -> None:
    p = argparse.ArgumentParser(description="Fine-tune a pre-trained YOLO model for golf ball detection")
    p.add_argument(
        "--dataset-root",
        type=Path,
        default=Path("yolo_dataset"),
        help="Dataset root containing images/ and labels/ folders.",
    )
    p.add_argument(
        "--model",
        type=str,
        default="yolov8n.pt",
        help="Pre-trained YOLO checkpoint to fine-tune.",
    )
    p.add_argument("--epochs", type=int, default=60)
    p.add_argument("--imgsz", type=int, default=1280)
    p.add_argument("--batch", type=int, default=8)
    p.add_argument("--device", type=str, default=None, help="Optional device override, e.g. cpu, 0, cuda:0")
    p.add_argument("--project", type=Path, default=Path("runs/yolo_ball"))
    p.add_argument("--name", type=str, default="golf_ball")
    args = p.parse_args()

    dataset_yaml = ensure_dataset_yaml(args.dataset_root)
    model = YOLO(args.model)
    model.train(
        data=str(dataset_yaml),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        project=str(args.project),
        name=args.name,
    )
    print("Training complete.")
    print("Best weights are typically under runs/.../weights/best.pt")


if __name__ == "__main__":
    main()
