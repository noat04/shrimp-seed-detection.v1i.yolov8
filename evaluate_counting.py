"""Compare predicted shrimp counts with ground-truth label counts for several inference settings.

Usage: python evaluate_counting.py [valid|test]
"""

import argparse
from pathlib import Path

from ultralytics import YOLO

from ai_service.main import DEFAULT_MODEL_PATH, detect_tiled


ROOT_DIR = Path(__file__).resolve().parent

# (name, conf, imgsz, tile_size); tile_size 0 = whole image
CONFIGS = [
    ("baseline conf=0.25", 0.25, 640, 0),
    ("conf=0.15", 0.15, 640, 0),
    ("imgsz=1280", 0.25, 1280, 0),
    ("tile=320", 0.25, 640, 320),
    ("tile=320 conf=0.15", 0.15, 640, 320),
    ("tile=480", 0.25, 640, 480),
]


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare counting settings against ground-truth labels.")
    parser.add_argument("split", nargs="?", choices=["valid", "test"], default="valid")
    parser.add_argument("--dataset-root", type=Path, default=ROOT_DIR)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL_PATH)
    args = parser.parse_args()
    split = args.split
    dataset_root = args.dataset_root.resolve()
    images = sorted(p for p in (dataset_root / split / "images").glob("*")
                    if p.suffix.lower() in {".jpg", ".jpeg", ".png"})
    if not images:
        parser.error("No images found in the selected split")
    if not args.model.is_file():
        parser.error(f"Model does not exist: {args.model}")

    truth = {}
    for image_path in images:
        label_path = dataset_root / split / "labels" / f"{image_path.stem}.txt"
        if not label_path.is_file():
            parser.error(f"Missing ground-truth label: {label_path}")
        truth[image_path] = sum(bool(line.strip()) for line in label_path.read_text().splitlines())

    model = YOLO(str(args.model))
    print(f"Model: {args.model.resolve()}")
    print(f"Dataset: {dataset_root}")
    print(f"{split}: {len(images)} images, {sum(truth.values())} ground-truth boxes")
    print(f"{'config':<22}{'MAE':>8}{'bias':>8}{'total':>8}")

    from PIL import Image

    for name, conf, imgsz, tile_size in CONFIGS:
        errors = []
        total = 0
        for image_path in images:
            image = Image.open(image_path).convert("RGB")
            if tile_size:
                _, detections = detect_tiled(model, image, conf, imgsz, 300, tile_size, 0.2, 0.5)
                count = len(detections)
            else:
                count = len(model.predict(source=image, conf=conf, imgsz=imgsz, verbose=False)[0].boxes)
            errors.append(count - truth[image_path])
            total += count
        mae = sum(abs(e) for e in errors) / len(errors)
        bias = sum(errors) / len(errors)
        print(f"{name:<22}{mae:>8.2f}{bias:>+8.2f}{total:>8}")


if __name__ == "__main__":
    main()
