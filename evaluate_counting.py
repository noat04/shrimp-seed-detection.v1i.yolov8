"""Compare predicted shrimp counts with ground-truth label counts for several inference settings.

Usage: python evaluate_counting.py [valid|test]
"""

import sys
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
    split = sys.argv[1] if len(sys.argv) > 1 else "valid"
    images = sorted((ROOT_DIR / split / "images").glob("*"))
    model = YOLO(str(DEFAULT_MODEL_PATH))

    truth = {}
    for image_path in images:
        label_path = ROOT_DIR / split / "labels" / f"{image_path.stem}.txt"
        truth[image_path] = len(label_path.read_text().splitlines()) if label_path.exists() else 0

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
