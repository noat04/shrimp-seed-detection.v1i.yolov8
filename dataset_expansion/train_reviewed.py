"""Train a separate checkpoint only after manual label review."""
import argparse
import math
from pathlib import Path

import yaml
from ultralytics import YOLO


def validate(data):
    config = yaml.safe_load(data.read_text(encoding="utf-8"))
    if config.get("nc") != 1 or config.get("names") not in (["shrimp-seed"], {0: "shrimp-seed"}):
        raise ValueError("Expected nc=1 and names: [shrimp-seed]")
    base = Path(config.get("path", data.parent))
    if not base.is_absolute():
        base = data.parent / base
    seen = {}
    import hashlib
    for split in ("train", "val", "test"):
        entry = config.get(split)
        if not isinstance(entry, str):
            raise ValueError(f"{split}: provide an images directory")
        directory = Path(entry)
        if not directory.is_absolute():
            directory = base / directory
        if directory.name != "images":
            raise ValueError(f"{split}: directory must end in images")
        images = [p for p in directory.glob("*") if p.suffix.lower() in (".jpg", ".jpeg", ".png")]
        if not images:
            raise ValueError(f"{split}: no images")
        total = 0
        for image in images:
            digest = hashlib.sha256(image.read_bytes()).hexdigest()
            if digest in seen and seen[digest] != split:
                raise ValueError(f"Duplicate image across splits: {image}")
            seen[digest] = split
            label = directory.parent / "labels" / (image.stem + ".txt")
            if not label.exists():
                raise ValueError(f"Missing label: {label}")
            for line in label.read_text().splitlines():
                if not line.strip():
                    continue
                values = list(map(float, line.split()))
                if len(values) != 5 or not all(math.isfinite(v) for v in values):
                    raise ValueError(f"Invalid YOLO row: {label}")
                c,x,y,w,h = values
                if c != 0 or not (0 <= x <= 1 and 0 <= y <= 1 and 0 < w <= 1 and 0 < h <= 1):
                    raise ValueError(f"Invalid coordinates/class: {label}")
                if min(x-w/2,y-h/2) < -0.00001 or max(x+w/2,y+h/2) > 1.00001:
                    raise ValueError(f"Box outside image: {label}")
                total += 1
        if not total:
            raise ValueError(f"{split}: no object labels")
        print(f"{split}: {len(images)} images, {total} boxes")


def default_device():
    import torch
    return "0" if torch.cuda.is_available() else "cpu"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--reviewed", action="store_true", help="Confirm labels and split groups were manually reviewed")
    parser.add_argument("--check-only", action="store_true")
    parser.add_argument("--model", default="yolov8s.pt", help="Pretrained weights (resolved in the project root first)")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--imgsz", type=int, default=640,
                        help="Tiles are upscaled to this size; match imgsz used by /predict")
    parser.add_argument("--batch", type=int, help="Default: 8 (a 4 GB GPU is nearly full at 16)")
    parser.add_argument("--device", help="'0' for the first GPU, 'cpu'; default: GPU if CUDA is available")
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--name", default="shrimp_seed_tiles_v3")
    args = parser.parse_args()
    if not args.reviewed:
        parser.error("Manual review required: pass --reviewed only after checking all labels and split groups")
    if args.epochs < 1:
        parser.error("epochs must be positive")
    data = args.data.resolve()
    validate(data)
    if args.check_only:
        return

    root = Path(__file__).resolve().parents[1]
    device = args.device or default_device()
    batch = args.batch or 8
    weights = root / args.model
    print(f"Training {args.model} on device={device}, batch={batch}, imgsz={args.imgsz}")
    # Top-down photos have no fixed orientation: flip both ways. Rotation is left off because
    # rotating an axis-aligned box around a thin, diagonal shrimp makes the box much looser.
    YOLO(str(weights if weights.exists() else args.model)).train(
        data=str(data), epochs=args.epochs, imgsz=args.imgsz, batch=batch, device=device,
        workers=args.workers, project=str(root / "runs/detect"), name=args.name, exist_ok=False,
        fliplr=0.5, flipud=0.5, degrees=0.0, hsv_h=0.015, hsv_s=0.5, hsv_v=0.4,
        scale=0.5, mosaic=1.0, close_mosaic=10, max_det=1000,
    )


if __name__ == "__main__":
    main()
