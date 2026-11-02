"""Measure counting error per image domain on the whole, held-out photos of a dataset built by
dataset_expansion/build_domain_dataset.py.

Usage:
  python evaluate_domains.py --dataset dataset_v3 --model runs/detect/shrimp_seed_yolov8n/weights/best.pt \
      [--model other.pt ...] [--tiles 0 320 200] [--confs 0.25 0.15] [--imgsz 640]
"""
import argparse
import json
from pathlib import Path

from PIL import Image
from ultralytics import YOLO

from ai_service.main import detect_tiled


def count_boxes(label_path):
    return sum(1 for line in label_path.read_text().splitlines() if line.strip())


def predict_count(model, image, conf, imgsz, tile):
    if tile:
        return len(detect_tiled(model, image, conf, imgsz, 2000, tile, 0.25, 0.5)[1])
    return len(model.predict(source=image, conf=conf, imgsz=imgsz, max_det=2000, verbose=False)[0].boxes)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--model", type=Path, action="append", required=True)
    parser.add_argument("--tiles", type=int, nargs="+", default=[0, 320, 200], help="0 = whole image")
    parser.add_argument("--confs", type=float, nargs="+", default=[0.25, 0.15])
    parser.add_argument("--imgsz", type=int, default=640)
    args = parser.parse_args()

    manifest = json.loads((args.dataset / "domain_manifest.json").read_text(encoding="utf-8"))
    tests = {}
    for entry in manifest["files"]:
        if entry["split"] == "test":
            tests.setdefault(entry["domain"], []).append(entry)
    for domain in sorted(manifest["summary"]):
        if domain not in tests:
            print(f"note: domain {domain} has no held-out test photos, so it cannot be measured")
    if not tests:
        parser.error("The dataset has no held-out test photos")

    for model_path in args.model:
        if not model_path.is_file():
            parser.error(f"Model does not exist: {model_path}")
        model = YOLO(str(model_path))
        print(f"\n=== {model_path}")
        print(f"{'domain':<8}{'n':>3}{'truth/img':>10}  {'config':<18}{'MAE':>8}{'rel.err':>9}{'bias':>8}")
        for domain in sorted(tests):
            samples = []
            for entry in tests[domain]:
                image_path = args.dataset / entry["file"]
                truth = count_boxes(args.dataset / "test/labels" / f"{image_path.stem}.txt")
                samples.append((Image.open(image_path).convert("RGB"), truth))
            mean_truth = sum(t for _, t in samples) / len(samples)
            best = None
            for tile in args.tiles:
                for conf in args.confs:
                    errors = [predict_count(model, img, conf, args.imgsz, tile) - truth for img, truth in samples]
                    mae = sum(abs(e) for e in errors) / len(errors)
                    rel = sum(abs(e) / truth for e, (_, truth) in zip(errors, samples) if truth) / max(1, sum(1 for _, t in samples if t))
                    bias = sum(errors) / len(errors)
                    name = f"tile={tile} conf={conf}" if tile else f"whole conf={conf}"
                    print(f"{domain:<8}{len(samples):>3}{mean_truth:>10.1f}  {name:<18}{mae:>8.1f}{100*rel:>8.0f}%{bias:>+8.1f}")
                    if best is None or rel < best[0]:
                        best = (rel, name)
            print(f"{'':<8}-> best for {domain}: {best[1]} (rel.err {100*best[0]:.0f}%)")


if __name__ == "__main__":
    main()
