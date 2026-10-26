"""Retrain copied existing labels; new pending images are excluded, not reviewed."""
from pathlib import Path
from train_reviewed import validate
from ultralytics import YOLO

if __name__ == "__main__":
    root = Path(__file__).resolve().parents[1]
    data = root / "dataset_v2/data.yaml"
    validate(data)
    YOLO(str(root / "yolov8n.pt")).train(
        data=str(data), epochs=50, imgsz=640, batch=8, device="cpu", workers=0,
        project=str(root / "runs/detect"), name="shrimp_seed_real_v2", exist_ok=False,
    )
