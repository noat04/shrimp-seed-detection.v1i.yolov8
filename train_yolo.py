from pathlib import Path

from ultralytics import YOLO


ROOT_DIR = Path(__file__).resolve().parent
DATA_YAML = ROOT_DIR / "data.yaml"


def main() -> None:
    model = YOLO("yolov8n.pt")
    model.train(
        data=str(DATA_YAML),
        epochs=30,
        imgsz=640,
        batch=8,
        name="shrimp_seed_yolov8n",
    )


if __name__ == "__main__":
    main()

