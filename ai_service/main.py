import os
from io import BytesIO
from pathlib import Path
from typing import Any, Optional

from fastapi import FastAPI, File, HTTPException, Query, UploadFile
from PIL import Image
from ultralytics import YOLO


ROOT_DIR = Path(__file__).resolve().parents[1]
DEFAULT_MODEL_PATH = ROOT_DIR / "runs" / "detect" / "shrimp_seed_yolov8n" / "weights" / "best.pt"
MODEL_PATH = Path(os.getenv("MODEL_PATH", str(DEFAULT_MODEL_PATH)))

app = FastAPI(
    title="Shrimp Seed Detection AI Service",
    description="YOLOv8 service for shrimp seed detection and counting.",
    version="0.1.0",
)

model: Optional[YOLO] = None


def get_model() -> YOLO:
    global model

    if model is None:
        if not MODEL_PATH.exists():
            raise HTTPException(
                status_code=503,
                detail=(
                    "Model file not found. Train first with `python train_yolo.py` "
                    f"or set MODEL_PATH. Expected: {MODEL_PATH}"
                ),
            )
        model = YOLO(str(MODEL_PATH))

    return model


def read_image(file_bytes: bytes) -> Image.Image:
    try:
        return Image.open(BytesIO(file_bytes)).convert("RGB")
    except Exception as exc:
        raise HTTPException(status_code=400, detail="Uploaded file is not a valid image.") from exc


def detection_to_dict(box: Any, names: dict[int, str]) -> dict[str, Any]:
    class_id = int(box.cls[0].item())
    x1, y1, x2, y2 = [float(value) for value in box.xyxy[0].tolist()]

    return {
        "class_id": class_id,
        "class": names.get(class_id, str(class_id)),
        "confidence": float(box.conf[0].item()),
        "box": {
            "x1": round(x1, 2),
            "y1": round(y1, 2),
            "x2": round(x2, 2),
            "y2": round(y2, 2),
        },
    }


@app.get("/health")
def health() -> dict[str, Any]:
    return {
        "status": "ok",
        "model_path": str(MODEL_PATH),
        "model_exists": MODEL_PATH.exists(),
        "model_loaded": model is not None,
    }


@app.post("/predict")
async def predict(
    file: UploadFile = File(...),
    conf: float = Query(0.25, ge=0.0, le=1.0),
    imgsz: int = Query(640, ge=64, le=2048),
) -> dict[str, Any]:
    file_bytes = await file.read()
    image = read_image(file_bytes)
    detector = get_model()

    result = detector.predict(source=image, conf=conf, imgsz=imgsz, verbose=False)[0]
    names = {int(key): value for key, value in result.names.items()}
    detections = [detection_to_dict(box, names) for box in result.boxes]

    return {
        "filename": file.filename,
        "image": {
            "width": image.width,
            "height": image.height,
        },
        "count": len(detections),
        "detections": detections,
    }
