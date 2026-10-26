import os
from io import BytesIO
from pathlib import Path
from typing import Annotated, Any, Optional

from fastapi import FastAPI, File, HTTPException, Query, UploadFile
from fastapi.concurrency import run_in_threadpool
import numpy as np
import torch
from PIL import Image
from ultralytics import YOLO
from ultralytics.engine.results import Results
import cloudinary.uploader

from ai_service.cloudinary_config import CLOUDINARY_CONFIGURED


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


def upload_original_to_cloudinary(file_bytes: bytes, filename: Optional[str]) -> dict[str, Any]:
    stream = BytesIO(file_bytes)
    safe_filename = (filename or "").replace("\\", "/").rsplit("/", 1)[-1].strip()
    stream.name = safe_filename or "shrimp-image.jpg"

    return cloudinary.uploader.upload(
        stream,
        folder="shrimp-detection/originals",
        resource_type="image",
    )



def upload_annotated_to_cloudinary(plotted_bgr: np.ndarray) -> dict[str, Any]:
    """Results.plot returns BGR; convert to RGB before encoding the PNG."""
    stream = BytesIO()
    Image.fromarray(plotted_bgr[..., ::-1]).save(stream, format="PNG")
    stream.seek(0)
    stream.name = "shrimp-annotated.png"
    return cloudinary.uploader.upload(
        stream, folder="shrimp-detection/annotated", resource_type="image"
    )


def plot_detections(image: Image.Image, names: dict[int, str], detections: list[dict[str, Any]]) -> np.ndarray:
    """Plot final merged tile detections on the original image, never individual tiles."""
    boxes = np.asarray([
        [det["box"]["x1"], det["box"]["y1"], det["box"]["x2"], det["box"]["y2"],
         det["confidence"], det["class_id"]]
        for det in detections
    ], dtype=np.float32).reshape(-1, 6)
    result = Results(
        orig_img=np.asarray(image)[..., ::-1].copy(), path="", names=names, boxes=torch.from_numpy(boxes)
    )
    return result.plot(labels=False, conf=False)


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


def tile_origins(length: int, tile: int, overlap: float) -> list[int]:
    if length <= tile:
        return [0]

    step = max(1, int(tile * (1 - overlap)))
    origins = list(range(0, length - tile, step))
    origins.append(length - tile)
    return origins


def merge_detections(detections: list[dict[str, Any]], iou_threshold: float) -> list[dict[str, Any]]:
    """Greedy NMS to drop duplicates of the same shrimp seen in overlapping tiles."""
    kept: list[dict[str, Any]] = []

    for det in sorted(detections, key=lambda item: item["confidence"], reverse=True):
        box = det["box"]
        area = (box["x2"] - box["x1"]) * (box["y2"] - box["y1"])
        duplicate = False

        for other in kept:
            ob = other["box"]
            iw = min(box["x2"], ob["x2"]) - max(box["x1"], ob["x1"])
            ih = min(box["y2"], ob["y2"]) - max(box["y1"], ob["y1"])
            if iw <= 0 or ih <= 0:
                continue
            inter = iw * ih
            other_area = (ob["x2"] - ob["x1"]) * (ob["y2"] - ob["y1"])
            if inter / (area + other_area - inter) >= iou_threshold:
                duplicate = True
                break

        if not duplicate:
            kept.append(det)

    return kept


def detect_tiled(
    detector: YOLO,
    image: Image.Image,
    conf: float,
    imgsz: int,
    max_det: int,
    tile_size: int,
    overlap: float,
    merge_iou: float,
) -> tuple[dict[int, str], list[dict[str, Any]]]:
    """Run YOLO on overlapping crops so small, dense shrimp seeds stay large enough to detect."""
    pixels = np.asarray(image)
    names: dict[int, str] = {}
    detections: list[dict[str, Any]] = []

    for y0 in tile_origins(image.height, tile_size, overlap):
        for x0 in tile_origins(image.width, tile_size, overlap):
            crop = pixels[y0 : y0 + tile_size, x0 : x0 + tile_size]
            result = detector.predict(source=crop, conf=conf, imgsz=imgsz, max_det=max_det, verbose=False)[0]
            names = {int(key): value for key, value in result.names.items()}

            for box in result.boxes:
                det = detection_to_dict(box, names)
                det["box"] = {
                    "x1": round(det["box"]["x1"] + x0, 2),
                    "y1": round(det["box"]["y1"] + y0, 2),
                    "x2": round(det["box"]["x2"] + x0, 2),
                    "y2": round(det["box"]["y2"] + y0, 2),
                }
                detections.append(det)

    return names, merge_detections(detections, merge_iou)


@app.get("/health")
def health() -> dict[str, Any]:
    return {
        "status": "ok",
        "model_path": str(MODEL_PATH),
        "model_exists": MODEL_PATH.exists(),
        "model_loaded": model is not None,
        "cloudinary_configured": CLOUDINARY_CONFIGURED,
    }


@app.post("/predict")
async def predict(
    file: UploadFile = File(...),
    conf: float = Query(0.25, ge=0.0, le=1.0),
    imgsz: int = Query(640, ge=64, le=2048),
    max_det: Annotated[int, Query(ge=1, le=5000)] = 300,
    tile_size: Annotated[int, Query(ge=0, le=4096, description="Crop size in px; 0 disables tiling.")] = 0,
    tile_overlap: Annotated[float, Query(ge=0.0, lt=1.0)] = 0.2,
    merge_iou: Annotated[float, Query(gt=0.0, le=1.0)] = 0.5,
) -> dict[str, Any]:
    file_bytes = await file.read()
    image = read_image(file_bytes)

    if not CLOUDINARY_CONFIGURED:
        raise HTTPException(
            status_code=503,
            detail={
                "message": "Cloudinary is not configured. Set all CLOUDINARY_* environment variables.",
                "image_saved": False,
            },
        )

    detector = get_model()

    if tile_size:
        names, detections = detect_tiled(
            detector, image, conf, imgsz, max_det, tile_size, tile_overlap, merge_iou
        )
    else:
        result = detector.predict(source=image, conf=conf, imgsz=imgsz, max_det=max_det, verbose=False)[0]
        names = {int(key): value for key, value in result.names.items()}
        detections = [detection_to_dict(box, names) for box in result.boxes]

    plotted_bgr = plot_detections(image, names, detections) if tile_size else result.plot(labels=False, conf=False)

    try:
        cloudinary_asset = await run_in_threadpool(
            upload_original_to_cloudinary,
            file_bytes,
            file.filename,
        )
        if not cloudinary_asset.get("secure_url") or not cloudinary_asset.get("public_id"):
            raise RuntimeError("Cloudinary upload response is missing the asset URL or public ID.")
    except Exception as exc:
        raise HTTPException(
            status_code=502,
            detail={
                "message": "YOLO prediction succeeded, but the original image could not be backed up to Cloudinary.",
                "image_saved": False,
            },
        ) from exc

    try:
        annotated_asset = await run_in_threadpool(upload_annotated_to_cloudinary, plotted_bgr)
        if not annotated_asset.get("secure_url") or not annotated_asset.get("public_id"):
            raise RuntimeError("Cloudinary annotated upload response is missing URL or public ID.")
    except Exception as exc:
        raise HTTPException(
            status_code=502,
            detail={
                "message": "Original image saved, but annotated image upload failed.",
                "image_saved": True,
                "image_url": cloudinary_asset["secure_url"],
                "public_id": cloudinary_asset["public_id"],
                "annotated_image_saved": False,
            },
        ) from exc

    return {
        "filename": file.filename,
        "image_saved": True,
        "annotated_image_saved": True,
        "annotated_image_url": annotated_asset["secure_url"],
        "annotated_public_id": annotated_asset["public_id"],
        "image_url": cloudinary_asset["secure_url"],
        "public_id": cloudinary_asset["public_id"],
        "image": {
            "width": image.width,
            "height": image.height,
        },
        "count": len(detections),
        "detections": detections,
    }
