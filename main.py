"""Road Damage Detection API - FastAPI + Ultralytics YOLO.

Run:  uvicorn main:app --reload
Docs: http://127.0.0.1:8000/docs      UI: http://127.0.0.1:8000/
"""
import base64
import os
import threading
import time
from collections import Counter
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
from fastapi import FastAPI, File, HTTPException, Query, UploadFile
from fastapi.responses import FileResponse, Response

BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = Path(os.getenv("MODEL_DIR", BASE_DIR / "models"))
DEFAULT_MODEL = os.getenv("DEFAULT_MODEL")          # optional: weight file name without ".pt"
DEFAULT_CONF = float(os.getenv("DEFAULT_CONF", "0.25"))
IMGSZ = int(os.getenv("IMGSZ", "640"))
MAX_UPLOAD_MB = 10

_models: dict = {}                 # loaded models, kept in memory (loaded once, not per request)
_lock = threading.Lock()           # Ultralytics predictors are not thread-safe


def available_models() -> dict:
    """All *.pt files in the models folder: {name: path}."""
    return {p.stem: p for p in sorted(MODEL_DIR.glob("*.pt"))}


def default_model_name(paths: dict) -> str:
    if DEFAULT_MODEL in paths:
        return DEFAULT_MODEL
    for name in paths:                              # prefer the best-performing model (YOLO11s)
        if "11s" in name:
            return name
    return next(iter(paths))


def get_model(name: Optional[str] = None):
    paths = available_models()
    if not paths:
        raise HTTPException(503, f"No .pt weights found in {MODEL_DIR}. Copy your trained weights there.")
    name = name or default_model_name(paths)
    if name not in paths:
        raise HTTPException(404, f"Unknown model '{name}'. Available: {list(paths)}")
    with _lock:
        if name not in _models:
            from ultralytics import YOLO            # imported lazily: keeps start-up fast
            _models[name] = YOLO(str(paths[name]))
    return name, _models[name]


@asynccontextmanager
async def lifespan(app: FastAPI):
    try:                                            # warm-up: load the default model at start-up
        get_model()
    except HTTPException as e:
        print("WARNING:", e.detail)
    yield


app = FastAPI(
    title="Road Damage Detection API",
    description="Detects road damage (alligator, block, crack, edge, longitudinal, pothole, transverse) "
                "in images using YOLO models trained on the Roboflow road-damage dataset.",
    version="1.0.0",
    lifespan=lifespan,
)


def read_image(upload: UploadFile) -> np.ndarray:
    data = upload.file.read(MAX_UPLOAD_MB * 1024 * 1024 + 1)
    if len(data) > MAX_UPLOAD_MB * 1024 * 1024:
        raise HTTPException(413, f"Image larger than {MAX_UPLOAD_MB} MB")
    img = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        raise HTTPException(400, "Could not decode the file as an image (use JPG, PNG, WEBP or BMP)")
    return img


def to_jpeg(img_bgr: np.ndarray, quality: int = 90) -> bytes:
    ok, buf = cv2.imencode(".jpg", img_bgr, [cv2.IMWRITE_JPEG_QUALITY, quality])
    if not ok:
        raise HTTPException(500, "Failed to encode the output image")
    return buf.tobytes()


def detect(img: np.ndarray, model_name: str, model, conf: float):
    t0 = time.perf_counter()
    with _lock:
        result = model.predict(img, imgsz=IMGSZ, conf=conf, verbose=False)[0]
    ms = (time.perf_counter() - t0) * 1000

    boxes = result.boxes
    n = len(boxes)
    xyxy = boxes.xyxy.cpu().numpy() if n else np.zeros((0, 4))
    scores = boxes.conf.cpu().numpy() if n else np.zeros(0)
    classes = boxes.cls.cpu().numpy().astype(int) if n else np.zeros(0, dtype=int)

    detections = [
        {
            "class_id": int(c),
            "class_name": result.names[int(c)],
            "confidence": round(float(s), 4),
            "bbox": {k: round(float(v), 1) for k, v in zip(("x1", "y1", "x2", "y2"), b)},
        }
        for b, s, c in zip(xyxy, scores, classes)
    ]
    detections.sort(key=lambda d: d["confidence"], reverse=True)
    info = {
        "model": model_name,
        "image": {"width": int(img.shape[1]), "height": int(img.shape[0])},
        "conf_threshold": conf,
        "inference_ms": round(ms, 1),
        "num_detections": n,
        "counts": dict(Counter(d["class_name"] for d in detections)),
        "detections": detections,
    }
    return info, result


@app.get("/", include_in_schema=False)
def index():
    return FileResponse(BASE_DIR / "static" / "index.html")


@app.get("/health")
def health():
    paths = available_models()
    return {
        "status": "ok" if paths else "no_models",
        "models": list(paths),
        "default_model": default_model_name(paths) if paths else None,
        "loaded": list(_models),
        "default_conf": DEFAULT_CONF,
    }


@app.get("/models")
def list_models():
    paths = available_models()
    return {"models": list(paths), "default": default_model_name(paths) if paths else None}


@app.post("/predict")
def predict(
    file: UploadFile = File(..., description="Road image (JPG/PNG/WEBP/BMP, max 10 MB)"),
    model: Optional[str] = Query(None, description="Weight file name without .pt (see /models)"),
    conf: float = Query(DEFAULT_CONF, ge=0.01, le=1.0, description="Confidence threshold"),
    annotate: bool = Query(True, description="Include the annotated image (base64 JPEG)"),
):
    """Returns detections as JSON (+ optional annotated image as a base64 data URL)."""
    img = read_image(file)
    name, mdl = get_model(model)
    info, result = detect(img, name, mdl, conf)
    if annotate:
        b64 = base64.b64encode(to_jpeg(result.plot())).decode()
        info["annotated_image"] = f"data:image/jpeg;base64,{b64}"
    return info


@app.post("/predict/image")
def predict_image(
    file: UploadFile = File(...),
    model: Optional[str] = Query(None),
    conf: float = Query(DEFAULT_CONF, ge=0.01, le=1.0),
):
    """Returns only the annotated image (image/jpeg)."""
    img = read_image(file)
    name, mdl = get_model(model)
    info, result = detect(img, name, mdl, conf)
    return Response(
        content=to_jpeg(result.plot()),
        media_type="image/jpeg",
        headers={"X-Detections": str(info["num_detections"]), "X-Model": name},
    )
