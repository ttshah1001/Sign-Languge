import base64
import os
from pathlib import Path

import cv2
import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from detector import SignLanguageDetector

STATIC_DIR = Path(__file__).parent / "static"
detector = SignLanguageDetector()
model_loaded = False
model_load_error = None


def ensure_model_loaded():
    global model_loaded, model_load_error
    if model_loaded:
        return True
    try:
        model_loaded = detector.load_model()
        if not model_loaded:
            model_load_error = "Model file not found on server"
    except Exception as exc:
        model_loaded = False
        model_load_error = str(exc)
    return model_loaded


ensure_model_loaded()

app = FastAPI(title="Sign Language Detector")
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


class PredictRequest(BaseModel):
    image: str


@app.get("/")
def index():
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/health")
def health():
    return {
        "status": "ok",
        "model_loaded": model_loaded,
        "model_path": os.environ.get("MODEL_PATH", "sign_model_26letters.pkl"),
        "model_error": model_load_error,
    }


@app.post("/predict")
def predict(body: PredictRequest):
    try:
        encoded = body.image.split(",", 1)[1] if "," in body.image else body.image
        image_bytes = base64.b64decode(encoded)
        np_arr = np.frombuffer(image_bytes, np.uint8)
        frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        if frame is None:
            raise ValueError("Could not decode image")

        ensure_model_loaded()
        letter, hand_detected = detector.predict_from_image(frame)
        return {
            "letter": letter,
            "hand_detected": hand_detected,
            "model_loaded": model_loaded,
            "model_error": model_load_error,
        }
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
