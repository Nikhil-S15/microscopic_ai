"""
FASTAPI MAIN - WITH CLASSIFICATION + DETECTION (2-STEP PIPELINE)
Step 1: /api/v1/classify  → 7-class bacteria identification (Keras .h5 model)
Step 2: /api/v1/detect    → bounding box detection (species-specific PyTorch models)
"""
import sys
import io as _io
from pathlib import Path
from typing import List, Dict, Any, Optional
import uuid
from datetime import datetime
import numpy as np
import cv2
import time

import torch
from fastapi import FastAPI, File, UploadFile, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from contextlib import asynccontextmanager
from PIL import Image as PILImage

sys.path.append(str(Path(__file__).parent))

from app.inference import BacteriaDetector
from app.utils import setup_logger, validate_image

logger = setup_logger(__name__)

# ── Globals ───────────────────────────────────────────────────────────────────
bacteria_detector = None
classifier_model  = None

# ── Class metadata ────────────────────────────────────────────────────────────
CLASS_NAMES = [
    "Acenetobacter_Preprocessed",
    "Aspergillus_Preprocessed",
    "Aureus_Preprocessed",
    "Bulkoder_Preprocessed",
    "Candida_Preprocessed",
    "Klebsiella_Preprocessed",
    "Pluribacter_Preprocessed",
]

CLASS_INFO = {
    "Acenetobacter_Preprocessed": {"color": "#ef4444", "risk": "High",   "gram": "Gram-negative", "shape": "Coccobacillus", "display": "Acinetobacter"},
    "Aspergillus_Preprocessed":   {"color": "#f97316", "risk": "Medium", "gram": "Fungal mold",   "shape": "Filamentous",   "display": "Aspergillus"},
    "Aureus_Preprocessed":        {"color": "#eab308", "risk": "High",   "gram": "Gram-positive", "shape": "Coccus",        "display": "Staphylococcus Aureus"},
    "Bulkoder_Preprocessed":      {"color": "#22c55e", "risk": "High",   "gram": "Gram-negative", "shape": "Rod",           "display": "Burkholderia"},
    "Candida_Preprocessed":       {"color": "#06b6d4", "risk": "Medium", "gram": "Fungal yeast",  "shape": "Oval",          "display": "Candida"},
    "Klebsiella_Preprocessed":    {"color": "#8b5cf6", "risk": "High",   "gram": "Gram-negative", "shape": "Rod",           "display": "Klebsiella"},
    "Pluribacter_Preprocessed":   {"color": "#ec4899", "risk": "Low",    "gram": "Gram-negative", "shape": "Rod",           "display": "Pluribacter"},
}

# ── Lifespan ──────────────────────────────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    global bacteria_detector, classifier_model

    logger.info("=" * 70)
    logger.info("🦠 Bacteria Detection API — 2-Step Pipeline")
    logger.info("=" * 70)

    # Load PyTorch detection models
    try:
        model_path = Path("app/model/final(tool)_best_model_fixed.pth")
        if not model_path.exists():
            raise FileNotFoundError(f"Detection model not found: {model_path}")
        bacteria_detector = BacteriaDetector(
            model_path=str(model_path),
            patch_size=48,
            device="cuda" if torch.cuda.is_available() else "cpu",
        )
        logger.info(f"✅ Detection models loaded on {bacteria_detector.device}")
    except Exception as e:
        logger.error(f"❌ Detection model failed: {e}")
        bacteria_detector = None

    # Load Keras classifier
    try:
        import keras
        cls_path = Path("app/model/bacteria_cnn_6class_model (1).h5")
        if not cls_path.exists():
            raise FileNotFoundError(f"Classifier model not found: {cls_path}")
        classifier_model = keras.saving.load_model(str(cls_path), compile=False)
        logger.info(f"✅ Classifier model loaded from {cls_path}")
    except Exception as e:
        logger.warning(f"⚠️  Classifier model not loaded: {e}")
        classifier_model = None

    yield
    logger.info("Shutting down...")


# ── App ───────────────────────────────────────────────────────────────────────
app = FastAPI(
    title="Bacteria Detection API — 2-Step Pipeline",
    description="Step 1: classify bacteria type. Step 2: detect with bounding boxes.",
    version="2.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Pydantic models ───────────────────────────────────────────────────────────
class ClassPrediction(BaseModel):
    class_id: int
    class_name: str
    probability: float
    color: str
    risk_level: str
    gram_type: str
    shape: str

class ClassifyResponse(BaseModel):
    success: bool
    predicted_class: str
    predicted_class_id: int
    confidence: float
    all_predictions: List[ClassPrediction]
    color: str
    risk_level: str
    gram_type: str
    shape: str
    processing_time: float

class TopProbability(BaseModel):
    class_id: int
    class_name: str
    probability: float

class DetectionResult(BaseModel):
    bbox: List[int]
    class_id: int
    class_name: str
    confidence: float
    top_probabilities: List[TopProbability]

class InferenceResponse(BaseModel):
    success: bool = True
    request_id: str
    timestamp: str
    image_dimensions: Dict[str, int]
    total_detections: int
    detections: List[DetectionResult]
    processing_time: float
    overall_top_probabilities: List[TopProbability]
    settings: Dict[str, Any]

class HealthResponse(BaseModel):
    status: str
    detection_model_loaded: bool
    classifier_model_loaded: bool
    device: str
    timestamp: str


# ── STEP 1: CLASSIFY ─────────────────────────────────────────────────────────
@app.post("/api/v1/classify", response_model=ClassifyResponse, tags=["Step 1 - Classification"])
async def classify_bacteria(file: UploadFile = File(...)):
    """Identify which bacteria species is present."""
    if classifier_model is None:
        raise HTTPException(status_code=503, detail="Classifier model not loaded")

    start = time.time()
    try:
        contents = await file.read()

        nparr   = np.frombuffer(contents, np.uint8)
        img_bgr = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if img_bgr is None:
            pil_img = PILImage.open(_io.BytesIO(contents)).convert("RGB")
            img_bgr = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)

        # Smart preprocessing — auto-detect dark vs bright image
        gray     = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
        mean_val = gray.mean()
        if mean_val > 127:
            # Bright background → invert so bacteria = white, background = black
            gray = cv2.bitwise_not(gray)
            logger.info(f"Bright image (mean={mean_val:.1f}) → inverted")
        else:
            logger.info(f"Dark image (mean={mean_val:.1f}) → no inversion")

        resized = cv2.resize(gray, (128, 128))
        arr     = resized.astype(np.float32) / 255.0
        arr     = arr.reshape(1, 128, 128, 1)
        logger.info(f"Classifier input: mean={arr.mean():.4f}")

        preds     = classifier_model(arr, training=False).numpy()[0]
        pred_idx  = int(np.argmax(preds))
        pred_key  = CLASS_NAMES[pred_idx]
        info      = CLASS_INFO[pred_key]
        pred_name = info.get("display", pred_key)

        all_preds = []
        for i, (key, prob) in enumerate(zip(CLASS_NAMES, preds.tolist())):
            ci = CLASS_INFO[key]
            all_preds.append(ClassPrediction(
                class_id=i,
                class_name=ci.get("display", key),
                probability=float(prob),
                color=ci["color"],
                risk_level=ci["risk"],
                gram_type=ci["gram"],
                shape=ci["shape"],
            ))
        all_preds.sort(key=lambda x: x.probability, reverse=True)

        logger.info(f"✅ Classified as '{pred_name}' ({preds[pred_idx]*100:.1f}%)")

        return ClassifyResponse(
            success=True,
            predicted_class=pred_name,
            predicted_class_id=pred_idx,
            confidence=float(preds[pred_idx]),
            all_predictions=all_preds,
            color=info["color"],
            risk_level=info["risk"],
            gram_type=info["gram"],
            shape=info["shape"],
            processing_time=time.time() - start,
        )

    except Exception as e:
        logger.error(f"Classification error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ── STEP 2: DETECT ────────────────────────────────────────────────────────────
@app.post("/api/v1/detect", response_model=InferenceResponse, tags=["Step 2 - Detection"])
async def detect_bacteria(
    file:                 UploadFile = File(...),
    confidence_threshold: float      = Query(0.75),
    nms_threshold:        float      = Query(0.3),
    use_tta:              bool       = Query(True),
    top_k:                int        = Query(3),
    species_name:         str        = Query(None),
):
    """Draw bounding boxes around individual bacteria using the species-specific model."""
    if bacteria_detector is None:
        raise HTTPException(status_code=503, detail="Detection model not loaded")

    try:
        request_id = str(uuid.uuid4())
        start_time = datetime.now()

        contents = await file.read()
        image    = validate_image(contents)
        if image is None:
            raise HTTPException(status_code=400, detail="Invalid image")

        logger.info(f"Detecting species: {species_name or 'generic'}")

        detections, overall_probs, proc_time = bacteria_detector.detect(
            image,
            confidence_threshold=confidence_threshold,
            nms_threshold=nms_threshold,
            use_tta=use_tta,
            top_k=top_k,
            species_name=species_name,
        )

        detection_results = []
        for det in detections:
            top_probs = [
                TopProbability(
                    class_id=p["class_id"],
                    class_name=p["class_name"],
                    probability=p["probability"],
                ) for p in det["top_probabilities"]
            ]
            detection_results.append(DetectionResult(
                bbox=det["bbox"],
                class_id=det["class_id"],
                class_name=det["class_name"],
                confidence=det["confidence"],
                top_probabilities=top_probs,
            ))

        overall_prob_results = [
            TopProbability(
                class_id=p["class_id"],
                class_name=p["class_name"],
                probability=p["probability"],
            ) for p in overall_probs
        ]

        total_time = (datetime.now() - start_time).total_seconds()

        return InferenceResponse(
            request_id=request_id,
            timestamp=start_time.isoformat(),
            image_dimensions={"height": image.shape[0], "width": image.shape[1]},
            total_detections=len(detections),
            detections=detection_results,
            processing_time=total_time,
            overall_top_probabilities=overall_prob_results,
            settings={
                "confidence_threshold": confidence_threshold,
                "nms_threshold": nms_threshold,
                "use_tta": use_tta,
                "top_k": top_k,
                "species": species_name or "generic",
            },
        )

    except Exception as e:
        logger.error(f"Detection error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ── HEALTH ────────────────────────────────────────────────────────────────────
@app.get("/health", response_model=HealthResponse, tags=["Health"])
async def health_check():
    return HealthResponse(
        status="healthy" if bacteria_detector else "partial",
        detection_model_loaded=bacteria_detector is not None,
        classifier_model_loaded=classifier_model is not None,
        device=str(bacteria_detector.device) if bacteria_detector else "none",
        timestamp=datetime.now().isoformat(),
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True, log_level="info")