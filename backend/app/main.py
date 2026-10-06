"""
FASTAPI MAIN — BACTERIA DETECTION
=================================
Detection: unified_bacteria_model.pth (one ResNet50) gated by noisy_background_model.pth.

The API reports *whether* bacteria are present and where — never which organism.
The model's internal class labels are used only inside the detector and are
stripped from every response.

Endpoints
  GET  /health
  GET  /api/v2/model                  model metadata
  POST /api/v2/analyze                one image   → detections + result
  POST /api/v2/analyze/batch          many images → per-image results + summary
  GET  /api/v2/progress/{request_id}  live progress (0-1) of an analyze call
"""
import io as _io
import threading
import time
import uuid
from contextlib import asynccontextmanager
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import cv2
import numpy as np
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image as PILImage

from app.unified import UnifiedDetector
from app.utils import setup_logger

logger = setup_logger(__name__)

MODEL_DIR       = Path(__file__).parent / "model"
UNIFIED_PATH    = MODEL_DIR / "unified_bacteria_model.pth"
NOISE_PATH      = MODEL_DIR / "noisy_background_model.pth"
MAX_BATCH       = 50
MAX_IMAGE_BYTES = 40 * 1024 * 1024
HIGH_CONFIDENCE = 0.95

detector: Optional[UnifiedDetector] = None
inference_lock = threading.Lock()      # torch already uses every core
progress: Dict[str, Dict[str, Any]] = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    global detector
    try:
        detector = UnifiedDetector(str(UNIFIED_PATH), str(NOISE_PATH))
        logger.info(f"Detector ready on {detector.device}")
    except Exception as e:
        logger.error(f"Detector failed to load: {e}")
        detector = None
    yield


app = FastAPI(
    title="Microscopy AI — Bacteria Detection API",
    description="Detects bacteria in microscope images and reports their locations and confidence.",
    version="4.0.0",
    lifespan=lifespan,
)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])


# ── helpers ───────────────────────────────────────────────────────────────────

def decode_image(data: bytes) -> np.ndarray:
    if not data:
        raise ValueError("Empty file")
    if len(data) > MAX_IMAGE_BYTES:
        raise ValueError("Image larger than 40 MB")
    img = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        try:
            pil = PILImage.open(_io.BytesIO(data)).convert("RGB")
        except Exception:
            raise ValueError("Unsupported or corrupt image")
        img = cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)
    return img


def public_result(raw: Dict) -> Dict:
    """Strip organism identity from the detector output; keep detection metrics."""
    dets = [{"id": d["id"], "bbox": d["bbox"], "confidence": d["confidence"]} for d in raw["detections"]]
    confs = [d["confidence"] for d in dets]
    w, h = raw["image"]["width"], raw["image"]["height"]
    covered = np.zeros((h, w), bool)
    for d in dets:
        x1, y1, x2, y2 = d["bbox"]
        covered[max(0, y1):y2, max(0, x1):x2] = True
    stats = raw["stats"]
    return {
        "bacteria_detected": bool(dets),
        "total_detections": len(dets),
        "detections": dets,
        "confidence": {
            "mean": round(float(np.mean(confs)), 4) if confs else None,
            "max": round(float(np.max(confs)), 4) if confs else None,
            "min": round(float(np.min(confs)), 4) if confs else None,
            "high_confidence_share": round(float(np.mean(np.array(confs) >= HIGH_CONFIDENCE)), 4) if confs else None,
        },
        "coverage": round(float(covered.mean()), 6),
        "image": raw["image"],
        "stats": {
            "windows_total": stats["windows_total"],
            "windows_skipped": stats["windows_flat_skipped"] + stats["windows_noise_skipped"],
            "windows_analysed": stats["windows_classified"],
            "raw_hits": int(sum(stats["raw_candidates"].values())),
        },
        "processing_time": raw["processing_time"],
    }


def run_analysis(data: bytes, filename: str, request_id: str, sensitivity: float, noise_gate: bool) -> Dict:
    progress[request_id] = {"status": "queued", "progress": 0.0, "updated": time.time()}

    def on_progress(p: float):
        progress[request_id].update(status="analyzing", progress=round(p, 3), updated=time.time())

    started = datetime.now()
    try:
        img = decode_image(data)
        with inference_lock:
            progress[request_id].update(status="analyzing", updated=time.time())
            raw = detector.detect(img, sensitivity=sensitivity, use_noise_gate=noise_gate,
                                  progress_cb=on_progress)
        progress[request_id].update(status="done", progress=1.0, updated=time.time())
        return {
            "success": True,
            "request_id": request_id,
            "filename": filename,
            "timestamp": started.isoformat(),
            **public_result(raw),
        }
    except ValueError as e:
        progress[request_id].update(status="error", updated=time.time())
        return {"success": False, "request_id": request_id, "filename": filename,
                "error": str(e), "error_type": "input"}
    except Exception as e:
        logger.exception(f"Analysis failed for {filename}")
        progress[request_id].update(status="error", updated=time.time())
        return {"success": False, "request_id": request_id, "filename": filename,
                "error": f"Analysis failed: {e}", "error_type": "server"}
    finally:
        cutoff = time.time() - 600
        for k in [k for k, v in progress.items() if v["updated"] < cutoff]:
            progress.pop(k, None)


def require_detector():
    if detector is None:
        raise HTTPException(503, "Detection model not loaded")


# ── endpoints ─────────────────────────────────────────────────────────────────

@app.get("/health", tags=["System"])
def health():
    return {
        "status": "healthy" if detector else "unavailable",
        "detector_loaded": detector is not None,
        "device": str(detector.device) if detector else None,
        "timestamp": datetime.now().isoformat(),
    }


@app.get("/api/v2/model", tags=["System"])
def model_info():
    require_detector()
    return {
        "name": "Bacteria Detector",
        "architecture": "ResNet50 · sliding-window patch classifier",
        "device": str(detector.device),
        "noise_gate": detector.noise_model is not None,
        "high_confidence": HIGH_CONFIDENCE,
        "limits": {"max_batch": MAX_BATCH, "max_image_mb": MAX_IMAGE_BYTES // (1024 * 1024)},
    }


@app.post("/api/v2/analyze", tags=["Analysis"])
def analyze(
    file: UploadFile = File(...),
    request_id: Optional[str] = Form(None),
    sensitivity: float = Form(0.0, ge=-0.2, le=0.3),
    noise_gate: bool = Form(True),
):
    require_detector()
    result = run_analysis(file.file.read(), file.filename or "image",
                          request_id or str(uuid.uuid4()), sensitivity, noise_gate)
    if not result["success"]:
        raise HTTPException(400 if result["error_type"] == "input" else 500, result["error"])
    return result


@app.post("/api/v2/analyze/batch", tags=["Analysis"])
def analyze_batch(
    files: List[UploadFile] = File(...),
    sensitivity: float = Form(0.0, ge=-0.2, le=0.3),
    noise_gate: bool = Form(True),
):
    require_detector()
    if len(files) > MAX_BATCH:
        raise HTTPException(400, f"At most {MAX_BATCH} images per batch")
    t0 = time.time()
    results = [run_analysis(f.file.read(), f.filename or f"image_{i}", str(uuid.uuid4()),
                            sensitivity, noise_gate)
               for i, f in enumerate(files)]
    ok = [r for r in results if r["success"]]
    return {
        "success": True,
        "results": results,
        "summary": {
            "images": len(results),
            "succeeded": len(ok),
            "failed": len(results) - len(ok),
            "positive_images": sum(r["bacteria_detected"] for r in ok),
            "total_detections": sum(r["total_detections"] for r in ok),
            "processing_time": round(time.time() - t0, 2),
        },
    }


@app.get("/api/v2/progress/{request_id}", tags=["Analysis"])
def get_progress(request_id: str):
    p = progress.get(request_id)
    if p is None:
        return {"status": "unknown", "progress": 0.0}
    return {"status": p["status"], "progress": p["progress"]}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True)
