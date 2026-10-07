"""
FASTAPI MAIN — BACTERIA DETECTION
=================================
Pipeline per image
  1. detection     unified_bacteria_model.pth (one ResNet50), sliding window,
                   at the per-type defaults or one global confidence threshold
  2. screening     every box crop through the noise CNN (M1); noise boxes dropped
  3. verification  every surviving box prompts MicroSAM; kept only if its mask
                   covers enough of the box. Masks saved under outputs/masks/.

The API reports *whether* bacteria are present and where — never which organism.
The model's internal class labels are used only inside the detector and are
stripped from every response.

Endpoints
  GET  /health
  GET  /api/v2/model                  model metadata
  POST /api/v2/analyze                one image   → detections + result
  POST /api/v2/analyze/batch          many images → per-image results + summary
  GET  /api/v2/progress/{request_id}  live progress (0-1) of an analyze call
  POST /api/v2/threshold-analysis     precision / recall / F1 per confidence threshold
  POST /api/v2/masks/export           saved masks of analysed images as a ZIP
"""
import io as _io
import re
import threading
import time
import uuid
import zipfile
from collections import OrderedDict
from contextlib import asynccontextmanager
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import cv2
import numpy as np
from fastapi import Depends, FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response
from PIL import Image as PILImage
from pydantic import BaseModel, Field

from app.evaluation import threshold_sweep
from app.screening import (
    NOISE_CUTOFF, SAM_MASK_THRESHOLD, SAM_MIN_COVERAGE, SamVerifier, append_log, mask_contour,
    save_masks, screen_noise,
)
from app.unified import CANDIDATE_FLOOR, UnifiedDetector
from app.utils import setup_logger

logger = setup_logger(__name__)

MODEL_DIR       = Path(__file__).parent / "model"
UNIFIED_PATH    = MODEL_DIR / "unified_bacteria_model.pth"
NOISE_PATH      = MODEL_DIR / "noisy_background_model.pth"
SAM_PATH        = MODEL_DIR / "microsam_vit_b_lm.pt"
OUTPUT_DIR      = Path(__file__).resolve().parent.parent / "outputs"
MASK_DIR        = OUTPUT_DIR / "masks"
SCREENING_LOG   = OUTPUT_DIR / "screening_log.csv"
MAX_BATCH       = 50
MAX_IMAGE_BYTES = 40 * 1024 * 1024
HIGH_CONFIDENCE = 0.95

detector: Optional[UnifiedDetector] = None
sam: Optional[SamVerifier] = None
inference_lock = threading.Lock()      # torch already uses every core
progress: Dict[str, Dict[str, Any]] = {}

# Candidate windows of recent analyses (request_id → candidates), so threshold
# analysis can re-score them without re-running the model. In memory only.
CANDIDATE_CACHE_SIZE = 500
candidate_cache: "OrderedDict[str, Dict]" = OrderedDict()
cache_lock = threading.Lock()


@asynccontextmanager
async def lifespan(app: FastAPI):
    global detector, sam
    try:
        detector = UnifiedDetector(str(UNIFIED_PATH), str(NOISE_PATH))
        logger.info(f"Detector ready on {detector.device}")
    except Exception as e:
        logger.error(f"Detector failed to load: {e}")
        detector = None
    if detector is not None:
        sam = SamVerifier(SAM_PATH, detector.device)
        if sam.available:
            try:
                sam.load()
            except Exception as e:
                sam.error = str(e)
                logger.error(f"SAM verifier failed to load: {e}")
        else:
            logger.warning(f"SAM verifier unavailable (needs segment-anything and {SAM_PATH.name})")
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


@dataclass
class Options:
    sensitivity: float = 0.0
    noise_gate: bool = True                    # window pre-filter (before the ResNet)
    threshold: Optional[float] = None          # one global threshold; None = per-type defaults
    box_screening: bool = True                 # stage 2 · M1 on every box
    noise_cutoff: float = NOISE_CUTOFF
    sam_verify: bool = True                    # stage 3 · SAM mask check
    mask_threshold: float = SAM_MASK_THRESHOLD
    min_coverage: float = SAM_MIN_COVERAGE


def public_box(d: Dict) -> Dict:
    """Detection without organism identity, plus screening scores and mask outline."""
    out = {"bbox": d["bbox"], "confidence": d["confidence"]}
    if "noise_score" in d:
        out["noise_score"] = d["noise_score"]
    if "mask" in d:
        out["mask"] = {"coverage": d["mask_coverage"], "area": d["mask_area"], "contour": mask_contour(d)}
    return out


def public_result(raw: Dict, kept: List[Dict], removed: List[Dict], screening: Dict) -> Dict:
    """Strip organism identity from the detector output; keep detection metrics."""
    dets = [{"id": d["id"], **public_box(d)} for d in kept]
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
        "screening": screening,
        "removed": [{"stage": d["removed_by"], **public_box(d)} for d in removed],
        "image": raw["image"],
        "stats": {
            "windows_total": stats["windows_total"],
            "windows_skipped": stats["windows_flat_skipped"] + stats["windows_noise_skipped"],
            "windows_analysed": stats["windows_classified"],
            "raw_hits": int(sum(stats["raw_candidates"].values())),
        },
        "processing_time": raw["processing_time"],
    }


def sam_ready() -> bool:
    return sam is not None and sam.available


def run_analysis(data: bytes, filename: str, request_id: str, opts: Options) -> Dict:
    progress[request_id] = {"status": "queued", "progress": 0.0, "updated": time.time()}
    use_sam = opts.sam_verify and sam_ready()
    detect_share = 0.85 if use_sam else 1.0    # SAM takes the last ~15% of the time

    def on_progress(p: float):
        progress[request_id].update(status="analyzing", progress=round(p * detect_share, 3), updated=time.time())

    started = datetime.now()
    try:
        img = decode_image(data)
        thresholds = {c: opts.threshold for c in detector.codes} if opts.threshold is not None else None
        with inference_lock:
            progress[request_id].update(status="analyzing", updated=time.time())
            # stage 1 · detection
            raw = detector.detect(img, thresholds=thresholds,
                                  sensitivity=0.0 if thresholds else opts.sensitivity,
                                  use_noise_gate=opts.noise_gate, progress_cb=on_progress)
            kept = [dict(d) for d in raw["detections"]]
            boxes_detected = len(kept)
            removed: List[Dict] = []
            # stage 2 · noise screening (M1)
            if opts.box_screening:
                kept, rm = screen_noise(detector, img, kept, opts.noise_cutoff)
                removed += [{**d, "removed_by": "noise"} for d in rm]
            # stage 3 · SAM mask verification
            sam_removed: List[Dict] = []
            if use_sam and kept:
                kept, sam_removed = sam.verify(img, kept, opts.mask_threshold, opts.min_coverage)
                removed += [{**d, "removed_by": "mask"} for d in sam_removed]
        for i, d in enumerate(kept):           # number the surviving detections 0…n-1
            d["id"] = i

        removed_noise = sum(d["removed_by"] == "noise" for d in removed)
        screening = {
            "threshold": opts.threshold,
            "boxes_detected": boxes_detected,
            "removed_noise": removed_noise,
            "removed_mask": len(sam_removed),
            "kept": len(kept),
            "box_screening": opts.box_screening,
            "noise_cutoff": opts.noise_cutoff,
            "sam_verify": use_sam,
            "mask_threshold": opts.mask_threshold,
            "min_coverage": opts.min_coverage,
        }
        folder = save_masks(MASK_DIR, request_id, filename, img.shape, kept, sam_removed, screening) if use_sam else None
        screening["masks_saved"] = folder is not None
        append_log(SCREENING_LOG, {
            "timestamp": started.isoformat(timespec="seconds"), "request_id": request_id, "filename": filename,
            "threshold": opts.threshold if opts.threshold is not None else "defaults",
            "boxes_detected": boxes_detected, "removed_noise": removed_noise,
            "removed_mask": len(sam_removed), "kept": len(kept),
            "noise_cutoff": opts.noise_cutoff if opts.box_screening else "off",
            "mask_threshold": opts.mask_threshold if use_sam else "off",
            "min_coverage": opts.min_coverage if use_sam else "off",
        })
        logger.info(f"{filename}: {boxes_detected} boxes → noise removed {removed_noise}, "
                    f"mask removed {len(sam_removed)}, kept {len(kept)}")
        raw["processing_time"] = round((datetime.now() - started).total_seconds(), 3)

        progress[request_id].update(status="done", progress=1.0, updated=time.time())
        with cache_lock:
            candidate_cache[request_id] = raw["candidates"]
            candidate_cache.move_to_end(request_id)
            while len(candidate_cache) > CANDIDATE_CACHE_SIZE:
                candidate_cache.popitem(last=False)
        return {
            "success": True,
            "request_id": request_id,
            "filename": filename,
            "timestamp": started.isoformat(),
            **public_result(raw, kept, removed, screening),
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
        "limits": {"max_batch": MAX_BATCH, "max_image_mb": MAX_IMAGE_BYTES // (1024 * 1024),
                   "min_threshold": CANDIDATE_FLOOR},
        "screening": {
            "box_screening": detector.noise_model is not None,
            "noise_cutoff": NOISE_CUTOFF,
            "sam": {
                "available": sam_ready(),
                "model": sam.name if sam else None,
                "mask_threshold": SAM_MASK_THRESHOLD,
                "min_coverage": SAM_MIN_COVERAGE,
                "error": sam.error if sam else None,
            },
        },
    }


def options_form(
    sensitivity: float = Form(0.0, ge=-0.2, le=0.3),
    noise_gate: bool = Form(True),
    threshold: Optional[float] = Form(None, ge=CANDIDATE_FLOOR, lt=1,
                                      description="One confidence threshold for every detection"),
    box_screening: bool = Form(True),
    noise_cutoff: float = Form(NOISE_CUTOFF, ge=0, le=1),
    sam_verify: bool = Form(True),
    mask_threshold: float = Form(SAM_MASK_THRESHOLD, ge=-20, le=20),
    min_coverage: float = Form(SAM_MIN_COVERAGE, ge=0, le=1),
) -> Options:
    return Options(sensitivity, noise_gate, threshold, box_screening, noise_cutoff,
                   sam_verify, mask_threshold, min_coverage)


REQUEST_ID = re.compile(r"^[A-Za-z0-9-]{1,64}$")   # also a folder name under outputs/masks


@app.post("/api/v2/analyze", tags=["Analysis"])
def analyze(
    file: UploadFile = File(...),
    request_id: Optional[str] = Form(None),
    opts: Options = Depends(options_form),
):
    require_detector()
    if request_id is not None and not REQUEST_ID.match(request_id):
        raise HTTPException(422, "request_id may only contain letters, digits and hyphens")
    result = run_analysis(file.file.read(), file.filename or "image",
                          request_id or str(uuid.uuid4()), opts)
    if not result["success"]:
        raise HTTPException(400 if result["error_type"] == "input" else 500, result["error"])
    return result


@app.post("/api/v2/analyze/batch", tags=["Analysis"])
def analyze_batch(
    files: List[UploadFile] = File(...),
    opts: Options = Depends(options_form),
):
    require_detector()
    if len(files) > MAX_BATCH:
        raise HTTPException(400, f"At most {MAX_BATCH} images per batch")
    t0 = time.time()
    results = [run_analysis(f.file.read(), f.filename or f"image_{i}", str(uuid.uuid4()), opts)
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


class SweepImage(BaseModel):
    request_id: str
    ground_truth: List[List[float]] = Field(
        default_factory=list, description="Labelled objects as [x1, y1, x2, y2] in pixels")


class SweepRequest(BaseModel):
    thresholds: List[float] = Field(..., min_length=1, max_length=101)
    images: List[SweepImage] = Field(..., min_length=1, max_length=CANDIDATE_CACHE_SIZE)


@app.post("/api/v2/threshold-analysis", tags=["Analysis"])
def threshold_analysis(req: SweepRequest):
    """
    Precision, recall, F1 and detection count at each confidence threshold, for
    images analysed earlier in this server session (by request_id) against
    their ground-truth boxes. One threshold is applied to every detection.
    """
    require_detector()
    thresholds = sorted({round(t, 4) for t in req.thresholds})
    if thresholds[0] < CANDIDATE_FLOOR or thresholds[-1] >= 1:
        raise HTTPException(422, f"Thresholds must be between {CANDIDATE_FLOOR} and 1")
    for im in req.images:
        if any(len(b) != 4 or b[2] <= b[0] or b[3] <= b[1] for b in im.ground_truth):
            raise HTTPException(422, "Each ground-truth box must be [x1, y1, x2, y2] with x2 > x1, y2 > y1")
    with cache_lock:
        found = {im.request_id: candidate_cache.get(im.request_id) for im in req.images}
    missing = [rid for rid, c in found.items() if c is None]
    images = [{"candidates": found[im.request_id], "ground_truth": im.ground_truth}
              for im in req.images if found[im.request_id] is not None]
    if not images:
        raise HTTPException(409, "None of these images are in the server's analysis cache — re-analyse them first")
    return {
        "images": len(images),
        "objects": sum(len(im["ground_truth"]) for im in images),
        "missing": missing,
        "results": threshold_sweep(images, thresholds),
    }


class MaskExportItem(BaseModel):
    request_id: str = Field(..., pattern=REQUEST_ID.pattern)
    name: str = Field(..., max_length=200, description="Folder name inside the ZIP (usually the image file name)")


class MaskExportRequest(BaseModel):
    items: List[MaskExportItem] = Field(..., min_length=1, max_length=CANDIDATE_CACHE_SIZE)


@app.post("/api/v2/masks/export", tags=["Analysis"])
def export_masks(req: MaskExportRequest):
    """ZIP of the saved masks (label map, per-box masks, screening.json) per image."""
    buf = _io.BytesIO()
    found = 0
    used = set()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        for it in req.items:
            folder = MASK_DIR / it.request_id
            if not folder.is_dir():
                continue
            found += 1
            base = re.sub(r"[^\w.-]+", "_", Path(it.name).stem)[:80] or "image"
            name, k = base, 2
            while name in used:
                name, k = f"{base}_{k}", k + 1
            used.add(name)
            for f in sorted(folder.iterdir()):
                zf.write(f, f"{name}/{f.name}")
        if SCREENING_LOG.exists():
            zf.write(SCREENING_LOG, "screening_log.csv")
    if not found:
        raise HTTPException(404, "No saved masks for these images — enable mask verification and re-analyse")
    return Response(buf.getvalue(), media_type="application/zip",
                    headers={"Content-Disposition": 'attachment; filename="masks.zip"'})


@app.get("/api/v2/progress/{request_id}", tags=["Analysis"])
def get_progress(request_id: str):
    p = progress.get(request_id)
    if p is None:
        return {"status": "unknown", "progress": 0.0}
    return {"status": p["status"], "progress": p["progress"]}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True)
