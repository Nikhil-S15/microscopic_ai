"""
BOX SCREENING — stages after detection
======================================

Stage 2 · noise screening (M1)
  Every detected box is cropped and classified by the noise CNN
  (noisy_background_model.pth). Boxes whose noise probability exceeds the cutoff
  are dropped.

Stage 3 · mask verification (MicroSAM)
  Each surviving box prompts Segment Anything with MicroSAM's light-microscopy
  weights (vit_b_lm). Mask logits above the mask threshold form the object mask;
  the box is kept only when the mask covers at least `min_coverage` of the box.
  Masks are saved per image.

Measured on held-out images (current detector, 1,172 boxes):
  M1 at 0.5   keeps 94% of boxes on labelled organisms, removes 23% of boxes on
              background-only images.
  SAM         a box prompt almost always yields a mask (median coverage ~65% for
              organisms and for debris alike), so coverage separates "something"
              from "nothing", not organisms from debris. The default minimum
              coverage only drops boxes with essentially no object in them.
"""
import csv
import json
import logging
import threading
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch
import torch.nn.functional as F

logger = logging.getLogger(__name__)

NOISE_CUTOFF = 0.50          # M1: noise probability above this → box dropped
SAM_MASK_THRESHOLD = 0.0     # SAM mask logit threshold (SAM's own default)
SAM_MIN_COVERAGE = 0.10      # keep a box only if its mask covers ≥ 10% of it
SAM_CHUNK = 32               # boxes decoded per batch (bounds memory)


# ══════════════════════════════════════════════════════════════
# Stage 2 · M1 noise screening
# ══════════════════════════════════════════════════════════════

def screen_noise(detector, image: np.ndarray, dets: List[Dict], cutoff: float) -> Tuple[List[Dict], List[Dict]]:
    """Score each box crop with the noise CNN. Returns (kept, removed); adds 'noise_score'."""
    if not dets or detector.noise_model is None:
        return dets, []
    crops = [image[d["bbox"][1]:d["bbox"][3], d["bbox"][0]:d["bbox"][2]] for d in dets]
    with torch.inference_mode():
        scores = []
        for i in range(0, len(crops), 256):
            t = detector._to_batch(crops[i:i + 256], 48).contiguous()
            scores.append(F.softmax(detector.noise_model(t), 1)[:, 0].cpu().numpy())
    scores = np.concatenate(scores)
    kept, removed = [], []
    for d, s in zip(dets, scores):
        d["noise_score"] = round(float(s), 4)
        (removed if s > cutoff else kept).append(d)
    return kept, removed


# ══════════════════════════════════════════════════════════════
# Stage 3 · SAM mask verification
# ══════════════════════════════════════════════════════════════

class SamVerifier:
    """MicroSAM (vit_b_lm) behind segment-anything's predictor; loaded on first use."""

    name = "MicroSAM · vit_b_lm (light microscopy)"

    def __init__(self, checkpoint: Path, device: torch.device):
        self.checkpoint = Path(checkpoint)
        self.device = device
        self._predictor = None
        self._lock = threading.Lock()
        self.error: Optional[str] = None

    @property
    def available(self) -> bool:
        if not self.checkpoint.exists():
            return False
        try:
            import segment_anything  # noqa: F401
        except ImportError:
            return False
        return self.error is None

    def load(self):
        with self._lock:
            if self._predictor is not None:
                return self._predictor
            from segment_anything import SamPredictor, sam_model_registry
            t0 = time.time()
            state = torch.load(self.checkpoint, map_location="cpu", weights_only=False)
            if "model_state" in state:            # torch_em training checkpoint
                state = state["model_state"]
            state = {k[4:] if k.startswith("sam.") else k: v for k, v in state.items()}
            sam = sam_model_registry["vit_b"]()
            sam.load_state_dict(state)
            self._predictor = SamPredictor(sam.to(self.device).eval())
            logger.info(f"SAM verifier loaded in {time.time() - t0:.1f}s")
            return self._predictor

    @torch.inference_mode()
    def verify(self, image: np.ndarray, dets: List[Dict], mask_threshold: float,
               min_coverage: float) -> Tuple[List[Dict], List[Dict]]:
        """
        Prompt SAM with every box. Adds 'mask' (box-sized bool array, private),
        'mask_coverage' and 'mask_area'. Returns (kept, removed).
        """
        if not dets:
            return dets, []
        pred = self.load()
        h, w = image.shape[:2]
        pred.set_image(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        ih, iw = pred.input_size                       # image size inside SAM's 1024 frame
        s = ih / h
        model = pred.model
        for i in range(0, len(dets), SAM_CHUNK):
            chunk = dets[i:i + SAM_CHUNK]
            boxes = torch.tensor([d["bbox"] for d in chunk], dtype=torch.float32, device=self.device)
            # predict_torch without its full-image upsampling (≈20 MB per box)
            sparse, dense = model.prompt_encoder(
                points=None, boxes=pred.transform.apply_boxes_torch(boxes, (h, w)), masks=None)
            low, _ = model.mask_decoder(
                image_embeddings=pred.features, image_pe=model.prompt_encoder.get_dense_pe(),
                sparse_prompt_embeddings=sparse, dense_prompt_embeddings=dense, multimask_output=False)
            up = F.interpolate(low, (model.image_encoder.img_size,) * 2, mode="bilinear",
                               align_corners=False)[:, 0, :ih, :iw].cpu().numpy()
            for d, lg in zip(chunk, up):
                x1, y1, x2, y2 = d["bbox"]
                # box region in SAM's frame → back to box pixel size
                region = lg[int(y1 * s):max(int(y1 * s) + 1, int(np.ceil(y2 * s))),
                            int(x1 * s):max(int(x1 * s) + 1, int(np.ceil(x2 * s)))]
                logits = cv2.resize(region, (x2 - x1, y2 - y1), interpolation=cv2.INTER_LINEAR)
                mask = logits > mask_threshold
                d["mask"] = mask
                d["mask_area"] = int(mask.sum())
                d["mask_coverage"] = round(float(mask.mean()), 4)
        kept = [d for d in dets if d["mask_coverage"] >= min_coverage]
        removed = [d for d in dets if d["mask_coverage"] < min_coverage]
        return kept, removed


def mask_contour(d: Dict, max_points: int = 120) -> List[List[int]]:
    """Outline of the largest mask region, in image pixels (for overlays)."""
    m = d.get("mask")
    if m is None or not m.any():
        return []
    cs, _ = cv2.findContours(m.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    c = max(cs, key=cv2.contourArea)
    eps = 0.5
    while len(c) > max_points:
        c = cv2.approxPolyDP(c, eps, True)
        eps *= 1.6
    x1, y1 = d["bbox"][:2]
    return [[int(p[0][0]) + x1, int(p[0][1]) + y1] for p in c]


# ══════════════════════════════════════════════════════════════
# Saving masks + the screening log
# ══════════════════════════════════════════════════════════════

def save_masks(out_dir: Path, request_id: str, filename: str, image_shape, kept: List[Dict],
               sam_removed: List[Dict], summary: Dict) -> Optional[Path]:
    """
    outputs/masks/<request_id>/
      label_map.png            16-bit, pixel value = detection number (1…n) of kept boxes
      kept_<n>.png             box-sized binary mask of kept detection n
      rejected_<k>.png         box-sized binary mask of a box removed by SAM
      screening.json           settings, counts and per-box scores
    """
    if not any("mask" in d for d in kept + sam_removed):
        return None
    folder = out_dir / request_id
    folder.mkdir(parents=True, exist_ok=True)
    h, w = image_shape[:2]
    label = np.zeros((h, w), np.uint16)
    for d in kept:
        if "mask" not in d:
            continue
        x1, y1, x2, y2 = d["bbox"]
        n = d["id"] + 1
        label[y1:y2, x1:x2][d["mask"]] = n
        cv2.imwrite(str(folder / f"kept_{n:03d}.png"), d["mask"].astype(np.uint8) * 255)
    for k, d in enumerate(sam_removed, 1):
        cv2.imwrite(str(folder / f"rejected_{k:03d}.png"), d["mask"].astype(np.uint8) * 255)
    cv2.imwrite(str(folder / "label_map.png"), label)
    box = lambda d, n: {"n": n, "bbox": d["bbox"], "confidence": d["confidence"],
                        "noise_score": d.get("noise_score"), "mask_coverage": d.get("mask_coverage"),
                        "mask_area": d.get("mask_area")}
    (folder / "screening.json").write_text(json.dumps({
        "filename": filename, **summary,
        "kept": [box(d, d["id"] + 1) for d in kept],
        "removed_by_mask": [box(d, k) for k, d in enumerate(sam_removed, 1)],
    }, indent=1))
    return folder


_log_lock = threading.Lock()


def append_log(log_path: Path, row: Dict):
    """One line per analysed image: how many boxes each stage removed."""
    with _log_lock:
        new = not log_path.exists()
        log_path.parent.mkdir(parents=True, exist_ok=True)
        with log_path.open("a", newline="") as f:
            wr = csv.DictWriter(f, fieldnames=list(row))
            if new:
                wr.writeheader()
            wr.writerow(row)
