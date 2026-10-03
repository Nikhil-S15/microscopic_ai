"""
MULTI-SPECIES BACTERIA DETECTOR — Exact pipeline from all detection scripts
===========================================================================

Candida    → CandidaDetector (ResNet50, frozen layer1+layer2, 64px)
             → NO NMS → merge_to_one_box_per_spore (merge_dist=100)
Staph      → BacteriaDetectorTransfer (ResNet50, 64px)
             → cluster distance NMS
Pluribacter→ BacteriaClassifier (ResNet18 3-class, 48px)
             → IoU NMS → filter_thread_clusters
Generic    → BacteriaDetectorCNN (simple CNN, 48px)
             → IoU NMS → filter_thread_clusters

Stage 1 (ALL species):
  NoiseDetectorCNN — patch always resized to 48x48 before noise check
  Threshold: 0.80 → skip if noise
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import cv2
import numpy as np
from typing import List, Dict, Tuple
import time
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


# ══════════════════════════════════════════════════════════════
# AUTO-PREPROCESSING
# Training images are MONOCHROME: dark background, bright bacteria
# Raw web images can be bright background (regular microscope)
# We auto-detect and convert to match training format
# ══════════════════════════════════════════════════════════════

def preprocess_image_for_detection(image_bgr: np.ndarray) -> np.ndarray:
    """
    Convert raw microscope image to match training data format:
    - Training images: dark background (~mean < 30), white bacteria dots
    - Raw web images: may be bright background (mean > 100)
    
    If image is bright → convert to monochrome dark-background format.
    If image is already dark → use as-is (matches training format).
    
    Returns BGR image in training-compatible format.
    """
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    mean = gray.mean()
    
    if mean > 80:
        # Bright background image — convert to dark background format
        # Step 1: invert so bacteria become bright
        inv = cv2.bitwise_not(gray)
        # Step 2: CLAHE to enhance bacteria contrast
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(inv)
        # Step 3: convert back to BGR (models trained on 3-channel images)
        processed = cv2.cvtColor(enhanced, cv2.COLOR_GRAY2BGR)
        logger.info(f"Image preprocessed: bright→dark (mean {mean:.1f} → {enhanced.mean():.1f})")
        return processed
    else:
        # Already dark background — matches training format, use as-is
        logger.info(f"Image already dark (mean={mean:.1f}) — no preprocessing needed")
        return image_bgr


# ══════════════════════════════════════════════════════════════
# MODEL ARCHITECTURES — exact from each training script
# ══════════════════════════════════════════════════════════════

class NoiseDetectorCNN(nn.Module):
    """Stage 1 noise filter. Class 0 = noise → skip patch."""
    def __init__(self):
        super().__init__()
        self.conv1   = nn.Conv2d(3, 32,  3, padding=1)
        self.bn1     = nn.BatchNorm2d(32)
        self.pool1   = nn.MaxPool2d(2, 2)
        self.conv2   = nn.Conv2d(32, 64, 3, padding=1)
        self.bn2     = nn.BatchNorm2d(64)
        self.pool2   = nn.MaxPool2d(2, 2)
        self.conv3   = nn.Conv2d(64, 128, 3, padding=1)
        self.bn3     = nn.BatchNorm2d(128)
        self.pool3   = nn.MaxPool2d(2, 2)
        self.fc1     = nn.Linear(128 * 6 * 6, 128)
        self.dropout = nn.Dropout(0.5)
        self.fc2     = nn.Linear(128, 2)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x))); x = self.pool1(x)
        x = F.relu(self.bn2(self.conv2(x))); x = self.pool2(x)
        x = F.relu(self.bn3(self.conv3(x))); x = self.pool3(x)
        x = x.view(-1, 128 * 6 * 6)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        return self.fc2(x)


class BacteriaDetectorCNN(nn.Module):
    """Generic CNN — fallback. 48px input."""
    def __init__(self):
        super().__init__()
        self.conv1   = nn.Conv2d(3, 32,  3, padding=1)
        self.bn1     = nn.BatchNorm2d(32)
        self.pool1   = nn.MaxPool2d(2, 2)
        self.conv2   = nn.Conv2d(32, 64, 3, padding=1)
        self.bn2     = nn.BatchNorm2d(64)
        self.pool2   = nn.MaxPool2d(2, 2)
        self.conv3   = nn.Conv2d(64, 128, 3, padding=1)
        self.bn3     = nn.BatchNorm2d(128)
        self.pool3   = nn.MaxPool2d(2, 2)
        self.fc1     = nn.Linear(128 * 6 * 6, 256)
        self.dropout = nn.Dropout(0.5)
        self.fc2     = nn.Linear(256, 2)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x))); x = self.pool1(x)
        x = F.relu(self.bn2(self.conv2(x))); x = self.pool2(x)
        x = F.relu(self.bn3(self.conv3(x))); x = self.pool3(x)
        x = x.view(-1, 128 * 6 * 6)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        return self.fc2(x)


class BacteriaDetectorTransfer(nn.Module):
    """ResNet50 — Staph model. 64px input upsampled to 224px."""
    def __init__(self):
        super().__init__()
        self.backbone = models.resnet50(weights=None)
        self.backbone.fc = nn.Linear(self.backbone.fc.in_features, 2)

    def forward(self, x):
        x = F.interpolate(x, size=(224, 224), mode='bilinear', align_corners=False)
        return self.backbone(x)


class CandidaDetector(nn.Module):
    """
    ResNet50 with frozen layer1+layer2 — EXACT from train_candida_model-3.py.
    64px input upsampled to 224px.
    """
    def __init__(self):
        super().__init__()
        self.backbone = models.resnet50(weights=None)
        # Freeze layer1 and layer2 (matches training script exactly)
        for p in self.backbone.layer1.parameters():
            p.requires_grad = False
        for p in self.backbone.layer2.parameters():
            p.requires_grad = False
        self.backbone.fc = nn.Linear(self.backbone.fc.in_features, 2)

    def forward(self, x):
        x = F.interpolate(x, size=(224, 224), mode='bilinear', align_corners=False)
        return self.backbone(x)


class PluralibacterDetector(nn.Module):
    """
    ResNet18 with 3-class head — EXACT from plurali-detector.py.
    Class 0 = Pluralibacter, class 1 = Spores, class 2 = Background.
    48px input.
    """
    def __init__(self):
        super().__init__()
        self.network = models.resnet18(weights=None)
        self.network.fc = nn.Linear(self.network.fc.in_features, 3)

    def forward(self, x):
        x = F.interpolate(x, size=(224, 224), mode='bilinear', align_corners=False)
        return self.network(x)


# ══════════════════════════════════════════════════════════════
# POST-PROCESSING — exact from detection scripts
# ══════════════════════════════════════════════════════════════

def nms_iou(detections, patch_size, iou_threshold=0.15):
    if not detections:
        return []
    detections = sorted(detections, key=lambda d: d['confidence'], reverse=True)
    keep = []
    for det in detections:
        suppress = False
        for kept in keep:
            x1 = max(det['x'], kept['x'])
            y1 = max(det['y'], kept['y'])
            x2 = min(det['x'] + patch_size, kept['x'] + patch_size)
            y2 = min(det['y'] + patch_size, kept['y'] + patch_size)
            if x2 > x1 and y2 > y1:
                inter = (x2 - x1) * (y2 - y1)
                union = 2 * (patch_size ** 2) - inter
                if inter / union > iou_threshold:
                    suppress = True
                    break
        if not suppress:
            keep.append(det)
    return keep


def nms_cluster(detections, patch_size, dist_factor=0.6):
    """Distance NMS for Staph grape-like clusters."""
    if not detections:
        return []
    detections = sorted(detections, key=lambda d: d['confidence'], reverse=True)
    keep   = []
    half   = patch_size // 2
    thresh = patch_size * dist_factor
    for det in detections:
        cx = det['x'] + half
        cy = det['y'] + half
        suppress = False
        for kept in keep:
            if ((cx - kept['x'] - half)**2 + (cy - kept['y'] - half)**2)**0.5 < thresh:
                suppress = True
                break
        if not suppress:
            keep.append(det)
    return keep


def filter_thread_clusters(detections, box_size=48, max_chain_length=3):
    """
    BFS groups touching detections, removes chains longer than max_chain_length.
    Exact from visual_detection_system-2.py.
    """
    if not detections:
        return []
    centres = np.array([[d['x'] + box_size // 2,
                         d['y'] + box_size // 2] for d in detections])
    n       = len(detections)
    visited = [False] * n
    final   = []
    for i in range(n):
        if not visited[i]:
            cluster_idx = []
            queue       = [i]
            visited[i]  = True
            while queue:
                curr = queue.pop(0)
                cluster_idx.append(curr)
                for j in range(n):
                    if not visited[j]:
                        if np.linalg.norm(centres[curr] - centres[j]) <= box_size * 1.2:
                            visited[j] = True
                            queue.append(j)
            if len(cluster_idx) <= max_chain_length:
                for idx in cluster_idx:
                    final.append(detections[idx])
    logger.info(f"thread_filter: {n} → {len(final)} (removed {n-len(final)} chains)")
    return final


def merge_to_one_box_per_spore(detections, patch_size, merge_distance=100):
    """
    BFS groups detections into clusters, ONE tight bbox per spore.
    Exact from visual_detection_candida-2.py.
    """
    if not detections:
        return []
    centres = np.array([[d['x'] + patch_size // 2,
                         d['y'] + patch_size // 2] for d in detections],
                        dtype=np.float32)
    n       = len(detections)
    visited = [False] * n
    result  = []
    for i in range(n):
        if visited[i]:
            continue
        cluster = []
        queue   = [i]
        visited[i] = True
        while queue:
            curr = queue.pop(0)
            cluster.append(curr)
            for j in range(n):
                if not visited[j]:
                    if np.linalg.norm(centres[curr] - centres[j]) <= merge_distance:
                        visited[j] = True
                        queue.append(j)
        dets      = [detections[k] for k in cluster]
        best_conf = max(d['confidence'] for d in dets)
        x1 = min(d['x']               for d in dets)
        y1 = min(d['y']               for d in dets)
        x2 = max(d['x'] + patch_size  for d in dets)
        y2 = max(d['y'] + patch_size  for d in dets)
        result.append({'x1': x1, 'y1': y1, 'x2': x2, 'y2': y2,
                       'confidence': best_conf})
    logger.info(f"merge_spores: {n} patches → {len(result)} spores")
    return result


# ══════════════════════════════════════════════════════════════
# SPECIES REGISTRY
# ══════════════════════════════════════════════════════════════

SPECIES_REGISTRY = {
    "Candida": {
        "model_file":     "candida_detection_model.pth",
        "model_class":    "candida",        # CandidaDetector
        "patch_size":     64,
        "stride":         24,
        "threshold":      0.75,  # lowered from 0.85 for raw image tolerance
        "nms":            None,             # NO NMS — go straight to merge
        "merge_spores":   True,
        "merge_distance": 100,
        "chain_filter":   False,
        "color":          "#06b6d4",
    },
    "Staphylococcus Aureus": {
        "model_file":     "staph_detection_model.pth",
        "model_class":    "resnet50",       # BacteriaDetectorTransfer
        "patch_size":     64,
        "stride":         16,
        "threshold":      0.75,  # lowered from 0.87 for raw image tolerance
        "nms":            "cluster",
        "merge_spores":   False,
        "chain_filter":   False,
        "color":          "#ef4444",
    },
    "Pluribacter": {
        "model_file":     "pluralibacter_detector.pth",
        "model_class":    "resnet18_3class", # PluralibacterDetector
        "patch_size":     48,
        "stride":         12,
        "threshold":      0.70,  # lowered from 0.80 for raw image tolerance
        "nms":            "iou",
        "iou_thresh":     0.10,             # NMS_THRESHOLD from pluralibacter_test.py
        "merge_spores":   False,
        "chain_filter":   True,
        "max_chain":      3,
        "color":          "#22c55e",
    },
    "Aspergillus": {
        "model_file":     "staph_detection_model.pth",
        "model_class":    "resnet50",
        "patch_size":     96,
        "stride":         24,
        "threshold":      0.82,
        "nms":            "iou",
        "iou_thresh":     0.25,
        "merge_spores":   False,
        "chain_filter":   False,
        "color":          "#f97316",
    },
    "Bacteria": {
        "model_file":     "final(tool)_best_model_fixed.pth",
        "model_class":    "cnn",
        "patch_size":     48,
        "stride":         24,
        "threshold":      0.75,
        "nms":            "iou",
        "iou_thresh":     0.15,
        "merge_spores":   False,
        "chain_filter":   True,
        "max_chain":      3,
        "color":          "#3b82f6",
    },
}

CLASSIFIER_TO_SPECIES = {
    "Staphylococcus Aureus": "Staphylococcus Aureus",
    "Candida":               "Candida",
    "Pluribacter":           "Pluribacter",
    "Aspergillus":           "Aspergillus",
    "Acinetobacter":         "Bacteria",
    "Burkholderia":          "Bacteria",
}

NOISE_PATCH_SIZE = 48
NOISE_THRESHOLD  = 0.80


# ══════════════════════════════════════════════════════════════
# MAIN DETECTOR
# ══════════════════════════════════════════════════════════════

class BacteriaDetector:

    def __init__(self, model_path: str, patch_size: int = 48, device: str = None):
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        self.patch_size  = patch_size
        self.stride      = 16
        self.model_dir   = str(Path(model_path).parent)
        self.class_names = {0: "Bacteria", 1: "Background"}

        self._noise_model  = self._load_noise()
        self._species_models: Dict[str, nn.Module] = {}
        for species, cfg in SPECIES_REGISTRY.items():
            m = self._load_species(species, cfg)
            if m:
                self._species_models[species] = m

        logger.info(f"BacteriaDetector ready on {self.device}")
        logger.info(f"Loaded: {list(self._species_models.keys())}")

    # ── Loaders ──────────────────────────────────────────────────────────────

    def _load_noise(self):
        p = Path(self.model_dir) / "noisy_background_model.pth"
        if not p.exists():
            logger.warning("noisy_background_model.pth not found — noise filter OFF")
            return None
        try:
            m = NoiseDetectorCNN()
            m.load_state_dict(torch.load(str(p), map_location=self.device))
            m.to(self.device); m.eval()
            logger.info("✅ Noise model loaded")
            return m
        except Exception as e:
            logger.warning(f"Noise model load failed: {e}")
            return None

    def _load_species(self, species: str, cfg: dict):
        p = Path(self.model_dir) / cfg["model_file"]
        if not p.exists():
            logger.warning(f"Model not found for {species}: {cfg['model_file']}")
            return None
        try:
            mc = cfg["model_class"]
            if mc == "candida":
                m = CandidaDetector()
            elif mc == "resnet50":
                m = BacteriaDetectorTransfer()
            elif mc == "resnet18_3class":
                m = PluralibacterDetector()
            else:
                m = BacteriaDetectorCNN()
            m.load_state_dict(torch.load(str(p), map_location=self.device))
            m.to(self.device); m.eval()
            logger.info(f"✅ {species} ({mc}) loaded")
            return m
        except Exception as e:
            logger.error(f"Failed {species}: {e}")
            return None

    # ── Inference helpers ─────────────────────────────────────────────────────

    def _to_tensor(self, patch: np.ndarray) -> torch.Tensor:
        return torch.from_numpy(patch.astype(np.float32) / 255.0)\
                     .permute(2, 0, 1).unsqueeze(0).to(self.device)

    def _is_noise(self, patch: np.ndarray) -> bool:
        if self._noise_model is None:
            return False
        # Always resize to 48x48 for noise model — exact from detection scripts
        np48 = cv2.resize(patch, (NOISE_PATCH_SIZE, NOISE_PATCH_SIZE))
        t    = self._to_tensor(np48)
        with torch.no_grad():
            return F.softmax(self._noise_model(t), dim=1)[0, 0].item() > NOISE_THRESHOLD

    def _bacteria_prob(self, model, patch, species, use_tta=False) -> float:
        """Get bacteria probability. Pluribacter uses class 0 of 3-class output."""
        t = self._to_tensor(patch)
        with torch.no_grad():
            if use_tta:
                preds = []
                for dims in [None, [3], [2]]:
                    inp = torch.flip(t, dims) if dims else t
                    preds.append(F.softmax(model(inp), dim=1))
                probs = torch.mean(torch.stack(preds), dim=0)
            else:
                probs = F.softmax(model(t), dim=1)
        # All models: class 0 = target bacteria
        return probs[0, 0].item()

    # ── Core scan ────────────────────────────────────────────────────────────

    def _scan(self, image, species, confidence_threshold=None, use_tta=False):
        cfg       = SPECIES_REGISTRY[species]
        model     = self._species_models.get(species)
        if model is None:
            logger.warning(f"No loaded model for {species}")
            return []

        ps        = cfg["patch_size"]
        stride    = cfg["stride"]
        threshold = confidence_threshold or cfg["threshold"]
        h, w      = image.shape[:2]
        candidates = []
        noise_ct   = 0

        for y in range(0, max(1, h - ps + 1), stride):
            for x in range(0, max(1, w - ps + 1), stride):
                patch = image[y:y+ps, x:x+ps]
                if patch.shape[0] != ps or patch.shape[1] != ps:
                    continue  # skip edge patches (exact from candida script)

                # Stage 1: noise filter — resize to 48px before checking
                if self._is_noise(patch):
                    noise_ct += 1
                    continue

                # Stage 2: bacteria detection
                prob = self._bacteria_prob(model, patch, species, use_tta)
                if prob >= threshold:
                    candidates.append({'x': x, 'y': y, 'confidence': prob})

        logger.info(f"{species}: {len(candidates)} raw, {noise_ct} noise-filtered")
        return candidates

    # ── Post-process ─────────────────────────────────────────────────────────

    def _postprocess(self, candidates, species):
        cfg = SPECIES_REGISTRY[species]
        ps  = cfg["patch_size"]

        # Candida: skip NMS, go straight to one-box-per-spore merge
        if cfg.get("merge_spores"):
            return merge_to_one_box_per_spore(
                candidates, ps, cfg.get("merge_distance", 100))

        # All others: NMS first
        if cfg.get("nms") == "cluster":
            dets = nms_cluster(candidates, ps, 0.6)
        elif cfg.get("nms") == "iou":
            dets = nms_iou(candidates, ps, cfg.get("iou_thresh", 0.15))
        else:
            dets = candidates

        logger.info(f"{species}: {len(candidates)} → {len(dets)} after NMS")

        # Rod bacteria: remove thread/fibre chains
        if cfg.get("chain_filter") and dets:
            dets = filter_thread_clusters(
                dets, box_size=ps,
                max_chain_length=cfg.get("max_chain", 3))

        return dets

    # ── Public API ────────────────────────────────────────────────────────────

    def detect(
        self,
        image:                np.ndarray,
        confidence_threshold: float = None,
        nms_threshold:        float = 0.3,
        use_tta:              bool  = True,
        top_k:                int   = 3,
        species_name:         str   = None,
    ) -> Tuple[List[Dict], List[Dict], float]:

        start = time.time()

        # Resolve species key
        key = CLASSIFIER_TO_SPECIES.get(species_name, "Bacteria") \
              if species_name else "Bacteria"
        if species_name and species_name in SPECIES_REGISTRY \
                and species_name in self._species_models:
            key = species_name

        cfg   = SPECIES_REGISTRY.get(key, SPECIES_REGISTRY["Bacteria"])
        ps    = cfg["patch_size"]
        color = cfg["color"]
        is_merged = cfg.get("merge_spores", False)

        logger.info(f"Detecting: species='{species_name}' → model='{key}' "
                    f"ps={ps} stride={cfg['stride']} thresh={cfg['threshold']}")

        # Auto-preprocess to match training data format (dark background)
        image = preprocess_image_for_detection(image)

        raw  = self._scan(image, key, confidence_threshold, use_tta)
        dets = self._postprocess(raw, key)

        api_detections = []
        for det in dets:
            # Candida merge produces x1/y1/x2/y2; others produce x/y
            if is_merged:
                x1, y1 = det["x1"], det["y1"]
                x2, y2 = det["x2"], det["y2"]
            else:
                x1, y1 = det["x"], det["y"]
                x2, y2 = x1 + ps, y1 + ps

            bact = det["confidence"]
            api_detections.append({
                "bbox":      [int(x1), int(y1), int(x2), int(y2)],
                "class_id":  0,
                "class_name": species_name or "Bacteria",
                "confidence": round(bact, 4),
                "top_probabilities": [
                    {"class_id": 0,
                     "class_name": species_name or "Bacteria",
                     "probability": round(bact, 4)},
                    {"class_id": 1, "class_name": "Background",
                     "probability": round(1.0 - bact, 4)},
                ][:top_k],
                "color": color,
            })

        overall_probs = [
            {"class_id": 0, "class_name": species_name or "Bacteria",
             "probability": 0.5},
            {"class_id": 1, "class_name": "Background", "probability": 0.5},
        ]

        proc_time = time.time() - start
        logger.info(f"detect() → {len(api_detections)} final in {proc_time:.2f}s")
        return api_detections, overall_probs, proc_time