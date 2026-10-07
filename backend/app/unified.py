"""
UNIFIED MULTI-SPECIES DETECTOR
==============================

One ResNet50 (unified_bacteria_model.pth) scores every species at once, gated by
the noise CNN (noisy_background_model.pth). Mirrors unified_train_all_species.py:

  crop (species patch size) → resize 64×64 → /255 → [model upsamples to 224] → softmax

Class order comes from the checkpoint ('class_names'), NOT from the hardcoded
indices in unified_detect_all_species.py, which disagree with training.

Speed (CPU): windows are skipped before the ResNet when they are
  1. flat background  — grayscale std below FLAT_STD (keeps 100% of labelled
     organisms on the training sets while skipping ~90-95% of windows), or
  2. rejected by the noise CNN (except species with skip_noise_gate).
Surviving crops are batched; one forward pass yields every species' score, so
species that share a patch size share the scan. Overlapping candidate windows
are merged into one box per organism (see MERGE_LINK / MIN_VOTES).
"""

import time
import logging
from pathlib import Path
from typing import Dict, List, Optional, Callable

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

logger = logging.getLogger(__name__)

MODEL_INPUT_SIZE = 64      # training resized every patch to this
NOISE_INPUT_SIZE = 48
NOISE_THRESHOLD  = 0.80
FLAT_STD         = 3.5
BATCH_SIZE       = 64

# Post-processing: candidate windows of one species whose centres are within
# MERGE_LINK × patch_size are merged into one box per organism; clusters with
# fewer than MIN_VOTES windows are dropped as isolated noise.
# Tuned on held-out labelled images (SA/CA/PG): recall 0.91–1.00, and ~2.3×
# faster than stride = patch/3 with no loss in accuracy.
MERGE_LINK = 1.0
MIN_VOTES  = 2

# Cross-species overlap: when boxes of different species overlap this much,
# keep only the more confident one (one label per organism).
CROSS_SPECIES_IOU = 0.40

# Lowest window confidence kept as a candidate, so results can be recomputed at
# any threshold from here up without re-running the model (threshold analysis).
CANDIDATE_FLOOR = 0.50

# Display metadata + detection settings per class code. Thresholds follow
# unified_detect_all_species.py. Colours are a CVD-checked categorical set.
SPECIES_CFG: Dict[str, Dict] = {
    "PG": {
        "name": "Pluralibacter gergoviae", "short": "Pluralibacter",
        "kind": "Bacterium", "gram": "Gram-negative", "morphology": "Rod",
        "patch_size": 48, "stride": 24, "threshold": 0.90,
        "color": "#2a78d6",
    },
    "SA": {
        "name": "Staphylococcus aureus", "short": "S. aureus",
        "kind": "Bacterium", "gram": "Gram-positive", "morphology": "Coccus (clusters)",
        "patch_size": 48, "stride": 24, "threshold": 0.75,
        # tiny cocci fill <1% of a 48px crop — the noise gate wrongly rejects them
        "skip_noise_gate": True,
        "color": "#eb6834",
    },
    "CA": {
        "name": "Candida albicans", "short": "Candida",
        "kind": "Fungus (yeast)", "gram": "Fungal", "morphology": "Oval yeast cell",
        "patch_size": 64, "stride": 32, "threshold": 0.85,
        "color": "#1baf7a",
    },
    "AS": {
        "name": "Aspergillus spores", "short": "Aspergillus",
        "kind": "Fungus (mould)", "gram": "Fungal", "morphology": "Round spore",
        "patch_size": 96, "stride": 48, "threshold": 0.82,
        "color": "#4a3aa7",
        # The checkpoint lists AS in 'species', but training skipped it: there is
        # no labels_aspergillus folder. The model's AS probability is 0.000 on
        # every Aspergillus image, so it cannot be detected. Set True once a
        # model is retrained with Aspergillus labels.
        "has_training_data": False,
    },
    # AL / BC: present in the model head, but the checkpoint does not list them
    # in 'species' (no training images).
    "AL": {
        "name": "Acinetobacter lwoffii", "short": "Acinetobacter",
        "kind": "Bacterium", "gram": "Gram-negative", "morphology": "Coccobacillus",
        "patch_size": 48, "stride": 24, "threshold": 0.88,
        "color": "#e87ba4",
    },
    "BC": {
        "name": "Burkholderia cepacia", "short": "Burkholderia",
        "kind": "Bacterium", "gram": "Gram-negative", "morphology": "Rod",
        "patch_size": 48, "stride": 24, "threshold": 0.88,
        "color": "#008300",
    },
}


# ══════════════════════════════════════════════════════════════
# MODELS — exact from unified_train_all_species.py
# ══════════════════════════════════════════════════════════════

class UnifiedBacteriaDetector(nn.Module):
    def __init__(self, num_classes: int):
        super().__init__()
        backbone = models.resnet50(weights=None)
        backbone.fc = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(backbone.fc.in_features, num_classes),
        )
        self.backbone = backbone

    def forward(self, x):
        x = F.interpolate(x, size=(224, 224), mode="bilinear", align_corners=False)
        return self.backbone(x)


class NoiseDetectorCNN(nn.Module):
    """Class 0 = noise/background, class 1 = organism. 48px input."""
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 32, 3, padding=1);   self.bn1 = nn.BatchNorm2d(32)
        self.pool1 = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1);  self.bn2 = nn.BatchNorm2d(64)
        self.pool2 = nn.MaxPool2d(2, 2)
        self.conv3 = nn.Conv2d(64, 128, 3, padding=1); self.bn3 = nn.BatchNorm2d(128)
        self.pool3 = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(128 * 6 * 6, 128); self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(128, 2)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x))); x = self.pool1(x)
        x = F.relu(self.bn2(self.conv2(x))); x = self.pool2(x)
        x = F.relu(self.bn3(self.conv3(x))); x = self.pool3(x)
        x = x.reshape(-1, 128 * 6 * 6)
        x = self.dropout(F.relu(self.fc1(x)))
        return self.fc2(x)


# ══════════════════════════════════════════════════════════════
# POST-PROCESSING
# ══════════════════════════════════════════════════════════════

def _iou_matrix(boxes: np.ndarray) -> np.ndarray:
    x1 = np.maximum(boxes[:, None, 0], boxes[None, :, 0])
    y1 = np.maximum(boxes[:, None, 1], boxes[None, :, 1])
    x2 = np.minimum(boxes[:, None, 2], boxes[None, :, 2])
    y2 = np.minimum(boxes[:, None, 3], boxes[None, :, 3])
    inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    area = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    return inter / (area[:, None] + area[None, :] - inter + 1e-9)


def merge_clusters(dets: List[Dict], link_px: float, min_votes: int) -> List[Dict]:
    """
    Union-find over one species' candidate windows: windows whose centres are
    within link_px belong to the same organism. Each cluster becomes one box
    (union of its windows) with the best confidence; clusters with fewer than
    min_votes windows are dropped.
    """
    n = len(dets)
    if n == 0:
        return []
    b = np.array([d["bbox"] for d in dets], np.float32)
    c = (b[:, :2] + b[:, 2:]) / 2
    parent = np.arange(n)

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    close = np.linalg.norm(c[:, None] - c[None], axis=2) <= link_px
    for i, j in zip(*np.nonzero(np.triu(close, 1))):
        parent[find(i)] = find(j)

    groups: Dict[int, List[int]] = {}
    for i in range(n):
        groups.setdefault(find(i), []).append(i)

    out = []
    for g in groups.values():
        if len(g) < min_votes:
            continue
        bb = b[g]
        out.append({
            "bbox": [*bb[:, :2].min(0).tolist(), *bb[:, 2:].max(0).tolist()],
            "confidence": max(dets[i]["confidence"] for i in g),
            "code": dets[g[0]]["code"],
            "votes": len(g),
        })
    return out


def resolve_cross_species(dets: List[Dict]) -> List[Dict]:
    """One label per organism: drop the less confident of overlapping boxes of different species."""
    if len(dets) < 2:
        return dets
    dets = sorted(dets, key=lambda d: (d["confidence"], d["votes"]), reverse=True)
    boxes = np.array([d["bbox"] for d in dets], np.float32)
    codes = np.array([d["code"] for d in dets])
    overlap = _iou_matrix(boxes) * (codes[:, None] != codes[None, :])
    keep, removed = [], np.zeros(len(dets), bool)
    for i in range(len(dets)):
        if removed[i]:
            continue
        keep.append(dets[i])
        removed |= overlap[i] > CROSS_SPECIES_IOU
    return keep


def post_process(candidates: Dict[str, List[Dict]], thr: Dict[str, float]) -> List[Dict]:
    """Candidate windows → final boxes at the given per-species thresholds."""
    final: List[Dict] = []
    for c, wins in candidates.items():
        kept = [d for d in wins if d["confidence"] >= thr[c]]
        final.extend(merge_clusters(kept, SPECIES_CFG[c]["patch_size"] * MERGE_LINK, MIN_VOTES))
    final = resolve_cross_species(final)
    final.sort(key=lambda d: (d["bbox"][1], d["bbox"][0]))
    return final


# ══════════════════════════════════════════════════════════════
# DETECTOR
# ══════════════════════════════════════════════════════════════

class UnifiedDetector:

    def __init__(self, model_path: str, noise_model_path: Optional[str] = None,
                 device: Optional[str] = None):
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        if self.device.type == "cpu":
            torch.set_num_threads(max(1, torch.get_num_threads()))

        ckpt = torch.load(model_path, map_location=self.device, weights_only=False)
        self.class_names: List[str] = list(ckpt["class_names"])
        self.trained_codes: List[str] = list(ckpt.get("species", self.class_names))
        self.val_acc = ckpt.get("val_acc")
        self.epoch = ckpt.get("epoch")
        self.model = UnifiedBacteriaDetector(len(self.class_names))
        self.model.load_state_dict(ckpt["model_state"])
        self.model.to(self.device).eval()
        if self.device.type == "cpu":
            self.model = self.model.to(memory_format=torch.channels_last)

        # Only species that had training data are detectable.
        self.codes = [c for c in self.class_names
                      if c in SPECIES_CFG and c in self.trained_codes
                      and SPECIES_CFG[c].get("has_training_data", True)]
        self.class_idx = {c: self.class_names.index(c) for c in self.codes}

        self.noise_model = None
        if noise_model_path and Path(noise_model_path).exists():
            m = NoiseDetectorCNN()
            m.load_state_dict(torch.load(noise_model_path, map_location=self.device))
            self.noise_model = m.to(self.device).eval()

        logger.info(f"Unified model: classes={self.class_names} trained={self.trained_codes} "
                    f"val_acc={self.val_acc} device={self.device} "
                    f"noise_gate={'on' if self.noise_model else 'off'}")

    # ── public metadata ─────────────────────────────────────────

    def species_info(self) -> List[Dict]:
        return [{
            "code": c,
            "name": SPECIES_CFG[c]["name"],
            "short": SPECIES_CFG[c]["short"],
            "kind": SPECIES_CFG[c]["kind"],
            "gram": SPECIES_CFG[c]["gram"],
            "morphology": SPECIES_CFG[c]["morphology"],
            "color": SPECIES_CFG[c]["color"],
            "patch_size": SPECIES_CFG[c]["patch_size"],
            "default_threshold": SPECIES_CFG[c]["threshold"],
        } for c in self.codes]

    # ── inference helpers ───────────────────────────────────────

    def _to_batch(self, crops: List[np.ndarray], size: int) -> torch.Tensor:
        arr = np.stack([cv2.resize(c, (size, size), interpolation=cv2.INTER_LINEAR)
                        for c in crops]).astype(np.float32) / 255.0
        t = torch.from_numpy(arr).permute(0, 3, 1, 2).to(self.device)
        return t.contiguous(memory_format=torch.channels_last) if self.device.type == "cpu" else t

    @torch.inference_mode()
    def _noise_mask(self, crops: List[np.ndarray]) -> np.ndarray:
        """True where the noise CNN says background/noise."""
        out = []
        for i in range(0, len(crops), 256):
            t = self._to_batch(crops[i:i + 256], NOISE_INPUT_SIZE).contiguous()
            out.append(F.softmax(self.noise_model(t), 1)[:, 0].cpu().numpy())
        return np.concatenate(out) > NOISE_THRESHOLD if out else np.zeros(0, bool)

    @torch.inference_mode()
    def _classify(self, crops: List[np.ndarray], progress: Callable[[int], None]) -> np.ndarray:
        out = []
        for i in range(0, len(crops), BATCH_SIZE):
            t = self._to_batch(crops[i:i + BATCH_SIZE], MODEL_INPUT_SIZE)
            out.append(F.softmax(self.model(t), 1).float().cpu().numpy())
            progress(min(len(crops), i + BATCH_SIZE))
        return np.concatenate(out) if out else np.zeros((0, len(self.class_names)), np.float32)

    # ── main entry ──────────────────────────────────────────────

    def detect(self, image: np.ndarray,
               species: Optional[List[str]] = None,
               thresholds: Optional[Dict[str, float]] = None,
               sensitivity: float = 0.0,
               use_noise_gate: bool = True,
               progress_cb: Optional[Callable[[float], None]] = None) -> Dict:
        """
        image       : BGR uint8, used as-is (training images were not inverted).
        species     : codes to detect (default: all trained).
        thresholds  : per-code overrides.
        sensitivity : shifts every threshold down (+) or up (−), e.g. 0.05.
        """
        t0 = time.time()
        active = [c for c in (species or self.codes) if c in self.codes]
        if not active:
            raise ValueError(f"No detectable species selected. Options: {self.codes}")
        thr = {c: float(np.clip((thresholds or {}).get(c, SPECIES_CFG[c]["threshold"]) - sensitivity,
                                0.30, 0.995)) for c in active}

        h, w = image.shape[:2]
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        noise_on = use_noise_gate and self.noise_model is not None

        # group species by patch geometry → one scan per group
        groups: Dict[tuple, List[str]] = {}
        for c in active:
            cfg = SPECIES_CFG[c]
            groups.setdefault((cfg["patch_size"], cfg["stride"]), []).append(c)

        # pass 1: collect candidate windows per group (cheap gates)
        plans = []
        total_windows = flat_skipped = noise_skipped = 0
        for (ps, stride), codes in groups.items():
            if h < ps or w < ps:
                continue
            ys = np.arange(0, h - ps + 1, stride)
            xs = np.arange(0, w - ps + 1, stride)
            total_windows += len(ys) * len(xs)
            # window std via integral images
            g = gray.astype(np.float64)
            s1 = cv2.integral(g)
            s2 = cv2.integral(g * g)
            Y, X = np.meshgrid(ys, xs, indexing="ij")
            Y, X = Y.ravel(), X.ravel()

            def box_sum(s):
                return s[Y + ps, X + ps] - s[Y, X + ps] - s[Y + ps, X] + s[Y, X]

            n = ps * ps
            mean = box_sum(s1) / n
            std = np.sqrt(np.maximum(box_sum(s2) / n - mean ** 2, 0))
            keep = std > FLAT_STD
            flat_skipped += int((~keep).sum())
            Y, X = Y[keep], X[keep]

            crops = [image[y:y + ps, x:x + ps] for y, x in zip(Y, X)]
            gated = np.zeros(len(crops), bool)
            if noise_on and crops:
                # noise gate always looks at the 48px crop at the anchor
                nc = crops if ps == NOISE_INPUT_SIZE else \
                    [image[y:y + NOISE_INPUT_SIZE, x:x + NOISE_INPUT_SIZE] for y, x in zip(Y, X)]
                gated = self._noise_mask(nc)
            bypass = [c for c in codes if SPECIES_CFG[c].get("skip_noise_gate")]
            if not bypass:
                noise_skipped += int(gated.sum())
                sel = ~gated
                Y, X, gated = Y[sel], X[sel], gated[sel]
                crops = [cr for cr, s in zip(crops, sel) if s]
            plans.append((ps, codes, Y, X, crops, gated))

        n_classify = sum(len(p[4]) for p in plans)
        done_before = 0

        def report(done_in_plan):
            if progress_cb and n_classify:
                progress_cb((done_before + done_in_plan) / n_classify)

        # pass 2: batched classification. Windows are kept down to CANDIDATE_FLOOR
        # (below the detection threshold) so post_process can re-run at any threshold.
        candidates: Dict[str, List[Dict]] = {c: [] for c in active}
        for ps, codes, Y, X, crops, gated in plans:
            probs = self._classify(crops, report)
            done_before += len(crops)
            for c in codes:
                p = probs[:, self.class_idx[c]] if len(crops) else np.zeros(0)
                hit = p >= min(thr[c], CANDIDATE_FLOOR)
                if not SPECIES_CFG[c].get("skip_noise_gate"):
                    hit &= ~gated
                # a crop gets the label of its most likely class only
                if len(crops):
                    hit &= probs.argmax(1) == self.class_idx[c]
                for i in np.flatnonzero(hit):
                    x, y = int(X[i]), int(Y[i])
                    candidates[c].append({"bbox": [x, y, x + ps, y + ps],
                                          "confidence": float(p[i]), "code": c})

        # pass 3: merge windows into one box per organism, then one label per organism
        final = post_process(candidates, thr)
        raw_counts = {c: sum(d["confidence"] >= thr[c] for d in candidates[c]) for c in active}

        counts = {c: 0 for c in active}
        conf_sum = {c: 0.0 for c in active}
        for d in final:
            counts[d["code"]] += 1
            conf_sum[d["code"]] += d["confidence"]

        dominant = max(active, key=lambda c: (counts[c], conf_sum[c])) if final else None
        elapsed = time.time() - t0
        logger.info(f"detect {w}x{h}: windows={total_windows} flat={flat_skipped} "
                    f"noise={noise_skipped} classified={n_classify} raw={raw_counts} "
                    f"final={counts} in {elapsed:.1f}s")

        return {
            "detections": [{
                "id": i,
                "code": d["code"],
                "species": SPECIES_CFG[d["code"]]["name"],
                "bbox": [int(v) for v in d["bbox"]],
                "confidence": round(d["confidence"], 4),
            } for i, d in enumerate(final)],
            "counts": counts,
            "mean_confidence": {c: round(conf_sum[c] / counts[c], 4) if counts[c] else None
                                for c in active},
            "dominant_species": dominant,
            "thresholds": {c: round(v, 3) for c, v in thr.items()},
            "stats": {
                "windows_total": total_windows,
                "windows_flat_skipped": flat_skipped,
                "windows_noise_skipped": noise_skipped,
                "windows_classified": n_classify,
                "raw_candidates": raw_counts,
            },
            "image": {"width": w, "height": h},
            "processing_time": round(elapsed, 3),
            # internal: for post_process at other thresholds (not part of the API)
            "candidates": candidates,
        }
