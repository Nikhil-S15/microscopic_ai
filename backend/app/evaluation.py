"""
THRESHOLD ANALYSIS
==================
Scores detections against ground-truth boxes at a range of confidence
thresholds. Detection-only: a detection is right if it lands on any labelled
object; the organism type plays no part.

Matching (a detection merges overlapping windows, so one box can cover a small
cluster of objects): a detection and a labelled object match when either
box's centre lies inside the other.

  precision = detections matching ≥1 labelled object / all detections
  recall    = labelled objects matched by ≥1 detection / all labelled objects
  F1        = 2·P·R / (P + R)
"""
from typing import Dict, List, Optional, Sequence

import numpy as np

from app.unified import post_process

Box = Sequence[float]


def match_matrix(dets: np.ndarray, gts: np.ndarray) -> np.ndarray:
    """(n_det, n_gt) bool: centre of either box inside the other."""
    if not len(dets) or not len(gts):
        return np.zeros((len(dets), len(gts)), bool)

    def centre_in(a, b):  # centre of each a inside each b → (len(a), len(b))
        c = (a[:, :2] + a[:, 2:]) / 2
        return ((b[None, :, 0] <= c[:, None, 0]) & (c[:, None, 0] <= b[None, :, 2]) &
                (b[None, :, 1] <= c[:, None, 1]) & (c[:, None, 1] <= b[None, :, 3]))

    return centre_in(dets, gts) | centre_in(gts, dets).T


def _ratio(a: int, b: int) -> Optional[float]:
    return round(a / b, 4) if b else None


def threshold_sweep(images: List[Dict], thresholds: List[float]) -> List[Dict]:
    """
    images: [{"candidates": detector candidates, "ground_truth": [[x1,y1,x2,y2], ...]}]
    One global confidence threshold is applied to every candidate window.
    """
    rows = []
    for t in thresholds:
        n_det = tp_det = n_obj = found = 0
        for im in images:
            final = post_process(im["candidates"], {c: t for c in im["candidates"]})
            dets = np.array([d["bbox"] for d in final], np.float32).reshape(-1, 4)
            gts = np.array(im["ground_truth"], np.float32).reshape(-1, 4)
            m = match_matrix(dets, gts)
            n_det += len(dets)
            n_obj += len(gts)
            tp_det += int(m.any(1).sum())
            found += int(m.any(0).sum())
        p, r = _ratio(tp_det, n_det), _ratio(found, n_obj)
        if r is None:            # no labelled objects: recall and F1 undefined
            f1 = None
        elif p is None or p + r == 0:   # nothing detected, or nothing right
            f1 = 0.0
        else:
            f1 = round(2 * p * r / (p + r), 4)
        rows.append({
            "threshold": round(t, 4),
            "precision": p,
            "recall": r,
            "f1": f1,
            "detections": n_det,
            "correct_detections": tp_det,
            "objects_found": found,
        })
    return rows
