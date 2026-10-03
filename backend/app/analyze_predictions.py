"""
Analyze what the model is learning
"""

import torch
import cv2
import numpy as np
from app.models import BacteriaDetectorCNN
from app.inference import BacteriaDetector
import matplotlib.pyplot as plt
import pickle


def analyze_confidence_distribution(detector, image_path):
    """See confidence distribution across image"""
    image = cv2.imread(image_path)
    detections, _, _ = detector.detect(image, confidence_threshold=0.0)  # Get ALL
    
    bacteria_conf = [d['confidence'] for d in detections if d['class_id'] == 0]
    background_conf = [d['confidence'] for d in detections if d['class_id'] == 1]
    
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.hist(bacteria_conf, bins=50, alpha=0.7, label='Bacteria')
    plt.xlabel('Confidence')
    plt.ylabel('Count')
    plt.title('Bacteria Confidence Distribution')
    plt.axvline(0.5, color='r', linestyle='--', label='Threshold 0.5')
    plt.axvline(0.7, color='orange', linestyle='--', label='Threshold 0.7')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.hist(background_conf, bins=50, alpha=0.7, label='Background', color='gray')
    plt.xlabel('Confidence')
    plt.ylabel('Count')
    plt.title('Background Confidence Distribution')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('confidence_analysis.png')
    print("✅ Saved confidence analysis to: confidence_analysis.png")


if __name__ == "__main__":
    detector = BacteriaDetector(
        "app/model/final_bacteria_detector_weights.pkl",
        patch_size=48
    )
    
    analyze_confidence_distribution(detector, "test_image.jpg")