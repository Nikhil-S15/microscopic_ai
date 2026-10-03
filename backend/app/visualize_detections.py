"""
Visualize detections to see what the model is doing
Usage: python visualize_detections.py <image_path>
"""

import sys
import cv2
import numpy as np
import requests
from pathlib import Path
import json


def visualize_detections(image_path, confidence_threshold=0.5):
    """Download detections and visualize them"""
    
    # Read image
    image = cv2.imread(image_path)
    if image is None:
        print(f"❌ Could not read image: {image_path}")
        return
    
    h, w = image.shape[:2]
    print(f"Image size: {w}x{h}")
    
    # Get detections from API
    print(f"\n🔍 Testing with confidence threshold: {confidence_threshold}")
    
    with open(image_path, 'rb') as f:
        files = {'file': f}
        params = {
            'confidence_threshold': confidence_threshold,
            'nms_threshold': 0.3,
            'top_k': 3
        }
        response = requests.post(
            'http://localhost:8000/api/v1/detect',
            files=files,
            params=params
        )
    
    if response.status_code != 200:
        print(f"❌ API Error: {response.status_code}")
        print(response.text)
        return
    
    data = response.json()
    detections = data['detections']
    
    print(f"\n📊 Results:")
    print(f"   Total detections: {len(detections)}")
    print(f"   Overall probabilities:")
    for prob in data['overall_top_probabilities']:
        print(f"      {prob['class_name']}: {prob['probability']*100:.1f}%")
    
    # Analyze confidence distribution
    if detections:
        confidences = [d['confidence'] for d in detections]
        print(f"\n📈 Confidence Statistics:")
        print(f"   Min:  {min(confidences):.3f}")
        print(f"   Max:  {max(confidences):.3f}")
        print(f"   Mean: {np.mean(confidences):.3f}")
        print(f"   Median: {np.median(confidences):.3f}")
    
    # Create visualization
    output = image.copy()
    
    # Draw all detections
    for i, det in enumerate(detections):
        x1, y1, x2, y2 = det['bbox']
        conf = det['confidence']
        
        # Color based on confidence
        if conf > 0.9:
            color = (0, 255, 0)  # Green - very confident
        elif conf > 0.7:
            color = (0, 255, 255)  # Yellow - confident
        else:
            color = (0, 165, 255)  # Orange - less confident
        
        # Draw box
        cv2.rectangle(output, (x1, y1), (x2, y2), color, 2)
        
        # Label
        label = f"#{i+1}: {conf:.2f}"
        cv2.putText(output, label, (x1, y1-5), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    
    # Save output
    output_path = f"detection_viz_{confidence_threshold}.jpg"
    cv2.imwrite(output_path, output)
    print(f"\n✅ Saved visualization to: {output_path}")
    
    # Save JSON
    json_path = f"detections_{confidence_threshold}.json"
    with open(json_path, 'w') as f:
        json.dump(data, f, indent=2)
    print(f"✅ Saved JSON to: {json_path}")
    
    return detections


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python visualize_detections.py <image_path>")
        sys.exit(1)
    
    image_path = sys.argv[1]
    
    # Test with different thresholds
    print("=" * 70)
    print("DETECTION VISUALIZATION")
    print("=" * 70)
    
    for threshold in [0.3, 0.5, 0.7, 0.9]:
        visualize_detections(image_path, threshold)
        print("\n" + "-" * 70 + "\n")