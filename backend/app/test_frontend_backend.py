"""
Test script to verify frontend and Colab give same results
"""

import requests
import cv2
import numpy as np
import json

def test_frontend_vs_colab(image_path):
    """Test if frontend API matches Colab results"""
    
    print("=" * 70)
    print("🔍 TESTING FRONTEND-BACKEND MATCH")
    print("=" * 70)
    
    # 1. First, run through Colab (simulate)
    print("\n1. Colab Simulation (using cv2.imread):")
    image_colab = cv2.imread(image_path)
    print(f"   Image shape: {image_colab.shape}")
    print(f"   Image dtype: {image_colab.dtype}")
    print(f"   Mean pixel value: {image_colab.mean():.2f}")
    print(f"   First pixel (BGR): {image_colab[0,0,:]}")
    
    # 2. Send to API (like frontend would)
    print("\n2. Frontend API Call:")
    with open(image_path, 'rb') as f:
        files = {'file': f}
        params = {
            'confidence_threshold': 0.5,
            'nms_threshold': 0.3,
            'use_tta': True,
            'top_k': 3
        }
        
        try:
            response = requests.post(
                'http://localhost:8000/api/v1/detect',
                files=files,
                params=params
            )
            
            if response.status_code == 200:
                data = response.json()
                print(f"   ✅ API Success: {len(data['detections'])} detections")
                
                # Show first detection
                if data['detections']:
                    first = data['detections'][0]
                    print(f"   First detection:")
                    print(f"     BBox: {first['bbox']}")
                    print(f"     Confidence: {first['confidence']:.3f}")
                
                # Save API response
                with open('api_response.json', 'w') as f:
                    json.dump(data, f, indent=2)
                print("   ✅ Saved API response to api_response.json")
                
            else:
                print(f"   ❌ API Error: {response.status_code}")
                print(response.text)
                
        except Exception as e:
            print(f"   ❌ Request failed: {str(e)}")
    
    # 3. Test direct endpoint
    print("\n3. Testing Direct Endpoint (no frontend conversion):")
    with open(image_path, 'rb') as f:
        files = {'file': f}
        response = requests.post(
            'http://localhost:8000/api/v1/test-colab-match',
            files=files
        )
        
        if response.status_code == 200:
            data = response.json()
            print(f"   ✅ Direct test: {data['total_detections']} detections")
            print(f"   Color format: {data['color_format']}")
    
    print("\n" + "=" * 70)
    print("📊 DIAGNOSIS:")
    print("-" * 70)
    print("If detections differ:")
    print("1. Check image format in API logs")
    print("2. Compare first pixel values")
    print("3. Try different confidence thresholds")
    print("=" * 70)

if __name__ == "__main__":
    # Test with your image
    test_frontend_vs_colab("test.jpg")