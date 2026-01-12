#!/usr/bin/env python3
"""
Test script for Emotion Detection API
Usage: python test_api.py <image_path>
"""

import requests
import sys
from pathlib import Path

BASE_URL = "http://127.0.0.1:5000"

def test_health():
    """Test health check endpoint"""
    try:
        response = requests.get(f"{BASE_URL}/health")
        print("✓ Health Check:")
        print(f"  Status: {response.json()}")
        return True
    except Exception as e:
        print(f"✗ Health Check Failed: {e}")
        return False

def test_emotion_detection(image_path):
    """Test emotion detection with an image"""
    if not Path(image_path).exists():
        print(f"✗ Image not found: {image_path}")
        return False
    
    try:
        with open(image_path, 'rb') as f:
            files = {'image': f}
            response = requests.post(f"{BASE_URL}/detect-emotion", files=files)
        
        if response.status_code == 200:
            print("✓ Emotion Detection:")
            result = response.json()
            print(f"  Emotion: {result['emotion']}")
            print(f"  Confidence: {result['confidence']:.2%}")
            print(f"  Faces Detected: {result['faces_detected']}")
            if result.get('all_emotions'):
                print(f"  All Emotions: {result['all_emotions']}")
            return True
        else:
            print(f"✗ Error: {response.json()}")
            return False
    except Exception as e:
        print(f"✗ Emotion Detection Failed: {e}")
        return False

if __name__ == "__main__":
    print("Emotion Detection API - Test Suite\n")
    
    # Test health
    if not test_health():
        print("Cannot connect to server. Make sure it's running on http://127.0.0.1:5000")
        sys.exit(1)
    
    # Test with image if provided
    if len(sys.argv) > 1:
        print()
        test_emotion_detection(sys.argv[1])
    else:
        print("\nUsage: python test_api.py <image_path>")
        print("Example: python test_api.py sample.jpg")
