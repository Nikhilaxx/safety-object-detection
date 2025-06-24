"""
Test script for the Safety Compliance Detection System
Run this to verify your installation and test detection capabilities
"""

import cv2
import numpy as np
from ultralytics import YOLO
import sys
import os
from utils import SafetyDetector, draw_enhanced_annotations

def test_installation():
    """Test if all required packages are installed correctly"""
    print("🔍 Testing installation...")
    
    try:
        import streamlit
        print("✅ Streamlit installed")
    except ImportError:
        print("❌ Streamlit not installed")
        return False
    
    try:
        import cv2
        print("✅ OpenCV installed")
    except ImportError:
        print("❌ OpenCV not installed")
        return False
    
    try:
        from ultralytics import YOLO
        print("✅ Ultralytics (YOLOv8) installed")
    except ImportError:
        print("❌ Ultralytics not installed")
        return False
    
    try:
        import numpy as np
        print("✅ NumPy installed")
    except ImportError:
        print("❌ NumPy not installed")
        return False
    
    try:
        from PIL import Image
        print("✅ Pillow installed")
    except ImportError:
        print("❌ Pillow not installed")
        return False
    
    try:
        import pandas as pd
        print("✅ Pandas installed")
    except ImportError:
        print("❌ Pandas not installed")
        return False
    
    print("✅ All packages installed successfully!")
    return True

def test_model_loading():
    """Test if YOLOv8 model can be loaded"""
    print("\n🤖 Testing model loading...")
    
    try:
        model = YOLO('yolov8n.pt')
        print("✅ YOLOv8 model loaded successfully")
        return model
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        print("💡 Make sure you have internet connection for first-time download")
        return None

def create_test_image():
    """Create a simple test image with basic shapes representing people"""
    print("\n🎨 Creating test image...")
    
    # Create a test image
    img = np.ones((400, 600, 3), dtype=np.uint8) * 240  # Light gray background
    
    # Add some basic shapes to represent people
    # Person 1 - compliant (with colored regions for helmet and vest)
    cv2.rectangle(img, (100, 200), (140, 350), (139, 69, 19), -1)  # Body
    cv2.circle(img, (120, 180), 20, (139, 69, 19), -1)  # Head
    cv2.circle(img, (120, 160), 25, (0, 255, 255), -1)  # Yellow helmet
    cv2.rectangle(img, (105, 220), (135, 280), (0, 165, 255), -1)  # Orange vest
    
    # Person 2 - non-compliant
    cv2.rectangle(img, (300, 200), (340, 350), (139, 69, 19), -1)  # Body
    cv2.circle(img, (320, 180), 20, (139, 69, 19), -1)  # Head (no helmet)
    cv2.rectangle(img, (305, 220), (335, 280), (100, 100, 100), -1)  # Gray shirt (no vest)
    
    # Person 3 - partially compliant
    cv2.rectangle(img, (450, 200), (490, 350), (139, 69, 19), -1)  # Body
    cv2.circle(img, (470, 180), 20, (139, 69, 19), -1)  # Head
    cv2.circle(img, (470, 160), 25, (0, 255, 255), -1)  # Yellow helmet
    # No vest
    
    # Add title
    cv2.putText(img, "TEST IMAGE - 3 PEOPLE", (150, 50), 
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
    
    print("✅ Test image created")
    return img


def test_detection(model, test_image):
    """Test the detection pipeline"""
    print("\n🔍 Testing detection pipeline...")
    
    if model is None:
        print("❌ Cannot test detection - model not loaded")
        return False
    
    try:
        # Test person detection
        results = model(test_image, classes=[0], conf=0.3)  # Class 0 is person
        
        num_detections = len(results[0].boxes) if len(results[0].boxes) > 0 else 0
        print(f"✅ Person detection test: {num_detections} people detected")
        
        # Test safety detector
        safety_detector = SafetyDetector()
        print("✅ Safety detector initialized")
        
        # Test color-based detection on a sample region
        if test_image.size > 0:
            helmet_color_detected = safety_detector.detect_helmet_by_color(sample_roi)
            helmet_bbox = [100, 160, 140, 190]  # Fake box just for test (in real scenario, compute from detection
            person_bbox = [100, 150, 140, 350]  # Test box for person
            if helmet_color_detected and is_helmet_on_head(person_bbox, helmet_bbox):
                helmet = True
            else:
                helmet = False
            vest_detected = safety_detector.detect_vest_by_color(sample_roi)
            print(f"✅ Color detection test - Helmet: {helmet}, Vest: {vest_detected}")
        return True
        
    except Exception as e:
        print(f"❌ Detection test failed: {e}")
        return False

def test_annotation(test_image):
    """Test the annotation drawing"""
    print("\n🎨 Testing annotation drawing...")
    
    try:
        # Create sample detection results
        sample_results = [
            {
                'person_id': 1,
                'bbox': [100, 150, 140, 350],
                'confidence': 0.85,
                'helmet': True,
                'mask': True,
                'vest': True,
                'compliance_rate': 1.0
            },
            {
                'person_id': 2,
                'bbox': [300, 150, 340, 350],
                'confidence': 0.78,
                'helmet': False,
                'mask': False,
                'vest': False,
                'compliance_rate': 0.0
            },
            {
                'person_id': 3,
                'bbox': [450, 150, 490, 350],
                'confidence': 0.92,
                'helmet': True,
                'mask': False,
                'vest': False,
                'compliance_rate': 0.33
            }
        ]
        
        annotated_image = draw_enhanced_annotations(test_image, sample_results)
        
        # Save test result
        cv2.imwrite("test_result.jpg", annotated_image)
        print("✅ Annotation test successful - saved as 'test_result.jpg'")
        return True
        
    except Exception as e:
        print(f"❌ Annotation test failed: {e}")
        return False

def run_comprehensive_test():
    """Run all tests"""
    print("🦺 SAFETY COMPLIANCE DETECTION SYSTEM - COMPREHENSIVE TEST")
    print("=" * 60)
    
    # Test 1: Installation
    if not test_installation():
        print("\n❌ Installation test failed. Please install missing packages.")
        return False
    
    # Test 2: Model loading
    model = test_model_loading()
    
    # Test 3: Create test image
    test_image = create_test_image()
    
    # Test 4: Detection
    detection_success = test_detection(model, test_image)
    
    # Test 5: Annotation
    annotation_success = test_annotation(test_image)
    
    # Summary
    print("\n" + "=" * 60)
    print("📊 TEST SUMMARY:")
    print("✅ Installation: PASSED")
    print(f"{'✅' if model is not None else '❌'} Model Loading: {'PASSED' if model is not None else 'FAILED'}")
    print("✅ Test Image Creation: PASSED")
    print(f"{'✅' if detection_success else '❌'} Detection Pipeline: {'PASSED' if detection_success else 'FAILED'}")
    print(f"{'✅' if annotation_success else '❌'} Annotation Drawing: {'PASSED' if annotation_success else 'FAILED'}")
    
    all_passed = model is not None and detection_success and annotation_success
    
    if all_passed:
        print("\n🎉 ALL TESTS PASSED! Your system is ready to use.")
        print("🚀 Run the main application with: streamlit run app.py")
    else:
        print("\n⚠️ Some tests failed. Please check the error messages above.")
        print("💡 Try reinstalling dependencies: pip install -r requirements.txt")
    
    return all_passed

def quick_demo():
    """Run a quick demo of the detection system"""
    print("\n🎭 QUICK DEMO MODE")
    print("-" * 30)
    
    # Create demo image
    demo_img = create_test_image()
    
    # Load model
    try:
        model = YOLO('yolov8n.pt')
        print("✅ Model loaded for demo")
    except:
        print("❌ Could not load model for demo")
        return
    
    # Run detection
    try:
        results = model(demo_img, classes=[0], conf=0.3)
        num_people = len(results[0].boxes) if len(results[0].boxes) > 0 else 0
        
        print(f"🔍 Detected {num_people} people in demo image")
        
        # Simulate compliance results
        demo_results = []
        for i in range(min(num_people, 3)):  # Max 3 for demo
            demo_results.append({
                'person_id': i + 1,
                'bbox': [100 + i*150, 150, 140 + i*150, 350],
                'confidence': 0.8 + i*0.05,
                'helmet': i != 1,  # Person 2 has no helmet
                'mask': i == 0,    # Only person 1 has mask
                'vest': i != 1,    # Person 2 has no vest
                'compliance_rate': (1.0 if i == 0 else 0.33 if i == 2 else 0.0)
            })
        
        # Draw annotations
        annotated = draw_enhanced_annotations(demo_img, demo_results)
        cv2.imwrite("demo_result.jpg", annotated)
        
        print("✅ Demo completed - result saved as 'demo_result.jpg'")
        
        # Print compliance summary
        print("\n📊 DEMO COMPLIANCE SUMMARY:")
        for result in demo_results:
            status = "COMPLIANT" if result['compliance_rate'] == 1.0 else "VIOLATION" if result['compliance_rate'] == 0.0 else "PARTIAL"
            print(f"Person {result['person_id']}: {status} ({result['compliance_rate']:.0%})")
            print(f"  Helmet: {'✅' if result['helmet'] else '❌'}, Mask: {'✅' if result['mask'] else '❌'}, Vest: {'✅' if result['vest'] else '❌'}")
        
    except Exception as e:
        print(f"❌ Demo failed: {e}")

if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--demo":
        quick_demo()
    else:
        run_comprehensive_test()
        
        # Ask if user wants to see demo
        try:
            response = input("\n🎭 Would you like to run a quick demo? (y/n): ").lower()
            if response in ['y', 'yes']:
                quick_demo()
        except KeyboardInterrupt:
            print("\n👋 Test completed!")