"""
Demo image generator for testing the Safety Compliance Detection System
This script creates sample scenarios for demonstration purposes
"""

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import os

def create_demo_construction_site(width=800, height=600, scenario="mixed"):
    """
    Create a demo construction site image with simulated people
    """
    # Create base construction site background
    img = np.ones((height, width, 3), dtype=np.uint8) * 200  # Light gray background
    
    # Add construction site elements
    # Ground
    cv2.rectangle(img, (0, height-100), (width, height), (139, 69, 19), -1)  # Brown ground
    
    # Building structure
    cv2.rectangle(img, (50, 200), (300, height-100), (169, 169, 169), -1)  # Gray building
    cv2.rectangle(img, (350, 150), (600, height-100), (169, 169, 169), -1)  # Another building
    
    # Construction equipment
    cv2.rectangle(img, (650, 300), (750, height-100), (255, 140, 0), -1)  # Orange equipment
    
    # Sky
    cv2.rectangle(img, (0, 0), (width, 200), (135, 206, 235), -1)  # Sky blue
    
    # Add text overlay
    cv2.putText(img, "DEMO CONSTRUCTION SITE", (width//2-150, 50), 
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
    cv2.putText(img, f"Scenario: {scenario.upper()}", (width//2-100, 80), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
    
    return img

def add_demo_person(img, x, y, width, height, helmet=True, mask=True, vest=True):
    """
    Add a simplified person figure to the image with safety equipment
    """
    # Person body (rectangle for simplicity)
    person_color = (101, 67, 33)  # Brown skin tone
    cv2.rectangle(img, (x, y), (x+width, y+height), person_color, -1)
    
    # Head
    head_size = width // 3
    head_x = x + width//2 - head_size//2
    head_y = y - head_size
    cv2.circle(img, (head_x + head_size//2, head_y + head_size//2), head_size//2, person_color, -1)
    
    # Safety equipment
    if helmet:
        # Yellow helmet
        helmet_color = (0, 255, 255)  # Yellow in BGR
        cv2.ellipse(img, (head_x + head_size//2, head_y + head_size//3), 
                   (head_size//2 + 5, head_size//3), 0, 180, 360, helmet_color, -1)
    
    if mask:
        # Blue mask
        mask_color = (255, 0, 0)  # Blue in BGR
        mask_y = head_y + head_size//2
        cv2.rectangle(img, (head_x + 5, mask_y), (head_x + head_size - 5, mask_y + head_size//4), mask_color, -1)
    
    if vest:
        # Orange safety vest
        vest_color = (0, 165, 255)  # Orange in BGR
        vest_y = y + height//4
        cv2.rectangle(img, (x + 5, vest_y), (x + width - 5, y + 3*height//4), vest_color, -1)
        
        # Reflective stripes
        stripe_color = (255, 255, 255)  # White
        cv2.rectangle(img, (x + 5, vest_y + 10), (x + width - 5, vest_y + 15), stripe_color, -1)
        cv2.rectangle(img, (x + 5, vest_y + 25), (x + width - 5, vest_y + 30), stripe_color, -1)

def generate_demo_scenarios():
    """
    Generate demo images for different compliance scenarios
    """
    scenarios = {
        "high_compliance": {
            "description": "High Compliance Site (90% compliant)",
            "people": [
                {"x": 100, "y": 300, "helmet": True, "mask": True, "vest": True},
                {"x": 200, "y": 320, "helmet": True, "mask": True, "vest": True},
                {"x": 300, "y": 310, "helmet": True, "mask": True, "vest": False},
                {"x": 450, "y": 290, "helmet": True, "mask": True, "vest": True},
                {"x": 550, "y": 305, "helmet": True, "mask": True, "vest": True},
            ]
        },
        "medium_compliance": {
            "description": "Medium Compliance Site (60% compliant)",
            "people": [
                {"x": 120, "y": 300, "helmet": True, "mask": True, "vest": False},
                {"x": 220, "y": 320, "helmet": True, "mask": False, "vest": True},
                {"x": 320, "y": 310, "helmet": False, "mask": True, "vest": True},
                {"x": 420, "y": 290, "helmet": True, "mask": True, "vest": True},
                {"x": 520, "y": 305, "helmet": False, "mask": False, "vest": True},
            ]
        },
        "low_compliance": {
            "description": "Low Compliance Site (30% compliant)",
            "people": [
                {"x": 110, "y": 300, "helmet": False, "mask": False, "vest": False},
                {"x": 210, "y": 320, "helmet": True, "mask": False, "vest": False},
                {"x": 310, "y": 310, "helmet": False, "mask": False, "vest": True},
                {"x": 410, "y": 290, "helmet": False, "mask": True, "vest": False},
                {"x": 510, "y": 305, "helmet": True, "mask": True, "vest": True},
            ]
        },
        "mixed_compliance": {
            "description": "Mixed Compliance Site (varied)",
            "people": [
                {"x": 90, "y": 300, "helmet": True, "mask": True, "vest": True},
                {"x": 180, "y": 320, "helmet": False, "mask": False, "vest": False},
                {"x": 270, "y": 310, "helmet": True, "mask": False, "vest": True},
                {"x": 360, "y": 290, "helmet": False, "mask": True, "vest": False},
                {"x": 450, "y": 305, "helmet": True, "mask": True, "vest": True},
                {"x": 540, "y": 315, "helmet": False, "mask": False, "vest": True},
            ]
        }
    }
    
    # Create demo_images directory if it doesn't exist
    os.makedirs("demo_images", exist_ok=True)
    
    for scenario_name, scenario_data in scenarios.items():
        # Create base construction site
        img = create_demo_construction_site(scenario=scenario_data["description"])
        
        # Add people with different compliance levels
        for person in scenario_data["people"]:
            add_demo_person(img, person["x"], person["y"], 40, 80, 
                          person["helmet"], person["mask"], person["vest"])
        
        # Save image
        filename = f"demo_images/{scenario_name}.jpg"
        cv2.imwrite(filename, img)
        print(f"Generated: {filename}")

def create_sample_real_images():
    """
    Create instructions for using real construction site images
    """
    instructions = """
# USING REAL CONSTRUCTION SITE IMAGES

For best results with the Safety Compliance Detection System, use images that have:

## IMAGE REQUIREMENTS:
- Clear visibility of people (not too far away)
- Good lighting conditions
- People wearing various safety equipment
- Construction site or industrial setting
- Resolution: 640x480 minimum, 1920x1080 recommended
- Format: JPG, PNG, BMP

## RECOMMENDED IMAGE SOURCES:
1. **Stock Photo Websites:**
   - Pexels.com (free construction site photos)
   - Unsplash.com (free industrial photos)
   - Shutterstock.com (premium construction photos)

2. **Search Terms:**
   - "construction workers safety equipment"
   - "construction site helmet vest"
   - "industrial workers safety gear"
   - "construction team safety compliance"

3. **Your Own Photos:**
   - Take photos at actual construction sites
   - Ensure proper permissions and safety
   - Include people with various compliance levels
   - Test different lighting conditions

## SAMPLE IMAGE CHARACTERISTICS:
- **High Compliance**: Most workers wearing all safety equipment
- **Medium Compliance**: Mixed compliance levels
- **Low Compliance**: Many workers missing safety equipment
- **Varied Scenarios**: Different equipment combinations

## TESTING TIPS:
1. Start with demo mode to understand the system
2. Use clear, well-lit images for best detection
3. Try images with 2-10 people for optimal results
4. Test different compliance scenarios
5. Adjust detection confidence as needed

The demo images generated by this script provide a starting point,
but real construction site photos will give the most accurate results.
    """
    
    with open("demo_images/README_REAL_IMAGES.txt", "w") as f:
        f.write(instructions)
    
    print("Created: demo_images/README_REAL_IMAGES.txt")

if __name__ == "__main__":
    print("🎭 Generating demo images for Safety Compliance Detection System...")
    generate_demo_scenarios()
    create_sample_real_images()
    print("✅ Demo image generation complete!")
    print("\nGenerated files:")
    print("- demo_images/high_compliance.jpg")
    print("- demo_images/medium_compliance.jpg") 
    print("- demo_images/low_compliance.jpg")
    print("- demo_images/mixed_compliance.jpg")
    print("- demo_images/README_REAL_IMAGES.txt")
    print("\n🚀 You can now run the main application: streamlit run app.py")