import cv2
import numpy as np
from typing import List, Dict, Tuple
import colorsys

class SafetyDetector:
    """Advanced safety equipment detection utilities using computer vision techniques"""
    
    def __init__(self):
        self.safety_colors = {
            'helmet_yellow': [(20, 100, 100), (30, 255, 255)],
            'helmet_orange': [(10, 100, 100), (20, 255, 255)],
            'helmet_white': [(0, 0, 200), (180, 30, 255)],
            'vest_green': [(45, 100, 100), (75, 255, 255)],
            'vest_orange': [(10, 100, 100), (20, 255, 255)],
            'vest_yellow': [(20, 100, 100), (30, 255, 255)]
        }

    def detect_helmet_by_color(self, person_roi: np.ndarray) -> bool:
        if person_roi.size == 0:
            return False
        try:
            hsv = cv2.cvtColor(person_roi, cv2.COLOR_BGR2HSV)
            height, width = person_roi.shape[:2]
            head_region = hsv[:height//3, width//4:3*width//4]
            helmet_ranges = [
                ([20, 150, 180], [35, 255, 255]),   # Bright yellow
                ([0, 0, 230], [180, 40, 255]),      # Very bright white
                ([5, 150, 180], [20, 255, 255])     # Bright orange
                ]
            combined_mask = np.zeros(head_region.shape[:2], dtype=np.uint8)
            for lower, upper in helmet_ranges:
                mask = cv2.inRange(head_region, np.array(lower), np.array(upper))
                combined_mask = cv2.bitwise_or(combined_mask, mask)
            kernel = np.ones((3, 3), np.uint8)
            combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_OPEN, kernel)
            combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_CLOSE, kernel)
            helmet_pixels = cv2.countNonZero(combined_mask)
            total_pixels = head_region.shape[0] * head_region.shape[1]
            helmet_ratio = helmet_pixels / total_pixels if total_pixels > 0 else 0
            return helmet_ratio > 0.13
        except Exception as e:
            print(f"Error in helmet detection: {e}")
            return False



    def detect_vest_by_color(self, person_crop):
        hsv = cv2.cvtColor(person_crop, cv2.COLOR_BGR2HSV)
        vest_ranges = [
            ([25, 180, 180], [35, 255, 255]),
            ([10, 180, 180], [25, 255, 255]),
            ([40, 180, 180], [0, 255, 255])
        ]
        combined_mask = None
        for lower, upper in vest_ranges:
            mask = cv2.inRange(hsv, np.array(lower), np.array(upper))
            combined_mask = mask if combined_mask is None else cv2.bitwise_or(combined_mask, mask)

        kernel = np.ones((3, 3), np.uint8)
        combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_CLOSE, kernel)
        combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_OPEN, kernel)

        vest_pixels = cv2.countNonZero(combined_mask)
        total_pixels = person_crop.shape[0] * person_crop.shape[1]
        vest_ratio = vest_pixels / total_pixels if total_pixels > 0 else 0

        return vest_ratio > 0.08

    def enhance_detection_accuracy(self, image: np.ndarray, results: List[Dict]) -> List[Dict]:
        enhanced_results = []

        for result in results:
            try:
                enhanced_result = result.copy()
                helmet_color = self.detect_helmet_by_color(person_roi)
                enhanced_result['helmet'] = helmet_color
                x1, y1, x2, y2 = result['bbox']
                person_roi = image[y1:y2, x1:x2]
                height = person_roi.shape[0]
                torso_crop = person_roi[height // 3 : 2 * height // 3, :]
                enhanced_result['vest'] = result.get('vest', False) or self.detect_vest_by_color(torso_crop)
                enhanced_result['mask'] = False

                enhanced_result = result.copy()
                compliance_items = {
                    'helmet': enhanced_result['helmet'],
                    'mask': False,
                    'vest': enhanced_result['vest']
                }

                enhanced_result['compliance_rate'] = sum(compliance_items.values()) / len(compliance_items)
                enhanced_result['compliant_items'] = sum(compliance_items.values())
                enhanced_results.append(enhanced_result)

            except Exception as e:
                print(f"Error enhancing detection for person {result.get('person_id', 'unknown')}: {e}")
                enhanced_results.append(result)

        return enhanced_results

def draw_enhanced_annotations(image, results):
    """
    Draw bounding boxes and compliance annotations on the image
    """
    annotated = image.copy()
    
    for result in results:
        x1, y1, x2, y2 = result['bbox']
        person_id = result['person_id']
        compliance = result['compliance_rate']
        
        # Color based on compliance level
        if compliance == 1.0:
            color = (0, 200, 0)       # Green
        elif compliance >= 0.33:
            color = (0, 165, 255)     # Orange
        else:
            color = (0, 0, 255)       # Red
        
        # Draw bounding box
        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
        
        label = f"ID {person_id} | {int(compliance * 100)}% Compliant"
        cv2.putText(annotated, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX,
                    0.6, color, 2, cv2.LINE_AA)
    
    return annotated


def is_helmet_on_head(person_box, helmet_box):
    px1, py1, px2, py2 = person_box
    hx1, hy1, hx2, hy2 = helmet_box
    person_height = py2 - py1
    head_zone_bottom = py1 + 0.3 * person_height
    helmet_center_y = (hy1 + hy2) / 2
    return py1 <= helmet_center_y <= head_zone_bottom