"""
Configuration file for the Safety Compliance Detection System
Modify these settings to customize the system behavior
"""

# Detection Configuration
DETECTION_CONFIG = {
    # YOLOv8 Model Settings
    'model_name': 'yolov8n.pt',  # Options: yolov8n.pt, yolov8s.pt, yolov8m.pt, yolov8l.pt, yolov8x.pt
    'confidence_threshold': 0.4,  # Minimum confidence for person detection (0.1 - 0.9)
    'person_class_id': 0,  # COCO dataset person class ID
    
    # Safety Equipment Detection Thresholds
    'helmet_threshold': 0.08,  # Percentage of head region for helmet detection
    'vest_threshold': 0.15,    # Percentage of torso region for vest detection
    'mask_threshold': 0.10,    # Percentage of face region for mask detection
    
    # Image Processing
    'max_image_size': (1920, 1080),  # Maximum image dimensions for processing
    'min_person_size': (30, 60),     # Minimum person bounding box size
}

# Color Ranges for Safety Equipment (HSV color space)
SAFETY_COLORS = {
    'helmet': {
        'yellow': [(20, 100, 100), (30, 255, 255)],
        'orange': [(10, 100, 100), (20, 255, 255)],
        'white': [(0, 0, 200), (180, 30, 255)],
        'red': [(0, 100, 100), (10, 255, 255), (170, 100, 100), (180, 255, 255)],
        'blue': [(100, 100, 100), (130, 255, 255)],
    },
    'vest': {
        'green': [(45, 100, 100), (75, 255, 255)],
        'orange': [(10, 100, 100), (20, 255, 255)],
        'yellow': [(20, 100, 100), (30, 255, 255)],
    },
    'mask': {
        'white': [(0, 0, 180), (180, 30, 255)],
        'blue': [(100, 50, 50), (130, 255, 255)],
        'cyan': [(80, 50, 50), (100, 255, 255)],
    }
}

# Compliance Scoring
COMPLIANCE_CONFIG = {
    'required_equipment': ['helmet', 'mask', 'vest'],  # List of required safety equipment
    'weights': {  # Importance weights for different equipment (must sum to 1.0)
        'helmet': 0.4,  # Helmet is most critical
        'mask': 0.3,    # Mask is important for health
        'vest': 0.3,    # Vest is important for visibility
    },
    'compliance_levels': {
        'full': 1.0,      # 100% compliance
        'partial_high': 0.67,  # 67% or higher
        'partial_low': 0.33,   # 33% or higher
        'violation': 0.0,      # Below 33%
    }
}

# UI Configuration
UI_CONFIG = {
    'page_title': 'Safety Compliance Detection System',
    'page_icon': '🦺',
    'layout': 'wide',
    'theme_colors': {
        'primary': '#FF6B35',
        'secondary': '#F7931E',
        'success': '#38a169',
        'warning': '#dd6b20',
        'danger': '#e53e3e',
        'info': '#4299e1',
    },
    'max_upload_size': 200,  # MB
    'supported_formats': ['jpg', 'jpeg', 'png', 'bmp'],
}

# Performance Configuration
PERFORMANCE_CONFIG = {
    'enable_gpu': True,  # Use GPU if available
    'batch_size': 1,     # Number of images to process simultaneously
    'num_threads': 4,    # Number of CPU threads for processing
    'cache_models': True,  # Cache loaded models in memory
    'optimize_for_speed': True,  # Optimize for speed vs accuracy
}

# Logging Configuration
LOGGING_CONFIG = {
    'enable_logging': True,
    'log_level': 'INFO',  # DEBUG, INFO, WARNING, ERROR, CRITICAL
    'log_file': 'safety_detection.log',
    'log_detections': True,  # Log detection results
    'log_compliance': True,  # Log compliance statistics
}

# Export Configuration
EXPORT_CONFIG = {
    'enable_exports': True,
    'export_formats': ['json', 'csv', 'pdf'],
    'include_images': True,  # Include annotated images in exports
    'export_directory': 'exports',
}

# Demo Configuration
DEMO_CONFIG = {
    'scenarios': {
        'high_compliance': {
            'name': 'High Compliance Site (90% compliant)',
            'compliance_rates': [1.0, 1.0, 0.67, 1.0, 1.0],
        },
        'medium_compliance': {
            'name': 'Medium Compliance Site (60% compliant)',
            'compliance_rates': [0.67, 1.0, 0.33, 0.67, 1.0],
        },
        'low_compliance': {
            'name': 'Low Compliance Site (30% compliant)',
            'compliance_rates': [0.33, 0.0, 0.33, 0.67, 0.33],
        },
        'mixed_compliance': {
            'name': 'Mixed Compliance Site (varied)',
            'compliance_rates': [1.0, 0.33, 0.67, 0.0, 1.0, 0.33],
        }
    }
}

# Advanced Features Configuration
ADVANCED_CONFIG = {
    'enable_tracking': False,  # Enable person tracking across frames (for video)
    'enable_alerts': False,    # Enable real-time alerts for violations
    'enable_database': False,  # Enable database logging
    'enable_api': False,       # Enable REST API endpoints
    'enable_multi_site': False,  # Enable multi-site monitoring
}

# Custom Model Configuration (for production use)
CUSTOM_MODELS = {
    'use_custom_models': False,  # Set to True to use custom trained models
    'helmet_model_path': 'models/helmet_detection.pt',
    'vest_model_path': 'models/vest_detection.pt',
    'mask_model_path': 'models/mask_detection.pt',
    'person_model_path': 'models/person_detection.pt',
}

# Validation functions
def validate_config():
    """Validate configuration settings"""
    errors = []
    
    # Validate compliance weights
    total_weight = sum(COMPLIANCE_CONFIG['weights'].values())
    if abs(total_weight - 1.0) > 0.01:
        errors.append(f"Compliance weights must sum to 1.0, got {total_weight}")
    
    # Validate thresholds
    for key, value in DETECTION_CONFIG.items():
        if 'threshold' in key and not (0.0 <= value <= 1.0):
            errors.append(f"{key} must be between 0.0 and 1.0, got {value}")
    
    # Validate color ranges
    for equipment, colors in SAFETY_COLORS.items():
        for color_name, ranges in colors.items():
            if isinstance(ranges[0], tuple) and len(ranges) == 2:
                # Single range
                lower, upper = ranges
                if len(lower) != 3 or len(upper) != 3:
                    errors.append(f"Invalid color range for {equipment}.{color_name}")
    
    if errors:
        raise ValueError("Configuration validation failed:\n" + "\n".join(errors))
    
    return True

# Load configuration on import
if __name__ == "__main__":
    try:
        validate_config()
        print("✅ Configuration validation passed")
    except ValueError as e:
        print(f"❌ Configuration validation failed: {e}")
else:
    validate_config()