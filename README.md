# 🦺 Real-Time Safety Compliance Detection System

A comprehensive AI-powered Streamlit application that uses YOLOv8 to detect people at construction sites and assess their safety equipment compliance in real-time.

![Safety Detection System](https://images.pexels.com/photos/1216589/pexels-photo-1216589.jpeg?auto=compress&cs=tinysrgb&w=800)

## 🎯 Features

### Core Detection Capabilities
- **👥 Person Detection**: Advanced YOLOv8-based person identification
- **🪖 Helmet Detection**: AI + computer vision helmet compliance checking
- **😷 Face Mask Detection**: Health safety mask verification
- **🦺 Safety Vest Detection**: High-visibility vest identification
- **📊 Compliance Analytics**: Detailed reporting and statistics

### User Interface
- **📱 Responsive Design**: Professional, modern interface
- **📷 Multiple Input Types**: Image upload, webcam capture, demo mode
- **🎨 Visual Indicators**: Color-coded compliance status
- **📈 Real-time Analytics**: Live compliance dashboards
- **📋 Detailed Reports**: Individual and aggregate compliance summaries

### Advanced Features
- **🔍 Enhanced Detection**: Combines AI with color-based computer vision
- **⚡ Real-time Processing**: Fast detection with confidence scoring
- **📊 Export Capabilities**: Compliance reports and statistics
- **🎛️ Configurable Settings**: Adjustable detection thresholds
- **🎭 Demo Mode**: Showcase capabilities with simulated scenarios

## 🚀 Quick Start

### Prerequisites
- Python 3.8 or higher
- 4GB RAM minimum (8GB recommended)
- 2GB free disk space
- Internet connection (for initial model download)
- Webcam (optional, for camera mode)

### Installation

1. **Clone or download the project**
```bash
git clone <repository-url>
cd safety-compliance-detection
```

2. **Create virtual environment (recommended)**
```bash
python -m venv safety_env

# Windows
safety_env\Scripts\activate

# macOS/Linux
source safety_env/bin/activate
```

3. **Install dependencies**
```bash
pip install -r requirements.txt
```

4. **Run the application**
```bash
streamlit run app.py
```

5. **Open your browser** to `http://localhost:8501`

## 📖 Usage Guide

### 1. Image Upload Mode
- Select "Upload Image" from the sidebar
- Upload a construction site image (JPG, JPEG, PNG, BMP)
- View real-time detection results and compliance analysis
- Download compliance reports

### 2. Webcam Mode
- Select "Use Webcam" from the sidebar
- Grant camera permissions when prompted
- Click "Take a picture" to capture and analyze
- View instant compliance results

### 3. Demo Mode
- Select "Demo Images" for testing
- Choose from different compliance scenarios:
  - High Compliance Site (90% compliant)
  - Medium Compliance Site (60% compliant)
  - Low Compliance Site (30% compliant)
  - Mixed Compliance Site (varied)

### 4. Configuration Options
- **Detection Confidence**: Adjust sensitivity (0.1-0.9)
- **Processing Settings**: Optimize for speed vs accuracy
- **Display Options**: Customize visual indicators

## 🔧 Technical Architecture

### Detection Pipeline
```
Image Input → Person Detection (YOLOv8) → ROI Extraction → 
Safety Equipment Analysis → Compliance Calculation → 
Visual Annotation → Report Generation
```

### Safety Equipment Detection Methods

#### 1. AI-Based Detection (Primary)
- **YOLOv8 Models**: Pretrained object detection
- **Person Detection**: COCO dataset person class
- **Equipment Classification**: Custom detection logic

#### 2. Computer Vision Enhancement (Secondary)
- **Color-based Detection**: HSV color space analysis
- **Morphological Operations**: Noise reduction and shape refinement
- **Region-based Analysis**: Focused detection areas (head, torso)

#### 3. Multi-modal Fusion
- **Confidence Scoring**: Weighted detection results
- **Ensemble Methods**: Combine AI and CV approaches
- **Validation Logic**: Cross-verification of detections

### Compliance Scoring Algorithm
```python
compliance_rate = (helmet_detected + mask_detected + vest_detected) / 3

Status Classification:
- Fully Compliant: 100% (3/3 items)
- Partial Compliance: 33-99% (1-2/3 items)  
- Violation: 0-33% (0-1/3 items)
```

## 🎨 Customization

### Adjusting Detection Sensitivity
```python
# In app.py, modify confidence thresholds
confidence_threshold = 0.4  # Range: 0.1 to 0.9

# In utils.py, adjust color detection thresholds
helmet_percentage > 0.08  # 8% of head region
vest_percentage > 0.15    # 15% of torso region
```

### Adding New Safety Equipment
1. **Extend the detection function**:
```python
def detect_new_equipment(person_roi):
    # Add your detection logic
    return equipment_detected
```

2. **Update compliance calculation**:
```python
compliance_items = {
    'helmet': helmet_detected,
    'mask': mask_detected,
    'vest': vest_detected,
    'new_equipment': new_equipment_detected  # Add here
}
```

### Custom Color Ranges
```python
# In utils.py, modify safety_colors dictionary
self.safety_colors = {
    'custom_helmet': [(hue_min, sat_min, val_min), (hue_max, sat_max, val_max)],
    # Add your custom color ranges
}
```

### Styling Customization
- **CSS Modifications**: Edit the `st.markdown()` CSS in `app.py`
- **Color Schemes**: Modify color variables in the CSS
- **Layout Changes**: Adjust Streamlit column layouts
- **Branding**: Add company logos and custom styling

## 📊 Performance Optimization

### Speed Optimization
```python
# Use smaller YOLOv8 models
model = YOLO('yolov8n.pt')  # Nano (fastest)
model = YOLO('yolov8s.pt')  # Small (balanced)
model = YOLO('yolov8m.pt')  # Medium (more accurate)
```

### Memory Optimization
- **Image Resizing**: Resize large images before processing
- **Batch Processing**: Process multiple images efficiently
- **Model Caching**: Use Streamlit's `@st.cache_resource`

### GPU Acceleration
```bash
# Install CUDA-enabled PyTorch
pip install torch torchvision --extra-index-url https://download.pytorch.org/whl/cu118
```

## 🔍 Troubleshooting

### Common Issues

**1. Models not loading**
```bash
# Solution: Check internet connection and reinstall
pip uninstall ultralytics
pip install ultralytics
```

**2. Camera not working**
- Grant browser camera permissions
- Close other applications using the camera
- Try different browsers (Chrome recommended)

**3. Performance issues**
- Reduce image size before upload
- Lower detection confidence threshold
- Use YOLOv8n (nano) model for speed
- Close unnecessary applications

**4. Import errors**
```bash
# Solution: Reinstall dependencies
pip install --upgrade pip
pip install -r requirements.txt --force-reinstall
```

### Performance Benchmarks
- **YOLOv8n**: ~50ms per image (CPU), ~10ms (GPU)
- **YOLOv8s**: ~100ms per image (CPU), ~15ms (GPU)
- **Memory Usage**: 2-4GB RAM typical
- **Model Size**: YOLOv8n (~6MB), YOLOv8s (~22MB)

## 🔬 Advanced Configuration

### Custom Model Integration
```python
# Replace with your custom trained model
model = YOLO('path/to/your/custom_model.pt')

# For safety equipment specific models
helmet_model = YOLO('helmet_detection_model.pt')
vest_model = YOLO('vest_detection_model.pt')
```

### Database Integration
```python
# Add database logging (example with SQLite)
import sqlite3

def log_compliance_result(results):
    conn = sqlite3.connect('compliance_log.db')
    # Insert compliance data
    conn.close()
```

### API Integration
```python
# Add REST API endpoints
from fastapi import FastAPI
app = FastAPI()

@app.post("/detect")
async def detect_compliance(image: UploadFile):
    # Process image and return results
    return compliance_results
```

## 📈 Future Enhancements

### Planned Features
- **🎥 Real-time Video Processing**: Live camera feed analysis
- **📱 Mobile App**: React Native mobile application
- **🌐 Multi-site Dashboard**: Centralized monitoring system
- **📧 Alert System**: Automated violation notifications
- **📊 Historical Analytics**: Trend analysis and reporting
- **🤖 Custom Model Training**: Site-specific model fine-tuning

### Integration Possibilities
- **Security Cameras**: RTSP stream processing
- **IoT Sensors**: Environmental data correlation
- **HR Systems**: Personnel tracking integration
- **Compliance Software**: Regulatory reporting tools
- **Mobile Devices**: Edge deployment capabilities

## 📄 File Structure
```
safety-compliance-detection/
├── app.py                 # Main Streamlit application
├── utils.py              # Detection utilities and helpers
├── requirements.txt      # Python dependencies
├── README.md            # This documentation
├── run_instructions.txt # Quick start guide
├── models/              # Custom model storage (optional)
├── data/               # Sample images and test data
├── exports/            # Generated reports and exports
└── logs/              # Application logs
```

## 🤝 Contributing

### Development Setup
1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests for new functionality
5. Submit a pull request

### Code Style
- Follow PEP 8 guidelines
- Use type hints where possible
- Add docstrings for functions
- Comment complex logic sections

## 📜 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🆘 Support

### Getting Help
1. **Documentation**: Check this README first
2. **Issues**: Create GitHub issues for bugs
3. **Discussions**: Use GitHub discussions for questions
4. **Email**: Contact support for enterprise inquiries

### System Requirements
- **Minimum**: Python 3.8, 4GB RAM, 2GB storage
- **Recommended**: Python 3.9+, 8GB RAM, 4GB storage, GPU
- **Optimal**: Python 3.10+, 16GB RAM, SSD storage, CUDA GPU

## 🏆 Acknowledgments

- **Ultralytics**: YOLOv8 implementation
- **Streamlit**: Web application framework
- **OpenCV**: Computer vision library
- **PyTorch**: Deep learning framework
- **Construction Industry**: Safety standards and requirements

---

**Built with ❤️ for construction site safety**

*Ensuring workplace safety through AI-powered monitoring and compliance detection.*