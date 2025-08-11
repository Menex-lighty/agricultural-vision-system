# 🌱 AI-Powered Agricultural Monitoring System
**TIFAN 2025 Winner - Budding Team Award | 3rd Place SRIJAN 2025**

## 🎯 Project Overview
Real-time plant detection and monitoring system for autonomous agricultural machinery. Integrates computer vision, IoT sensors, and mobile applications to achieve 95% planting verification accuracy and reduce planting errors by 40%.

## 🏆 Achievements
- **Budding Team Award** - TIFAN 2025 (SAE India Agricultural Innovation)
- **3rd Place** - Rural Technology Category, SRIJAN 2025
- **95% Detection Accuracy** with real-time processing
- **40% Reduction** in planting errors

## 🚀 Features
- **Real-time Plant Detection** using custom YOLO model
- **Cross-platform Mobile App** built with Flutter
- **REST API Backend** with Flask
- **Live Data Streaming** and field monitoring
- **CSV Data Logging** with automatic sapling classification
- **Web Dashboard** for real-time statistics

## 🛠️ Tech Stack
- **Backend**: Python, Flask, OpenCV, PyTorch
- **AI/ML**: YOLO v8, Computer Vision, Custom Object Detection
- **Frontend**: Flutter (Mobile), HTML/CSS/JavaScript (Web)
- **Database**: CSV-based data logging, Pandas for analysis
- **Hardware Integration**: Arduino, Camera systems

## 📁 Repository Structure
```
agricultural-vision-system/
├── README.md                          # This file
├── requirements.txt                   # Python dependencies
├── .gitignore                        # Git ignore file
├── 
├── backend/
│   ├── combined.py                   # Main server + detection system
│   ├── api_server.py                 # Standalone Flask API
│   ├── sapling_detection.py          # Standalone detection script
│   └── utils/
│       ├── detection_utils.py        # Detection helper functions
│       └── data_processing.py        # CSV and data utilities
├── 
├── models/
│   ├── best1.pt                     # YOLO model weights (if < 100MB)
│   ├── model_info.md                # Model architecture details
│   └── training/                    # Training scripts and configs
│       ├── train_config.yaml
│       └── dataset_preparation.py
├── 
├── mobile_app/                       # Flutter application
│   ├── lib/
│   │   ├── main.dart
│   │   ├── services/
│   │   │   ├── api_service.dart
│   │   │   └── data_models.dart
│   │   ├── screens/
│   │   │   ├── dashboard_screen.dart
│   │   │   ├── monitoring_screen.dart
│   │   │   └── settings_screen.dart
│   │   └── widgets/
│   ├── pubspec.yaml
│   └── android/
├── 
├── web_dashboard/                    # Web interface
│   ├── static/
│   │   ├── css/
│   │   ├── js/
│   │   └── images/
│   └── templates/
│       ├── index.html
│       └── dashboard.html
├── 
├── data/
│   ├── sample_data/                 # Sample CSV files
│   └── test_images/                 # Test images for demo
├── 
├── docs/
│   ├── API_DOCUMENTATION.md
│   ├── SETUP_GUIDE.md
│   ├── HARDWARE_INTEGRATION.md
│   └── images/                      # Screenshots and diagrams
│       ├── system_architecture.png
│       ├── mobile_app_demo.png
│       └── detection_results.png
└── 
└── scripts/
    ├── setup.sh                     # Linux/Mac setup script
    ├── setup.bat                    # Windows setup script
    └── run_demo.py                  # Quick demo script
```

## 🔧 Quick Start

### Prerequisites
- Python 3.8+
- Flutter SDK (for mobile app)
- Webcam or IP camera
- CUDA-compatible GPU (recommended)

### Installation
```bash
# Clone the repository
git clone https://github.com/yourusername/agricultural-vision-system.git
cd agricultural-vision-system

# Install Python dependencies
pip install -r requirements.txt

# Run the combined system (Flask server + detection)
python backend/combined.py

# Or run components separately
python backend/api_server.py          # Flask API only
python backend/sapling_detection.py   # Detection only
```

### Mobile App Setup
```bash
cd mobile_app
flutter pub get
flutter run
```

## 📊 System Architecture
```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Camera Feed   │───▶│  YOLO Detection │───▶│   Flask API     │
└─────────────────┘    └─────────────────┘    └─────────────────┘
                              │                        │
                              ▼                        ▼
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│  CSV Logging    │◀───│  Data Processing│◀───│  Flutter App    │
└─────────────────┘    └─────────────────┘    └─────────────────┘
```

## 🎥 Demo
### Detection in Action
- Real-time sapling classification (Properly Planted vs Tilted)
- Angle measurement with temporal smoothing
- Automatic ID assignment and CSV logging
- Live statistics and monitoring

### API Endpoints
- `GET /api/stats` - Current planting statistics
- `GET /api/saplings` - All sapling data
- `GET /api/simulate_detection` - Add simulated detection
- `GET /api/reset` - Reset with sample data

## 📈 Performance Metrics
- **Detection Accuracy**: 95%
- **Processing Speed**: 30+ FPS
- **Classification Categories**: Properly Planted, Tilted
- **Real-time Processing**: ✅
- **Mobile Integration**: ✅

## 🔬 Technical Details

### Computer Vision Pipeline
- **YOLO v8** custom model for plant detection
- **Keypoint detection** for stem orientation analysis
- **Temporal smoothing** for robust angle measurement
- **Multi-threading** for real-time performance

### Data Processing
- **Automatic classification** based on planting angle (60-120° = proper)
- **Weighted angle calculation** prioritizing stem bottom
- **CSV logging** with timestamp and metadata
- **Statistical analysis** and reporting

### Mobile Integration
- **Real-time updates** via REST API
- **Cross-platform** Flutter application
- **Offline capability** with local data caching
- **User-friendly interface** for field monitoring

## 🚀 Future Enhancements
- [ ] Multi-crop detection support
- [ ] Machine learning-based angle optimization
- [ ] GPS integration for field mapping
- [ ] Cloud deployment with AWS/Azure
- [ ] Advanced analytics dashboard
- [ ] Integration with farming equipment APIs

## 📄 License
MIT License - See [LICENSE](LICENSE) file for details

## 🤝 Contributing
Contributions welcome! Please read our [Contributing Guidelines](docs/CONTRIBUTING.md) first.

## 📞 Contact
**Rishabh Sinha** - Vision System Lead  
📧 rishabhsinha1712@gmail.com  
🔗 [LinkedIn](https://linkedin.com/in/rishabh-sinha)

---

*This project was developed for TIFAN 2025 (SAE India Agricultural Innovation Challenge) and showcases practical applications of AI in agriculture.*
