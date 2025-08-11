#!/usr/bin/env python3
"""
Agricultural Vision System - Easy Setup Script
TIFAN 2025 Winner - Automated setup and verification

This script helps set up the agricultural vision system and verifies
all dependencies are correctly installed.
"""

import os
import sys
import subprocess
import json
from pathlib import Path

def print_header():
    """Print welcome header"""
    print("🌱" + "="*60 + "🌱")
    print("  AGRICULTURAL VISION SYSTEM - SETUP")
    print("  TIFAN 2025 Winner | AI-Powered Plant Detection")
    print("🌱" + "="*60 + "🌱")
    print()

def check_python_version():
    """Check if Python version is compatible"""
    print("🔍 Checking Python version...")
    
    version = sys.version_info
    if version.major < 3 or (version.major == 3 and version.minor < 8):
        print("❌ Python 3.8+ is required. Current version:", sys.version)
        return False
    
    print(f"✅ Python {version.major}.{version.minor}.{version.micro} is compatible")
    return True

def create_directories():
    """Create necessary directories"""
    print("📁 Creating project directories...")
    
    directories = [
        "data",
        "data/sample_data", 
        "data/test_images",
        "models",
        "docs/images",
        "mobile_app",
        "scripts"
    ]
    
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
        print(f"   ✅ Created: {directory}/")
    
    print("📁 All directories created successfully!")

def install_requirements():
    """Install Python requirements"""
    print("📦 Installing Python dependencies...")
    
    if not Path("requirements.txt").exists():
        print("❌ requirements.txt not found!")
        return False
    
    try:
        subprocess.check_call([
            sys.executable, "-m", "pip", "install", "-r", "requirements.txt"
        ])
        print("✅ All Python dependencies installed successfully!")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to install dependencies: {e}")
        return False

def check_model_file():
    """Check if YOLO model file exists"""
    print("🤖 Checking for YOLO model...")
    
    model_path = Path("models/best1.pt")
    if model_path.exists():
        size_mb = model_path.stat().st_size / (1024 * 1024)
        print(f"✅ Model found: {model_path} ({size_mb:.1f} MB)")
        return True
    else:
        print("⚠️  Model file not found at models/best1.pt")
        print("   Please copy your YOLO model to the models/ directory")
        print("   You can still run the system - it will show an error message")
        return False

def test_camera():
    """Test camera access"""
    print("📷 Testing camera access...")
    
    try:
        import cv2
        cap = cv2.VideoCapture(0)
        
        if cap.isOpened():
            print("✅ Camera access successful!")
            cap.release()
            return True
        else:
            print("⚠️  Camera not accessible")
            print("   Please check:")
            print("   - Camera is connected")
            print("   - Camera permissions are granted")
            print("   - No other app is using the camera")
            return False
    except ImportError:
        print("⚠️  OpenCV not installed - skipping camera test")
        return False

def test_dependencies():
    """Test that all major dependencies can be imported"""
    print("🔧 Testing dependencies...")
    
    dependencies = [
        ("flask", "Flask web framework"),
        ("cv2", "OpenCV computer vision"),
        ("torch", "PyTorch deep learning"),
        ("ultralytics", "YOLO object detection"),
        ("pandas", "Data analysis"),
        ("numpy", "Numerical computing")
    ]
    
    all_good = True
    for module, description in dependencies:
        try:
            __import__(module)
            print(f"   ✅ {module} - {description}")
        except ImportError:
            print(f"   ❌ {module} - {description} (MISSING)")
            all_good = False
    
    return all_good

def create_sample_data():
    """Create sample data for testing"""
    print("📊 Creating sample data...")
    
    try:
        import pandas as pd
        import random
        from datetime import datetime, timedelta
        
        # Generate sample sapling data
        sample_data = []
        base_time = datetime.now() - timedelta(hours=1)
        
        for i in range(1, 21):  # 20 sample saplings
            position = "Left" if i % 2 == 0 else "Right"
            is_proper = random.random() < 0.7  # 70% properly planted
            status = "Properly Planted" if is_proper else "Tilted"
            
            angle = random.uniform(80, 100) if is_proper else random.choice([
                random.uniform(30, 50), random.uniform(130, 150)
            ])
            
            sample_data.append({
                'timestamp': (base_time + timedelta(minutes=i*2)).strftime('%Y-%m-%d %H:%M:%S'),
                'frame_number': i,
                'sapling_id': i,
                'plantation_status': status,
                'angle': round(angle, 2),
                'confidence': round(random.uniform(0.7, 0.95), 2),
                'position': position
            })
        
        # Save to CSV
        df = pd.DataFrame(sample_data)
        csv_path = f"data/sample_data/sample_saplings_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        df.to_csv(csv_path, index=False)
        
        print(f"✅ Sample data created: {csv_path}")
        print(f"   - 20 sample saplings")
        print(f"   - {sum(1 for d in sample_data if d['plantation_status'] == 'Properly Planted')} properly planted")
        print(f"   - {sum(1 for d in sample_data if d['plantation_status'] == 'Tilted')} tilted")
        
        return True
    except Exception as e:
        print(f"⚠️  Could not create sample data: {e}")
        return False

def create_run_script():
    """Create a convenient run script"""
    print("🚀 Creating run script...")
    
    script_content = '''#!/usr/bin/env python3
"""
Quick start script for Agricultural Vision System
"""

import os
import sys

def main():
    print("🌱 Starting Agricultural Vision System...")
    print("TIFAN 2025 Winner - AI-Powered Plant Detection")
    print("="*50)
    
    # Check if we're in the right directory
    if not os.path.exists("backend/combined.py"):
        print("❌ Please run this script from the project root directory")
        return
    
    # Check if model exists
    if not os.path.exists("models/best1.pt"):
        print("⚠️  Model file not found at models/best1.pt")
        print("   The system will run but detection may not work")
        print("   Please copy your YOLO model to models/best1.pt")
        print()
    
    print("🚀 Starting the system...")
    print("   - Flask server will start on http://localhost:5000")
    print("   - Computer vision window will open")
    print("   - Press 'q' in the vision window to quit")
    print("   - Press Ctrl+C in terminal to stop")
    print()
    
    # Import and run the main system
    sys.path.insert(0, 'backend')
    
    try:
        from combined import *
        # This will start both Flask server and detection
    except ImportError as e:
        print(f"❌ Import error: {e}")
        print("Please ensure all dependencies are installed:")
        print("pip install -r requirements.txt")
    except KeyboardInterrupt:
        print("\\n👋 System stopped by user")
    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    main()
'''
    
    with open("run_system.py", "w") as f:
        f.write(script_content)
    
    # Make executable on Unix systems
    if os.name != 'nt':  # Not Windows
        os.chmod("run_system.py", 0o755)
    
    print("✅ Created run_system.py - easy startup script")

def print_next_steps():
    """Print what to do next"""
    print("\n🎉" + "="*60 + "🎉")
    print("  SETUP COMPLETE!")
    print("🎉" + "="*60 + "🎉")
    print()
    print("🚀 NEXT STEPS:")
    print()
    print("1. 📁 Copy your YOLO model:")
    print("   - Place your best1.pt file in the models/ directory")
    print("   - Or download a pre-trained YOLO model")
    print()
    print("2. 🏃 Run the system:")
    print("   python run_system.py")
    print("   OR")
    print("   python backend/combined.py")
    print()
    print("3. 🌐 Access the web interface:")
    print("   Open browser: http://localhost:5000")
    print()
    print("4. 📱 Connect your mobile app:")
    print("   Point your Flutter app to: http://YOUR_IP:5000")
    print()
    print("📚 DOCUMENTATION:")
    print("   - README.md - Complete project documentation")
    print("   - docs/ - Additional guides and images")
    print()
    print("🐛 TROUBLESHOOTING:")
    print("   - Check that camera is connected and accessible")
    print("   - Ensure no other apps are using the camera")
    print("   - Run in virtual environment if dependencies conflict")
    print()
    print("🏆 TIFAN 2025 Winner | 95% Detection Accuracy")
    print("📧 Support: rishabhsinha1712@gmail.com")

def main():
    """Main setup function"""
    print_header()
    
    # Run setup steps
    steps = [
        ("Python Version", check_python_version),
        ("Create Directories", create_directories),
        ("Install Dependencies", install_requirements),
        ("Check Model File", check_model_file),
        ("Test Dependencies", test_dependencies),
        ("Test Camera", test_camera),
        ("Create Sample Data", create_sample_data),
        ("Create Run Script", create_run_script),
    ]
    
    results = {}
    for step_name, step_func in steps:
        print(f"\n{'='*20} {step_name} {'='*20}")
        try:
            results[step_name] = step_func()
        except Exception as e:
            print(f"❌ Error in {step_name}: {e}")
            results[step_name] = False
    
    # Summary
    print(f"\n{'='*20} SETUP SUMMARY {'='*20}")
    for step_name, success in results.items():
        status = "✅" if success else "⚠️"
        print(f"{status} {step_name}")
    
    print_next_steps()

if __name__ == "__main__":
    main()