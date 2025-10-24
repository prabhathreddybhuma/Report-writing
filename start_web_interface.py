#!/usr/bin/env python3
"""
Startup script for Cybersecurity ML Framework Web Interface
"""

import os
import sys
import subprocess
from pathlib import Path

def check_dependencies():
    """Check if required packages are installed."""
    required_packages = ['flask', 'numpy', 'pandas', 'scikit-learn', 'matplotlib', 'seaborn']
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package)
        except ImportError:
            missing_packages.append(package)
    
    if missing_packages:
        print(f"❌ Missing packages: {', '.join(missing_packages)}")
        print("Installing missing packages...")
        subprocess.check_call([sys.executable, '-m', 'pip', 'install'] + missing_packages)
        print("✅ All packages installed successfully!")
    else:
        print("✅ All required packages are available!")

def start_web_interface():
    """Start the web interface."""
    print("🚀 Starting Cybersecurity ML Framework Web Interface...")
    print("=" * 60)
    
    # Change to frontend directory
    frontend_dir = Path(__file__).parent / 'frontend'
    os.chdir(frontend_dir)
    
    # Start Flask app
    try:
        from app import app
        print("📊 Web interface is starting...")
        print("🌐 Open your browser and go to: http://localhost:5000")
        print("🛑 Press Ctrl+C to stop the server")
        print("=" * 60)
        
        app.run(debug=True, host='0.0.0.0', port=5000)
        
    except Exception as e:
        print(f"❌ Error starting web interface: {e}")
        print("Make sure you're in the correct directory and all dependencies are installed.")

def main():
    """Main function."""
    print("🔒 Cybersecurity ML Framework - Web Interface")
    print("=" * 60)
    
    # Check dependencies
    check_dependencies()
    
    # Start web interface
    start_web_interface()

if __name__ == "__main__":
    main()
