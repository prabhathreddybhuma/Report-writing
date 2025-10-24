#!/usr/bin/env python3
"""
Test script for the Cybersecurity ML Framework Web Interface
"""

import requests
import json
import time
import subprocess
import sys
from pathlib import Path

def test_web_interface():
    """Test the web interface functionality."""
    print("🧪 Testing Cybersecurity ML Framework Web Interface")
    print("=" * 60)
    
    base_url = "http://localhost:5000"
    
    try:
        # Test 1: Check if server is running
        print("1. Testing server status...")
        response = requests.get(f"{base_url}/api/get_status", timeout=5)
        if response.status_code == 200:
            print("   ✅ Server is running")
            status_data = response.json()
            print(f"   📊 Models: {len(status_data['models_trained'])}")
            print(f"   📊 Datasets: {len(status_data['datasets_available'])}")
        else:
            print(f"   ❌ Server returned status code: {response.status_code}")
            return False
        
        # Test 2: Generate synthetic data
        print("\n2. Testing data generation...")
        data_payload = {
            "n_samples": 500,
            "n_features": 10,
            "n_classes": 2
        }
        
        response = requests.post(
            f"{base_url}/api/generate_data",
            json=data_payload,
            timeout=10
        )
        
        if response.status_code == 200:
            result = response.json()
            if result['success']:
                print("   ✅ Data generation successful")
                print(f"   📊 Generated {result['dataset_info']['n_samples']} samples")
            else:
                print(f"   ❌ Data generation failed: {result['error']}")
                return False
        else:
            print(f"   ❌ Data generation request failed: {response.status_code}")
            return False
        
        # Test 3: Train a model
        print("\n3. Testing model training...")
        model_payload = {
            "model_type": "random_forest",
            "params": {
                "n_estimators": 50,
                "max_depth": 5
            }
        }
        
        response = requests.post(
            f"{base_url}/api/train_model",
            json=model_payload,
            timeout=15
        )
        
        if response.status_code == 200:
            result = response.json()
            if result['success']:
                accuracy = result['results']['accuracy']
                print(f"   ✅ Model training successful")
                print(f"   📊 Accuracy: {accuracy:.4f}")
            else:
                print(f"   ❌ Model training failed: {result['error']}")
                return False
        else:
            print(f"   ❌ Model training request failed: {response.status_code}")
            return False
        
        # Test 4: Make a prediction
        print("\n4. Testing prediction...")
        prediction_payload = {
            "model_type": "random_forest",
            "input_data": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]
        }
        
        response = requests.post(
            f"{base_url}/api/predict",
            json=prediction_payload,
            timeout=10
        )
        
        if response.status_code == 200:
            result = response.json()
            if result['success']:
                prediction = result['prediction']
                print(f"   ✅ Prediction successful")
                print(f"   📊 Prediction: {prediction}")
            else:
                print(f"   ❌ Prediction failed: {result['error']}")
                return False
        else:
            print(f"   ❌ Prediction request failed: {response.status_code}")
            return False
        
        # Test 5: Generate visualization
        print("\n5. Testing visualization...")
        viz_payload = {
            "plot_type": "confusion_matrix",
            "model_type": "random_forest"
        }
        
        response = requests.post(
            f"{base_url}/api/visualize",
            json=viz_payload,
            timeout=10
        )
        
        if response.status_code == 200:
            result = response.json()
            if result['success']:
                print("   ✅ Visualization generation successful")
                print(f"   📊 Plot type: {result['plot_type']}")
            else:
                print(f"   ❌ Visualization failed: {result['error']}")
                return False
        else:
            print(f"   ❌ Visualization request failed: {response.status_code}")
            return False
        
        print("\n" + "=" * 60)
        print("🎉 ALL TESTS PASSED! Web interface is working correctly!")
        print("=" * 60)
        return True
        
    except requests.exceptions.ConnectionError:
        print("❌ Cannot connect to server. Make sure the web interface is running.")
        print("   Start it with: python start_web_interface.py")
        return False
    except Exception as e:
        print(f"❌ Test failed with error: {e}")
        return False

def start_server_if_needed():
    """Start the server if it's not running."""
    try:
        response = requests.get("http://localhost:5000/api/get_status", timeout=2)
        print("✅ Server is already running")
        return True
    except:
        print("🚀 Starting web server...")
        try:
            # Start server in background
            subprocess.Popen([
                sys.executable, 
                str(Path(__file__).parent / "frontend" / "app.py")
            ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            
            # Wait for server to start
            print("⏳ Waiting for server to start...")
            for i in range(10):
                time.sleep(1)
                try:
                    response = requests.get("http://localhost:5000/api/get_status", timeout=2)
                    print("✅ Server started successfully")
                    return True
                except:
                    continue
            
            print("❌ Server failed to start")
            return False
            
        except Exception as e:
            print(f"❌ Error starting server: {e}")
            return False

def main():
    """Main function."""
    print("🔒 Cybersecurity ML Framework - Web Interface Test")
    print("=" * 60)
    
    # Start server if needed
    if not start_server_if_needed():
        return
    
    # Run tests
    success = test_web_interface()
    
    if success:
        print("\n🌐 Web Interface is ready!")
        print("   Open your browser and go to: http://localhost:5000")
        print("   Features available:")
        print("   • Generate synthetic cybersecurity datasets")
        print("   • Train ML models (Random Forest, SVM)")
        print("   • Train anomaly detectors")
        print("   • Make predictions on new data")
        print("   • Generate visualizations")
    else:
        print("\n❌ Web Interface test failed. Check the errors above.")

if __name__ == "__main__":
    main()
