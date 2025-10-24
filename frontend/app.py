#!/usr/bin/env python3
"""
Cybersecurity ML Framework - Web Frontend

A Flask-based web interface for the cybersecurity ML framework.
"""

from flask import Flask, render_template, request, jsonify, send_file
import numpy as np
import pandas as pd
import json
import base64
import io
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier, IsolationForest
from sklearn.svm import SVC, OneClassSVM
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_classification
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_curve, roc_auc_score
import logging
from datetime import datetime
import os

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
app.secret_key = 'cybersecurity_ml_framework_2024'

# Global variables to store trained models
trained_models = {}
current_dataset = None

class MLModelManager:
    """Manager for ML models and operations."""
    
    def __init__(self):
        self.models = {}
        self.datasets = {}
        self.results = {}
    
    def generate_synthetic_data(self, n_samples=1000, n_features=20, n_classes=2):
        """Generate synthetic cybersecurity data."""
        X, y = make_classification(
            n_samples=n_samples, 
            n_features=n_features, 
            n_classes=n_classes,
            n_clusters_per_class=1, 
            random_state=42
        )
        
        feature_names = [f"Feature_{i}" for i in range(n_features)]
        class_names = ['Normal', 'Attack'] if n_classes == 2 else [f'Class_{i}' for i in range(n_classes)]
        
        return X, y, feature_names, class_names
    
    def train_classification_model(self, model_type, X_train, y_train, **params):
        """Train a classification model."""
        if model_type == 'random_forest':
            model = RandomForestClassifier(
                n_estimators=params.get('n_estimators', 100),
                max_depth=params.get('max_depth', None),
                random_state=42
            )
        elif model_type == 'svm':
            model = SVC(
                C=params.get('C', 1.0),
                kernel=params.get('kernel', 'rbf'),
                probability=True,
                random_state=42
            )
        else:
            raise ValueError(f"Unknown model type: {model_type}")
        
        model.fit(X_train, y_train)
        return model
    
    def train_anomaly_detector(self, detector_type, X_train, **params):
        """Train an anomaly detection model."""
        if detector_type == 'isolation_forest':
            detector = IsolationForest(
                contamination=params.get('contamination', 0.1),
                random_state=42
            )
        elif detector_type == 'one_class_svm':
            detector = OneClassSVM(
                nu=params.get('nu', 0.1),
                kernel=params.get('kernel', 'rbf')
            )
        else:
            raise ValueError(f"Unknown detector type: {detector_type}")
        
        detector.fit(X_train)
        return detector
    
    def evaluate_model(self, model, X_test, y_test):
        """Evaluate model performance."""
        predictions = model.predict(X_test)
        probabilities = model.predict_proba(X_test) if hasattr(model, 'predict_proba') else None
        
        accuracy = accuracy_score(y_test, predictions)
        
        results = {
            'accuracy': accuracy,
            'predictions': predictions.tolist(),
            'confusion_matrix': confusion_matrix(y_test, predictions).tolist()
        }
        
        if probabilities is not None:
            results['probabilities'] = probabilities.tolist()
            results['roc_auc'] = roc_auc_score(y_test, probabilities[:, 1]) if probabilities.shape[1] > 1 else 0.0
        
        return results

# Initialize model manager
model_manager = MLModelManager()

@app.route('/')
def index():
    """Main dashboard page."""
    return render_template('index.html')

@app.route('/api/generate_data', methods=['POST'])
def generate_data():
    """Generate synthetic cybersecurity data."""
    try:
        data = request.get_json()
        n_samples = data.get('n_samples', 1000)
        n_features = data.get('n_features', 20)
        n_classes = data.get('n_classes', 2)
        
        X, y, feature_names, class_names = model_manager.generate_synthetic_data(
            n_samples, n_features, n_classes
        )
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        # Store dataset
        model_manager.datasets['current'] = {
            'X_train': X_train.tolist(),
            'X_test': X_test.tolist(),
            'y_train': y_train.tolist(),
            'y_test': y_test.tolist(),
            'feature_names': feature_names,
            'class_names': class_names,
            'n_samples': n_samples,
            'n_features': n_features,
            'n_classes': n_classes
        }
        
        return jsonify({
            'success': True,
            'message': f'Generated dataset with {n_samples} samples, {n_features} features',
            'dataset_info': {
                'n_samples': n_samples,
                'n_features': n_features,
                'n_classes': n_classes,
                'train_samples': len(X_train),
                'test_samples': len(X_test),
                'feature_names': feature_names,
                'class_names': class_names
            }
        })
        
    except Exception as e:
        logger.error(f"Error generating data: {e}")
        return jsonify({'success': False, 'error': str(e)})

@app.route('/api/train_model', methods=['POST'])
def train_model():
    """Train a machine learning model."""
    try:
        data = request.get_json()
        model_type = data.get('model_type')
        params = data.get('params', {})
        
        if 'current' not in model_manager.datasets:
            return jsonify({'success': False, 'error': 'No dataset available. Please generate data first.'})
        
        dataset = model_manager.datasets['current']
        X_train = np.array(dataset['X_train'])
        y_train = np.array(dataset['y_train'])
        X_test = np.array(dataset['X_test'])
        y_test = np.array(dataset['y_test'])
        
        # Train model
        model = model_manager.train_classification_model(model_type, X_train, y_train, **params)
        
        # Evaluate model
        results = model_manager.evaluate_model(model, X_test, y_test)
        
        # Store model and results
        model_manager.models[model_type] = model
        model_manager.results[model_type] = results
        
        return jsonify({
            'success': True,
            'message': f'{model_type.replace("_", " ").title()} model trained successfully',
            'results': results
        })
        
    except Exception as e:
        logger.error(f"Error training model: {e}")
        return jsonify({'success': False, 'error': str(e)})

@app.route('/api/train_anomaly_detector', methods=['POST'])
def train_anomaly_detector():
    """Train an anomaly detection model."""
    try:
        data = request.get_json()
        detector_type = data.get('detector_type')
        params = data.get('params', {})
        
        if 'current' not in model_manager.datasets:
            return jsonify({'success': False, 'error': 'No dataset available. Please generate data first.'})
        
        dataset = model_manager.datasets['current']
        X_train = np.array(dataset['X_train'])
        X_test = np.array(dataset['X_test'])
        y_test = np.array(dataset['y_test'])
        
        # Use only normal samples for training (assuming class 0 is normal)
        y_train = np.array(dataset['y_train'])
        normal_mask = y_train == 0
        X_normal = X_train[normal_mask]
        
        # Train detector
        detector = model_manager.train_anomaly_detector(detector_type, X_normal, **params)
        
        # Evaluate detector
        predictions = detector.predict(X_test)
        scores = detector.score_samples(X_test) if hasattr(detector, 'score_samples') else None
        
        # Convert predictions to binary (1 for normal, 0 for anomaly)
        binary_predictions = (predictions == 1).astype(int)
        binary_y_test = (y_test == 0).astype(int)
        
        accuracy = accuracy_score(binary_y_test, binary_predictions)
        
        results = {
            'accuracy': accuracy,
            'predictions': binary_predictions.tolist(),
            'scores': scores.tolist() if scores is not None else None,
            'anomalies_detected': int(np.sum(binary_predictions == 0))
        }
        
        # Store detector and results
        model_manager.models[f'anomaly_{detector_type}'] = detector
        model_manager.results[f'anomaly_{detector_type}'] = results
        
        return jsonify({
            'success': True,
            'message': f'{detector_type.replace("_", " ").title()} anomaly detector trained successfully',
            'results': results
        })
        
    except Exception as e:
        logger.error(f"Error training anomaly detector: {e}")
        return jsonify({'success': False, 'error': str(e)})

@app.route('/api/predict', methods=['POST'])
def predict():
    """Make predictions on new data."""
    try:
        data = request.get_json()
        model_type = data.get('model_type')
        input_data = data.get('input_data')
        
        if model_type not in model_manager.models:
            return jsonify({'success': False, 'error': f'Model {model_type} not found. Please train the model first.'})
        
        model = model_manager.models[model_type]
        
        # Convert input data to numpy array
        if isinstance(input_data, list):
            X = np.array(input_data).reshape(1, -1)
        else:
            X = np.array([input_data]).reshape(1, -1)
        
        # Make prediction
        prediction = model.predict(X)[0]
        probability = model.predict_proba(X)[0].tolist() if hasattr(model, 'predict_proba') else None
        
        return jsonify({
            'success': True,
            'prediction': int(prediction),
            'probability': probability,
            'model_type': model_type
        })
        
    except Exception as e:
        logger.error(f"Error making prediction: {e}")
        return jsonify({'success': False, 'error': str(e)})

@app.route('/api/visualize', methods=['POST'])
def visualize():
    """Generate visualization plots."""
    try:
        data = request.get_json()
        plot_type = data.get('plot_type')
        model_type = data.get('model_type')
        
        if model_type not in model_manager.results:
            return jsonify({'success': False, 'error': f'No results found for model {model_type}'})
        
        results = model_manager.results[model_type]
        dataset = model_manager.datasets['current']
        
        # Create plot
        plt.figure(figsize=(10, 8))
        
        if plot_type == 'confusion_matrix':
            cm = np.array(results['confusion_matrix'])
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
            plt.title(f'{model_type.replace("_", " ").title()} - Confusion Matrix')
            plt.xlabel('Predicted')
            plt.ylabel('Actual')
            
        elif plot_type == 'feature_importance' and model_type in model_manager.models:
            model = model_manager.models[model_type]
            if hasattr(model, 'feature_importances_'):
                feature_importance = model.feature_importances_
                top_features = np.argsort(feature_importance)[-10:][::-1]
                
                plt.bar(range(len(top_features)), feature_importance[top_features])
                plt.title(f'{model_type.replace("_", " ").title()} - Top 10 Feature Importance')
                plt.xlabel('Feature Rank')
                plt.ylabel('Importance')
                plt.xticks(range(len(top_features)), [f'F{i}' for i in top_features])
        
        elif plot_type == 'roc_curve' and 'probabilities' in results:
            y_test = np.array(dataset['y_test'])
            probabilities = np.array(results['probabilities'])
            
            if probabilities.shape[1] > 1:
                fpr, tpr, _ = roc_curve(y_test, probabilities[:, 1])
                auc_score = roc_auc_score(y_test, probabilities[:, 1])
                
                plt.plot(fpr, tpr, label=f'ROC Curve (AUC = {auc_score:.3f})')
                plt.plot([0, 1], [0, 1], 'k--', label='Random Classifier')
                plt.xlabel('False Positive Rate')
                plt.ylabel('True Positive Rate')
                plt.title(f'{model_type.replace("_", " ").title()} - ROC Curve')
                plt.legend()
        
        # Convert plot to base64 string
        img_buffer = io.BytesIO()
        plt.savefig(img_buffer, format='png', dpi=300, bbox_inches='tight')
        img_buffer.seek(0)
        img_base64 = base64.b64encode(img_buffer.getvalue()).decode()
        plt.close()
        
        return jsonify({
            'success': True,
            'image': img_base64,
            'plot_type': plot_type
        })
        
    except Exception as e:
        logger.error(f"Error generating visualization: {e}")
        return jsonify({'success': False, 'error': str(e)})

@app.route('/api/get_status')
def get_status():
    """Get current system status."""
    return jsonify({
        'models_trained': list(model_manager.models.keys()),
        'datasets_available': list(model_manager.datasets.keys()),
        'results_available': list(model_manager.results.keys()),
        'timestamp': datetime.now().isoformat()
    })

if __name__ == '__main__':
    print("🚀 Starting Cybersecurity ML Framework Web Interface...")
    print("📊 Open your browser and go to: http://localhost:5000")
    print("🛑 Press Ctrl+C to stop the server")
    
    app.run(debug=True, host='0.0.0.0', port=5000)
