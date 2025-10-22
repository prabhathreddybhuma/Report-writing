# Cybersecurity ML Framework - Documentation

## Table of Contents
1. [Overview](#overview)
2. [Installation](#installation)
3. [Quick Start Guide](#quick-start-guide)
4. [Core Components](#core-components)
5. [Usage Examples](#usage-examples)
6. [API Reference](#api-reference)
7. [Configuration](#configuration)
8. [Performance Optimization](#performance-optimization)
9. [Troubleshooting](#troubleshooting)
10. [Contributing](#contributing)

## Overview

The Cybersecurity ML Framework is a comprehensive machine learning system designed for threat detection, anomaly analysis, and security protocol enhancement. It integrates advanced ML/DL methodologies including supervised, unsupervised, and ensemble approaches to detect, classify, and respond to cyber threats across large-scale network environments.

### Key Features

- **Hybrid ML-DL Models**: Random Forest, SVM, CNN, RNN, and Ensemble methods
- **Federated Learning**: Privacy-preserving distributed training
- **Explainable AI**: SHAP, LIME, and feature importance analysis
- **Real-time Monitoring**: Continuous threat detection pipeline
- **Anomaly Detection**: Multiple algorithms for threat identification
- **Adversarial Robustness**: Attack and defense mechanisms
- **Performance Analysis**: Comprehensive evaluation metrics

## Installation

### Prerequisites
- Python 3.8+
- CUDA-capable GPU (recommended for deep learning)
- 8GB+ RAM
- Linux/macOS/Windows

### Setup

```bash
# Clone the repository
git clone <repository-url>
cd cybersecurity-ml-framework

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Download datasets (optional - will be downloaded automatically on first run)
python scripts/download_datasets.py
```

### Docker Installation

```bash
# Build Docker image
docker build -t cybersecurity-ml .

# Run container
docker run -it --gpus all cybersecurity-ml
```

## Quick Start Guide

### 1. Basic Threat Detection

```python
from data.preprocessing import DataPreprocessor
from models.ml_models import RandomForestModel
from evaluation.performance_analysis import CybersecurityEvaluator

# Load and preprocess data
preprocessor = DataPreprocessor()
features, labels = preprocessor.load_nsl_kdd()
processed_features = preprocessor.preprocess_features(features, 'nsl_kdd')
processed_labels = preprocessor.preprocess_labels(labels, 'nsl_kdd')

# Split data
splits = preprocessor.split_data(processed_features, processed_labels)

# Train model
model = RandomForestModel()
model.train(splits['X_train'], splits['y_train'], splits['X_val'], splits['y_val'])

# Evaluate model
evaluator = CybersecurityEvaluator()
performance = evaluator.evaluate_threat_detection_performance(
    model, splits['X_test'], splits['y_test']
)

print(f"Model accuracy: {performance['basic_performance'].metrics.accuracy:.4f}")
```

### 2. Real-time Monitoring

```python
from real_time_monitoring.realtime_pipeline import RealTimeMonitoringPipeline
from anomaly_detection.anomaly_detection import IsolationForestDetector

# Create anomaly detector
detector = IsolationForestDetector()
detector.fit(normal_data)  # Train on normal data

# Create monitoring pipeline
pipeline = RealTimeMonitoringPipeline(detector)

# Start monitoring
pipeline.start_monitoring(update_interval=1.0)

# Monitor for 60 seconds
import time
time.sleep(60)

# Get statistics
stats = pipeline.get_monitoring_stats()
print(f"Detections: {stats['total_detections']}")
print(f"Active alerts: {stats['active_alerts']}")

# Stop monitoring
pipeline.stop_monitoring()
```

### 3. Federated Learning

```python
from federated_learning.federated_learning import PrivacyPreservingFederatedLearning
from federated_learning.federated_learning import create_simple_nn

# Create neural network
model = create_simple_nn(input_size=20, num_classes=2)

# Initialize federated learning
fl_system = PrivacyPreservingFederatedLearning(model, num_clients=3)

# Add client data
fl_system.add_client_data("client_1", X1, y1)
fl_system.add_client_data("client_2", X2, y2)
fl_system.add_client_data("client_3", X3, y3)

# Train federated model
training_history = fl_system.train_federated(num_rounds=10, enable_privacy=True)

print(f"Final accuracy: {training_history['global_accuracy'][-1]:.4f}")
```

## Core Components

### Data Preprocessing (`data/preprocessing.py`)

The data preprocessing module handles loading, cleaning, and feature engineering for cybersecurity datasets.

**Key Classes:**
- `DataPreprocessor`: Main preprocessing class
- Methods: `load_nsl_kdd()`, `load_cicids2017()`, `load_ton_iot()`

**Example:**
```python
preprocessor = DataPreprocessor()
features, labels = preprocessor.load_nsl_kdd()
processed_features = preprocessor.preprocess_features(features, 'nsl_kdd')
```

### ML Models (`models/ml_models.py`)

Comprehensive ML/DL model implementations for threat detection.

**Key Classes:**
- `RandomForestModel`: Random Forest classifier
- `SVMModel`: Support Vector Machine
- `CNNModel`: Convolutional Neural Network
- `RNNModel`: Recurrent Neural Network
- `EnsembleModel`: Ensemble of multiple models

**Example:**
```python
# Random Forest
rf_model = RandomForestModel()
rf_model.train(X_train, y_train, X_val, y_val)

# Ensemble
ensemble = EnsembleModel([rf_model, svm_model, cnn_model])
ensemble.train(X_train, y_train, X_val, y_val)
```

### Federated Learning (`federated_learning/federated_learning.py`)

Privacy-preserving distributed learning framework.

**Key Classes:**
- `PrivacyPreservingFederatedLearning`: Main federated learning system
- `FederatedClient`: Individual client implementation
- `FederatedServer`: Central server coordination

**Example:**
```python
fl_system = PrivacyPreservingFederatedLearning(model, num_clients=5)
fl_system.add_client_data("client_1", X1, y1)
training_history = fl_system.train_federated(num_rounds=10)
```

### Explainable AI (`explainable_ai/xai_components.py`)

Model interpretability and explanation tools.

**Key Classes:**
- `SHAPExplainer`: SHAP-based explanations
- `LIMEExplainer`: LIME-based explanations
- `ModelInterpretabilityDashboard`: Comprehensive dashboard

**Example:**
```python
dashboard = ModelInterpretabilityDashboard(model, feature_names, training_data)
explanation = dashboard.generate_comprehensive_explanation(X, y, instance_idx=0)
dashboard.plot_comprehensive_dashboard(explanation)
```

### Anomaly Detection (`anomaly_detection/anomaly_detection.py`)

Multiple anomaly detection algorithms for threat identification.

**Key Classes:**
- `IsolationForestDetector`: Isolation Forest algorithm
- `OneClassSVMDetector`: One-Class SVM
- `AutoencoderDetector`: Autoencoder-based detection
- `CybersecurityAnomalyDetectionSystem`: Main system

**Example:**
```python
system = create_default_anomaly_detection_system()
system.train_system(X_normal, X_mixed, y_mixed)
analysis_results = system.comprehensive_analysis(X_test)
```

### Real-time Monitoring (`real_time_monitoring/realtime_pipeline.py`)

Continuous monitoring and threat detection pipeline.

**Key Classes:**
- `RealTimeMonitoringPipeline`: Main monitoring system
- `NetworkTrafficCollector`: Network data collection
- `SystemLogCollector`: System log collection
- `AlertManager`: Alert management

**Example:**
```python
pipeline = RealTimeMonitoringPipeline(detector, classifier)
pipeline.start_monitoring(update_interval=1.0)
```

### Performance Analysis (`evaluation/performance_analysis.py`)

Comprehensive evaluation metrics and benchmarking tools.

**Key Classes:**
- `PerformanceAnalyzer`: Performance analysis
- `BenchmarkSuite`: Model benchmarking
- `CybersecurityEvaluator`: Specialized cybersecurity evaluation

**Example:**
```python
evaluator = CybersecurityEvaluator()
benchmark_results = evaluator.benchmark_suite.run_benchmark(
    models, X_train, y_train, X_test, y_test
)
```

### Adversarial Robustness (`adversarial_robustness/adversarial_testing.py`)

Adversarial attack and defense mechanisms.

**Key Classes:**
- `FGSMAttack`: Fast Gradient Sign Method attack
- `PGDAttack`: Projected Gradient Descent attack
- `AdversarialTraining`: Adversarial training defense
- `AdversarialRobustnessTester`: Comprehensive testing

**Example:**
```python
tester = AdversarialRobustnessTester()
fgsm_attack = FGSMAttack(epsilon=0.1)
attack_result = tester.test_attack(model, X_test, y_test, fgsm_attack)
```

## Usage Examples

### Example 1: Complete Threat Detection Pipeline

```python
import numpy as np
from data.preprocessing import DataPreprocessor
from models.ml_models import create_ensemble_models
from evaluation.performance_analysis import CybersecurityEvaluator
from explainable_ai.xai_components import ModelInterpretabilityDashboard

def complete_threat_detection_pipeline():
    """Complete threat detection pipeline example."""
    
    # 1. Data preprocessing
    print("Step 1: Data Preprocessing")
    preprocessor = DataPreprocessor()
    features, labels = preprocessor.load_nsl_kdd()
    
    processed_features = preprocessor.preprocess_features(features, 'nsl_kdd')
    processed_labels = preprocessor.preprocess_labels(labels, 'nsl_kdd')
    selected_features = preprocessor.feature_selection(processed_features, processed_labels, 'nsl_kdd')
    
    splits = preprocessor.split_data(selected_features, processed_labels)
    
    # 2. Model training
    print("Step 2: Model Training")
    ensemble = create_ensemble_models(selected_features.shape[1:], 2)
    ensemble.train(splits['X_train'], splits['y_train'], splits['X_val'], splits['y_val'])
    
    # 3. Model evaluation
    print("Step 3: Model Evaluation")
    evaluator = CybersecurityEvaluator()
    performance = evaluator.evaluate_threat_detection_performance(
        ensemble, splits['X_test'], splits['y_test']
    )
    
    print(f"Model Accuracy: {performance['basic_performance'].metrics.accuracy:.4f}")
    print(f"F1-Score: {performance['basic_performance'].metrics.f1_score:.4f}")
    
    # 4. Model interpretation
    print("Step 4: Model Interpretation")
    dashboard = ModelInterpretabilityDashboard(
        ensemble, 
        feature_names=selected_features.columns.tolist(),
        training_data=splits['X_train']
    )
    
    explanation = dashboard.generate_comprehensive_explanation(
        splits['X_test'], splits['y_test'], instance_idx=0
    )
    
    report = dashboard.generate_explanation_report(explanation)
    print(report)
    
    return ensemble, performance, explanation

# Run the pipeline
model, performance, explanation = complete_threat_detection_pipeline()
```

### Example 2: Federated Learning for Privacy-Preserving Training

```python
from federated_learning.federated_learning import PrivacyPreservingFederatedLearning, create_simple_nn
import numpy as np

def federated_learning_example():
    """Federated learning example for privacy-preserving training."""
    
    # Create neural network
    model = create_simple_nn(input_size=20, num_classes=2)
    
    # Initialize federated learning system
    fl_system = PrivacyPreservingFederatedLearning(model, num_clients=5)
    
    # Generate synthetic client data (in practice, this would be real data from different organizations)
    np.random.seed(42)
    
    clients_data = []
    for i in range(5):
        # Each client has different data distribution
        X_client = np.random.randn(200, 20) + np.random.randn(1, 20) * 0.5
        y_client = np.random.randint(0, 2, 200)
        clients_data.append((X_client, y_client))
    
    # Add client data to federated system
    for i, (X_client, y_client) in enumerate(clients_data):
        fl_system.add_client_data(f"client_{i+1}", X_client, y_client)
    
    # Train federated model
    print("Starting federated training...")
    training_history = fl_system.train_federated(num_rounds=10, enable_privacy=True)
    
    # Print results
    print(f"Federated training completed!")
    print(f"Final global accuracy: {training_history['global_accuracy'][-1]:.4f}")
    print(f"Privacy budget remaining: {training_history['privacy_budget_used'][-1]:.4f}")
    
    # Get trained global model
    global_model = fl_system.get_global_model()
    
    return global_model, training_history

# Run federated learning
global_model, history = federated_learning_example()
```

### Example 3: Real-time Cybersecurity Monitoring

```python
from real_time_monitoring.realtime_pipeline import RealTimeMonitoringPipeline
from anomaly_detection.anomaly_detection import IsolationForestDetector
from models.ml_models import RandomForestModel
import time
import numpy as np

def real_time_monitoring_example():
    """Real-time cybersecurity monitoring example."""
    
    # Generate synthetic normal data for training
    np.random.seed(42)
    normal_data = np.random.randn(1000, 10)
    
    # Create anomaly detector
    detector = IsolationForestDetector(contamination=0.1)
    detector.fit(normal_data)
    
    # Create threat classifier (simplified)
    classifier = RandomForestModel()
    X_train = np.random.randn(500, 10)
    y_train = np.random.randint(0, 5, 500)
    classifier.train(X_train, y_train)
    
    # Create monitoring pipeline
    pipeline = RealTimeMonitoringPipeline(detector, classifier)
    
    try:
        # Start monitoring
        print("Starting real-time monitoring...")
        pipeline.start_monitoring(update_interval=2.0)
        
        # Monitor for 30 seconds
        print("Monitoring for 30 seconds...")
        for i in range(15):  # 15 iterations of 2 seconds each
            time.sleep(2)
            
            # Get current statistics
            stats = pipeline.get_monitoring_stats()
            print(f"Iteration {i+1}: Detections={stats['total_detections']}, Alerts={stats['active_alerts']}")
            
            # Get recent alerts
            alerts = pipeline.get_recent_alerts(limit=3)
            for alert in alerts:
                print(f"  Alert: {alert.description} (Severity: {alert.severity})")
    
    finally:
        # Stop monitoring
        print("Stopping monitoring...")
        pipeline.stop_monitoring()
        
        # Final statistics
        final_stats = pipeline.get_monitoring_stats()
        print(f"Final Statistics:")
        print(f"  Total Detections: {final_stats['total_detections']}")
        print(f"  Active Alerts: {final_stats['active_alerts']}")
        print(f"  Collectors Status: {final_stats['collectors_status']}")

# Run real-time monitoring
real_time_monitoring_example()
```

### Example 4: Adversarial Robustness Testing

```python
from adversarial_robustness.adversarial_testing import (
    AdversarialRobustnessTester, FGSMAttack, PGDAttack, 
    AdversarialTraining, DefensiveDistillation
)
from models.ml_models import RandomForestModel
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split

def adversarial_robustness_example():
    """Adversarial robustness testing example."""
    
    # Generate synthetic data
    X, y = make_classification(n_samples=2000, n_features=20, n_classes=2, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Create and train model
    model = RandomForestModel()
    model.train(X_train, y_train)
    
    # Create robustness tester
    tester = AdversarialRobustnessTester()
    
    # Test individual attacks
    print("Testing individual attacks...")
    
    # FGSM attack
    fgsm_attack = FGSMAttack(epsilon=0.1)
    fgsm_result = tester.test_attack(model, X_test, y_test, fgsm_attack)
    print(f"FGSM Attack - Success Rate: {fgsm_result.success_rate:.4f}")
    
    # PGD attack
    pgd_attack = PGDAttack(epsilon=0.1, iterations=20)
    pgd_result = tester.test_attack(model, X_test, y_test, pgd_attack)
    print(f"PGD Attack - Success Rate: {pgd_result.success_rate:.4f}")
    
    # Test defenses
    print("\nTesting defenses...")
    
    # Adversarial training defense
    adversarial_training = AdversarialTraining(fgsm_attack)
    defense_result = tester.test_defense(
        model, X_train, y_train, X_test, y_test, fgsm_attack, adversarial_training
    )
    print(f"Adversarial Training - Robustness Improvement: {defense_result.robustness_improvement:.4f}")
    
    # Comprehensive testing
    print("\nRunning comprehensive robustness test...")
    comprehensive_results = tester.comprehensive_robustness_test(
        model, X_train, y_train, X_test, y_test
    )
    
    # Print summary
    summary = comprehensive_results['summary']
    print(f"\nComprehensive Results Summary:")
    print(f"Most Effective Attack: {summary['attack_summary']['most_effective_attack']}")
    print(f"Most Effective Defense: {summary['defense_summary']['most_effective_defense']}")
    print(f"Overall Robustness Score: {summary['overall_robustness_score']:.4f}")
    
    # Plot results
    tester.plot_robustness_results(comprehensive_results)
    
    return comprehensive_results

# Run adversarial robustness testing
robustness_results = adversarial_robustness_example()
```

### Example 5: Performance Benchmarking

```python
from evaluation.performance_analysis import CybersecurityEvaluator
from models.ml_models import RandomForestModel, SVMModel, CNNModel
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split

def performance_benchmarking_example():
    """Performance benchmarking example."""
    
    # Generate synthetic data
    X, y = make_classification(n_samples=2000, n_features=20, n_classes=5, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Create models
    models = {
        'Random Forest': RandomForestModel(),
        'SVM': SVMModel(),
        'CNN': CNNModel(input_shape=(20,), num_classes=5)
    }
    
    # Create evaluator
    evaluator = CybersecurityEvaluator()
    
    # Run benchmark
    print("Running performance benchmark...")
    benchmark_results = evaluator.benchmark_suite.run_benchmark(
        models, X_train, y_train, X_test, y_test, "Cybersecurity ML Benchmark"
    )
    
    # Generate report
    report = evaluator.benchmark_suite.generate_benchmark_report("Cybersecurity ML Benchmark")
    print(report)
    
    # Plot comparisons
    performances = list(benchmark_results.values())
    evaluator.performance_analyzer.plot_performance_comparison(performances)
    evaluator.performance_analyzer.plot_confusion_matrices(performances)
    evaluator.performance_analyzer.plot_roc_curves(performances)
    
    # Save results
    evaluator.benchmark_suite.save_benchmark_results(
        "Cybersecurity ML Benchmark", "benchmark_results.json"
    )
    
    return benchmark_results

# Run performance benchmarking
benchmark_results = performance_benchmarking_example()
```

## API Reference

### Data Preprocessing

#### `DataPreprocessor`

```python
class DataPreprocessor:
    def __init__(self, data_dir: str = "data")
    def load_nsl_kdd(self) -> Tuple[pd.DataFrame, pd.DataFrame]
    def load_cicids2017(self) -> Tuple[pd.DataFrame, pd.DataFrame]
    def load_ton_iot(self) -> Tuple[pd.DataFrame, pd.DataFrame]
    def preprocess_features(self, features: pd.DataFrame, dataset_name: str) -> pd.DataFrame
    def preprocess_labels(self, labels: pd.Series, dataset_name: str) -> pd.Series
    def feature_selection(self, features: pd.DataFrame, labels: pd.Series, dataset_name: str, k: int = 20) -> pd.DataFrame
    def split_data(self, features: pd.DataFrame, labels: pd.Series, test_size: float = 0.2, val_size: float = 0.1) -> Dict[str, np.ndarray]
```

### ML Models

#### `BaseModel`

```python
class BaseModel:
    def __init__(self, model_name: str)
    def train(self, X_train: np.ndarray, y_train: np.ndarray, X_val: Optional[np.ndarray] = None, y_val: Optional[np.ndarray] = None) -> Dict[str, Any]
    def predict(self, X: np.ndarray) -> np.ndarray
    def predict_proba(self, X: np.ndarray) -> np.ndarray
    def save_model(self, filepath: str)
    def load_model(self, filepath: str)
```

#### `EnsembleModel`

```python
class EnsembleModel(BaseModel):
    def __init__(self, models: List[BaseModel], weights: List[float] = None)
    def train(self, X_train: np.ndarray, y_train: np.ndarray, X_val: Optional[np.ndarray] = None, y_val: Optional[np.ndarray] = None) -> Dict[str, Any]
    def predict(self, X: np.ndarray) -> np.ndarray
    def predict_proba(self, X: np.ndarray) -> np.ndarray
```

### Federated Learning

#### `PrivacyPreservingFederatedLearning`

```python
class PrivacyPreservingFederatedLearning:
    def __init__(self, model: nn.Module, num_clients: int = 5)
    def add_client_data(self, client_id: str, X_train: np.ndarray, y_train: np.ndarray, config: Optional[ClientConfig] = None)
    def train_federated(self, num_rounds: int = 10, enable_privacy: bool = True) -> Dict[str, Any]
    def get_global_model(self) -> nn.Module
    def save_federated_model(self, filepath: str)
    def load_federated_model(self, filepath: str)
```

### Explainable AI

#### `ModelInterpretabilityDashboard`

```python
class ModelInterpretabilityDashboard:
    def __init__(self, model, feature_names: List[str] = None, training_data: np.ndarray = None)
    def generate_comprehensive_explanation(self, X: np.ndarray, y: np.ndarray = None, instance_idx: int = 0) -> Dict[str, Any]
    def plot_comprehensive_dashboard(self, explanation: Dict[str, Any])
    def generate_explanation_report(self, explanation: Dict[str, Any]) -> str
```

### Anomaly Detection

#### `CybersecurityAnomalyDetectionSystem`

```python
class CybersecurityAnomalyDetectionSystem:
    def __init__(self)
    def add_anomaly_detector(self, name: str, detector: AnomalyDetector)
    def train_system(self, X_normal: np.ndarray, X_mixed: np.ndarray = None, y_mixed: np.ndarray = None) -> Dict[str, Any]
    def detect_anomalies(self, X: np.ndarray) -> Dict[str, Any]
    def classify_threats(self, X: np.ndarray) -> Dict[str, Any]
    def comprehensive_analysis(self, X: np.ndarray) -> Dict[str, Any]
```

### Real-time Monitoring

#### `RealTimeMonitoringPipeline`

```python
class RealTimeMonitoringPipeline:
    def __init__(self, anomaly_detector=None, threat_classifier=None)
    def start_monitoring(self, update_interval: float = 1.0)
    def stop_monitoring(self)
    def get_monitoring_stats(self) -> Dict[str, Any]
    def get_recent_alerts(self, limit: int = 10) -> List[Alert]
```

### Performance Analysis

#### `CybersecurityEvaluator`

```python
class CybersecurityEvaluator:
    def __init__(self)
    def evaluate_threat_detection_performance(self, model, X_test: np.ndarray, y_test: np.ndarray, threat_types: List[str] = None) -> Dict[str, Any]
    def evaluate_anomaly_detection_performance(self, detector, X_test: np.ndarray, y_test: np.ndarray) -> Dict[str, Any]
```

### Adversarial Robustness

#### `AdversarialRobustnessTester`

```python
class AdversarialRobustnessTester:
    def __init__(self)
    def test_attack(self, model, X_test: np.ndarray, y_test: np.ndarray, attack: AdversarialAttack) -> AttackResult
    def test_defense(self, model, X_train: np.ndarray, y_train: np.ndarray, X_test: np.ndarray, y_test: np.ndarray, attack: AdversarialAttack, defense: AdversarialDefense) -> DefenseResult
    def comprehensive_robustness_test(self, model, X_train: np.ndarray, y_train: np.ndarray, X_test: np.ndarray, y_test: np.ndarray) -> Dict[str, Any]
    def plot_robustness_results(self, results: Dict[str, Any])
```

## Configuration

### Configuration File (`config/settings.py`)

The framework uses a comprehensive configuration system with the following main components:

#### Model Configuration

```python
@dataclass
class ModelConfig:
    # Random Forest parameters
    rf_n_estimators: int = 100
    rf_max_depth: Optional[int] = None
    rf_min_samples_split: int = 2
    
    # SVM parameters
    svm_C: float = 1.0
    svm_kernel: str = 'rbf'
    svm_gamma: str = 'scale'
    
    # Neural Network parameters
    nn_hidden_layers: List[int] = None
    nn_activation: str = 'relu'
    nn_learning_rate: float = 0.001
    nn_epochs: int = 100
```

#### Data Configuration

```python
@dataclass
class DataConfig:
    data_dir: str = "data"
    test_size: float = 0.2
    val_size: float = 0.1
    feature_selection_k: int = 20
    random_state: int = 42
    enable_balancing: bool = True
```

#### Federated Learning Configuration

```python
@dataclass
class FederatedLearningConfig:
    num_clients: int = 5
    num_rounds: int = 10
    local_epochs: int = 5
    learning_rate: float = 0.01
    differential_privacy: bool = True
    epsilon: float = 1.0
```

#### Real-time Configuration

```python
@dataclass
class RealTimeConfig:
    update_interval: float = 1.0
    batch_size: int = 100
    max_queue_size: int = 1000
    threat_threshold: float = 0.8
    anomaly_threshold: float = 0.7
```

### Using Configuration

```python
from config.settings import Config

# Load configuration
config = Config("config.yaml")

# Use configuration
model_config = config.model
data_config = config.data
federated_config = config.federated

# Create model with configuration
rf_model = RandomForestModel(
    n_estimators=model_config.rf_n_estimators,
    max_depth=model_config.rf_max_depth
)
```

## Performance Optimization

### GPU Acceleration

```python
# Enable GPU for TensorFlow
import tensorflow as tf
tf.config.experimental.set_memory_growth(tf.config.experimental.list_physical_devices('GPU')[0], True)

# Enable GPU for PyTorch
import torch
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)
```

### Memory Optimization

```python
# Use data generators for large datasets
def data_generator(X, y, batch_size=32):
    for i in range(0, len(X), batch_size):
        yield X[i:i+batch_size], y[i:i+batch_size]

# Clear memory
import gc
gc.collect()
```

### Parallel Processing

```python
# Use joblib for parallel processing
from joblib import Parallel, delayed

def train_model(model, X, y):
    return model.fit(X, y)

# Train multiple models in parallel
models = [RandomForestModel() for _ in range(5)]
results = Parallel(n_jobs=-1)(delayed(train_model)(model, X, y) for model in models)
```

## Troubleshooting

### Common Issues

#### 1. Memory Issues

**Problem**: Out of memory errors during training
**Solution**: 
- Reduce batch size
- Use data generators
- Enable memory growth for GPU
- Use smaller models

```python
# Reduce batch size
model.train(X_train, y_train, batch_size=16)

# Use memory growth
tf.config.experimental.set_memory_growth(gpu, True)
```

#### 2. Dataset Loading Issues

**Problem**: Cannot load datasets
**Solution**:
- Check internet connection
- Verify dataset URLs
- Use synthetic data for testing

```python
# Use synthetic data
from sklearn.datasets import make_classification
X, y = make_classification(n_samples=1000, n_features=20, n_classes=2)
```

#### 3. Model Training Issues

**Problem**: Models not converging
**Solution**:
- Adjust learning rate
- Increase epochs
- Check data preprocessing
- Use different optimizers

```python
# Adjust learning rate
model = CNNModel(learning_rate=0.0001)

# Increase epochs
model.train(X_train, y_train, epochs=200)
```

#### 4. Real-time Monitoring Issues

**Problem**: Monitoring pipeline not working
**Solution**:
- Check system permissions
- Verify network interfaces
- Use simplified collectors

```python
# Use simplified monitoring
pipeline = RealTimeMonitoringPipeline()
pipeline.start_monitoring(update_interval=5.0)
```

### Debug Mode

```python
# Enable debug logging
import logging
logging.basicConfig(level=logging.DEBUG)

# Enable verbose output
model.train(X_train, y_train, verbose=1)
```

### Performance Profiling

```python
# Profile model performance
import cProfile
cProfile.run('model.train(X_train, y_train)')

# Memory profiling
from memory_profiler import profile

@profile
def train_model():
    model.train(X_train, y_train)
```

## Contributing

### Development Setup

```bash
# Clone repository
git clone <repository-url>
cd cybersecurity-ml-framework

# Create development environment
python -m venv dev_env
source dev_env/bin/activate

# Install development dependencies
pip install -r requirements-dev.txt

# Install pre-commit hooks
pre-commit install
```

### Code Style

- Follow PEP 8
- Use type hints
- Write docstrings
- Add unit tests

### Testing

```bash
# Run tests
pytest tests/

# Run with coverage
pytest --cov=. tests/

# Run specific test
pytest tests/test_models.py::test_random_forest
```

### Submitting Changes

1. Fork the repository
2. Create a feature branch
3. Make changes
4. Add tests
5. Run tests
6. Submit pull request

### Documentation

- Update README.md
- Add docstrings
- Update API reference
- Add examples

---

For more information, please refer to the individual module documentation or contact the development team.
