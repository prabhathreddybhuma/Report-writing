#!/usr/bin/env python3
"""
Test Script for Cybersecurity ML Framework

This script tests all components of the cybersecurity ML framework to ensure
they work correctly together.
"""

import sys
import os
import traceback
import logging
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def test_imports():
    """Test that all modules can be imported."""
    logger.info("Testing imports...")
    
    try:
        # Test data preprocessing
        from data.preprocessing import DataPreprocessor
        logger.info("✓ Data preprocessing imported successfully")
        
        # Test ML models
        from models.ml_models import RandomForestModel, SVMModel, CNNModel, RNNModel, EnsembleModel
        logger.info("✓ ML models imported successfully")
        
        # Test federated learning
        from federated_learning.federated_learning import PrivacyPreservingFederatedLearning
        logger.info("✓ Federated learning imported successfully")
        
        # Test explainable AI
        from explainable_ai.xai_components import ModelInterpretabilityDashboard
        logger.info("✓ Explainable AI imported successfully")
        
        # Test anomaly detection
        from anomaly_detection.anomaly_detection import IsolationForestDetector
        logger.info("✓ Anomaly detection imported successfully")
        
        # Test real-time monitoring
        from real_time_monitoring.realtime_pipeline import RealTimeMonitoringPipeline
        logger.info("✓ Real-time monitoring imported successfully")
        
        # Test evaluation
        from evaluation.performance_analysis import CybersecurityEvaluator
        logger.info("✓ Performance analysis imported successfully")
        
        # Test adversarial robustness
        from adversarial_robustness.adversarial_testing import AdversarialRobustnessTester
        logger.info("✓ Adversarial robustness imported successfully")
        
        return True
        
    except Exception as e:
        logger.error(f"✗ Import test failed: {e}")
        traceback.print_exc()
        return False


def test_data_preprocessing():
    """Test data preprocessing functionality."""
    logger.info("Testing data preprocessing...")
    
    try:
        from data.preprocessing import DataPreprocessor
        import numpy as np
        import pandas as pd
        
        # Create test data
        preprocessor = DataPreprocessor()
        
        # Test NSL-KDD loading (synthetic)
        features, labels = preprocessor.load_nsl_kdd()
        logger.info(f"✓ NSL-KDD loaded: {features.shape[0]} samples, {features.shape[1]} features")
        
        # Test preprocessing
        processed_features = preprocessor.preprocess_features(features, 'nsl_kdd')
        processed_labels = preprocessor.preprocess_labels(labels, 'nsl_kdd')
        logger.info(f"✓ Features preprocessed: {processed_features.shape}")
        
        # Test feature selection
        selected_features = preprocessor.feature_selection(processed_features, processed_labels, 'nsl_kdd')
        logger.info(f"✓ Feature selection completed: {selected_features.shape[1]} features selected")
        
        # Test data splitting
        splits = preprocessor.split_data(selected_features, processed_labels)
        logger.info(f"✓ Data split: Train={splits['X_train'].shape[0]}, Val={splits['X_val'].shape[0]}, Test={splits['X_test'].shape[0]}")
        
        return True
        
    except Exception as e:
        logger.error(f"✗ Data preprocessing test failed: {e}")
        traceback.print_exc()
        return False


def test_ml_models():
    """Test ML model functionality."""
    logger.info("Testing ML models...")
    
    try:
        from models.ml_models import RandomForestModel, SVMModel, EnsembleModel
        import numpy as np
        from sklearn.datasets import make_classification
        
        # Generate test data
        X, y = make_classification(n_samples=1000, n_features=20, n_classes=2, random_state=42)
        
        # Test Random Forest
        rf_model = RandomForestModel()
        rf_model.train(X, y)
        rf_predictions = rf_model.predict(X)
        logger.info(f"✓ Random Forest trained and tested: {np.mean(rf_predictions == y):.4f} accuracy")
        
        # Test SVM
        svm_model = SVMModel()
        svm_model.train(X, y)
        svm_predictions = svm_model.predict(X)
        logger.info(f"✓ SVM trained and tested: {np.mean(svm_predictions == y):.4f} accuracy")
        
        # Test Ensemble
        ensemble = EnsembleModel([rf_model, svm_model])
        ensemble.train(X, y)
        ensemble_predictions = ensemble.predict(X)
        logger.info(f"✓ Ensemble trained and tested: {np.mean(ensemble_predictions == y):.4f} accuracy")
        
        return True
        
    except Exception as e:
        logger.error(f"✗ ML models test failed: {e}")
        traceback.print_exc()
        return False


def test_anomaly_detection():
    """Test anomaly detection functionality."""
    logger.info("Testing anomaly detection...")
    
    try:
        from anomaly_detection.anomaly_detection import IsolationForestDetector, create_default_anomaly_detection_system
        import numpy as np
        
        # Generate test data
        normal_data = np.random.randn(500, 10)
        anomalous_data = np.random.randn(100, 10) + 3
        
        # Test Isolation Forest
        detector = IsolationForestDetector()
        detector.fit(normal_data)
        predictions = detector.predict(anomalous_data)
        logger.info(f"✓ Isolation Forest: {np.sum(predictions == -1)}/{len(anomalous_data)} anomalies detected")
        
        # Test anomaly detection system
        system = create_default_anomaly_detection_system()
        system.train_system(normal_data)
        
        test_data = np.vstack([normal_data[:50], anomalous_data[:50]])
        results = system.comprehensive_analysis(test_data)
        logger.info(f"✓ Anomaly detection system: {len(results['anomaly_detection'])} detectors tested")
        
        return True
        
    except Exception as e:
        logger.error(f"✗ Anomaly detection test failed: {e}")
        traceback.print_exc()
        return False


def test_evaluation():
    """Test evaluation functionality."""
    logger.info("Testing evaluation...")
    
    try:
        from evaluation.performance_analysis import CybersecurityEvaluator
        from models.ml_models import RandomForestModel
        import numpy as np
        from sklearn.datasets import make_classification
        
        # Generate test data
        X, y = make_classification(n_samples=1000, n_features=20, n_classes=2, random_state=42)
        
        # Train model
        model = RandomForestModel()
        model.train(X, y)
        
        # Test evaluator
        evaluator = CybersecurityEvaluator()
        performance = evaluator.evaluate_threat_detection_performance(model, X, y)
        
        logger.info(f"✓ Evaluation completed: {performance['basic_performance'].metrics.accuracy:.4f} accuracy")
        
        return True
        
    except Exception as e:
        logger.error(f"✗ Evaluation test failed: {e}")
        traceback.print_exc()
        return False


def test_adversarial_robustness():
    """Test adversarial robustness functionality."""
    logger.info("Testing adversarial robustness...")
    
    try:
        from adversarial_robustness.adversarial_testing import AdversarialRobustnessTester, FGSMAttack
        from models.ml_models import RandomForestModel
        import numpy as np
        from sklearn.datasets import make_classification
        
        # Generate test data
        X, y = make_classification(n_samples=500, n_features=20, n_classes=2, random_state=42)
        
        # Train model
        model = RandomForestModel()
        model.train(X, y)
        
        # Test adversarial robustness
        tester = AdversarialRobustnessTester()
        attack = FGSMAttack(epsilon=0.1)
        
        result = tester.test_attack(model, X, y, attack)
        logger.info(f"✓ Adversarial attack test: {result.success_rate:.4f} success rate")
        
        return True
        
    except Exception as e:
        logger.error(f"✗ Adversarial robustness test failed: {e}")
        traceback.print_exc()
        return False


def test_configuration():
    """Test configuration functionality."""
    logger.info("Testing configuration...")
    
    try:
        from config.settings import Config, ModelConfig, DataConfig
        
        # Test configuration creation
        config = Config()
        logger.info(f"✓ Configuration created: {config.model.rf_n_estimators} RF estimators")
        
        # Test individual configs
        model_config = ModelConfig()
        data_config = DataConfig()
        
        logger.info(f"✓ Model config: {model_config.rf_n_estimators} estimators")
        logger.info(f"✓ Data config: {data_config.test_size} test size")
        
        return True
        
    except Exception as e:
        logger.error(f"✗ Configuration test failed: {e}")
        traceback.print_exc()
        return False


def run_all_tests():
    """Run all tests."""
    logger.info("=" * 60)
    logger.info("CYBERSECURITY ML FRAMEWORK - COMPREHENSIVE TESTING")
    logger.info("=" * 60)
    
    tests = [
        ("Import Test", test_imports),
        ("Data Preprocessing", test_data_preprocessing),
        ("ML Models", test_ml_models),
        ("Anomaly Detection", test_anomaly_detection),
        ("Evaluation", test_evaluation),
        ("Adversarial Robustness", test_adversarial_robustness),
        ("Configuration", test_configuration)
    ]
    
    passed_tests = 0
    total_tests = len(tests)
    
    for test_name, test_func in tests:
        logger.info(f"\n--- {test_name} ---")
        try:
            if test_func():
                logger.info(f"✓ {test_name} PASSED")
                passed_tests += 1
            else:
                logger.error(f"✗ {test_name} FAILED")
        except Exception as e:
            logger.error(f"✗ {test_name} FAILED with exception: {e}")
    
    logger.info("\n" + "=" * 60)
    logger.info(f"TEST RESULTS: {passed_tests}/{total_tests} tests passed")
    logger.info("=" * 60)
    
    if passed_tests == total_tests:
        logger.info("🎉 ALL TESTS PASSED! Framework is ready to use.")
        return True
    else:
        logger.error(f"❌ {total_tests - passed_tests} tests failed. Please check the errors above.")
        return False


def main():
    """Main function."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Test Cybersecurity ML Framework')
    parser.add_argument('--test', type=str, choices=[
        'imports', 'preprocessing', 'models', 'anomaly', 'evaluation', 
        'adversarial', 'config', 'all'
    ], default='all', help='Specific test to run')
    
    args = parser.parse_args()
    
    if args.test == 'all':
        success = run_all_tests()
    else:
        test_map = {
            'imports': test_imports,
            'preprocessing': test_data_preprocessing,
            'models': test_ml_models,
            'anomaly': test_anomaly_detection,
            'evaluation': test_evaluation,
            'adversarial': test_adversarial_robustness,
            'config': test_configuration
        }
        
        if args.test in test_map:
            success = test_map[args.test]()
        else:
            logger.error(f"Unknown test: {args.test}")
            success = False
    
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
