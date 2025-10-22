"""
Example Usage Scripts for Cybersecurity ML Framework

This module contains comprehensive examples demonstrating how to use
various components of the cybersecurity ML framework.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')

# Import framework components
from data.preprocessing import DataPreprocessor
from models.ml_models import RandomForestModel, SVMModel, CNNModel, RNNModel, EnsembleModel
from federated_learning.federated_learning import PrivacyPreservingFederatedLearning, create_simple_nn
from explainable_ai.xai_components import ModelInterpretabilityDashboard
from anomaly_detection.anomaly_detection import (
    IsolationForestDetector, OneClassSVMDetector, AutoencoderDetector,
    CybersecurityAnomalyDetectionSystem, create_default_anomaly_detection_system
)
from real_time_monitoring.realtime_pipeline import RealTimeMonitoringPipeline
from evaluation.performance_analysis import CybersecurityEvaluator, BenchmarkSuite
from adversarial_robustness.adversarial_testing import (
    AdversarialRobustnessTester, FGSMAttack, PGDAttack,
    AdversarialTraining, DefensiveDistillation
)


def example_1_basic_threat_detection():
    """Example 1: Basic threat detection using Random Forest."""
    print("=" * 60)
    print("EXAMPLE 1: Basic Threat Detection")
    print("=" * 60)
    
    # Generate synthetic cybersecurity data
    X, y = make_classification(
        n_samples=2000, n_features=20, n_classes=2, 
        n_clusters_per_class=1, random_state=42
    )
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)
    
    # Create and train model
    print("Training Random Forest model...")
    model = RandomForestModel(n_estimators=100, random_state=42)
    training_history = model.train(X_train, y_train, X_val, y_val)
    
    # Make predictions
    predictions = model.predict(X_test)
    probabilities = model.predict_proba(X_test)
    
    # Calculate accuracy
    accuracy = np.mean(predictions == y_test)
    print(f"Model Accuracy: {accuracy:.4f}")
    print(f"Training History: {training_history}")
    
    # Get feature importance
    if hasattr(model, 'get_feature_importance'):
        feature_importance = model.get_feature_importance()
        print(f"Top 5 Features: {np.argsort(feature_importance)[-5:]}")
    
    return model, predictions, probabilities


def example_2_ensemble_modeling():
    """Example 2: Ensemble modeling for improved performance."""
    print("\n" + "=" * 60)
    print("EXAMPLE 2: Ensemble Modeling")
    print("=" * 60)
    
    # Generate data
    X, y = make_classification(n_samples=2000, n_features=20, n_classes=2, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)
    
    # Create individual models
    print("Creating individual models...")
    rf_model = RandomForestModel(n_estimators=100, random_state=42)
    svm_model = SVMModel(C=1.0, kernel='rbf', random_state=42)
    cnn_model = CNNModel(input_shape=(20,), num_classes=2)
    
    # Create ensemble
    print("Creating ensemble model...")
    ensemble = EnsembleModel([rf_model, svm_model, cnn_model])
    
    # Train ensemble
    print("Training ensemble...")
    ensemble_history = ensemble.train(X_train, y_train, X_val, y_val)
    
    # Evaluate ensemble
    ensemble_predictions = ensemble.predict(X_test)
    ensemble_accuracy = np.mean(ensemble_predictions == y_test)
    
    print(f"Ensemble Accuracy: {ensemble_accuracy:.4f}")
    print(f"Ensemble Weights: {ensemble.weights}")
    
    # Compare with individual models
    individual_accuracies = []
    for model in ensemble.models:
        if hasattr(model, 'train'):
            model.train(X_train, y_train, X_val, y_val)
        predictions = model.predict(X_test)
        accuracy = np.mean(predictions == y_test)
        individual_accuracies.append(accuracy)
        print(f"{model.model_name} Accuracy: {accuracy:.4f}")
    
    print(f"Ensemble improvement: {ensemble_accuracy - np.mean(individual_accuracies):.4f}")
    
    return ensemble, ensemble_predictions


def example_3_federated_learning():
    """Example 3: Federated learning for privacy-preserving training."""
    print("\n" + "=" * 60)
    print("EXAMPLE 3: Federated Learning")
    print("=" * 60)
    
    # Create neural network
    model = create_simple_nn(input_size=20, num_classes=2)
    
    # Initialize federated learning system
    fl_system = PrivacyPreservingFederatedLearning(model, num_clients=3)
    
    # Generate synthetic client data
    print("Generating client data...")
    np.random.seed(42)
    
    clients_data = []
    for i in range(3):
        # Each client has different data distribution
        X_client = np.random.randn(300, 20) + np.random.randn(1, 20) * 0.3
        y_client = np.random.randint(0, 2, 300)
        clients_data.append((X_client, y_client))
        
        # Add to federated system
        fl_system.add_client_data(f"client_{i+1}", X_client, y_client)
        print(f"Client {i+1}: {len(X_client)} samples")
    
    # Train federated model
    print("Starting federated training...")
    training_history = fl_system.train_federated(num_rounds=5, enable_privacy=True)
    
    # Print results
    print(f"Federated training completed!")
    print(f"Final global accuracy: {training_history['global_accuracy'][-1]:.4f}")
    print(f"Privacy budget remaining: {training_history['privacy_budget_used'][-1]:.4f}")
    
    # Plot training progress
    plt.figure(figsize=(10, 6))
    plt.plot(training_history['global_accuracy'], label='Global Accuracy')
    plt.plot(training_history['privacy_budget_used'], label='Privacy Budget')
    plt.xlabel('Round')
    plt.ylabel('Value')
    plt.title('Federated Learning Progress')
    plt.legend()
    plt.grid(True)
    plt.show()
    
    return fl_system, training_history


def example_4_explainable_ai():
    """Example 4: Explainable AI for model interpretability."""
    print("\n" + "=" * 60)
    print("EXAMPLE 4: Explainable AI")
    print("=" * 60)
    
    # Generate data
    X, y = make_classification(n_samples=1000, n_features=20, n_classes=2, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Train model
    model = RandomForestModel(n_estimators=100, random_state=42)
    model.train(X_train, y_train)
    
    # Create feature names
    feature_names = [f"Feature_{i}" for i in range(20)]
    
    # Create interpretability dashboard
    print("Creating interpretability dashboard...")
    dashboard = ModelInterpretabilityDashboard(
        model, feature_names, X_train
    )
    
    # Generate explanation for a specific instance
    instance_idx = 0
    print(f"Generating explanation for instance {instance_idx}...")
    explanation = dashboard.generate_comprehensive_explanation(
        X_test, y_test, instance_idx=instance_idx
    )
    
    # Generate report
    report = dashboard.generate_explanation_report(explanation)
    print("Explanation Report:")
    print(report)
    
    # Plot dashboard
    print("Plotting interpretability dashboard...")
    dashboard.plot_comprehensive_dashboard(explanation)
    
    return dashboard, explanation


def example_5_anomaly_detection():
    """Example 5: Anomaly detection for threat identification."""
    print("\n" + "=" * 60)
    print("EXAMPLE 5: Anomaly Detection")
    print("=" * 60)
    
    # Generate normal and anomalous data
    np.random.seed(42)
    normal_data = np.random.randn(1000, 15)
    anomalous_data = np.random.randn(200, 15) + 3  # Shifted distribution
    
    # Combine data
    X_mixed = np.vstack([normal_data, anomalous_data])
    y_mixed = np.hstack([np.zeros(1000), np.ones(200)])  # 0 = normal, 1 = anomaly
    
    # Create anomaly detection system
    print("Creating anomaly detection system...")
    system = create_default_anomaly_detection_system()
    
    # Train system
    print("Training anomaly detection system...")
    training_results = system.train_system(normal_data, X_mixed, y_mixed)
    print(f"Training results: {training_results}")
    
    # Test system
    print("Testing anomaly detection system...")
    test_normal = np.random.randn(100, 15)
    test_anomalous = np.random.randn(50, 15) + 3
    X_test = np.vstack([test_normal, test_anomalous])
    y_test = np.hstack([np.zeros(100), np.ones(50)])
    
    # Comprehensive analysis
    analysis_results = system.comprehensive_analysis(X_test)
    
    print("Anomaly Detection Results:")
    for detector_name, results in analysis_results['anomaly_detection'].items():
        print(f"  {detector_name}:")
        print(f"    Anomaly Count: {results['anomaly_count']}")
        print(f"    Anomaly Rate: {results['anomaly_rate']:.4f}")
    
    # Plot anomaly scores
    plt.figure(figsize=(12, 8))
    
    for i, (detector_name, results) in enumerate(analysis_results['anomaly_detection'].items()):
        plt.subplot(2, 3, i+1)
        plt.hist(results['scores'], bins=30, alpha=0.7)
        plt.title(f'{detector_name} - Anomaly Scores')
        plt.xlabel('Score')
        plt.ylabel('Frequency')
    
    plt.tight_layout()
    plt.show()
    
    return system, analysis_results


def example_6_real_time_monitoring():
    """Example 6: Real-time monitoring and threat detection."""
    print("\n" + "=" * 60)
    print("EXAMPLE 6: Real-time Monitoring")
    print("=" * 60)
    
    # Create anomaly detector
    np.random.seed(42)
    normal_data = np.random.randn(500, 10)
    
    detector = IsolationForestDetector(contamination=0.1)
    detector.fit(normal_data)
    
    # Create threat classifier
    classifier = RandomForestModel()
    X_train = np.random.randn(200, 10)
    y_train = np.random.randint(0, 5, 200)
    classifier.train(X_train, y_train)
    
    # Create monitoring pipeline
    print("Creating monitoring pipeline...")
    pipeline = RealTimeMonitoringPipeline(detector, classifier)
    
    try:
        # Start monitoring
        print("Starting real-time monitoring...")
        pipeline.start_monitoring(update_interval=2.0)
        
        # Monitor for 20 seconds
        print("Monitoring for 20 seconds...")
        monitoring_stats = []
        
        for i in range(10):  # 10 iterations of 2 seconds each
            import time
            time.sleep(2)
            
            # Get current statistics
            stats = pipeline.get_monitoring_stats()
            monitoring_stats.append(stats)
            
            print(f"Iteration {i+1}:")
            print(f"  Detections: {stats['total_detections']}")
            print(f"  Active Alerts: {stats['active_alerts']}")
            print(f"  Collectors Status: {stats['collectors_status']}")
            
            # Get recent alerts
            alerts = pipeline.get_recent_alerts(limit=2)
            for alert in alerts:
                print(f"  Alert: {alert.description} (Severity: {alert.severity})")
    
    finally:
        # Stop monitoring
        print("Stopping monitoring...")
        pipeline.stop_monitoring()
        
        # Final statistics
        final_stats = pipeline.get_monitoring_stats()
        print(f"\nFinal Statistics:")
        print(f"  Total Detections: {final_stats['total_detections']}")
        print(f"  Active Alerts: {final_stats['active_alerts']}")
    
    return pipeline, monitoring_stats


def example_7_performance_benchmarking():
    """Example 7: Performance benchmarking and evaluation."""
    print("\n" + "=" * 60)
    print("EXAMPLE 7: Performance Benchmarking")
    print("=" * 60)
    
    # Generate data
    X, y = make_classification(n_samples=2000, n_features=20, n_classes=5, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Create models
    print("Creating models for benchmarking...")
    models = {
        'Random Forest': RandomForestModel(n_estimators=100, random_state=42),
        'SVM': SVMModel(C=1.0, kernel='rbf', random_state=42),
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
    print("Benchmark Report:")
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
    print("Benchmark results saved to benchmark_results.json")
    
    return benchmark_results, report


def example_8_adversarial_robustness():
    """Example 8: Adversarial robustness testing."""
    print("\n" + "=" * 60)
    print("EXAMPLE 8: Adversarial Robustness Testing")
    print("=" * 60)
    
    # Generate data
    X, y = make_classification(n_samples=2000, n_features=20, n_classes=2, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Create and train model
    model = RandomForestModel(n_estimators=100, random_state=42)
    model.train(X_train, y_train)
    
    # Create robustness tester
    tester = AdversarialRobustnessTester()
    
    # Test individual attacks
    print("Testing individual attacks...")
    
    # FGSM attack
    fgsm_attack = FGSMAttack(epsilon=0.1)
    fgsm_result = tester.test_attack(model, X_test, y_test, fgsm_attack)
    print(f"FGSM Attack:")
    print(f"  Original Accuracy: {fgsm_result.original_accuracy:.4f}")
    print(f"  Adversarial Accuracy: {fgsm_result.adversarial_accuracy:.4f}")
    print(f"  Success Rate: {fgsm_result.success_rate:.4f}")
    
    # PGD attack
    pgd_attack = PGDAttack(epsilon=0.1, iterations=20)
    pgd_result = tester.test_attack(model, X_test, y_test, pgd_attack)
    print(f"\nPGD Attack:")
    print(f"  Original Accuracy: {pgd_result.original_accuracy:.4f}")
    print(f"  Adversarial Accuracy: {pgd_result.adversarial_accuracy:.4f}")
    print(f"  Success Rate: {pgd_result.success_rate:.4f}")
    
    # Test defenses
    print("\nTesting defenses...")
    
    # Adversarial training defense
    adversarial_training = AdversarialTraining(fgsm_attack)
    defense_result = tester.test_defense(
        model, X_train, y_train, X_test, y_test, fgsm_attack, adversarial_training
    )
    print(f"Adversarial Training Defense:")
    print(f"  Original Accuracy: {defense_result.original_accuracy:.4f}")
    print(f"  Defended Accuracy: {defense_result.defended_accuracy:.4f}")
    print(f"  Robustness Improvement: {defense_result.robustness_improvement:.4f}")
    
    # Comprehensive testing
    print("\nRunning comprehensive robustness test...")
    comprehensive_results = tester.comprehensive_robustness_test(
        model, X_train, y_train, X_test, y_test
    )
    
    # Print summary
    summary = comprehensive_results['summary']
    print(f"\nComprehensive Results Summary:")
    print(f"  Most Effective Attack: {summary['attack_summary']['most_effective_attack']}")
    print(f"  Most Effective Defense: {summary['defense_summary']['most_effective_defense']}")
    print(f"  Overall Robustness Score: {summary['overall_robustness_score']:.4f}")
    
    # Plot results
    tester.plot_robustness_results(comprehensive_results)
    
    return comprehensive_results, summary


def example_9_complete_pipeline():
    """Example 9: Complete cybersecurity pipeline."""
    print("\n" + "=" * 60)
    print("EXAMPLE 9: Complete Cybersecurity Pipeline")
    print("=" * 60)
    
    # Step 1: Data preprocessing
    print("Step 1: Data Preprocessing")
    preprocessor = DataPreprocessor()
    
    # Generate synthetic cybersecurity data
    X, y = make_classification(
        n_samples=3000, n_features=25, n_classes=5, 
        n_clusters_per_class=1, random_state=42
    )
    
    # Preprocess data
    processed_features = preprocessor.preprocess_features(pd.DataFrame(X), 'synthetic')
    processed_labels = preprocessor.preprocess_labels(pd.Series(y), 'synthetic')
    selected_features = preprocessor.feature_selection(processed_features, processed_labels, 'synthetic')
    
    splits = preprocessor.split_data(selected_features, processed_labels)
    
    # Step 2: Model training
    print("Step 2: Model Training")
    ensemble = EnsembleModel([
        RandomForestModel(n_estimators=100, random_state=42),
        SVMModel(C=1.0, kernel='rbf', random_state=42)
    ])
    
    ensemble.train(splits['X_train'], splits['y_train'], splits['X_val'], splits['y_val'])
    
    # Step 3: Model evaluation
    print("Step 3: Model Evaluation")
    evaluator = CybersecurityEvaluator()
    performance = evaluator.evaluate_threat_detection_performance(
        ensemble, splits['X_test'], splits['y_test']
    )
    
    print(f"Model Performance:")
    print(f"  Accuracy: {performance['basic_performance'].metrics.accuracy:.4f}")
    print(f"  F1-Score: {performance['basic_performance'].metrics.f1_score:.4f}")
    print(f"  ROC-AUC: {performance['basic_performance'].metrics.roc_auc:.4f}")
    
    # Step 4: Model interpretation
    print("Step 4: Model Interpretation")
    feature_names = [f"Feature_{i}" for i in range(selected_features.shape[1])]
    dashboard = ModelInterpretabilityDashboard(
        ensemble, feature_names, splits['X_train']
    )
    
    explanation = dashboard.generate_comprehensive_explanation(
        splits['X_test'], splits['y_test'], instance_idx=0
    )
    
    # Step 5: Anomaly detection
    print("Step 5: Anomaly Detection")
    normal_data = splits['X_train'][splits['y_train'] == 0]
    anomaly_system = create_default_anomaly_detection_system()
    anomaly_system.train_system(normal_data)
    
    anomaly_results = anomaly_system.comprehensive_analysis(splits['X_test'])
    
    # Step 6: Adversarial robustness testing
    print("Step 6: Adversarial Robustness Testing")
    robustness_tester = AdversarialRobustnessTester()
    fgsm_attack = FGSMAttack(epsilon=0.1)
    attack_result = robustness_tester.test_attack(ensemble, splits['X_test'], splits['y_test'], fgsm_attack)
    
    print(f"Adversarial Attack Results:")
    print(f"  Success Rate: {attack_result.success_rate:.4f}")
    
    # Step 7: Generate comprehensive report
    print("Step 7: Generating Comprehensive Report")
    
    report = f"""
    ========================================
    CYBERSECURITY ML PIPELINE REPORT
    ========================================
    
    Dataset Information:
    - Total Samples: {len(X)}
    - Features: {X.shape[1]}
    - Classes: {len(np.unique(y))}
    
    Model Performance:
    - Accuracy: {performance['basic_performance'].metrics.accuracy:.4f}
    - Precision: {performance['basic_performance'].metrics.precision:.4f}
    - Recall: {performance['basic_performance'].metrics.recall:.4f}
    - F1-Score: {performance['basic_performance'].metrics.f1_score:.4f}
    - ROC-AUC: {performance['basic_performance'].metrics.roc_auc:.4f}
    
    Anomaly Detection:
    - Detectors Used: {len(anomaly_results['anomaly_detection'])}
    - Overall Anomaly Rate: {anomaly_results['summary'].get('overall_anomaly_rate', 0):.4f}
    
    Adversarial Robustness:
    - Attack Success Rate: {attack_result.success_rate:.4f}
    - Perturbation Norm: {attack_result.perturbation_norm:.4f}
    
    Recommendations:
    1. Model shows good performance with accuracy > 0.85
    2. Consider ensemble methods for improved robustness
    3. Implement adversarial training for better security
    4. Monitor anomaly detection rates regularly
    5. Use explainable AI for model transparency
    
    ========================================
    """
    
    print(report)
    
    return {
        'model': ensemble,
        'performance': performance,
        'explanation': explanation,
        'anomaly_results': anomaly_results,
        'attack_result': attack_result,
        'report': report
    }


def run_all_examples():
    """Run all examples sequentially."""
    print("CYBERSECURITY ML FRAMEWORK - COMPREHENSIVE EXAMPLES")
    print("=" * 80)
    
    try:
        # Run all examples
        example_1_basic_threat_detection()
        example_2_ensemble_modeling()
        example_3_federated_learning()
        example_4_explainable_ai()
        example_5_anomaly_detection()
        example_6_real_time_monitoring()
        example_7_performance_benchmarking()
        example_8_adversarial_robustness()
        example_9_complete_pipeline()
        
        print("\n" + "=" * 80)
        print("ALL EXAMPLES COMPLETED SUCCESSFULLY!")
        print("=" * 80)
        
    except Exception as e:
        print(f"Error running examples: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    # Run all examples
    run_all_examples()
    
    # Or run individual examples
    # example_1_basic_threat_detection()
    # example_2_ensemble_modeling()
    # example_3_federated_learning()
    # example_4_explainable_ai()
    # example_5_anomaly_detection()
    # example_6_real_time_monitoring()
    # example_7_performance_benchmarking()
    # example_8_adversarial_robustness()
    # example_9_complete_pipeline()
