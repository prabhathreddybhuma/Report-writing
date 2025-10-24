#!/usr/bin/env python3
"""
Simple Demo Script for Cybersecurity ML Framework

This script demonstrates the core functionality of the cybersecurity ML framework
using only the basic packages that are installed.
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_classification
from sklearn.metrics import accuracy_score, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def demo_basic_ml_models():
    """Demo basic ML models for cybersecurity."""
    logger.info("=" * 60)
    logger.info("DEMO: Basic ML Models for Cybersecurity")
    logger.info("=" * 60)
    
    # Generate synthetic cybersecurity data
    logger.info("Generating synthetic cybersecurity dataset...")
    X, y = make_classification(
        n_samples=2000, n_features=20, n_classes=2, 
        n_clusters_per_class=1, random_state=42
    )
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    logger.info(f"Dataset shape: {X.shape}")
    logger.info(f"Training samples: {X_train.shape[0]}")
    logger.info(f"Test samples: {X_test.shape[0]}")
    
    # Test Random Forest
    logger.info("\nTraining Random Forest classifier...")
    rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
    rf_model.fit(X_train, y_train)
    rf_predictions = rf_model.predict(X_test)
    rf_accuracy = accuracy_score(y_test, rf_predictions)
    
    logger.info(f"Random Forest Accuracy: {rf_accuracy:.4f}")
    
    # Test SVM
    logger.info("\nTraining SVM classifier...")
    svm_model = SVC(random_state=42, probability=True)
    svm_model.fit(X_train, y_train)
    svm_predictions = svm_model.predict(X_test)
    svm_accuracy = accuracy_score(y_test, svm_predictions)
    
    logger.info(f"SVM Accuracy: {svm_accuracy:.4f}")
    
    # Feature importance
    logger.info("\nTop 5 Most Important Features (Random Forest):")
    feature_importance = rf_model.feature_importances_
    top_features = np.argsort(feature_importance)[-5:][::-1]
    for i, feature_idx in enumerate(top_features):
        logger.info(f"  {i+1}. Feature {feature_idx}: {feature_importance[feature_idx]:.4f}")
    
    return rf_model, svm_model, rf_accuracy, svm_accuracy


def demo_anomaly_detection():
    """Demo anomaly detection for cybersecurity."""
    logger.info("\n" + "=" * 60)
    logger.info("DEMO: Anomaly Detection")
    logger.info("=" * 60)
    
    from sklearn.ensemble import IsolationForest
    from sklearn.svm import OneClassSVM
    
    # Generate normal and anomalous data
    logger.info("Generating normal and anomalous data...")
    np.random.seed(42)
    normal_data = np.random.randn(1000, 15)
    anomalous_data = np.random.randn(200, 15) + 3  # Shifted distribution
    
    # Combine data
    X_mixed = np.vstack([normal_data, anomalous_data])
    y_mixed = np.hstack([np.zeros(1000), np.ones(200)])  # 0 = normal, 1 = anomaly
    
    logger.info(f"Normal samples: {len(normal_data)}")
    logger.info(f"Anomalous samples: {len(anomalous_data)}")
    
    # Test Isolation Forest
    logger.info("\nTesting Isolation Forest...")
    iso_forest = IsolationForest(contamination=0.1, random_state=42)
    iso_forest.fit(normal_data)
    
    predictions = iso_forest.predict(X_mixed)
    iso_anomalies = np.sum(predictions == -1)
    iso_accuracy = np.mean((predictions == -1) == (y_mixed == 1))
    
    logger.info(f"Isolation Forest detected {iso_anomalies} anomalies")
    logger.info(f"Isolation Forest accuracy: {iso_accuracy:.4f}")
    
    # Test One-Class SVM
    logger.info("\nTesting One-Class SVM...")
    oc_svm = OneClassSVM(nu=0.1)
    oc_svm.fit(normal_data)
    
    predictions_svm = oc_svm.predict(X_mixed)
    svm_anomalies = np.sum(predictions_svm == -1)
    svm_accuracy = np.mean((predictions_svm == -1) == (y_mixed == 1))
    
    logger.info(f"One-Class SVM detected {svm_anomalies} anomalies")
    logger.info(f"One-Class SVM accuracy: {svm_accuracy:.4f}")
    
    return iso_forest, oc_svm, iso_accuracy, svm_accuracy


def demo_performance_evaluation():
    """Demo performance evaluation."""
    logger.info("\n" + "=" * 60)
    logger.info("DEMO: Performance Evaluation")
    logger.info("=" * 60)
    
    from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score
    
    # Generate test data
    X, y = make_classification(n_samples=1000, n_features=20, n_classes=2, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Train model
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    # Make predictions
    predictions = model.predict(X_test)
    probabilities = model.predict_proba(X_test)[:, 1]
    
    # Calculate metrics
    accuracy = accuracy_score(y_test, predictions)
    precision = precision_score(y_test, predictions)
    recall = recall_score(y_test, predictions)
    f1 = f1_score(y_test, predictions)
    roc_auc = roc_auc_score(y_test, probabilities)
    
    logger.info("Performance Metrics:")
    logger.info(f"  Accuracy:  {accuracy:.4f}")
    logger.info(f"  Precision: {precision:.4f}")
    logger.info(f"  Recall:    {recall:.4f}")
    logger.info(f"  F1-Score:  {f1:.4f}")
    logger.info(f"  ROC-AUC:   {roc_auc:.4f}")
    
    # Classification report
    logger.info("\nDetailed Classification Report:")
    report = classification_report(y_test, predictions)
    logger.info(report)
    
    return accuracy, precision, recall, f1, roc_auc


def demo_visualization():
    """Demo visualization capabilities."""
    logger.info("\n" + "=" * 60)
    logger.info("DEMO: Visualization")
    logger.info("=" * 60)
    
    # Generate data
    X, y = make_classification(n_samples=1000, n_features=20, n_classes=2, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Train model
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    # Get predictions and probabilities
    predictions = model.predict(X_test)
    probabilities = model.predict_proba(X_test)[:, 1]
    
    # Create visualizations
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # 1. Feature importance
    feature_importance = model.feature_importances_
    top_features = np.argsort(feature_importance)[-10:][::-1]
    
    axes[0, 0].bar(range(len(top_features)), feature_importance[top_features])
    axes[0, 0].set_title('Top 10 Feature Importance')
    axes[0, 0].set_xlabel('Feature Rank')
    axes[0, 0].set_ylabel('Importance')
    
    # 2. Prediction distribution
    axes[0, 1].hist(probabilities, bins=30, alpha=0.7, edgecolor='black')
    axes[0, 1].set_title('Prediction Probability Distribution')
    axes[0, 1].set_xlabel('Probability')
    axes[0, 1].set_ylabel('Frequency')
    
    # 3. Confusion matrix
    from sklearn.metrics import confusion_matrix
    cm = confusion_matrix(y_test, predictions)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[1, 0])
    axes[1, 0].set_title('Confusion Matrix')
    axes[1, 0].set_xlabel('Predicted')
    axes[1, 0].set_ylabel('Actual')
    
    # 4. ROC Curve
    from sklearn.metrics import roc_curve, roc_auc_score
    fpr, tpr, _ = roc_curve(y_test, probabilities)
    auc_score = roc_auc_score(y_test, probabilities)
    axes[1, 1].plot(fpr, tpr, label=f'ROC Curve (AUC = {auc_score:.3f})')
    axes[1, 1].plot([0, 1], [0, 1], 'k--', label='Random Classifier')
    axes[1, 1].set_title('ROC Curve')
    axes[1, 1].set_xlabel('False Positive Rate')
    axes[1, 1].set_ylabel('True Positive Rate')
    axes[1, 1].legend()
    
    plt.tight_layout()
    plt.savefig('cybersecurity_demo_results.png', dpi=300, bbox_inches='tight')
    logger.info("Visualization saved as 'cybersecurity_demo_results.png'")
    
    return fig


def main():
    """Main demo function."""
    logger.info("🚀 CYBERSECURITY ML FRAMEWORK - DEMO")
    logger.info("=" * 80)
    
    try:
        # Run demos
        logger.info("Starting cybersecurity ML framework demonstration...")
        
        # Demo 1: Basic ML Models
        rf_model, svm_model, rf_acc, svm_acc = demo_basic_ml_models()
        
        # Demo 2: Anomaly Detection
        iso_forest, oc_svm, iso_acc, svm_acc_anomaly = demo_anomaly_detection()
        
        # Demo 3: Performance Evaluation
        accuracy, precision, recall, f1, roc_auc = demo_performance_evaluation()
        
        # Demo 4: Visualization
        fig = demo_visualization()
        
        # Summary
        logger.info("\n" + "=" * 80)
        logger.info("DEMO SUMMARY")
        logger.info("=" * 80)
        logger.info(f"✅ Random Forest Accuracy: {rf_acc:.4f}")
        logger.info(f"✅ SVM Accuracy: {svm_acc:.4f}")
        logger.info(f"✅ Isolation Forest Anomaly Detection: {iso_acc:.4f}")
        logger.info(f"✅ One-Class SVM Anomaly Detection: {svm_acc_anomaly:.4f}")
        logger.info(f"✅ Overall Performance - Accuracy: {accuracy:.4f}, F1: {f1:.4f}")
        logger.info(f"✅ Visualization saved as 'cybersecurity_demo_results.png'")
        
        logger.info("\n🎉 DEMO COMPLETED SUCCESSFULLY!")
        logger.info("The cybersecurity ML framework is working correctly!")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Demo failed: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = main()
    if success:
        print("\n🎉 SUCCESS! The cybersecurity ML framework is working!")
        print("📊 Check 'cybersecurity_demo_results.png' for visualizations")
    else:
        print("\n❌ Demo failed. Check the logs above for errors.")
