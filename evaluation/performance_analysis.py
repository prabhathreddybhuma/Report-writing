"""
Evaluation Metrics and Performance Analysis Tools for Cybersecurity ML Framework.

This module provides comprehensive evaluation metrics, performance analysis,
and benchmarking tools for cybersecurity ML models.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, roc_curve, precision_recall_curve, auc,
    confusion_matrix, classification_report, matthews_corrcoef,
    cohen_kappa_score, log_loss, brier_score_loss
)
from sklearn.model_selection import cross_val_score, validation_curve, learning_curve
import logging
from typing import Dict, List, Tuple, Optional, Any, Union
import time
import psutil
from dataclasses import dataclass
import json
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class PerformanceMetrics:
    """Data class for performance metrics."""
    accuracy: float
    precision: float
    recall: float
    f1_score: float
    roc_auc: float
    pr_auc: float
    specificity: float
    sensitivity: float
    matthews_corrcoef: float
    cohen_kappa: float
    log_loss: float
    brier_score: float
    false_positive_rate: float
    false_negative_rate: float


@dataclass
class ModelPerformance:
    """Data class for comprehensive model performance."""
    model_name: str
    metrics: PerformanceMetrics
    confusion_matrix: np.ndarray
    roc_curve: Tuple[np.ndarray, np.ndarray, np.ndarray]
    pr_curve: Tuple[np.ndarray, np.ndarray]
    training_time: float
    prediction_time: float
    memory_usage: float
    cross_val_scores: List[float]


class MetricsCalculator:
    """Calculator for various performance metrics."""
    
    def __init__(self):
        self.metrics_history = []
    
    def calculate_basic_metrics(self, y_true: np.ndarray, y_pred: np.ndarray, 
                               y_proba: np.ndarray = None) -> PerformanceMetrics:
        """Calculate basic performance metrics."""
        # Basic classification metrics
        accuracy = accuracy_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred, average='weighted', zero_division=0)
        recall = recall_score(y_true, y_pred, average='weighted', zero_division=0)
        f1 = f1_score(y_true, y_pred, average='weighted', zero_division=0)
        
        # Confusion matrix for additional metrics
        cm = confusion_matrix(y_true, y_pred)
        tn, fp, fn, tp = cm.ravel() if cm.size == 4 else (0, 0, 0, 0)
        
        # Specificity and sensitivity
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
        sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
        
        # False positive and negative rates
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
        fnr = fn / (fn + tp) if (fn + tp) > 0 else 0
        
        # Additional metrics
        mcc = matthews_corrcoef(y_true, y_pred)
        kappa = cohen_kappa_score(y_true, y_pred)
        
        # Probability-based metrics
        roc_auc = 0.0
        pr_auc = 0.0
        log_loss_val = 0.0
        brier_score = 0.0
        
        if y_proba is not None:
            try:
                # ROC AUC
                if len(np.unique(y_true)) == 2:  # Binary classification
                    roc_auc = roc_auc_score(y_true, y_proba[:, 1] if y_proba.shape[1] > 1 else y_proba)
                else:  # Multi-class
                    roc_auc = roc_auc_score(y_true, y_proba, multi_class='ovr', average='weighted')
                
                # Precision-Recall AUC
                precision_vals, recall_vals, _ = precision_recall_curve(y_true, y_proba[:, 1] if y_proba.shape[1] > 1 else y_proba)
                pr_auc = auc(recall_vals, precision_vals)
                
                # Log loss
                log_loss_val = log_loss(y_true, y_proba)
                
                # Brier score
                brier_score = brier_score_loss(y_true, y_proba[:, 1] if y_proba.shape[1] > 1 else y_proba)
                
            except Exception as e:
                logger.warning(f"Error calculating probability-based metrics: {e}")
        
        return PerformanceMetrics(
            accuracy=accuracy,
            precision=precision,
            recall=recall,
            f1_score=f1,
            roc_auc=roc_auc,
            pr_auc=pr_auc,
            specificity=specificity,
            sensitivity=sensitivity,
            matthews_corrcoef=mcc,
            cohen_kappa=kappa,
            log_loss=log_loss_val,
            brier_score=brier_score,
            false_positive_rate=fpr,
            false_negative_rate=fnr
        )
    
    def calculate_curves(self, y_true: np.ndarray, y_proba: np.ndarray) -> Tuple[Tuple, Tuple]:
        """Calculate ROC and Precision-Recall curves."""
        try:
            # ROC curve
            if len(np.unique(y_true)) == 2:  # Binary classification
                fpr, tpr, roc_thresholds = roc_curve(y_true, y_proba[:, 1] if y_proba.shape[1] > 1 else y_proba)
            else:  # Multi-class - use macro average
                fpr, tpr, roc_thresholds = roc_curve(y_true, y_proba, multi_class='ovr', average='macro')
            
            # Precision-Recall curve
            precision_vals, recall_vals, pr_thresholds = precision_recall_curve(
                y_true, y_proba[:, 1] if y_proba.shape[1] > 1 else y_proba
            )
            
            return (fpr, tpr, roc_thresholds), (precision_vals, recall_vals)
            
        except Exception as e:
            logger.error(f"Error calculating curves: {e}")
            return (np.array([]), np.array([]), np.array([])), (np.array([]), np.array([]))
    
    def calculate_cross_validation_scores(self, model, X: np.ndarray, y: np.ndarray, 
                                        cv: int = 5, scoring: str = 'accuracy') -> List[float]:
        """Calculate cross-validation scores."""
        try:
            scores = cross_val_score(model, X, y, cv=cv, scoring=scoring)
            return scores.tolist()
        except Exception as e:
            logger.error(f"Error calculating cross-validation scores: {e}")
            return []


class PerformanceAnalyzer:
    """Comprehensive performance analyzer for ML models."""
    
    def __init__(self):
        self.metrics_calculator = MetricsCalculator()
        self.performance_history = []
    
    def evaluate_model(self, model, X_test: np.ndarray, y_test: np.ndarray,
                      X_train: np.ndarray = None, y_train: np.ndarray = None,
                      model_name: str = "Model") -> ModelPerformance:
        """Evaluate a model comprehensively."""
        logger.info(f"Evaluating {model_name}...")
        
        # Measure training time
        training_time = 0.0
        if X_train is not None and y_train is not None:
            start_time = time.time()
            model.fit(X_train, y_train)
            training_time = time.time() - start_time
        
        # Measure prediction time
        start_time = time.time()
        y_pred = model.predict(X_test)
        prediction_time = time.time() - start_time
        
        # Get prediction probabilities if available
        y_proba = None
        if hasattr(model, 'predict_proba'):
            y_proba = model.predict_proba(X_test)
        
        # Calculate metrics
        metrics = self.metrics_calculator.calculate_basic_metrics(y_test, y_pred, y_proba)
        
        # Calculate curves
        roc_curve_data, pr_curve_data = self.metrics_calculator.calculate_curves(y_test, y_proba)
        
        # Calculate confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        
        # Calculate cross-validation scores
        cv_scores = []
        if X_train is not None and y_train is not None:
            cv_scores = self.metrics_calculator.calculate_cross_validation_scores(model, X_train, y_train)
        
        # Measure memory usage
        memory_usage = psutil.Process().memory_info().rss / 1024 / 1024  # MB
        
        # Create performance object
        performance = ModelPerformance(
            model_name=model_name,
            metrics=metrics,
            confusion_matrix=cm,
            roc_curve=roc_curve_data,
            pr_curve=pr_curve_data,
            training_time=training_time,
            prediction_time=prediction_time,
            memory_usage=memory_usage,
            cross_val_scores=cv_scores
        )
        
        # Store in history
        self.performance_history.append(performance)
        
        logger.info(f"Evaluation completed for {model_name}")
        return performance
    
    def compare_models(self, performances: List[ModelPerformance]) -> pd.DataFrame:
        """Compare multiple model performances."""
        comparison_data = []
        
        for perf in performances:
            comparison_data.append({
                'Model': perf.model_name,
                'Accuracy': perf.metrics.accuracy,
                'Precision': perf.metrics.precision,
                'Recall': perf.metrics.recall,
                'F1-Score': perf.metrics.f1_score,
                'ROC-AUC': perf.metrics.roc_auc,
                'PR-AUC': perf.metrics.pr_auc,
                'Specificity': perf.metrics.specificity,
                'Sensitivity': perf.metrics.sensitivity,
                'MCC': perf.metrics.matthews_corrcoef,
                'Cohen Kappa': perf.metrics.cohen_kappa,
                'Training Time (s)': perf.training_time,
                'Prediction Time (s)': perf.prediction_time,
                'Memory Usage (MB)': perf.memory_usage,
                'CV Score Mean': np.mean(perf.cross_val_scores) if perf.cross_val_scores else 0,
                'CV Score Std': np.std(perf.cross_val_scores) if perf.cross_val_scores else 0
            })
        
        return pd.DataFrame(comparison_data)
    
    def plot_performance_comparison(self, performances: List[ModelPerformance], 
                                  metrics: List[str] = None):
        """Plot performance comparison charts."""
        if metrics is None:
            metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'ROC-AUC']
        
        comparison_df = self.compare_models(performances)
        
        # Create subplots
        fig = make_subplots(
            rows=2, cols=3,
            subplot_titles=metrics[:6],
            specs=[[{"type": "bar"}, {"type": "bar"}, {"type": "bar"}],
                   [{"type": "bar"}, {"type": "bar"}, {"type": "bar"}]]
        )
        
        # Plot each metric
        for i, metric in enumerate(metrics[:6]):
            row = (i // 3) + 1
            col = (i % 3) + 1
            
            fig.add_trace(
                go.Bar(
                    x=comparison_df['Model'],
                    y=comparison_df[metric],
                    name=metric,
                    showlegend=False
                ),
                row=row, col=col
            )
        
        fig.update_layout(
            title="Model Performance Comparison",
            height=600,
            showlegend=False
        )
        
        fig.show()
    
    def plot_confusion_matrices(self, performances: List[ModelPerformance]):
        """Plot confusion matrices for all models."""
        n_models = len(performances)
        fig, axes = plt.subplots(1, n_models, figsize=(5 * n_models, 4))
        
        if n_models == 1:
            axes = [axes]
        
        for i, perf in enumerate(performances):
            sns.heatmap(perf.confusion_matrix, annot=True, fmt='d', 
                       cmap='Blues', ax=axes[i])
            axes[i].set_title(f'{perf.model_name} Confusion Matrix')
            axes[i].set_xlabel('Predicted')
            axes[i].set_ylabel('Actual')
        
        plt.tight_layout()
        plt.show()
    
    def plot_roc_curves(self, performances: List[ModelPerformance]):
        """Plot ROC curves for all models."""
        plt.figure(figsize=(10, 8))
        
        for perf in performances:
            fpr, tpr, _ = perf.roc_curve
            if len(fpr) > 0 and len(tpr) > 0:
                plt.plot(fpr, tpr, label=f'{perf.model_name} (AUC = {perf.metrics.roc_auc:.3f})')
        
        plt.plot([0, 1], [0, 1], 'k--', label='Random Classifier')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curves Comparison')
        plt.legend()
        plt.grid(True)
        plt.show()
    
    def plot_precision_recall_curves(self, performances: List[ModelPerformance]):
        """Plot Precision-Recall curves for all models."""
        plt.figure(figsize=(10, 8))
        
        for perf in performances:
            precision, recall = perf.pr_curve
            if len(precision) > 0 and len(recall) > 0:
                plt.plot(recall, precision, label=f'{perf.model_name} (PR-AUC = {perf.metrics.pr_auc:.3f})')
        
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title('Precision-Recall Curves Comparison')
        plt.legend()
        plt.grid(True)
        plt.show()


class BenchmarkSuite:
    """Benchmark suite for cybersecurity ML models."""
    
    def __init__(self):
        self.benchmark_results = {}
        self.performance_analyzer = PerformanceAnalyzer()
    
    def run_benchmark(self, models: Dict[str, Any], X_train: np.ndarray, y_train: np.ndarray,
                     X_test: np.ndarray, y_test: np.ndarray, 
                     benchmark_name: str = "Cybersecurity Benchmark") -> Dict[str, ModelPerformance]:
        """Run benchmark on multiple models."""
        logger.info(f"Running benchmark: {benchmark_name}")
        
        benchmark_results = {}
        
        for model_name, model in models.items():
            logger.info(f"Benchmarking {model_name}...")
            
            try:
                performance = self.performance_analyzer.evaluate_model(
                    model, X_test, y_test, X_train, y_train, model_name
                )
                benchmark_results[model_name] = performance
                
            except Exception as e:
                logger.error(f"Error benchmarking {model_name}: {e}")
                continue
        
        self.benchmark_results[benchmark_name] = benchmark_results
        
        logger.info(f"Benchmark {benchmark_name} completed")
        return benchmark_results
    
    def generate_benchmark_report(self, benchmark_name: str) -> str:
        """Generate a comprehensive benchmark report."""
        if benchmark_name not in self.benchmark_results:
            return f"Benchmark '{benchmark_name}' not found"
        
        results = self.benchmark_results[benchmark_name]
        performances = list(results.values())
        
        report = []
        report.append(f"=== {benchmark_name} Report ===\n")
        
        # Summary statistics
        comparison_df = self.performance_analyzer.compare_models(performances)
        
        report.append("=== Performance Summary ===")
        report.append(comparison_df.to_string(index=False))
        
        # Best performing models
        report.append("\n=== Best Performing Models ===")
        
        metrics_to_rank = ['Accuracy', 'F1-Score', 'ROC-AUC', 'Precision', 'Recall']
        for metric in metrics_to_rank:
            best_model = comparison_df.loc[comparison_df[metric].idxmax(), 'Model']
            best_score = comparison_df[metric].max()
            report.append(f"{metric}: {best_model} ({best_score:.4f})")
        
        # Performance insights
        report.append("\n=== Performance Insights ===")
        
        # Speed analysis
        fastest_model = comparison_df.loc[comparison_df['Prediction Time (s)'].idxmin(), 'Model']
        fastest_time = comparison_df['Prediction Time (s)'].min()
        report.append(f"Fastest Model: {fastest_model} ({fastest_time:.4f}s)")
        
        # Memory analysis
        most_efficient_model = comparison_df.loc[comparison_df['Memory Usage (MB)'].idxmin(), 'Model']
        most_efficient_memory = comparison_df['Memory Usage (MB)'].min()
        report.append(f"Most Memory Efficient: {most_efficient_model} ({most_efficient_memory:.2f}MB)")
        
        # Cross-validation analysis
        cv_scores = comparison_df[['Model', 'CV Score Mean', 'CV Score Std']].dropna()
        if not cv_scores.empty:
            most_stable_model = cv_scores.loc[cv_scores['CV Score Std'].idxmin(), 'Model']
            most_stable_std = cv_scores['CV Score Std'].min()
            report.append(f"Most Stable Model: {most_stable_model} (CV Std: {most_stable_std:.4f})")
        
        return "\n".join(report)
    
    def save_benchmark_results(self, benchmark_name: str, filepath: str):
        """Save benchmark results to file."""
        if benchmark_name not in self.benchmark_results:
            logger.error(f"Benchmark '{benchmark_name}' not found")
            return
        
        results = self.benchmark_results[benchmark_name]
        
        # Convert to serializable format
        serializable_results = {}
        for model_name, performance in results.items():
            serializable_results[model_name] = {
                'model_name': performance.model_name,
                'metrics': {
                    'accuracy': performance.metrics.accuracy,
                    'precision': performance.metrics.precision,
                    'recall': performance.metrics.recall,
                    'f1_score': performance.metrics.f1_score,
                    'roc_auc': performance.metrics.roc_auc,
                    'pr_auc': performance.metrics.pr_auc,
                    'specificity': performance.metrics.specificity,
                    'sensitivity': performance.metrics.sensitivity,
                    'matthews_corrcoef': performance.metrics.matthews_corrcoef,
                    'cohen_kappa': performance.metrics.cohen_kappa,
                    'log_loss': performance.metrics.log_loss,
                    'brier_score': performance.metrics.brier_score,
                    'false_positive_rate': performance.metrics.false_positive_rate,
                    'false_negative_rate': performance.metrics.false_negative_rate
                },
                'training_time': performance.training_time,
                'prediction_time': performance.prediction_time,
                'memory_usage': performance.memory_usage,
                'cross_val_scores': performance.cross_val_scores,
                'confusion_matrix': performance.confusion_matrix.tolist()
            }
        
        # Save to JSON file
        with open(filepath, 'w') as f:
            json.dump(serializable_results, f, indent=2)
        
        logger.info(f"Benchmark results saved to {filepath}")


class CybersecurityEvaluator:
    """Specialized evaluator for cybersecurity applications."""
    
    def __init__(self):
        self.benchmark_suite = BenchmarkSuite()
        self.performance_analyzer = PerformanceAnalyzer()
    
    def evaluate_threat_detection_performance(self, model, X_test: np.ndarray, y_test: np.ndarray,
                                           threat_types: List[str] = None) -> Dict[str, Any]:
        """Evaluate threat detection performance with cybersecurity-specific metrics."""
        logger.info("Evaluating threat detection performance...")
        
        # Basic evaluation
        performance = self.performance_analyzer.evaluate_model(model, X_test, y_test, model_name="Threat Detection")
        
        # Cybersecurity-specific analysis
        y_pred = model.predict(X_test)
        
        # Detection rate by threat type
        threat_detection_rates = {}
        if threat_types:
            for threat_type in threat_types:
                threat_mask = y_test == threat_type
                if np.any(threat_mask):
                    threat_predictions = y_pred[threat_mask]
                    detection_rate = np.mean(threat_predictions == threat_type)
                    threat_detection_rates[threat_type] = detection_rate
        
        # False positive analysis
        normal_mask = y_test == 'Normal'
        if np.any(normal_mask):
            normal_predictions = y_pred[normal_mask]
            false_positive_rate = np.mean(normal_predictions != 'Normal')
        else:
            false_positive_rate = 0.0
        
        # Response time analysis (simulated)
        response_times = np.random.exponential(0.1, len(y_test))  # Simulated response times
        
        cybersecurity_metrics = {
            'basic_performance': performance,
            'threat_detection_rates': threat_detection_rates,
            'false_positive_rate': false_positive_rate,
            'average_response_time': np.mean(response_times),
            'max_response_time': np.max(response_times),
            'threat_types_analyzed': threat_types
        }
        
        return cybersecurity_metrics
    
    def evaluate_anomaly_detection_performance(self, detector, X_test: np.ndarray, 
                                            y_test: np.ndarray) -> Dict[str, Any]:
        """Evaluate anomaly detection performance."""
        logger.info("Evaluating anomaly detection performance...")
        
        # Get predictions and scores
        y_pred = detector.predict(X_test)
        y_scores = detector.score_samples(X_test)
        
        # Convert to binary (1 for normal, 0 for anomaly)
        y_binary = (y_pred == 1).astype(int)
        y_true_binary = (y_test == 'Normal').astype(int)
        
        # Calculate metrics
        metrics = self.performance_analyzer.metrics_calculator.calculate_basic_metrics(
            y_true_binary, y_binary, y_scores
        )
        
        # Anomaly detection specific metrics
        anomaly_mask = y_test != 'Normal'
        if np.any(anomaly_mask):
            anomaly_detection_rate = np.mean(y_pred[anomaly_mask] == -1)
        else:
            anomaly_detection_rate = 0.0
        
        normal_mask = y_test == 'Normal'
        if np.any(normal_mask):
            false_alarm_rate = np.mean(y_pred[normal_mask] == -1)
        else:
            false_alarm_rate = 0.0
        
        anomaly_metrics = {
            'detection_metrics': metrics,
            'anomaly_detection_rate': anomaly_detection_rate,
            'false_alarm_rate': false_alarm_rate,
            'score_distribution': {
                'mean': np.mean(y_scores),
                'std': np.std(y_scores),
                'min': np.min(y_scores),
                'max': np.max(y_scores)
            }
        }
        
        return anomaly_metrics


def main():
    """Example usage of evaluation metrics and performance analysis."""
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.svm import SVC
    from sklearn.datasets import make_classification
    
    # Generate synthetic data
    X, y = make_classification(n_samples=2000, n_features=20, n_classes=5, random_state=42)
    
    # Split data
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Create models
    models = {
        'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
        'SVM': SVC(probability=True, random_state=42)
    }
    
    # Create evaluator
    evaluator = CybersecurityEvaluator()
    
    # Run benchmark
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


if __name__ == "__main__":
    main()
