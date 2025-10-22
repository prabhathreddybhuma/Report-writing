"""
Anomaly Detection and Threat Classification System.

This module implements various anomaly detection algorithms and threat
classification systems for cybersecurity applications.
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.cluster import DBSCAN, KMeans
from sklearn.svm import OneClassSVM
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models
import logging
from typing import Dict, List, Tuple, Optional, Any, Union
import warnings
from scipy import stats
from scipy.spatial.distance import pdist, squareform
import matplotlib.pyplot as plt
import seaborn as sns

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class AnomalyDetector:
    """Base class for anomaly detection algorithms."""
    
    def __init__(self, name: str):
        self.name = name
        self.model = None
        self.is_trained = False
        self.scaler = StandardScaler()
        
    def fit(self, X: np.ndarray) -> 'AnomalyDetector':
        """Fit the anomaly detector to normal data."""
        raise NotImplementedError
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict anomalies (1 for normal, -1 for anomaly)."""
        raise NotImplementedError
    
    def score_samples(self, X: np.ndarray) -> np.ndarray:
        """Get anomaly scores for samples."""
        raise NotImplementedError


class IsolationForestDetector(AnomalyDetector):
    """Isolation Forest anomaly detector."""
    
    def __init__(self, contamination: float = 0.1, n_estimators: int = 100, 
                 max_samples: Union[int, float] = 'auto', random_state: int = 42):
        super().__init__("IsolationForest")
        self.model = IsolationForest(
            contamination=contamination,
            n_estimators=n_estimators,
            max_samples=max_samples,
            random_state=random_state
        )
        self.contamination = contamination
    
    def fit(self, X: np.ndarray) -> 'IsolationForestDetector':
        """Fit the Isolation Forest model."""
        logger.info(f"Training {self.name} anomaly detector...")
        
        # Scale the data
        X_scaled = self.scaler.fit_transform(X)
        
        # Fit the model
        self.model.fit(X_scaled)
        self.is_trained = True
        
        logger.info(f"{self.name} training completed")
        return self
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict anomalies."""
        if not self.is_trained:
            raise ValueError("Model must be trained before making predictions")
        
        X_scaled = self.scaler.transform(X)
        return self.model.predict(X_scaled)
    
    def score_samples(self, X: np.ndarray) -> np.ndarray:
        """Get anomaly scores."""
        if not self.is_trained:
            raise ValueError("Model must be trained before scoring")
        
        X_scaled = self.scaler.transform(X)
        return self.model.score_samples(X_scaled)


class OneClassSVMDetector(AnomalyDetector):
    """One-Class SVM anomaly detector."""
    
    def __init__(self, kernel: str = 'rbf', gamma: str = 'scale', nu: float = 0.1):
        super().__init__("OneClassSVM")
        self.model = OneClassSVM(kernel=kernel, gamma=gamma, nu=nu)
        self.nu = nu
    
    def fit(self, X: np.ndarray) -> 'OneClassSVMDetector':
        """Fit the One-Class SVM model."""
        logger.info(f"Training {self.name} anomaly detector...")
        
        # Scale the data
        X_scaled = self.scaler.fit_transform(X)
        
        # Fit the model
        self.model.fit(X_scaled)
        self.is_trained = True
        
        logger.info(f"{self.name} training completed")
        return self
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict anomalies."""
        if not self.is_trained:
            raise ValueError("Model must be trained before making predictions")
        
        X_scaled = self.scaler.transform(X)
        return self.model.predict(X_scaled)
    
    def score_samples(self, X: np.ndarray) -> np.ndarray:
        """Get anomaly scores."""
        if not self.is_trained:
            raise ValueError("Model must be trained before scoring")
        
        X_scaled = self.scaler.transform(X)
        return self.model.decision_function(X_scaled)


class DBSCANDetector(AnomalyDetector):
    """DBSCAN-based anomaly detector."""
    
    def __init__(self, eps: float = 0.5, min_samples: int = 5):
        super().__init__("DBSCAN")
        self.model = DBSCAN(eps=eps, min_samples=min_samples)
        self.eps = eps
        self.min_samples = min_samples
    
    def fit(self, X: np.ndarray) -> 'DBSCANDetector':
        """Fit the DBSCAN model."""
        logger.info(f"Training {self.name} anomaly detector...")
        
        # Scale the data
        X_scaled = self.scaler.fit_transform(X)
        
        # Fit the model
        self.model.fit(X_scaled)
        self.is_trained = True
        
        logger.info(f"{self.name} training completed")
        return self
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict anomalies."""
        if not self.is_trained:
            raise ValueError("Model must be trained before making predictions")
        
        X_scaled = self.scaler.transform(X)
        labels = self.model.fit_predict(X_scaled)
        
        # Convert to anomaly format (-1 for anomaly, 1 for normal)
        predictions = np.where(labels == -1, -1, 1)
        return predictions
    
    def score_samples(self, X: np.ndarray) -> np.ndarray:
        """Get anomaly scores based on distance to nearest cluster."""
        if not self.is_trained:
            raise ValueError("Model must be trained before scoring")
        
        X_scaled = self.scaler.transform(X)
        
        # Calculate distances to core samples
        core_samples = self.model.core_sample_indices_
        if len(core_samples) == 0:
            # If no core samples, use all samples
            core_samples = np.arange(len(self.model.components_))
        
        distances = []
        for sample in X_scaled:
            min_dist = np.min(np.linalg.norm(self.model.components_[core_samples] - sample, axis=1))
            distances.append(min_dist)
        
        return np.array(distances)


class AutoencoderDetector(AnomalyDetector):
    """Autoencoder-based anomaly detector."""
    
    def __init__(self, input_dim: int, encoding_dim: int = 32, 
                 hidden_dims: List[int] = None, learning_rate: float = 0.001):
        super().__init__("Autoencoder")
        
        if hidden_dims is None:
            hidden_dims = [64, 32]
        
        self.input_dim = input_dim
        self.encoding_dim = encoding_dim
        self.hidden_dims = hidden_dims
        self.learning_rate = learning_rate
        self.threshold = None
        
        self.model = self._build_autoencoder()
    
    def _build_autoencoder(self) -> keras.Model:
        """Build the autoencoder architecture."""
        # Encoder
        encoder_input = keras.Input(shape=(self.input_dim,))
        x = encoder_input
        
        for dim in self.hidden_dims:
            x = layers.Dense(dim, activation='relu')(x)
            x = layers.Dropout(0.2)(x)
        
        encoded = layers.Dense(self.encoding_dim, activation='relu')(x)
        
        # Decoder
        x = encoded
        for dim in reversed(self.hidden_dims):
            x = layers.Dense(dim, activation='relu')(x)
            x = layers.Dropout(0.2)(x)
        
        decoded = layers.Dense(self.input_dim, activation='sigmoid')(x)
        
        # Autoencoder
        autoencoder = keras.Model(encoder_input, decoded)
        autoencoder.compile(optimizer=keras.optimizers.Adam(self.learning_rate), loss='mse')
        
        return autoencoder
    
    def fit(self, X: np.ndarray, epochs: int = 100, batch_size: int = 32,
            validation_split: float = 0.1) -> 'AutoencoderDetector':
        """Fit the autoencoder model."""
        logger.info(f"Training {self.name} anomaly detector...")
        
        # Scale the data
        X_scaled = self.scaler.fit_transform(X)
        
        # Train the autoencoder
        history = self.model.fit(
            X_scaled, X_scaled,
            epochs=epochs,
            batch_size=batch_size,
            validation_split=validation_split,
            verbose=0
        )
        
        # Calculate reconstruction error threshold
        X_reconstructed = self.model.predict(X_scaled)
        reconstruction_errors = np.mean(np.square(X_scaled - X_reconstructed), axis=1)
        self.threshold = np.percentile(reconstruction_errors, 95)  # 95th percentile
        
        self.is_trained = True
        
        logger.info(f"{self.name} training completed. Threshold: {self.threshold:.4f}")
        return self
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict anomalies based on reconstruction error."""
        if not self.is_trained:
            raise ValueError("Model must be trained before making predictions")
        
        X_scaled = self.scaler.transform(X)
        X_reconstructed = self.model.predict(X_scaled)
        reconstruction_errors = np.mean(np.square(X_scaled - X_reconstructed), axis=1)
        
        # Anomaly if reconstruction error > threshold
        predictions = np.where(reconstruction_errors > self.threshold, -1, 1)
        return predictions
    
    def score_samples(self, X: np.ndarray) -> np.ndarray:
        """Get anomaly scores based on reconstruction error."""
        if not self.is_trained:
            raise ValueError("Model must be trained before scoring")
        
        X_scaled = self.scaler.transform(X)
        X_reconstructed = self.model.predict(X_scaled)
        reconstruction_errors = np.mean(np.square(X_scaled - X_reconstructed), axis=1)
        
        return reconstruction_errors


class StatisticalAnomalyDetector(AnomalyDetector):
    """Statistical-based anomaly detector using Z-score and IQR methods."""
    
    def __init__(self, method: str = 'zscore', threshold: float = 3.0):
        super().__init__(f"Statistical_{method}")
        self.method = method
        self.threshold = threshold
        self.stats_params = {}
    
    def fit(self, X: np.ndarray) -> 'StatisticalAnomalyDetector':
        """Fit the statistical model."""
        logger.info(f"Training {self.name} anomaly detector...")
        
        # Scale the data
        X_scaled = self.scaler.fit_transform(X)
        
        if self.method == 'zscore':
            self.stats_params = {
                'mean': np.mean(X_scaled, axis=0),
                'std': np.std(X_scaled, axis=0)
            }
        elif self.method == 'iqr':
            self.stats_params = {
                'q25': np.percentile(X_scaled, 25, axis=0),
                'q75': np.percentile(X_scaled, 75, axis=0),
                'iqr': np.percentile(X_scaled, 75, axis=0) - np.percentile(X_scaled, 25, axis=0)
            }
        
        self.is_trained = True
        logger.info(f"{self.name} training completed")
        return self
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict anomalies using statistical methods."""
        if not self.is_trained:
            raise ValueError("Model must be trained before making predictions")
        
        X_scaled = self.scaler.transform(X)
        
        if self.method == 'zscore':
            z_scores = np.abs((X_scaled - self.stats_params['mean']) / self.stats_params['std'])
            anomalies = np.any(z_scores > self.threshold, axis=1)
        elif self.method == 'iqr':
            lower_bound = self.stats_params['q25'] - 1.5 * self.stats_params['iqr']
            upper_bound = self.stats_params['q75'] + 1.5 * self.stats_params['iqr']
            anomalies = np.any((X_scaled < lower_bound) | (X_scaled > upper_bound), axis=1)
        
        predictions = np.where(anomalies, -1, 1)
        return predictions
    
    def score_samples(self, X: np.ndarray) -> np.ndarray:
        """Get anomaly scores."""
        if not self.is_trained:
            raise ValueError("Model must be trained before scoring")
        
        X_scaled = self.scaler.transform(X)
        
        if self.method == 'zscore':
            z_scores = np.abs((X_scaled - self.stats_params['mean']) / self.stats_params['std'])
            scores = np.max(z_scores, axis=1)
        elif self.method == 'iqr':
            lower_bound = self.stats_params['q25'] - 1.5 * self.stats_params['iqr']
            upper_bound = self.stats_params['q75'] + 1.5 * self.stats_params['iqr']
            distances = np.maximum(
                np.maximum(0, lower_bound - X_scaled),
                np.maximum(0, X_scaled - upper_bound)
            )
            scores = np.max(distances, axis=1)
        
        return scores


class EnsembleAnomalyDetector(AnomalyDetector):
    """Ensemble anomaly detector combining multiple methods."""
    
    def __init__(self, detectors: List[AnomalyDetector], weights: List[float] = None):
        super().__init__("Ensemble")
        self.detectors = detectors
        self.weights = weights or [1.0 / len(detectors)] * len(detectors)
        
        if len(self.weights) != len(detectors):
            raise ValueError("Number of weights must match number of detectors")
    
    def fit(self, X: np.ndarray) -> 'EnsembleAnomalyDetector':
        """Fit all detectors in the ensemble."""
        logger.info(f"Training {self.name} anomaly detector with {len(self.detectors)} detectors...")
        
        for i, detector in enumerate(self.detectors):
            logger.info(f"Training detector {i+1}/{len(self.detectors)}: {detector.name}")
            detector.fit(X)
        
        self.is_trained = True
        logger.info(f"{self.name} training completed")
        return self
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make ensemble predictions."""
        if not self.is_trained:
            raise ValueError("Ensemble must be trained before making predictions")
        
        predictions = []
        for detector in self.detectors:
            pred = detector.predict(X)
            predictions.append(pred)
        
        # Weighted voting
        weighted_predictions = []
        for i, pred in enumerate(predictions):
            weighted_predictions.append(pred * self.weights[i])
        
        ensemble_pred = np.sum(weighted_predictions, axis=0)
        final_predictions = np.where(ensemble_pred < 0, -1, 1)
        
        return final_predictions
    
    def score_samples(self, X: np.ndarray) -> np.ndarray:
        """Get ensemble anomaly scores."""
        if not self.is_trained:
            raise ValueError("Ensemble must be trained before scoring")
        
        scores = []
        for detector in self.detectors:
            score = detector.score_samples(X)
            scores.append(score)
        
        # Weighted average of scores
        weighted_scores = []
        for i, score in enumerate(scores):
            weighted_scores.append(score * self.weights[i])
        
        ensemble_scores = np.sum(weighted_scores, axis=0)
        return ensemble_scores


class ThreatClassifier:
    """Threat classification system for cybersecurity."""
    
    def __init__(self, model=None):
        self.model = model
        self.is_trained = False
        self.threat_types = ['Normal', 'DoS', 'Probe', 'R2L', 'U2R']
        self.scaler = StandardScaler()
    
    def train(self, X_train: np.ndarray, y_train: np.ndarray,
              X_val: np.ndarray = None, y_val: np.ndarray = None) -> Dict[str, Any]:
        """Train the threat classifier."""
        logger.info("Training threat classifier...")
        
        # Scale the data
        X_train_scaled = self.scaler.fit_transform(X_train)
        
        if X_val is not None:
            X_val_scaled = self.scaler.transform(X_val)
        
        # Train the model
        if self.model is None:
            from sklearn.ensemble import RandomForestClassifier
            self.model = RandomForestClassifier(n_estimators=100, random_state=42)
        
        self.model.fit(X_train_scaled, y_train)
        self.is_trained = True
        
        # Calculate performance metrics
        train_score = self.model.score(X_train_scaled, y_train)
        val_score = self.model.score(X_val_scaled, y_val) if X_val is not None else None
        
        training_info = {
            'train_accuracy': train_score,
            'val_accuracy': val_score,
            'num_classes': len(np.unique(y_train)),
            'class_distribution': dict(zip(*np.unique(y_train, return_counts=True)))
        }
        
        logger.info(f"Threat classifier training completed. Train accuracy: {train_score:.4f}")
        if val_score:
            logger.info(f"Validation accuracy: {val_score:.4f}")
        
        return training_info
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict threat types."""
        if not self.is_trained:
            raise ValueError("Model must be trained before making predictions")
        
        X_scaled = self.scaler.transform(X)
        return self.model.predict(X_scaled)
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict threat type probabilities."""
        if not self.is_trained:
            raise ValueError("Model must be trained before making predictions")
        
        X_scaled = self.scaler.transform(X)
        return self.model.predict_proba(X_scaled)
    
    def get_threat_severity(self, predictions: np.ndarray) -> np.ndarray:
        """Get threat severity scores."""
        severity_map = {
            'Normal': 0,
            'DoS': 3,
            'Probe': 2,
            'R2L': 4,
            'U2R': 5
        }
        
        severities = []
        for pred in predictions:
            if pred in severity_map:
                severities.append(severity_map[pred])
            else:
                severities.append(1)  # Default severity
        
        return np.array(severities)


class CybersecurityAnomalyDetectionSystem:
    """Main system for cybersecurity anomaly detection and threat classification."""
    
    def __init__(self):
        self.anomaly_detectors = {}
        self.threat_classifier = None
        self.is_trained = False
    
    def add_anomaly_detector(self, name: str, detector: AnomalyDetector):
        """Add an anomaly detector to the system."""
        self.anomaly_detectors[name] = detector
        logger.info(f"Added anomaly detector: {name}")
    
    def train_system(self, X_normal: np.ndarray, X_mixed: np.ndarray = None, 
                    y_mixed: np.ndarray = None) -> Dict[str, Any]:
        """Train the entire cybersecurity system."""
        logger.info("Training cybersecurity anomaly detection system...")
        
        training_results = {}
        
        # Train anomaly detectors on normal data
        for name, detector in self.anomaly_detectors.items():
            logger.info(f"Training anomaly detector: {name}")
            detector.fit(X_normal)
            training_results[f"{name}_trained"] = True
        
        # Train threat classifier if mixed data is provided
        if X_mixed is not None and y_mixed is not None:
            logger.info("Training threat classifier...")
            self.threat_classifier = ThreatClassifier()
            classifier_results = self.threat_classifier.train(X_mixed, y_mixed)
            training_results['threat_classifier'] = classifier_results
        
        self.is_trained = True
        logger.info("Cybersecurity system training completed")
        
        return training_results
    
    def detect_anomalies(self, X: np.ndarray) -> Dict[str, Any]:
        """Detect anomalies using all detectors."""
        if not self.is_trained:
            raise ValueError("System must be trained before detection")
        
        results = {}
        
        for name, detector in self.anomaly_detectors.items():
            predictions = detector.predict(X)
            scores = detector.score_samples(X)
            
            results[name] = {
                'predictions': predictions,
                'scores': scores,
                'anomaly_count': np.sum(predictions == -1),
                'anomaly_rate': np.mean(predictions == -1)
            }
        
        return results
    
    def classify_threats(self, X: np.ndarray) -> Dict[str, Any]:
        """Classify threats if threat classifier is available."""
        if self.threat_classifier is None or not self.threat_classifier.is_trained:
            logger.warning("Threat classifier not available")
            return {}
        
        predictions = self.threat_classifier.predict(X)
        probabilities = self.threat_classifier.predict_proba(X)
        severities = self.threat_classifier.get_threat_severity(predictions)
        
        return {
            'predictions': predictions,
            'probabilities': probabilities,
            'severities': severities,
            'threat_types': self.threat_classifier.threat_types
        }
    
    def comprehensive_analysis(self, X: np.ndarray) -> Dict[str, Any]:
        """Perform comprehensive cybersecurity analysis."""
        logger.info("Performing comprehensive cybersecurity analysis...")
        
        analysis_results = {
            'anomaly_detection': self.detect_anomalies(X),
            'threat_classification': self.classify_threats(X),
            'summary': {}
        }
        
        # Generate summary
        anomaly_results = analysis_results['anomaly_detection']
        threat_results = analysis_results['threat_classification']
        
        # Calculate overall anomaly rate
        if anomaly_results:
            anomaly_rates = [result['anomaly_rate'] for result in anomaly_results.values()]
            analysis_results['summary']['overall_anomaly_rate'] = np.mean(anomaly_rates)
            analysis_results['summary']['max_anomaly_rate'] = np.max(anomaly_rates)
        
        # Calculate threat statistics
        if threat_results and 'predictions' in threat_results:
            threat_predictions = threat_results['predictions']
            unique_threats, threat_counts = np.unique(threat_predictions, return_counts=True)
            analysis_results['summary']['threat_distribution'] = dict(zip(unique_threats, threat_counts))
            analysis_results['summary']['threat_rate'] = np.mean(threat_predictions != 'Normal')
        
        logger.info("Comprehensive analysis completed")
        return analysis_results


def create_default_anomaly_detection_system() -> CybersecurityAnomalyDetectionSystem:
    """Create a default anomaly detection system with multiple detectors."""
    system = CybersecurityAnomalyDetectionSystem()
    
    # Add various anomaly detectors
    system.add_anomaly_detector('isolation_forest', IsolationForestDetector())
    system.add_anomaly_detector('one_class_svm', OneClassSVMDetector())
    system.add_anomaly_detector('dbscan', DBSCANDetector())
    system.add_anomaly_detector('statistical_zscore', StatisticalAnomalyDetector('zscore'))
    system.add_anomaly_detector('statistical_iqr', StatisticalAnomalyDetector('iqr'))
    
    return system


def main():
    """Example usage of the anomaly detection and threat classification system."""
    from sklearn.datasets import make_classification
    
    # Generate synthetic data
    X_normal, _ = make_classification(n_samples=1000, n_features=20, n_classes=2, 
                                    n_clusters_per_class=1, random_state=42)
    
    X_mixed, y_mixed = make_classification(n_samples=2000, n_features=20, n_classes=5,
                                         n_clusters_per_class=1, random_state=42)
    
    # Create and train the system
    system = create_default_anomaly_detection_system()
    
    # Train the system
    training_results = system.train_system(X_normal, X_mixed, y_mixed)
    print("Training completed:", training_results)
    
    # Test the system
    X_test, y_test = make_classification(n_samples=500, n_features=20, n_classes=5,
                                        n_clusters_per_class=1, random_state=123)
    
    # Comprehensive analysis
    analysis_results = system.comprehensive_analysis(X_test)
    
    print("\n=== Anomaly Detection Results ===")
    for detector_name, results in analysis_results['anomaly_detection'].items():
        print(f"{detector_name}: {results['anomaly_count']} anomalies ({results['anomaly_rate']:.2%})")
    
    print("\n=== Threat Classification Results ===")
    if analysis_results['threat_classification']:
        threat_dist = analysis_results['summary']['threat_distribution']
        print(f"Threat distribution: {threat_dist}")
        print(f"Overall threat rate: {analysis_results['summary']['threat_rate']:.2%}")


if __name__ == "__main__":
    main()
