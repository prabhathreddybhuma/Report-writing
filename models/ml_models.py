"""
Machine Learning and Deep Learning Models for Cybersecurity Threat Detection.

This module implements various ML/DL models including Random Forest, SVM,
CNN, RNN, and Ensemble methods for threat detection and anomaly analysis.
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, VotingClassifier, BaggingClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import cross_val_score, GridSearchCV
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models, optimizers, callbacks
import logging
from typing import Dict, List, Tuple, Optional, Any
import joblib
import os
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class BaseModel:
    """Base class for all ML models."""
    
    def __init__(self, model_name: str):
        self.model_name = model_name
        self.model = None
        self.is_trained = False
        self.training_history = None
        
    def train(self, X_train: np.ndarray, y_train: np.ndarray, 
              X_val: Optional[np.ndarray] = None, y_val: Optional[np.ndarray] = None) -> Dict[str, Any]:
        """Train the model."""
        raise NotImplementedError
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions."""
        if not self.is_trained:
            raise ValueError("Model must be trained before making predictions")
        raise NotImplementedError
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict class probabilities."""
        if not self.is_trained:
            raise ValueError("Model must be trained before making predictions")
        raise NotImplementedError
    
    def save_model(self, filepath: str):
        """Save the trained model."""
        raise NotImplementedError
    
    def load_model(self, filepath: str):
        """Load a pre-trained model."""
        raise NotImplementedError


class RandomForestModel(BaseModel):
    """Random Forest classifier for threat detection."""
    
    def __init__(self, n_estimators: int = 100, max_depth: Optional[int] = None,
                 min_samples_split: int = 2, min_samples_leaf: int = 1,
                 random_state: int = 42):
        super().__init__("RandomForest")
        self.model = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            random_state=random_state,
            n_jobs=-1
        )
    
    def train(self, X_train: np.ndarray, y_train: np.ndarray, 
              X_val: Optional[np.ndarray] = None, y_val: Optional[np.ndarray] = None) -> Dict[str, Any]:
        """Train the Random Forest model."""
        logger.info(f"Training {self.model_name}...")
        
        self.model.fit(X_train, y_train)
        self.is_trained = True
        
        # Calculate training accuracy
        train_score = self.model.score(X_train, y_train)
        val_score = self.model.score(X_val, y_val) if X_val is not None else None
        
        self.training_history = {
            'train_accuracy': train_score,
            'val_accuracy': val_score,
            'feature_importance': self.model.feature_importances_
        }
        
        logger.info(f"Training completed. Train accuracy: {train_score:.4f}")
        if val_score:
            logger.info(f"Validation accuracy: {val_score:.4f}")
        
        return self.training_history
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions."""
        return self.model.predict(X)
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict class probabilities."""
        return self.model.predict_proba(X)
    
    def get_feature_importance(self) -> np.ndarray:
        """Get feature importance scores."""
        if not self.is_trained:
            raise ValueError("Model must be trained first")
        return self.model.feature_importances_
    
    def save_model(self, filepath: str):
        """Save the trained model."""
        joblib.dump(self.model, filepath)
        logger.info(f"Model saved to {filepath}")
    
    def load_model(self, filepath: str):
        """Load a pre-trained model."""
        self.model = joblib.load(filepath)
        self.is_trained = True
        logger.info(f"Model loaded from {filepath}")


class SVMModel(BaseModel):
    """Support Vector Machine classifier for threat detection."""
    
    def __init__(self, C: float = 1.0, kernel: str = 'rbf', gamma: str = 'scale',
                 random_state: int = 42):
        super().__init__("SVM")
        self.model = SVC(
            C=C,
            kernel=kernel,
            gamma=gamma,
            random_state=random_state,
            probability=True
        )
    
    def train(self, X_train: np.ndarray, y_train: np.ndarray, 
              X_val: Optional[np.ndarray] = None, y_val: Optional[np.ndarray] = None) -> Dict[str, Any]:
        """Train the SVM model."""
        logger.info(f"Training {self.model_name}...")
        
        self.model.fit(X_train, y_train)
        self.is_trained = True
        
        # Calculate training accuracy
        train_score = self.model.score(X_train, y_train)
        val_score = self.model.score(X_val, y_val) if X_val is not None else None
        
        self.training_history = {
            'train_accuracy': train_score,
            'val_accuracy': val_score
        }
        
        logger.info(f"Training completed. Train accuracy: {train_score:.4f}")
        if val_score:
            logger.info(f"Validation accuracy: {val_score:.4f}")
        
        return self.training_history
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions."""
        return self.model.predict(X)
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict class probabilities."""
        return self.model.predict_proba(X)
    
    def save_model(self, filepath: str):
        """Save the trained model."""
        joblib.dump(self.model, filepath)
        logger.info(f"Model saved to {filepath}")
    
    def load_model(self, filepath: str):
        """Load a pre-trained model."""
        self.model = joblib.load(filepath)
        self.is_trained = True
        logger.info(f"Model loaded from {filepath}")


class CNNModel(BaseModel):
    """Convolutional Neural Network for threat detection."""
    
    def __init__(self, input_shape: Tuple[int, ...], num_classes: int = 2,
                 filters: List[int] = None, kernel_sizes: List[int] = None,
                 pool_sizes: List[int] = None, dropout: float = 0.2):
        super().__init__("CNN")
        
        if filters is None:
            filters = [32, 64, 128]
        if kernel_sizes is None:
            kernel_sizes = [3, 3, 3]
        if pool_sizes is None:
            pool_sizes = [2, 2, 2]
        
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.model = self._build_model(filters, kernel_sizes, pool_sizes, dropout)
    
    def _build_model(self, filters: List[int], kernel_sizes: List[int], 
                    pool_sizes: List[int], dropout: float) -> keras.Model:
        """Build the CNN architecture."""
        model = models.Sequential()
        
        # Reshape input for CNN
        model.add(layers.Reshape((*self.input_shape, 1), input_shape=self.input_shape))
        
        # Convolutional layers
        for i, (f, k, p) in enumerate(zip(filters, kernel_sizes, pool_sizes)):
            model.add(layers.Conv2D(f, k, activation='relu', padding='same'))
            model.add(layers.BatchNormalization())
            model.add(layers.MaxPooling2D(p))
            model.add(layers.Dropout(dropout))
        
        # Flatten and dense layers
        model.add(layers.Flatten())
        model.add(layers.Dense(128, activation='relu'))
        model.add(layers.Dropout(dropout))
        model.add(layers.Dense(64, activation='relu'))
        model.add(layers.Dropout(dropout))
        
        # Output layer
        if self.num_classes == 2:
            model.add(layers.Dense(1, activation='sigmoid'))
            model.compile(
                optimizer='adam',
                loss='binary_crossentropy',
                metrics=['accuracy']
            )
        else:
            model.add(layers.Dense(self.num_classes, activation='softmax'))
            model.compile(
                optimizer='adam',
                loss='sparse_categorical_crossentropy',
                metrics=['accuracy']
            )
        
        return model
    
    def train(self, X_train: np.ndarray, y_train: np.ndarray, 
              X_val: Optional[np.ndarray] = None, y_val: Optional[np.ndarray] = None,
              epochs: int = 100, batch_size: int = 32) -> Dict[str, Any]:
        """Train the CNN model."""
        logger.info(f"Training {self.model_name}...")
        
        # Prepare validation data
        validation_data = (X_val, y_val) if X_val is not None else None
        
        # Callbacks
        callbacks_list = [
            callbacks.EarlyStopping(patience=10, restore_best_weights=True),
            callbacks.ReduceLROnPlateau(factor=0.5, patience=5)
        ]
        
        # Train the model
        history = self.model.fit(
            X_train, y_train,
            validation_data=validation_data,
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks_list,
            verbose=1
        )
        
        self.is_trained = True
        self.training_history = history.history
        
        logger.info(f"Training completed. Final accuracy: {history.history['accuracy'][-1]:.4f}")
        
        return self.training_history
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions."""
        predictions = self.model.predict(X)
        if self.num_classes == 2:
            return (predictions > 0.5).astype(int).flatten()
        else:
            return np.argmax(predictions, axis=1)
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict class probabilities."""
        return self.model.predict(X)
    
    def save_model(self, filepath: str):
        """Save the trained model."""
        self.model.save(filepath)
        logger.info(f"Model saved to {filepath}")
    
    def load_model(self, filepath: str):
        """Load a pre-trained model."""
        self.model = keras.models.load_model(filepath)
        self.is_trained = True
        logger.info(f"Model loaded from {filepath}")


class RNNModel(BaseModel):
    """Recurrent Neural Network for threat detection."""
    
    def __init__(self, input_shape: Tuple[int, ...], num_classes: int = 2,
                 rnn_units: List[int] = None, dropout: float = 0.2,
                 recurrent_dropout: float = 0.2):
        super().__init__("RNN")
        
        if rnn_units is None:
            rnn_units = [64, 32]
        
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.model = self._build_model(rnn_units, dropout, recurrent_dropout)
    
    def _build_model(self, rnn_units: List[int], dropout: float, 
                    recurrent_dropout: float) -> keras.Model:
        """Build the RNN architecture."""
        model = models.Sequential()
        
        # Reshape input for RNN (sequence_length, features)
        if len(self.input_shape) == 1:
            # If 1D input, treat as sequence
            model.add(layers.Reshape((self.input_shape[0], 1), input_shape=self.input_shape))
        
        # LSTM layers
        for i, units in enumerate(rnn_units):
            return_sequences = i < len(rnn_units) - 1
            model.add(layers.LSTM(
                units,
                return_sequences=return_sequences,
                dropout=dropout,
                recurrent_dropout=recurrent_dropout
            ))
            model.add(layers.BatchNormalization())
        
        # Dense layers
        model.add(layers.Dense(64, activation='relu'))
        model.add(layers.Dropout(dropout))
        model.add(layers.Dense(32, activation='relu'))
        model.add(layers.Dropout(dropout))
        
        # Output layer
        if self.num_classes == 2:
            model.add(layers.Dense(1, activation='sigmoid'))
            model.compile(
                optimizer='adam',
                loss='binary_crossentropy',
                metrics=['accuracy']
            )
        else:
            model.add(layers.Dense(self.num_classes, activation='softmax'))
            model.compile(
                optimizer='adam',
                loss='sparse_categorical_crossentropy',
                metrics=['accuracy']
            )
        
        return model
    
    def train(self, X_train: np.ndarray, y_train: np.ndarray, 
              X_val: Optional[np.ndarray] = None, y_val: Optional[np.ndarray] = None,
              epochs: int = 100, batch_size: int = 32) -> Dict[str, Any]:
        """Train the RNN model."""
        logger.info(f"Training {self.model_name}...")
        
        # Prepare validation data
        validation_data = (X_val, y_val) if X_val is not None else None
        
        # Callbacks
        callbacks_list = [
            callbacks.EarlyStopping(patience=10, restore_best_weights=True),
            callbacks.ReduceLROnPlateau(factor=0.5, patience=5)
        ]
        
        # Train the model
        history = self.model.fit(
            X_train, y_train,
            validation_data=validation_data,
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks_list,
            verbose=1
        )
        
        self.is_trained = True
        self.training_history = history.history
        
        logger.info(f"Training completed. Final accuracy: {history.history['accuracy'][-1]:.4f}")
        
        return self.training_history
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions."""
        predictions = self.model.predict(X)
        if self.num_classes == 2:
            return (predictions > 0.5).astype(int).flatten()
        else:
            return np.argmax(predictions, axis=1)
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict class probabilities."""
        return self.model.predict(X)
    
    def save_model(self, filepath: str):
        """Save the trained model."""
        self.model.save(filepath)
        logger.info(f"Model saved to {filepath}")
    
    def load_model(self, filepath: str):
        """Load a pre-trained model."""
        self.model = keras.models.load_model(filepath)
        self.is_trained = True
        logger.info(f"Model loaded from {filepath}")


class EnsembleModel(BaseModel):
    """Ensemble model combining multiple classifiers."""
    
    def __init__(self, models: List[BaseModel], voting: str = 'soft'):
        super().__init__("Ensemble")
        self.models = models
        self.voting = voting
        self.weights = None
    
    def train(self, X_train: np.ndarray, y_train: np.ndarray, 
              X_val: Optional[np.ndarray] = None, y_val: Optional[np.ndarray] = None) -> Dict[str, Any]:
        """Train all models in the ensemble."""
        logger.info(f"Training {self.model_name} with {len(self.models)} models...")
        
        training_histories = {}
        
        # Train each model
        for i, model in enumerate(self.models):
            logger.info(f"Training model {i+1}/{len(self.models)}: {model.model_name}")
            history = model.train(X_train, y_train, X_val, y_val)
            training_histories[f"model_{i}_{model.model_name}"] = history
        
        self.is_trained = True
        self.training_history = training_histories
        
        # Calculate ensemble weights based on validation performance
        if X_val is not None and y_val is not None:
            self._calculate_weights(X_val, y_val)
        
        logger.info("Ensemble training completed")
        return self.training_history
    
    def _calculate_weights(self, X_val: np.ndarray, y_val: np.ndarray):
        """Calculate weights for ensemble voting based on validation performance."""
        weights = []
        
        for model in self.models:
            if model.is_trained:
                val_score = model.model.score(X_val, y_val) if hasattr(model.model, 'score') else None
                if val_score is None:
                    # For neural networks, calculate accuracy manually
                    predictions = model.predict(X_val)
                    val_score = np.mean(predictions == y_val)
                weights.append(val_score)
            else:
                weights.append(0.0)
        
        # Normalize weights
        total_weight = sum(weights)
        if total_weight > 0:
            self.weights = [w / total_weight for w in weights]
        else:
            self.weights = [1.0 / len(self.models)] * len(self.models)
        
        logger.info(f"Ensemble weights: {self.weights}")
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make ensemble predictions."""
        if not self.is_trained:
            raise ValueError("Ensemble must be trained before making predictions")
        
        predictions = []
        
        for model in self.models:
            if model.is_trained:
                pred = model.predict(X)
                predictions.append(pred)
        
        if not predictions:
            raise ValueError("No trained models available for prediction")
        
        # Weighted voting
        if self.weights:
            weighted_predictions = []
            for i, pred in enumerate(predictions):
                weighted_predictions.append(pred * self.weights[i])
            ensemble_pred = np.sum(weighted_predictions, axis=0)
            return (ensemble_pred > 0.5).astype(int)
        else:
            # Simple majority voting
            ensemble_pred = np.mean(predictions, axis=0)
            return (ensemble_pred > 0.5).astype(int)
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict ensemble class probabilities."""
        if not self.is_trained:
            raise ValueError("Ensemble must be trained before making predictions")
        
        probabilities = []
        
        for model in self.models:
            if model.is_trained:
                prob = model.predict_proba(X)
                probabilities.append(prob)
        
        if not probabilities:
            raise ValueError("No trained models available for prediction")
        
        # Weighted average of probabilities
        if self.weights:
            weighted_probs = []
            for i, prob in enumerate(probabilities):
                weighted_probs.append(prob * self.weights[i])
            ensemble_prob = np.sum(weighted_probs, axis=0)
        else:
            ensemble_prob = np.mean(probabilities, axis=0)
        
        return ensemble_prob
    
    def save_model(self, filepath: str):
        """Save all models in the ensemble."""
        model_dir = Path(filepath)
        model_dir.mkdir(parents=True, exist_ok=True)
        
        for i, model in enumerate(self.models):
            model_path = model_dir / f"model_{i}_{model.model_name}.pkl"
            model.save_model(str(model_path))
        
        # Save ensemble metadata
        ensemble_metadata = {
            'weights': self.weights,
            'voting': self.voting,
            'model_names': [model.model_name for model in self.models]
        }
        
        import json
        with open(model_dir / "ensemble_metadata.json", 'w') as f:
            json.dump(ensemble_metadata, f)
        
        logger.info(f"Ensemble saved to {filepath}")
    
    def load_model(self, filepath: str):
        """Load all models in the ensemble."""
        model_dir = Path(filepath)
        
        # Load ensemble metadata
        import json
        with open(model_dir / "ensemble_metadata.json", 'r') as f:
            ensemble_metadata = json.load(f)
        
        self.weights = ensemble_metadata['weights']
        self.voting = ensemble_metadata['voting']
        
        # Load individual models
        for i, model_name in enumerate(ensemble_metadata['model_names']):
            model_path = model_dir / f"model_{i}_{model_name}.pkl"
            if model_path.exists():
                self.models[i].load_model(str(model_path))
        
        self.is_trained = True
        logger.info(f"Ensemble loaded from {filepath}")


def create_ensemble_models(input_shape: Tuple[int, ...], num_classes: int = 2) -> EnsembleModel:
    """Create a default ensemble of models."""
    models = [
        RandomForestModel(),
        SVMModel(),
        CNNModel(input_shape, num_classes),
        RNNModel(input_shape, num_classes)
    ]
    
    return EnsembleModel(models)


def main():
    """Example usage of the ML models."""
    # Generate synthetic data
    np.random.seed(42)
    X = np.random.randn(1000, 20)
    y = np.random.randint(0, 2, 1000)
    
    # Split data
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)
    
    # Test Random Forest
    print("Testing Random Forest...")
    rf_model = RandomForestModel()
    rf_model.train(X_train, y_train, X_val, y_val)
    rf_predictions = rf_model.predict(X_test)
    print(f"Random Forest Accuracy: {np.mean(rf_predictions == y_test):.4f}")
    
    # Test SVM
    print("Testing SVM...")
    svm_model = SVMModel()
    svm_model.train(X_train, y_train, X_val, y_val)
    svm_predictions = svm_model.predict(X_test)
    print(f"SVM Accuracy: {np.mean(svm_predictions == y_test):.4f}")
    
    # Test Ensemble
    print("Testing Ensemble...")
    ensemble = create_ensemble_models((20,), 2)
    ensemble.train(X_train, y_train, X_val, y_val)
    ensemble_predictions = ensemble.predict(X_test)
    print(f"Ensemble Accuracy: {np.mean(ensemble_predictions == y_test):.4f}")


if __name__ == "__main__":
    main()
