"""
Configuration module for the Cybersecurity ML Framework.

This module contains configuration settings, hyperparameters, and constants
used throughout the project.
"""

import os
from dataclasses import dataclass
from typing import Dict, List, Optional
import yaml


@dataclass
class ModelConfig:
    """Configuration for ML/DL models."""
    
    # Random Forest parameters
    rf_n_estimators: int = 100
    rf_max_depth: Optional[int] = None
    rf_min_samples_split: int = 2
    rf_min_samples_leaf: int = 1
    rf_random_state: int = 42
    
    # SVM parameters
    svm_C: float = 1.0
    svm_kernel: str = 'rbf'
    svm_gamma: str = 'scale'
    svm_random_state: int = 42
    
    # Neural Network parameters
    nn_hidden_layers: List[int] = None
    nn_activation: str = 'relu'
    nn_optimizer: str = 'adam'
    nn_learning_rate: float = 0.001
    nn_batch_size: int = 32
    nn_epochs: int = 100
    nn_dropout: float = 0.2
    
    # CNN parameters
    cnn_filters: List[int] = None
    cnn_kernel_sizes: List[int] = None
    cnn_pool_sizes: List[int] = None
    
    # RNN parameters
    rnn_units: List[int] = None
    rnn_dropout: float = 0.2
    rnn_recurrent_dropout: float = 0.2
    
    def __post_init__(self):
        if self.nn_hidden_layers is None:
            self.nn_hidden_layers = [128, 64, 32]
        
        if self.cnn_filters is None:
            self.cnn_filters = [32, 64, 128]
        
        if self.cnn_kernel_sizes is None:
            self.cnn_kernel_sizes = [3, 3, 3]
        
        if self.cnn_pool_sizes is None:
            self.cnn_pool_sizes = [2, 2, 2]
        
        if self.rnn_units is None:
            self.rnn_units = [64, 32]


@dataclass
class DataConfig:
    """Configuration for data processing."""
    
    # Dataset paths
    data_dir: str = "data"
    nsl_kdd_path: str = "data/KDDTrain+.txt"
    cicids2017_path: str = "data/CICIDS2017"
    ton_iot_path: str = "data/TON_IoT"
    
    # Preprocessing parameters
    test_size: float = 0.2
    val_size: float = 0.1
    feature_selection_k: int = 20
    random_state: int = 42
    
    # Data augmentation
    enable_augmentation: bool = True
    augmentation_factor: float = 1.5
    
    # Balancing parameters
    enable_balancing: bool = True
    balancing_method: str = "smote"  # smote, undersample, oversample


@dataclass
class FederatedLearningConfig:
    """Configuration for federated learning."""
    
    # Federated learning parameters
    num_clients: int = 5
    num_rounds: int = 10
    local_epochs: int = 5
    learning_rate: float = 0.01
    batch_size: int = 32
    
    # Privacy parameters
    differential_privacy: bool = True
    epsilon: float = 1.0
    delta: float = 1e-5
    
    # Communication parameters
    compression_enabled: bool = True
    compression_ratio: float = 0.1


@dataclass
class EvaluationConfig:
    """Configuration for model evaluation."""
    
    # Metrics to compute
    metrics: List[str] = None
    
    # Cross-validation
    cv_folds: int = 5
    
    # Performance thresholds
    min_accuracy: float = 0.85
    max_false_positive_rate: float = 0.05
    
    # Visualization
    plot_confusion_matrix: bool = True
    plot_roc_curve: bool = True
    plot_precision_recall: bool = True
    
    def __post_init__(self):
        if self.metrics is None:
            self.metrics = [
                'accuracy', 'precision', 'recall', 'f1_score',
                'roc_auc', 'pr_auc', 'confusion_matrix'
            ]


@dataclass
class RealTimeConfig:
    """Configuration for real-time monitoring."""
    
    # Monitoring parameters
    update_interval: float = 1.0  # seconds
    batch_size: int = 100
    max_queue_size: int = 1000
    
    # Alert thresholds
    threat_threshold: float = 0.8
    anomaly_threshold: float = 0.7
    
    # Performance parameters
    max_processing_time: float = 0.1  # seconds
    enable_parallel_processing: bool = True
    num_workers: int = 4


@dataclass
class AdversarialConfig:
    """Configuration for adversarial robustness testing."""
    
    # Attack parameters
    attack_methods: List[str] = None
    epsilon_values: List[float] = None
    max_iterations: int = 100
    
    # Defense parameters
    enable_defense: bool = True
    defense_methods: List[str] = None
    
    def __post_init__(self):
        if self.attack_methods is None:
            self.attack_methods = ['fgsm', 'pgd', 'carlini_wagner']
        
        if self.epsilon_values is None:
            self.epsilon_values = [0.01, 0.05, 0.1, 0.2]
        
        if self.defense_methods is None:
            self.defense_methods = ['adversarial_training', 'defensive_distillation']


class Config:
    """Main configuration class."""
    
    def __init__(self, config_file: Optional[str] = None):
        """
        Initialize configuration.
        
        Args:
            config_file: Path to YAML configuration file
        """
        self.model = ModelConfig()
        self.data = DataConfig()
        self.federated = FederatedLearningConfig()
        self.evaluation = EvaluationConfig()
        self.real_time = RealTimeConfig()
        self.adversarial = AdversarialConfig()
        
        if config_file and os.path.exists(config_file):
            self.load_from_file(config_file)
    
    def load_from_file(self, config_file: str):
        """Load configuration from YAML file."""
        with open(config_file, 'r') as f:
            config_dict = yaml.safe_load(f)
        
        # Update configurations
        if 'model' in config_dict:
            self.model = ModelConfig(**config_dict['model'])
        
        if 'data' in config_dict:
            self.data = DataConfig(**config_dict['data'])
        
        if 'federated' in config_dict:
            self.federated = FederatedLearningConfig(**config_dict['federated'])
        
        if 'evaluation' in config_dict:
            self.evaluation = EvaluationConfig(**config_dict['evaluation'])
        
        if 'real_time' in config_dict:
            self.real_time = RealTimeConfig(**config_dict['real_time'])
        
        if 'adversarial' in config_dict:
            self.adversarial = AdversarialConfig(**config_dict['adversarial'])
    
    def save_to_file(self, config_file: str):
        """Save configuration to YAML file."""
        config_dict = {
            'model': self.model.__dict__,
            'data': self.data.__dict__,
            'federated': self.federated.__dict__,
            'evaluation': self.evaluation.__dict__,
            'real_time': self.real_time.__dict__,
            'adversarial': self.adversarial.__dict__
        }
        
        with open(config_file, 'w') as f:
            yaml.dump(config_dict, f, default_flow_style=False)


# Global configuration instance
config = Config()


# Dataset-specific configurations
DATASET_CONFIGS = {
    'nsl_kdd': {
        'num_features': 41,
        'num_classes': 5,
        'attack_types': ['Normal', 'DoS', 'Probe', 'R2L', 'U2R'],
        'binary_classes': ['Normal', 'Attack']
    },
    'cicids2017': {
        'num_features': 78,
        'num_classes': 8,
        'attack_types': ['BENIGN', 'DDoS', 'PortScan', 'Bot', 'Infiltration', 'WebAttack', 'BruteForce', 'XSS'],
        'binary_classes': ['BENIGN', 'Attack']
    },
    'ton_iot': {
        'num_features': 45,
        'num_classes': 6,
        'attack_types': ['Normal', 'Backdoor', 'DDoS', 'Injection', 'MITM', 'Password'],
        'binary_classes': ['Normal', 'Attack']
    }
}

# Model performance targets
PERFORMANCE_TARGETS = {
    'accuracy': 0.95,
    'precision': 0.90,
    'recall': 0.90,
    'f1_score': 0.90,
    'false_positive_rate': 0.05,
    'detection_time': 0.1  # seconds
}

# Feature importance thresholds
FEATURE_IMPORTANCE_THRESHOLDS = {
    'high': 0.1,
    'medium': 0.05,
    'low': 0.01
}

# Logging configuration
LOGGING_CONFIG = {
    'version': 1,
    'disable_existing_loggers': False,
    'formatters': {
        'standard': {
            'format': '%(asctime)s [%(levelname)s] %(name)s: %(message)s'
        },
    },
    'handlers': {
        'default': {
            'level': 'INFO',
            'formatter': 'standard',
            'class': 'logging.StreamHandler',
        },
        'file': {
            'level': 'DEBUG',
            'formatter': 'standard',
            'class': 'logging.FileHandler',
            'filename': 'logs/cybersecurity_ml.log',
            'mode': 'a',
        },
    },
    'loggers': {
        '': {
            'handlers': ['default', 'file'],
            'level': 'DEBUG',
            'propagate': False
        }
    }
}
