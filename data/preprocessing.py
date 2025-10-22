"""
Data Preprocessing Module for Cybersecurity ML Framework

This module handles loading, preprocessing, and feature engineering for
NSL-KDD, TON_IoT, and CICIDS2017 datasets used in threat detection.
"""

import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder, MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import SelectKBest, f_classif
import logging
from typing import Tuple, Dict, List, Optional
import requests
import zipfile
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DataPreprocessor:
    """
    Main class for preprocessing cybersecurity datasets.
    Handles data loading, cleaning, feature engineering, and preparation for ML models.
    """
    
    def __init__(self, data_dir: str = "data"):
        """
        Initialize the DataPreprocessor.
        
        Args:
            data_dir: Directory to store datasets
        """
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(exist_ok=True)
        self.scalers = {}
        self.label_encoders = {}
        self.feature_selectors = {}
        
    def load_nsl_kdd(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Load and preprocess NSL-KDD dataset.
        
        Returns:
            Tuple of (features, labels) DataFrames
        """
        logger.info("Loading NSL-KDD dataset...")
        
        # Download dataset if not exists
        train_file = self.data_dir / "KDDTrain+.txt"
        test_file = self.data_dir / "KDDTest+.txt"
        
        if not train_file.exists() or not test_file.exists():
            self._download_nsl_kdd()
        
        # Load data
        train_data = pd.read_csv(train_file, header=None)
        test_data = pd.read_csv(test_file, header=None)
        
        # Define column names
        columns = [
            'duration', 'protocol_type', 'service', 'flag', 'src_bytes',
            'dst_bytes', 'land', 'wrong_fragment', 'urgent', 'hot',
            'num_failed_logins', 'logged_in', 'num_compromised', 'root_shell',
            'su_attempted', 'num_root', 'num_file_creations', 'num_shells',
            'num_access_files', 'num_outbound_cmds', 'is_host_login',
            'is_guest_login', 'count', 'srv_count', 'serror_rate',
            'srv_serror_rate', 'rerror_rate', 'srv_rerror_rate',
            'same_srv_rate', 'diff_srv_rate', 'srv_diff_host_rate',
            'dst_host_count', 'dst_host_srv_count', 'dst_host_same_srv_rate',
            'dst_host_diff_srv_rate', 'dst_host_same_src_port_rate',
            'dst_host_srv_diff_host_rate', 'dst_host_serror_rate',
            'dst_host_srv_serror_rate', 'dst_host_rerror_rate',
            'dst_host_srv_rerror_rate', 'attack_type'
        ]
        
        train_data.columns = columns
        test_data.columns = columns
        
        # Combine train and test data
        data = pd.concat([train_data, test_data], ignore_index=True)
        
        # Separate features and labels
        features = data.drop('attack_type', axis=1)
        labels = data['attack_type']
        
        logger.info(f"NSL-KDD loaded: {features.shape[0]} samples, {features.shape[1]} features")
        return features, labels
    
    def load_cicids2017(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Load and preprocess CICIDS2017 dataset.
        
        Returns:
            Tuple of (features, labels) DataFrames
        """
        logger.info("Loading CICIDS2017 dataset...")
        
        # This is a placeholder - CICIDS2017 is a large dataset
        # In practice, you would download and process the actual files
        logger.warning("CICIDS2017 dataset loading not fully implemented - using synthetic data")
        
        # Generate synthetic data for demonstration
        n_samples = 10000
        n_features = 78
        
        features = pd.DataFrame(
            np.random.randn(n_samples, n_features),
            columns=[f'feature_{i}' for i in range(n_features)]
        )
        
        # Generate labels (BENIGN, DDoS, PortScan, etc.)
        attack_types = ['BENIGN', 'DDoS', 'PortScan', 'Bot', 'Infiltration']
        labels = pd.Series(np.random.choice(attack_types, n_samples))
        
        logger.info(f"CICIDS2017 loaded: {features.shape[0]} samples, {features.shape[1]} features")
        return features, labels
    
    def load_ton_iot(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Load and preprocess TON_IoT dataset.
        
        Returns:
            Tuple of (features, labels) DataFrames
        """
        logger.info("Loading TON_IoT dataset...")
        
        # This is a placeholder - TON_IoT is a large dataset
        # In practice, you would download and process the actual files
        logger.warning("TON_IoT dataset loading not fully implemented - using synthetic data")
        
        # Generate synthetic data for demonstration
        n_samples = 8000
        n_features = 45
        
        features = pd.DataFrame(
            np.random.randn(n_samples, n_features),
            columns=[f'iot_feature_{i}' for i in range(n_features)]
        )
        
        # Generate labels for IoT attacks
        attack_types = ['Normal', 'Backdoor', 'DDoS', 'Injection', 'MITM', 'Password']
        labels = pd.Series(np.random.choice(attack_types, n_samples))
        
        logger.info(f"TON_IoT loaded: {features.shape[0]} samples, {features.shape[1]} features")
        return features, labels
    
    def preprocess_features(self, features: pd.DataFrame, dataset_name: str) -> pd.DataFrame:
        """
        Preprocess features for ML models.
        
        Args:
            features: Raw feature DataFrame
            dataset_name: Name of the dataset for scaler storage
            
        Returns:
            Preprocessed feature DataFrame
        """
        logger.info(f"Preprocessing features for {dataset_name}...")
        
        processed_features = features.copy()
        
        # Handle categorical variables
        categorical_columns = processed_features.select_dtypes(include=['object']).columns
        
        for col in categorical_columns:
            if col not in self.label_encoders:
                self.label_encoders[col] = LabelEncoder()
                processed_features[col] = self.label_encoders[col].fit_transform(processed_features[col])
            else:
                processed_features[col] = self.label_encoders[col].transform(processed_features[col])
        
        # Handle missing values
        processed_features = processed_features.fillna(processed_features.mean())
        
        # Scale features
        if dataset_name not in self.scalers:
            self.scalers[dataset_name] = StandardScaler()
            processed_features = pd.DataFrame(
                self.scalers[dataset_name].fit_transform(processed_features),
                columns=processed_features.columns
            )
        else:
            processed_features = pd.DataFrame(
                self.scalers[dataset_name].transform(processed_features),
                columns=processed_features.columns
            )
        
        logger.info(f"Features preprocessed: {processed_features.shape}")
        return processed_features
    
    def preprocess_labels(self, labels: pd.Series, dataset_name: str) -> pd.Series:
        """
        Preprocess labels for ML models.
        
        Args:
            labels: Raw label Series
            dataset_name: Name of the dataset
            
        Returns:
            Preprocessed label Series
        """
        logger.info(f"Preprocessing labels for {dataset_name}...")
        
        # Create binary classification (Normal vs Attack)
        binary_labels = labels.copy()
        binary_labels = binary_labels.apply(lambda x: 0 if x == 'Normal' or x == 'BENIGN' else 1)
        
        logger.info(f"Binary labels created: {binary_labels.value_counts().to_dict()}")
        return binary_labels
    
    def feature_selection(self, features: pd.DataFrame, labels: pd.Series, 
                        dataset_name: str, k: int = 20) -> pd.DataFrame:
        """
        Perform feature selection using statistical tests.
        
        Args:
            features: Feature DataFrame
            labels: Label Series
            dataset_name: Name of the dataset
            k: Number of features to select
            
        Returns:
            DataFrame with selected features
        """
        logger.info(f"Performing feature selection for {dataset_name}...")
        
        if dataset_name not in self.feature_selectors:
            self.feature_selectors[dataset_name] = SelectKBest(score_func=f_classif, k=k)
            selected_features = self.feature_selectors[dataset_name].fit_transform(features, labels)
            
            # Get selected feature names
            feature_names = features.columns[self.feature_selectors[dataset_name].get_support()]
            selected_df = pd.DataFrame(selected_features, columns=feature_names)
        else:
            selected_features = self.feature_selectors[dataset_name].transform(features)
            feature_names = features.columns[self.feature_selectors[dataset_name].get_support()]
            selected_df = pd.DataFrame(selected_features, columns=feature_names)
        
        logger.info(f"Selected {selected_df.shape[1]} features from {features.shape[1]} original features")
        return selected_df
    
    def split_data(self, features: pd.DataFrame, labels: pd.Series, 
                  test_size: float = 0.2, val_size: float = 0.1) -> Dict[str, np.ndarray]:
        """
        Split data into train, validation, and test sets.
        
        Args:
            features: Feature DataFrame
            labels: Label Series
            test_size: Proportion of data for testing
            val_size: Proportion of training data for validation
            
        Returns:
            Dictionary containing train/val/test splits
        """
        logger.info("Splitting data into train/validation/test sets...")
        
        # First split: train+val vs test
        X_temp, X_test, y_temp, y_test = train_test_split(
            features, labels, test_size=test_size, random_state=42, stratify=labels
        )
        
        # Second split: train vs val
        X_train, X_val, y_train, y_val = train_test_split(
            X_temp, y_temp, test_size=val_size/(1-test_size), random_state=42, stratify=y_temp
        )
        
        splits = {
            'X_train': X_train.values,
            'X_val': X_val.values,
            'X_test': X_test.values,
            'y_train': y_train.values,
            'y_val': y_val.values,
            'y_test': y_test.values
        }
        
        logger.info(f"Data split: Train={X_train.shape[0]}, Val={X_val.shape[0]}, Test={X_test.shape[0]}")
        return splits
    
    def _download_nsl_kdd(self):
        """Download NSL-KDD dataset if not present."""
        logger.info("Downloading NSL-KDD dataset...")
        
        # URLs for NSL-KDD dataset
        urls = {
            'KDDTrain+.txt': 'https://www.unb.ca/cic/datasets/nsl.html',
            'KDDTest+.txt': 'https://www.unb.ca/cic/datasets/nsl.html'
        }
        
        # Note: In practice, you would implement actual download logic
        # For now, we'll create placeholder files
        logger.warning("NSL-KDD download not implemented - using placeholder files")
        
        # Create placeholder files
        train_file = self.data_dir / "KDDTrain+.txt"
        test_file = self.data_dir / "KDDTest+.txt"
        
        # Generate synthetic NSL-KDD-like data
        n_train = 125973
        n_test = 22544
        n_features = 41
        
        # Generate synthetic training data
        train_data = np.random.randn(n_train, n_features)
        train_data[:, 1] = np.random.choice(['tcp', 'udp', 'icmp'], n_train)  # protocol_type
        train_data[:, 2] = np.random.choice(['http', 'ftp', 'smtp'], n_train)  # service
        train_data[:, 3] = np.random.choice(['SF', 'S0', 'REJ'], n_train)    # flag
        train_data[:, -1] = np.random.choice(['Normal', 'DoS', 'Probe', 'R2L', 'U2R'], n_train)  # attack_type
        
        # Generate synthetic test data
        test_data = np.random.randn(n_test, n_features)
        test_data[:, 1] = np.random.choice(['tcp', 'udp', 'icmp'], n_test)
        test_data[:, 2] = np.random.choice(['http', 'ftp', 'smtp'], n_test)
        test_data[:, 3] = np.random.choice(['SF', 'S0', 'REJ'], n_test)
        test_data[:, -1] = np.random.choice(['Normal', 'DoS', 'Probe', 'R2L', 'U2R'], n_test)
        
        # Save synthetic data
        np.savetxt(train_file, train_data, fmt='%s', delimiter=',')
        np.savetxt(test_file, test_data, fmt='%s', delimiter=',')
        
        logger.info("NSL-KDD dataset files created")


def main():
    """Example usage of the DataPreprocessor."""
    preprocessor = DataPreprocessor()
    
    # Load and preprocess NSL-KDD dataset
    features, labels = preprocessor.load_nsl_kdd()
    
    # Preprocess features and labels
    processed_features = preprocessor.preprocess_features(features, 'nsl_kdd')
    processed_labels = preprocessor.preprocess_labels(labels, 'nsl_kdd')
    
    # Feature selection
    selected_features = preprocessor.feature_selection(processed_features, processed_labels, 'nsl_kdd')
    
    # Split data
    splits = preprocessor.split_data(selected_features, processed_labels)
    
    print("Data preprocessing completed successfully!")
    print(f"Final dataset shape: {selected_features.shape}")
    print(f"Label distribution: {processed_labels.value_counts().to_dict()}")


if __name__ == "__main__":
    main()
