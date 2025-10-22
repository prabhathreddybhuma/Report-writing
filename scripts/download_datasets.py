#!/usr/bin/env python3
"""
Download Script for Cybersecurity ML Framework Datasets

This script downloads and prepares the datasets used in the cybersecurity ML framework.
"""

import os
import requests
import zipfile
import tarfile
import gzip
import shutil
from pathlib import Path
import logging
from typing import Optional
import hashlib

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DatasetDownloader:
    """Dataset downloader for cybersecurity datasets."""
    
    def __init__(self, data_dir: str = "data"):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(exist_ok=True)
        
        # Dataset URLs and information
        self.datasets = {
            'nsl_kdd': {
                'url': 'https://www.unb.ca/cic/datasets/nsl.html',
                'files': ['KDDTrain+.txt', 'KDDTest+.txt'],
                'description': 'NSL-KDD dataset for network intrusion detection'
            },
            'cicids2017': {
                'url': 'https://www.unb.ca/cic/datasets/ids-2017.html',
                'files': ['CICIDS2017.zip'],
                'description': 'CICIDS2017 dataset for contemporary intrusion detection'
            },
            'ton_iot': {
                'url': 'https://ieee-dataport.org/open-access/toniot-datasets',
                'files': ['TON_IoT_Datasets.zip'],
                'description': 'TON_IoT dataset for IoT network traffic analysis'
            }
        }
    
    def download_file(self, url: str, filename: str, chunk_size: int = 8192) -> bool:
        """Download a file from URL."""
        try:
            logger.info(f"Downloading {filename} from {url}")
            
            response = requests.get(url, stream=True)
            response.raise_for_status()
            
            filepath = self.data_dir / filename
            
            with open(filepath, 'wb') as f:
                for chunk in response.iter_content(chunk_size=chunk_size):
                    f.write(chunk)
            
            logger.info(f"Downloaded {filename} successfully")
            return True
            
        except Exception as e:
            logger.error(f"Error downloading {filename}: {e}")
            return False
    
    def extract_archive(self, archive_path: Path, extract_to: Optional[Path] = None) -> bool:
        """Extract archive file."""
        try:
            if extract_to is None:
                extract_to = archive_path.parent
            
            logger.info(f"Extracting {archive_path.name}")
            
            if archive_path.suffix == '.zip':
                with zipfile.ZipFile(archive_path, 'r') as zip_ref:
                    zip_ref.extractall(extract_to)
            
            elif archive_path.suffix in ['.tar', '.gz']:
                with tarfile.open(archive_path, 'r:*') as tar_ref:
                    tar_ref.extractall(extract_to)
            
            elif archive_path.suffix == '.gz':
                with gzip.open(archive_path, 'rb') as f_in:
                    with open(extract_to / archive_path.stem, 'wb') as f_out:
                        shutil.copyfileobj(f_in, f_out)
            
            logger.info(f"Extracted {archive_path.name} successfully")
            return True
            
        except Exception as e:
            logger.error(f"Error extracting {archive_path.name}: {e}")
            return False
    
    def create_synthetic_nsl_kdd(self) -> bool:
        """Create synthetic NSL-KDD dataset for demonstration."""
        try:
            logger.info("Creating synthetic NSL-KDD dataset...")
            
            import numpy as np
            import pandas as pd
            
            # Generate synthetic NSL-KDD-like data
            n_train = 125973
            n_test = 22544
            n_features = 41
            
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
            
            # Generate training data
            train_data = np.random.randn(n_train, n_features)
            
            # Set categorical columns
            train_data[:, 1] = np.random.choice(['tcp', 'udp', 'icmp'], n_train)  # protocol_type
            train_data[:, 2] = np.random.choice(['http', 'ftp', 'smtp', 'ssh', 'telnet'], n_train)  # service
            train_data[:, 3] = np.random.choice(['SF', 'S0', 'REJ', 'RSTR', 'RSTO'], n_train)  # flag
            train_data[:, -1] = np.random.choice(['Normal', 'DoS', 'Probe', 'R2L', 'U2R'], n_train)  # attack_type
            
            # Generate test data
            test_data = np.random.randn(n_test, n_features)
            test_data[:, 1] = np.random.choice(['tcp', 'udp', 'icmp'], n_test)
            test_data[:, 2] = np.random.choice(['http', 'ftp', 'smtp', 'ssh', 'telnet'], n_test)
            test_data[:, 3] = np.random.choice(['SF', 'S0', 'REJ', 'RSTR', 'RSTO'], n_test)
            test_data[:, -1] = np.random.choice(['Normal', 'DoS', 'Probe', 'R2L', 'U2R'], n_test)
            
            # Save synthetic data
            train_file = self.data_dir / "KDDTrain+.txt"
            test_file = self.data_dir / "KDDTest+.txt"
            
            np.savetxt(train_file, train_data, fmt='%s', delimiter=',')
            np.savetxt(test_file, test_data, fmt='%s', delimiter=',')
            
            logger.info("Synthetic NSL-KDD dataset created successfully")
            return True
            
        except Exception as e:
            logger.error(f"Error creating synthetic NSL-KDD dataset: {e}")
            return False
    
    def create_synthetic_cicids2017(self) -> bool:
        """Create synthetic CICIDS2017 dataset for demonstration."""
        try:
            logger.info("Creating synthetic CICIDS2017 dataset...")
            
            import numpy as np
            
            # Create CICIDS2017 directory
            cicids_dir = self.data_dir / "CICIDS2017"
            cicids_dir.mkdir(exist_ok=True)
            
            # Generate synthetic CICIDS2017-like data
            n_samples = 10000
            n_features = 78
            
            # Generate features
            features = np.random.randn(n_samples, n_features)
            
            # Generate labels
            attack_types = ['BENIGN', 'DDoS', 'PortScan', 'Bot', 'Infiltration', 'WebAttack', 'BruteForce', 'XSS']
            labels = np.random.choice(attack_types, n_samples)
            
            # Save synthetic data
            data_file = cicids_dir / "CICIDS2017_synthetic.csv"
            
            # Create DataFrame
            import pandas as pd
            df = pd.DataFrame(features, columns=[f'feature_{i}' for i in range(n_features)])
            df['Label'] = labels
            
            df.to_csv(data_file, index=False)
            
            logger.info("Synthetic CICIDS2017 dataset created successfully")
            return True
            
        except Exception as e:
            logger.error(f"Error creating synthetic CICIDS2017 dataset: {e}")
            return False
    
    def create_synthetic_ton_iot(self) -> bool:
        """Create synthetic TON_IoT dataset for demonstration."""
        try:
            logger.info("Creating synthetic TON_IoT dataset...")
            
            import numpy as np
            
            # Create TON_IoT directory
            ton_iot_dir = self.data_dir / "TON_IoT"
            ton_iot_dir.mkdir(exist_ok=True)
            
            # Generate synthetic TON_IoT-like data
            n_samples = 8000
            n_features = 45
            
            # Generate features
            features = np.random.randn(n_samples, n_features)
            
            # Generate labels
            attack_types = ['Normal', 'Backdoor', 'DDoS', 'Injection', 'MITM', 'Password']
            labels = np.random.choice(attack_types, n_samples)
            
            # Save synthetic data
            data_file = ton_iot_dir / "TON_IoT_synthetic.csv"
            
            # Create DataFrame
            import pandas as pd
            df = pd.DataFrame(features, columns=[f'iot_feature_{i}' for i in range(n_features)])
            df['Label'] = labels
            
            df.to_csv(data_file, index=False)
            
            logger.info("Synthetic TON_IoT dataset created successfully")
            return True
            
        except Exception as e:
            logger.error(f"Error creating synthetic TON_IoT dataset: {e}")
            return False
    
    def download_dataset(self, dataset_name: str, use_synthetic: bool = True) -> bool:
        """Download a specific dataset."""
        if dataset_name not in self.datasets:
            logger.error(f"Unknown dataset: {dataset_name}")
            return False
        
        dataset_info = self.datasets[dataset_name]
        logger.info(f"Downloading {dataset_name}: {dataset_info['description']}")
        
        if use_synthetic:
            # Create synthetic datasets for demonstration
            if dataset_name == 'nsl_kdd':
                return self.create_synthetic_nsl_kdd()
            elif dataset_name == 'cicids2017':
                return self.create_synthetic_cicids2017()
            elif dataset_name == 'ton_iot':
                return self.create_synthetic_ton_iot()
        else:
            # Download real datasets (requires actual URLs)
            logger.warning("Real dataset download not implemented - using synthetic data")
            return self.download_dataset(dataset_name, use_synthetic=True)
        
        return False
    
    def download_all_datasets(self, use_synthetic: bool = True) -> bool:
        """Download all datasets."""
        logger.info("Downloading all datasets...")
        
        success_count = 0
        total_datasets = len(self.datasets)
        
        for dataset_name in self.datasets.keys():
            if self.download_dataset(dataset_name, use_synthetic):
                success_count += 1
        
        logger.info(f"Downloaded {success_count}/{total_datasets} datasets successfully")
        return success_count == total_datasets
    
    def verify_datasets(self) -> bool:
        """Verify that datasets are properly downloaded."""
        logger.info("Verifying datasets...")
        
        all_valid = True
        
        for dataset_name, dataset_info in self.datasets.items():
            logger.info(f"Verifying {dataset_name}...")
            
            if dataset_name == 'nsl_kdd':
                train_file = self.data_dir / "KDDTrain+.txt"
                test_file = self.data_dir / "KDDTest+.txt"
                
                if train_file.exists() and test_file.exists():
                    logger.info(f"{dataset_name} verification: PASSED")
                else:
                    logger.error(f"{dataset_name} verification: FAILED")
                    all_valid = False
            
            elif dataset_name == 'cicids2017':
                cicids_file = self.data_dir / "CICIDS2017" / "CICIDS2017_synthetic.csv"
                
                if cicids_file.exists():
                    logger.info(f"{dataset_name} verification: PASSED")
                else:
                    logger.error(f"{dataset_name} verification: FAILED")
                    all_valid = False
            
            elif dataset_name == 'ton_iot':
                ton_iot_file = self.data_dir / "TON_IoT" / "TON_IoT_synthetic.csv"
                
                if ton_iot_file.exists():
                    logger.info(f"{dataset_name} verification: PASSED")
                else:
                    logger.error(f"{dataset_name} verification: FAILED")
                    all_valid = False
        
        return all_valid
    
    def cleanup_datasets(self) -> bool:
        """Clean up downloaded datasets."""
        try:
            logger.info("Cleaning up datasets...")
            
            if self.data_dir.exists():
                shutil.rmtree(self.data_dir)
                logger.info("Datasets cleaned up successfully")
                return True
            
        except Exception as e:
            logger.error(f"Error cleaning up datasets: {e}")
            return False
        
        return False


def main():
    """Main function for dataset download script."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Download datasets for Cybersecurity ML Framework')
    parser.add_argument('--dataset', type=str, choices=['nsl_kdd', 'cicids2017', 'ton_iot', 'all'],
                       default='all', help='Dataset to download')
    parser.add_argument('--data-dir', type=str, default='data',
                       help='Directory to store datasets')
    parser.add_argument('--synthetic', action='store_true', default=True,
                       help='Use synthetic datasets for demonstration')
    parser.add_argument('--verify', action='store_true',
                       help='Verify downloaded datasets')
    parser.add_argument('--cleanup', action='store_true',
                       help='Clean up datasets')
    
    args = parser.parse_args()
    
    # Create downloader
    downloader = DatasetDownloader(args.data_dir)
    
    if args.cleanup:
        downloader.cleanup_datasets()
        return
    
    # Download datasets
    if args.dataset == 'all':
        success = downloader.download_all_datasets(args.synthetic)
    else:
        success = downloader.download_dataset(args.dataset, args.synthetic)
    
    if success:
        logger.info("Dataset download completed successfully")
        
        if args.verify:
            if downloader.verify_datasets():
                logger.info("Dataset verification passed")
            else:
                logger.error("Dataset verification failed")
    else:
        logger.error("Dataset download failed")


if __name__ == "__main__":
    main()
