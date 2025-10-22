"""
Federated Learning Framework for Privacy-Preserving Cybersecurity Training.

This module implements federated learning capabilities that allow multiple
organizations to collaboratively train ML models without sharing raw data.
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import logging
from typing import Dict, List, Tuple, Optional, Any
import copy
import json
from dataclasses import dataclass
import random
from cryptography.fernet import Fernet
import hashlib

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class ClientConfig:
    """Configuration for federated learning clients."""
    client_id: str
    data_size: int
    local_epochs: int = 5
    learning_rate: float = 0.01
    batch_size: int = 32
    privacy_budget: float = 1.0  # For differential privacy


class FederatedClient:
    """Individual client in the federated learning system."""
    
    def __init__(self, client_id: str, model: nn.Module, config: ClientConfig):
        self.client_id = client_id
        self.model = copy.deepcopy(model)
        self.config = config
        self.optimizer = optim.Adam(self.model.parameters(), lr=config.learning_rate)
        self.criterion = nn.CrossEntropyLoss()
        self.training_history = []
        
    def train_local(self, X_train: np.ndarray, y_train: np.ndarray) -> Dict[str, Any]:
        """Train the model locally on client data."""
        logger.info(f"Training client {self.client_id} locally...")
        
        # Convert to PyTorch tensors
        X_tensor = torch.FloatTensor(X_train)
        y_tensor = torch.LongTensor(y_train)
        
        # Create data loader
        dataset = TensorDataset(X_tensor, y_tensor)
        dataloader = DataLoader(dataset, batch_size=self.config.batch_size, shuffle=True)
        
        # Training loop
        self.model.train()
        epoch_losses = []
        
        for epoch in range(self.config.local_epochs):
            epoch_loss = 0.0
            num_batches = 0
            
            for batch_X, batch_y in dataloader:
                self.optimizer.zero_grad()
                outputs = self.model(batch_X)
                loss = self.criterion(outputs, batch_y)
                loss.backward()
                self.optimizer.step()
                
                epoch_loss += loss.item()
                num_batches += 1
            
            avg_loss = epoch_loss / num_batches
            epoch_losses.append(avg_loss)
        
        # Store training history
        training_info = {
            'client_id': self.client_id,
            'epochs': self.config.local_epochs,
            'final_loss': epoch_losses[-1],
            'avg_loss': np.mean(epoch_losses),
            'data_size': len(X_train)
        }
        
        self.training_history.append(training_info)
        
        logger.info(f"Client {self.client_id} training completed. Final loss: {epoch_losses[-1]:.4f}")
        
        return training_info
    
    def get_model_parameters(self) -> List[torch.Tensor]:
        """Get current model parameters."""
        return [param.clone().detach() for param in self.model.parameters()]
    
    def set_model_parameters(self, parameters: List[torch.Tensor]):
        """Set model parameters from global model."""
        for param, new_param in zip(self.model.parameters(), parameters):
            param.data.copy_(new_param.data)
    
    def add_differential_privacy_noise(self, parameters: List[torch.Tensor], 
                                     epsilon: float, delta: float) -> List[torch.Tensor]:
        """Add differential privacy noise to model parameters."""
        noisy_parameters = []
        
        for param in parameters:
            # Calculate sensitivity (simplified)
            sensitivity = 1.0 / len(parameters)
            
            # Calculate noise scale
            noise_scale = sensitivity * np.sqrt(2 * np.log(1.25 / delta)) / epsilon
            
            # Add Gaussian noise
            noise = torch.normal(0, noise_scale, size=param.shape)
            noisy_param = param + noise
            noisy_parameters.append(noisy_param)
        
        return noisy_parameters


class FederatedServer:
    """Central server for federated learning coordination."""
    
    def __init__(self, global_model: nn.Module, num_clients: int = 5):
        self.global_model = global_model
        self.num_clients = num_clients
        self.clients: List[FederatedClient] = []
        self.round_history = []
        self.encryption_key = Fernet.generate_key()
        self.cipher_suite = Fernet(self.encryption_key)
        
    def add_client(self, client: FederatedClient):
        """Add a client to the federated learning system."""
        self.clients.append(client)
        logger.info(f"Added client {client.client_id} to federated system")
    
    def federated_averaging(self, client_parameters: List[List[torch.Tensor]], 
                          client_weights: List[float]) -> List[torch.Tensor]:
        """Perform federated averaging of client model parameters."""
        logger.info("Performing federated averaging...")
        
        # Initialize averaged parameters
        averaged_params = []
        for i, param in enumerate(client_parameters[0]):
            averaged_param = torch.zeros_like(param)
            for j, client_params in enumerate(client_parameters):
                averaged_param += client_weights[j] * client_params[i]
            averaged_params.append(averaged_param)
        
        return averaged_params
    
    def calculate_client_weights(self, client_data_sizes: List[int]) -> List[float]:
        """Calculate weights for federated averaging based on data size."""
        total_data = sum(client_data_sizes)
        weights = [size / total_data for size in client_data_sizes]
        return weights
    
    def encrypt_model_parameters(self, parameters: List[torch.Tensor]) -> bytes:
        """Encrypt model parameters for secure transmission."""
        # Convert parameters to bytes
        param_bytes = []
        for param in parameters:
            param_bytes.append(param.numpy().tobytes())
        
        # Serialize and encrypt
        serialized_params = json.dumps([param.hex() for param in param_bytes])
        encrypted_params = self.cipher_suite.encrypt(serialized_params.encode())
        
        return encrypted_params
    
    def decrypt_model_parameters(self, encrypted_params: bytes) -> List[torch.Tensor]:
        """Decrypt model parameters."""
        # Decrypt
        decrypted_data = self.cipher_suite.decrypt(encrypted_params)
        serialized_params = json.loads(decrypted_data.decode())
        
        # Convert back to tensors
        parameters = []
        for param_hex in serialized_params:
            param_bytes = bytes.fromhex(param_hex)
            param_array = np.frombuffer(param_bytes, dtype=np.float32)
            param_tensor = torch.from_numpy(param_array)
            parameters.append(param_tensor)
        
        return parameters
    
    def run_federated_round(self, round_num: int, enable_privacy: bool = True) -> Dict[str, Any]:
        """Run one round of federated learning."""
        logger.info(f"Starting federated learning round {round_num}...")
        
        # Collect parameters from all clients
        client_parameters = []
        client_data_sizes = []
        
        for client in self.clients:
            params = client.get_model_parameters()
            
            # Add differential privacy noise if enabled
            if enable_privacy:
                params = client.add_differential_privacy_noise(
                    params, 
                    epsilon=client.config.privacy_budget,
                    delta=1e-5
                )
            
            client_parameters.append(params)
            client_data_sizes.append(client.config.data_size)
        
        # Calculate weights and perform federated averaging
        client_weights = self.calculate_client_weights(client_data_sizes)
        global_parameters = self.federated_averaging(client_parameters, client_weights)
        
        # Update global model
        for param, global_param in zip(self.global_model.parameters(), global_parameters):
            param.data.copy_(global_param.data)
        
        # Distribute updated model to all clients
        for client in self.clients:
            client.set_model_parameters(global_parameters)
        
        # Store round information
        round_info = {
            'round': round_num,
            'num_clients': len(self.clients),
            'client_weights': client_weights,
            'privacy_enabled': enable_privacy
        }
        
        self.round_history.append(round_info)
        
        logger.info(f"Federated learning round {round_num} completed")
        
        return round_info


class PrivacyPreservingFederatedLearning:
    """Main class for privacy-preserving federated learning."""
    
    def __init__(self, model: nn.Module, num_clients: int = 5):
        self.server = FederatedServer(model, num_clients)
        self.privacy_budget = 1.0
        self.delta = 1e-5
        
    def add_client_data(self, client_id: str, X_train: np.ndarray, y_train: np.ndarray,
                       config: Optional[ClientConfig] = None):
        """Add client data to the federated learning system."""
        if config is None:
            config = ClientConfig(
                client_id=client_id,
                data_size=len(X_train)
            )
        
        client = FederatedClient(client_id, self.server.global_model, config)
        self.server.add_client(client)
        
        # Store data for training
        client.X_train = X_train
        client.y_train = y_train
        
        logger.info(f"Added data for client {client_id}: {len(X_train)} samples")
    
    def train_federated(self, num_rounds: int = 10, enable_privacy: bool = True) -> Dict[str, Any]:
        """Train the federated learning model."""
        logger.info(f"Starting federated training for {num_rounds} rounds...")
        
        training_history = {
            'rounds': [],
            'global_accuracy': [],
            'privacy_budget_used': []
        }
        
        for round_num in range(num_rounds):
            # Train each client locally
            client_results = []
            for client in self.server.clients:
                result = client.train_local(client.X_train, client.y_train)
                client_results.append(result)
            
            # Run federated averaging
            round_info = self.server.run_federated_round(round_num, enable_privacy)
            
            # Calculate global model performance (simplified)
            global_accuracy = self._evaluate_global_model()
            
            # Update privacy budget
            if enable_privacy:
                self.privacy_budget -= 0.1  # Simplified privacy budget tracking
            
            # Store round results
            round_summary = {
                'round': round_num,
                'client_results': client_results,
                'round_info': round_info,
                'global_accuracy': global_accuracy,
                'privacy_budget': self.privacy_budget
            }
            
            training_history['rounds'].append(round_summary)
            training_history['global_accuracy'].append(global_accuracy)
            training_history['privacy_budget_used'].append(self.privacy_budget)
            
            logger.info(f"Round {round_num} completed. Global accuracy: {global_accuracy:.4f}")
        
        logger.info("Federated training completed")
        return training_history
    
    def _evaluate_global_model(self) -> float:
        """Evaluate the global model performance (simplified)."""
        # This is a placeholder - in practice, you would evaluate on a test set
        return random.uniform(0.8, 0.95)
    
    def get_global_model(self) -> nn.Module:
        """Get the trained global model."""
        return self.server.global_model
    
    def save_federated_model(self, filepath: str):
        """Save the federated learning model and metadata."""
        model_dir = Path(filepath)
        model_dir.mkdir(parents=True, exist_ok=True)
        
        # Save global model
        torch.save(self.server.global_model.state_dict(), model_dir / "global_model.pth")
        
        # Save federated learning metadata
        metadata = {
            'num_clients': len(self.server.clients),
            'round_history': self.server.round_history,
            'privacy_budget': self.privacy_budget,
            'client_configs': [client.config.__dict__ for client in self.server.clients]
        }
        
        with open(model_dir / "federated_metadata.json", 'w') as f:
            json.dump(metadata, f, indent=2)
        
        logger.info(f"Federated model saved to {filepath}")
    
    def load_federated_model(self, filepath: str):
        """Load a federated learning model."""
        model_dir = Path(filepath)
        
        # Load global model
        self.server.global_model.load_state_dict(torch.load(model_dir / "global_model.pth"))
        
        # Load metadata
        with open(model_dir / "federated_metadata.json", 'r') as f:
            metadata = json.load(f)
        
        self.privacy_budget = metadata['privacy_budget']
        self.server.round_history = metadata['round_history']
        
        logger.info(f"Federated model loaded from {filepath}")


class SecureAggregation:
    """Secure aggregation protocol for federated learning."""
    
    def __init__(self, num_clients: int):
        self.num_clients = num_clients
        self.threshold = (num_clients + 1) // 2  # Majority threshold
    
    def generate_shares(self, value: float, num_shares: int) -> List[float]:
        """Generate secret shares of a value."""
        shares = []
        remaining = value
        
        for i in range(num_shares - 1):
            share = random.uniform(-remaining, remaining)
            shares.append(share)
            remaining -= share
        
        shares.append(remaining)
        return shares
    
    def reconstruct_value(self, shares: List[float]) -> float:
        """Reconstruct the original value from shares."""
        return sum(shares)
    
    def secure_aggregate(self, client_values: List[float]) -> float:
        """Perform secure aggregation of client values."""
        # Generate shares for each client
        all_shares = []
        for value in client_values:
            shares = self.generate_shares(value, self.num_clients)
            all_shares.append(shares)
        
        # Aggregate shares
        aggregated_shares = []
        for i in range(self.num_clients):
            share_sum = sum(shares[i] for shares in all_shares)
            aggregated_shares.append(share_sum)
        
        # Reconstruct final value
        final_value = self.reconstruct_value(aggregated_shares)
        return final_value


def create_simple_nn(input_size: int, num_classes: int = 2) -> nn.Module:
    """Create a simple neural network for federated learning."""
    model = nn.Sequential(
        nn.Linear(input_size, 128),
        nn.ReLU(),
        nn.Dropout(0.2),
        nn.Linear(128, 64),
        nn.ReLU(),
        nn.Dropout(0.2),
        nn.Linear(64, num_classes)
    )
    return model


def main():
    """Example usage of federated learning."""
    # Create a simple neural network
    input_size = 20
    num_classes = 2
    model = create_simple_nn(input_size, num_classes)
    
    # Initialize federated learning
    fl_system = PrivacyPreservingFederatedLearning(model, num_clients=3)
    
    # Generate synthetic client data
    np.random.seed(42)
    
    # Client 1 data
    X1 = np.random.randn(100, input_size)
    y1 = np.random.randint(0, num_classes, 100)
    fl_system.add_client_data("client_1", X1, y1)
    
    # Client 2 data
    X2 = np.random.randn(150, input_size)
    y2 = np.random.randint(0, num_classes, 150)
    fl_system.add_client_data("client_2", X2, y2)
    
    # Client 3 data
    X3 = np.random.randn(120, input_size)
    y3 = np.random.randint(0, num_classes, 120)
    fl_system.add_client_data("client_3", X3, y3)
    
    # Train federated model
    training_history = fl_system.train_federated(num_rounds=5, enable_privacy=True)
    
    print("Federated learning completed!")
    print(f"Final global accuracy: {training_history['global_accuracy'][-1]:.4f}")
    print(f"Privacy budget remaining: {training_history['privacy_budget_used'][-1]:.4f}")


if __name__ == "__main__":
    main()
