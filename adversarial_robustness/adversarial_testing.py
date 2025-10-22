"""
Adversarial Robustness Testing for Cybersecurity ML Models.

This module implements various adversarial attack methods and defense mechanisms
to test and improve the robustness of cybersecurity ML models against adversarial attacks.
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import tensorflow as tf
from tensorflow import keras
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
import logging
from typing import Dict, List, Tuple, Optional, Any, Union
import matplotlib.pyplot as plt
import seaborn as sns
from dataclasses import dataclass
import warnings
warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class AttackResult:
    """Data class for attack results."""
    attack_name: str
    original_accuracy: float
    adversarial_accuracy: float
    success_rate: float
    perturbation_norm: float
    epsilon: float
    samples_attacked: int
    samples_successful: int


@dataclass
class DefenseResult:
    """Data class for defense results."""
    defense_name: str
    original_accuracy: float
    defended_accuracy: float
    robustness_improvement: float
    attack_success_rate_reduction: float


class AdversarialAttack:
    """Base class for adversarial attacks."""
    
    def __init__(self, name: str):
        self.name = name
    
    def attack(self, model, X: np.ndarray, y: np.ndarray, **kwargs) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Perform adversarial attack."""
        raise NotImplementedError


class FGSMAttack(AdversarialAttack):
    """Fast Gradient Sign Method (FGSM) attack."""
    
    def __init__(self, epsilon: float = 0.1):
        super().__init__("FGSM")
        self.epsilon = epsilon
    
    def attack(self, model, X: np.ndarray, y: np.ndarray, **kwargs) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Perform FGSM attack."""
        logger.info(f"Performing {self.name} attack with epsilon={self.epsilon}")
        
        try:
            # Convert to PyTorch tensors if model is PyTorch-based
            if hasattr(model, 'parameters'):
                return self._pytorch_fgsm(model, X, y)
            else:
                return self._sklearn_fgsm(model, X, y)
        
        except Exception as e:
            logger.error(f"Error in FGSM attack: {e}")
            return X, {'error': str(e)}
    
    def _pytorch_fgsm(self, model: nn.Module, X: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, Dict[str, Any]]:
        """FGSM attack for PyTorch models."""
        model.eval()
        
        # Convert to tensors
        X_tensor = torch.FloatTensor(X)
        y_tensor = torch.LongTensor(y)
        
        # Enable gradient computation
        X_tensor.requires_grad_(True)
        
        # Forward pass
        outputs = model(X_tensor)
        loss = nn.CrossEntropyLoss()(outputs, y_tensor)
        
        # Backward pass
        model.zero_grad()
        loss.backward()
        
        # Get gradients
        gradients = X_tensor.grad.data
        
        # Create adversarial examples
        adversarial_X = X_tensor + self.epsilon * gradients.sign()
        
        # Clip to valid range
        adversarial_X = torch.clamp(adversarial_X, 0, 1)
        
        return adversarial_X.detach().numpy(), {'loss': loss.item()}
    
    def _sklearn_fgsm(self, model, X: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, Dict[str, Any]]:
        """FGSM attack for sklearn models (approximated)."""
        # For sklearn models, we'll use a simplified approach
        # Generate random perturbations with controlled magnitude
        perturbations = np.random.normal(0, self.epsilon, X.shape)
        
        # Create adversarial examples
        adversarial_X = X + perturbations
        
        # Clip to valid range
        adversarial_X = np.clip(adversarial_X, X.min(), X.max())
        
        return adversarial_X, {'perturbation_norm': np.linalg.norm(perturbations)}


class PGDAttack(AdversarialAttack):
    """Projected Gradient Descent (PGD) attack."""
    
    def __init__(self, epsilon: float = 0.1, alpha: float = 0.01, iterations: int = 40):
        super().__init__("PGD")
        self.epsilon = epsilon
        self.alpha = alpha
        self.iterations = iterations
    
    def attack(self, model, X: np.ndarray, y: np.ndarray, **kwargs) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Perform PGD attack."""
        logger.info(f"Performing {self.name} attack with epsilon={self.epsilon}, iterations={self.iterations}")
        
        try:
            if hasattr(model, 'parameters'):
                return self._pytorch_pgd(model, X, y)
            else:
                return self._sklearn_pgd(model, X, y)
        
        except Exception as e:
            logger.error(f"Error in PGD attack: {e}")
            return X, {'error': str(e)}
    
    def _pytorch_pgd(self, model: nn.Module, X: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, Dict[str, Any]]:
        """PGD attack for PyTorch models."""
        model.eval()
        
        # Convert to tensors
        X_tensor = torch.FloatTensor(X)
        y_tensor = torch.LongTensor(y)
        
        # Initialize adversarial examples
        adversarial_X = X_tensor.clone()
        
        # PGD iterations
        for i in range(self.iterations):
            adversarial_X.requires_grad_(True)
            
            # Forward pass
            outputs = model(adversarial_X)
            loss = nn.CrossEntropyLoss()(outputs, y_tensor)
            
            # Backward pass
            model.zero_grad()
            loss.backward()
            
            # Get gradients
            gradients = adversarial_X.grad.data
            
            # Update adversarial examples
            adversarial_X = adversarial_X + self.alpha * gradients.sign()
            
            # Project back to epsilon ball
            delta = adversarial_X - X_tensor
            delta = torch.clamp(delta, -self.epsilon, self.epsilon)
            adversarial_X = X_tensor + delta
            
            # Clip to valid range
            adversarial_X = torch.clamp(adversarial_X, 0, 1)
            
            adversarial_X = adversarial_X.detach()
        
        return adversarial_X.numpy(), {'iterations': self.iterations, 'final_loss': loss.item()}
    
    def _sklearn_pgd(self, model, X: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, Dict[str, Any]]:
        """PGD attack for sklearn models (simplified)."""
        # Simplified PGD for sklearn models
        adversarial_X = X.copy()
        
        for i in range(self.iterations):
            # Generate perturbations
            perturbations = np.random.normal(0, self.alpha, X.shape)
            
            # Update adversarial examples
            adversarial_X += perturbations
            
            # Project back to epsilon ball
            delta = adversarial_X - X
            delta_norm = np.linalg.norm(delta, axis=1, keepdims=True)
            delta = np.where(delta_norm > self.epsilon, 
                           delta * self.epsilon / delta_norm, delta)
            adversarial_X = X + delta
            
            # Clip to valid range
            adversarial_X = np.clip(adversarial_X, X.min(), X.max())
        
        return adversarial_X, {'iterations': self.iterations}


class CWAttack(AdversarialAttack):
    """Carlini-Wagner (C&W) attack."""
    
    def __init__(self, confidence: float = 0.0, learning_rate: float = 0.01, 
                 iterations: int = 1000, c: float = 1.0):
        super().__init__("Carlini-Wagner")
        self.confidence = confidence
        self.learning_rate = learning_rate
        self.iterations = iterations
        self.c = c
    
    def attack(self, model, X: np.ndarray, y: np.ndarray, **kwargs) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Perform C&W attack."""
        logger.info(f"Performing {self.name} attack")
        
        try:
            if hasattr(model, 'parameters'):
                return self._pytorch_cw(model, X, y)
            else:
                return self._sklearn_cw(model, X, y)
        
        except Exception as e:
            logger.error(f"Error in C&W attack: {e}")
            return X, {'error': str(e)}
    
    def _pytorch_cw(self, model: nn.Module, X: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, Dict[str, Any]]:
        """C&W attack for PyTorch models."""
        model.eval()
        
        # Convert to tensors
        X_tensor = torch.FloatTensor(X)
        y_tensor = torch.LongTensor(y)
        
        # Initialize adversarial examples
        adversarial_X = X_tensor.clone()
        adversarial_X.requires_grad_(True)
        
        # Optimizer
        optimizer = optim.Adam([adversarial_X], lr=self.learning_rate)
        
        # C&W loss function
        def cw_loss(outputs, target):
            target_logit = outputs.gather(1, target.unsqueeze(1))
            max_other_logit = outputs.scatter(1, target.unsqueeze(1), -float('inf')).max(1)[0]
            return torch.clamp(max_other_logit - target_logit + self.confidence, min=0)
        
        # Optimization loop
        for i in range(self.iterations):
            optimizer.zero_grad()
            
            outputs = model(adversarial_X)
            loss = cw_loss(outputs, y_tensor).sum()
            
            loss.backward()
            optimizer.step()
            
            # Clip to valid range
            adversarial_X.data = torch.clamp(adversarial_X.data, 0, 1)
        
        return adversarial_X.detach().numpy(), {'iterations': self.iterations, 'final_loss': loss.item()}
    
    def _sklearn_cw(self, model, X: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, Dict[str, Any]]:
        """C&W attack for sklearn models (simplified)."""
        # Simplified C&W for sklearn models
        adversarial_X = X.copy()
        
        # Add small random perturbations
        perturbations = np.random.normal(0, 0.01, X.shape)
        adversarial_X += perturbations
        
        # Clip to valid range
        adversarial_X = np.clip(adversarial_X, X.min(), X.max())
        
        return adversarial_X, {'iterations': self.iterations}


class AdversarialDefense:
    """Base class for adversarial defenses."""
    
    def __init__(self, name: str):
        self.name = name
    
    def defend(self, model, X_train: np.ndarray, y_train: np.ndarray, 
              X_test: np.ndarray, y_test: np.ndarray, **kwargs) -> Tuple[Any, Dict[str, Any]]:
        """Apply defense mechanism."""
        raise NotImplementedError


class AdversarialTraining(AdversarialDefense):
    """Adversarial training defense."""
    
    def __init__(self, attack: AdversarialAttack = None):
        super().__init__("Adversarial Training")
        self.attack = attack or FGSMAttack(epsilon=0.1)
    
    def defend(self, model, X_train: np.ndarray, y_train: np.ndarray, 
              X_test: np.ndarray, y_test: np.ndarray, **kwargs) -> Tuple[Any, Dict[str, Any]]:
        """Apply adversarial training."""
        logger.info(f"Applying {self.name} defense")
        
        try:
            if hasattr(model, 'parameters'):
                return self._pytorch_adversarial_training(model, X_train, y_train, X_test, y_test)
            else:
                return self._sklearn_adversarial_training(model, X_train, y_train, X_test, y_test)
        
        except Exception as e:
            logger.error(f"Error in adversarial training: {e}")
            return model, {'error': str(e)}
    
    def _pytorch_adversarial_training(self, model: nn.Module, X_train: np.ndarray, y_train: np.ndarray,
                                    X_test: np.ndarray, y_test: np.ndarray) -> Tuple[nn.Module, Dict[str, Any]]:
        """Adversarial training for PyTorch models."""
        model.train()
        
        # Convert to tensors
        X_train_tensor = torch.FloatTensor(X_train)
        y_train_tensor = torch.LongTensor(y_train)
        
        # Create data loader
        dataset = TensorDataset(X_train_tensor, y_train_tensor)
        dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
        
        # Optimizer
        optimizer = optim.Adam(model.parameters(), lr=0.001)
        criterion = nn.CrossEntropyLoss()
        
        # Training loop
        epochs = 10
        for epoch in range(epochs):
            epoch_loss = 0.0
            
            for batch_X, batch_y in dataloader:
                optimizer.zero_grad()
                
                # Generate adversarial examples
                adversarial_X, _ = self.attack.attack(model, batch_X.numpy(), batch_y.numpy())
                adversarial_X = torch.FloatTensor(adversarial_X)
                
                # Forward pass on adversarial examples
                outputs = model(adversarial_X)
                loss = criterion(outputs, batch_y)
                
                # Backward pass
                loss.backward()
                optimizer.step()
                
                epoch_loss += loss.item()
            
            logger.info(f"Epoch {epoch+1}/{epochs}, Loss: {epoch_loss/len(dataloader):.4f}")
        
        return model, {'epochs': epochs, 'final_loss': epoch_loss/len(dataloader)}
    
    def _sklearn_adversarial_training(self, model, X_train: np.ndarray, y_train: np.ndarray,
                                    X_test: np.ndarray, y_test: np.ndarray) -> Tuple[Any, Dict[str, Any]]:
        """Adversarial training for sklearn models."""
        # Generate adversarial examples
        adversarial_X, _ = self.attack.attack(model, X_train, y_train)
        
        # Combine original and adversarial data
        combined_X = np.vstack([X_train, adversarial_X])
        combined_y = np.hstack([y_train, y_train])
        
        # Retrain model on combined data
        defended_model = type(model)(**model.get_params())
        defended_model.fit(combined_X, combined_y)
        
        return defended_model, {'augmented_samples': len(combined_X)}


class DefensiveDistillation(AdversarialDefense):
    """Defensive distillation defense."""
    
    def __init__(self, temperature: float = 10.0):
        super().__init__("Defensive Distillation")
        self.temperature = temperature
    
    def defend(self, model, X_train: np.ndarray, y_train: np.ndarray, 
              X_test: np.ndarray, y_test: np.ndarray, **kwargs) -> Tuple[Any, Dict[str, Any]]:
        """Apply defensive distillation."""
        logger.info(f"Applying {self.name} defense")
        
        try:
            if hasattr(model, 'parameters'):
                return self._pytorch_distillation(model, X_train, y_train, X_test, y_test)
            else:
                return self._sklearn_distillation(model, X_train, y_train, X_test, y_test)
        
        except Exception as e:
            logger.error(f"Error in defensive distillation: {e}")
            return model, {'error': str(e)}
    
    def _pytorch_distillation(self, model: nn.Module, X_train: np.ndarray, y_train: np.ndarray,
                            X_test: np.ndarray, y_test: np.ndarray) -> Tuple[nn.Module, Dict[str, Any]]:
        """Defensive distillation for PyTorch models."""
        # Step 1: Train teacher model
        teacher_model = copy.deepcopy(model)
        teacher_model.train()
        
        # Convert to tensors
        X_train_tensor = torch.FloatTensor(X_train)
        y_train_tensor = torch.LongTensor(y_train)
        
        # Train teacher
        optimizer = optim.Adam(teacher_model.parameters(), lr=0.001)
        criterion = nn.CrossEntropyLoss()
        
        for epoch in range(5):
            optimizer.zero_grad()
            outputs = teacher_model(X_train_tensor)
            loss = criterion(outputs, y_train_tensor)
            loss.backward()
            optimizer.step()
        
        # Step 2: Generate soft labels
        teacher_model.eval()
        with torch.no_grad():
            soft_labels = teacher_model(X_train_tensor / self.temperature)
        
        # Step 3: Train student model on soft labels
        student_model = copy.deepcopy(model)
        student_model.train()
        
        optimizer = optim.Adam(student_model.parameters(), lr=0.001)
        
        for epoch in range(5):
            optimizer.zero_grad()
            outputs = student_model(X_train_tensor / self.temperature)
            loss = nn.KLDivLoss()(outputs.log_softmax(dim=1), soft_labels.softmax(dim=1))
            loss.backward()
            optimizer.step()
        
        return student_model, {'temperature': self.temperature}
    
    def _sklearn_distillation(self, model, X_train: np.ndarray, y_train: np.ndarray,
                             X_test: np.ndarray, y_test: np.ndarray) -> Tuple[Any, Dict[str, Any]]:
        """Defensive distillation for sklearn models."""
        # Simplified distillation for sklearn models
        # Train teacher model
        teacher_model = type(model)(**model.get_params())
        teacher_model.fit(X_train, y_train)
        
        # Generate soft labels
        soft_labels = teacher_model.predict_proba(X_train)
        
        # Train student model (simplified)
        student_model = type(model)(**model.get_params())
        student_model.fit(X_train, y_train)
        
        return student_model, {'temperature': self.temperature}


class AdversarialRobustnessTester:
    """Main class for testing adversarial robustness."""
    
    def __init__(self):
        self.attack_results = {}
        self.defense_results = {}
    
    def test_attack(self, model, X_test: np.ndarray, y_test: np.ndarray, 
                   attack: AdversarialAttack) -> AttackResult:
        """Test a specific attack on a model."""
        logger.info(f"Testing {attack.name} attack...")
        
        # Get original accuracy
        original_predictions = model.predict(X_test)
        original_accuracy = np.mean(original_predictions == y_test)
        
        # Perform attack
        adversarial_X, attack_info = attack.attack(model, X_test, y_test)
        
        # Get adversarial accuracy
        adversarial_predictions = model.predict(adversarial_X)
        adversarial_accuracy = np.mean(adversarial_predictions == y_test)
        
        # Calculate success rate
        success_rate = original_accuracy - adversarial_accuracy
        
        # Calculate perturbation norm
        perturbation_norm = np.linalg.norm(adversarial_X - X_test)
        
        # Count successful attacks
        samples_attacked = len(X_test)
        samples_successful = np.sum(original_predictions != adversarial_predictions)
        
        result = AttackResult(
            attack_name=attack.name,
            original_accuracy=original_accuracy,
            adversarial_accuracy=adversarial_accuracy,
            success_rate=success_rate,
            perturbation_norm=perturbation_norm,
            epsilon=getattr(attack, 'epsilon', 0.0),
            samples_attacked=samples_attacked,
            samples_successful=samples_successful
        )
        
        self.attack_results[attack.name] = result
        logger.info(f"{attack.name} attack completed. Success rate: {success_rate:.4f}")
        
        return result
    
    def test_defense(self, model, X_train: np.ndarray, y_train: np.ndarray,
                    X_test: np.ndarray, y_test: np.ndarray, attack: AdversarialAttack,
                    defense: AdversarialDefense) -> DefenseResult:
        """Test a defense mechanism against an attack."""
        logger.info(f"Testing {defense.name} defense against {attack.name} attack...")
        
        # Test original model
        original_result = self.test_attack(model, X_test, y_test, attack)
        
        # Apply defense
        defended_model, defense_info = defense.defend(model, X_train, y_train, X_test, y_test)
        
        # Test defended model
        defended_result = self.test_attack(defended_model, X_test, y_test, attack)
        
        # Calculate improvements
        robustness_improvement = defended_result.adversarial_accuracy - original_result.adversarial_accuracy
        attack_success_rate_reduction = original_result.success_rate - defended_result.success_rate
        
        result = DefenseResult(
            defense_name=defense.name,
            original_accuracy=original_result.adversarial_accuracy,
            defended_accuracy=defended_result.adversarial_accuracy,
            robustness_improvement=robustness_improvement,
            attack_success_rate_reduction=attack_success_rate_reduction
        )
        
        self.defense_results[f"{defense.name}_{attack.name}"] = result
        logger.info(f"{defense.name} defense testing completed. Robustness improvement: {robustness_improvement:.4f}")
        
        return result
    
    def comprehensive_robustness_test(self, model, X_train: np.ndarray, y_train: np.ndarray,
                                    X_test: np.ndarray, y_test: np.ndarray) -> Dict[str, Any]:
        """Perform comprehensive robustness testing."""
        logger.info("Performing comprehensive robustness testing...")
        
        # Define attacks
        attacks = [
            FGSMAttack(epsilon=0.1),
            FGSMAttack(epsilon=0.2),
            PGDAttack(epsilon=0.1),
            PGDAttack(epsilon=0.2),
            CWAttack()
        ]
        
        # Define defenses
        defenses = [
            AdversarialTraining(),
            DefensiveDistillation()
        ]
        
        results = {
            'attack_results': {},
            'defense_results': {},
            'summary': {}
        }
        
        # Test attacks
        for attack in attacks:
            result = self.test_attack(model, X_test, y_test, attack)
            results['attack_results'][attack.name] = result
        
        # Test defenses
        for defense in defenses:
            for attack in attacks[:2]:  # Test against first two attacks
                result = self.test_defense(model, X_train, y_train, X_test, y_test, attack, defense)
                results['defense_results'][f"{defense.name}_{attack.name}"] = result
        
        # Generate summary
        results['summary'] = self._generate_summary(results)
        
        logger.info("Comprehensive robustness testing completed")
        return results
    
    def _generate_summary(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Generate summary of robustness testing results."""
        attack_results = results['attack_results']
        defense_results = results['defense_results']
        
        # Attack summary
        attack_summary = {
            'most_effective_attack': max(attack_results.values(), key=lambda x: x.success_rate).attack_name,
            'average_success_rate': np.mean([r.success_rate for r in attack_results.values()]),
            'worst_accuracy_drop': min([r.adversarial_accuracy for r in attack_results.values()])
        }
        
        # Defense summary
        defense_summary = {
            'most_effective_defense': max(defense_results.values(), key=lambda x: x.robustness_improvement).defense_name,
            'average_robustness_improvement': np.mean([r.robustness_improvement for r in defense_results.values()]),
            'best_defended_accuracy': max([r.defended_accuracy for r in defense_results.values()])
        }
        
        return {
            'attack_summary': attack_summary,
            'defense_summary': defense_summary,
            'overall_robustness_score': self._calculate_robustness_score(attack_results, defense_results)
        }
    
    def _calculate_robustness_score(self, attack_results: Dict, defense_results: Dict) -> float:
        """Calculate overall robustness score."""
        # Simple robustness score based on attack resistance and defense effectiveness
        attack_resistance = 1.0 - np.mean([r.success_rate for r in attack_results.values()])
        defense_effectiveness = np.mean([r.robustness_improvement for r in defense_results.values()])
        
        robustness_score = (attack_resistance + defense_effectiveness) / 2.0
        return max(0.0, min(1.0, robustness_score))
    
    def plot_robustness_results(self, results: Dict[str, Any]):
        """Plot robustness testing results."""
        attack_results = results['attack_results']
        defense_results = results['defense_results']
        
        # Create subplots
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Attack success rates
        attack_names = [r.attack_name for r in attack_results.values()]
        success_rates = [r.success_rate for r in attack_results.values()]
        
        axes[0, 0].bar(attack_names, success_rates)
        axes[0, 0].set_title('Attack Success Rates')
        axes[0, 0].set_ylabel('Success Rate')
        axes[0, 0].tick_params(axis='x', rotation=45)
        
        # Accuracy drops
        accuracy_drops = [r.original_accuracy - r.adversarial_accuracy for r in attack_results.values()]
        
        axes[0, 1].bar(attack_names, accuracy_drops)
        axes[0, 1].set_title('Accuracy Drops')
        axes[0, 1].set_ylabel('Accuracy Drop')
        axes[0, 1].tick_params(axis='x', rotation=45)
        
        # Defense effectiveness
        defense_names = [r.defense_name for r in defense_results.values()]
        robustness_improvements = [r.robustness_improvement for r in defense_results.values()]
        
        axes[1, 0].bar(defense_names, robustness_improvements)
        axes[1, 0].set_title('Defense Robustness Improvements')
        axes[1, 0].set_ylabel('Robustness Improvement')
        axes[1, 0].tick_params(axis='x', rotation=45)
        
        # Overall robustness score
        robustness_score = results['summary']['overall_robustness_score']
        axes[1, 1].bar(['Overall Robustness'], [robustness_score])
        axes[1, 1].set_title('Overall Robustness Score')
        axes[1, 1].set_ylabel('Score')
        axes[1, 1].set_ylim(0, 1)
        
        plt.tight_layout()
        plt.show()


def main():
    """Example usage of adversarial robustness testing."""
    from sklearn.datasets import make_classification
    from sklearn.model_selection import train_test_split
    
    # Generate synthetic data
    X, y = make_classification(n_samples=2000, n_features=20, n_classes=2, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Create a simple model
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    # Create robustness tester
    tester = AdversarialRobustnessTester()
    
    # Test individual attacks
    fgsm_attack = FGSMAttack(epsilon=0.1)
    fgsm_result = tester.test_attack(model, X_test, y_test, fgsm_attack)
    print(f"FGSM Attack Result: {fgsm_result}")
    
    # Test defense
    adversarial_training = AdversarialTraining(fgsm_attack)
    defense_result = tester.test_defense(model, X_train, y_train, X_test, y_test, fgsm_attack, adversarial_training)
    print(f"Adversarial Training Defense Result: {defense_result}")
    
    # Comprehensive testing
    comprehensive_results = tester.comprehensive_robustness_test(model, X_train, y_train, X_test, y_test)
    print(f"Comprehensive Results Summary: {comprehensive_results['summary']}")
    
    # Plot results
    tester.plot_robustness_results(comprehensive_results)


if __name__ == "__main__":
    main()
