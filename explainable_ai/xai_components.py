"""
Explainable AI (XAI) Components for Cybersecurity ML Framework.

This module provides tools for interpreting and explaining ML model decisions,
including SHAP, LIME, feature importance analysis, and decision tree visualization.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import shap
import lime
import lime.lime_tabular
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.inspection import permutation_importance, partial_dependence
from sklearn.metrics import confusion_matrix
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import logging
from typing import Dict, List, Tuple, Optional, Any, Union
import warnings
warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ModelExplainer:
    """Base class for model explanation methods."""
    
    def __init__(self, model, feature_names: List[str] = None):
        self.model = model
        self.feature_names = feature_names
        self.explainer = None
        
    def explain_prediction(self, X: np.ndarray, instance_idx: int = 0) -> Dict[str, Any]:
        """Explain a single prediction."""
        raise NotImplementedError
    
    def explain_global(self, X: np.ndarray) -> Dict[str, Any]:
        """Explain global model behavior."""
        raise NotImplementedError


class SHAPExplainer(ModelExplainer):
    """SHAP-based model explainer."""
    
    def __init__(self, model, feature_names: List[str] = None, background_data: np.ndarray = None):
        super().__init__(model, feature_names)
        self.background_data = background_data
        self._initialize_explainer()
    
    def _initialize_explainer(self):
        """Initialize SHAP explainer based on model type."""
        try:
            if hasattr(self.model, 'predict_proba'):
                # For sklearn models
                if self.background_data is not None:
                    self.explainer = shap.Explainer(self.model, self.background_data)
                else:
                    self.explainer = shap.Explainer(self.model)
            else:
                # For neural networks or other models
                self.explainer = shap.DeepExplainer(self.model, self.background_data)
        except Exception as e:
            logger.warning(f"Could not initialize SHAP explainer: {e}")
            self.explainer = None
    
    def explain_prediction(self, X: np.ndarray, instance_idx: int = 0) -> Dict[str, Any]:
        """Explain a single prediction using SHAP."""
        if self.explainer is None:
            logger.error("SHAP explainer not initialized")
            return {}
        
        try:
            # Get SHAP values
            shap_values = self.explainer(X[instance_idx:instance_idx+1])
            
            # Extract explanation data
            explanation = {
                'instance_idx': instance_idx,
                'prediction': self.model.predict(X[instance_idx:instance_idx+1])[0],
                'prediction_proba': self.model.predict_proba(X[instance_idx:instance_idx+1])[0] if hasattr(self.model, 'predict_proba') else None,
                'shap_values': shap_values.values[0] if hasattr(shap_values, 'values') else shap_values[0],
                'feature_names': self.feature_names,
                'feature_values': X[instance_idx]
            }
            
            return explanation
            
        except Exception as e:
            logger.error(f"Error explaining prediction: {e}")
            return {}
    
    def explain_global(self, X: np.ndarray, max_samples: int = 100) -> Dict[str, Any]:
        """Explain global model behavior using SHAP."""
        if self.explainer is None:
            logger.error("SHAP explainer not initialized")
            return {}
        
        try:
            # Sample data for global explanation
            if len(X) > max_samples:
                sample_indices = np.random.choice(len(X), max_samples, replace=False)
                X_sample = X[sample_indices]
            else:
                X_sample = X
            
            # Get SHAP values
            shap_values = self.explainer(X_sample)
            
            # Calculate feature importance
            if hasattr(shap_values, 'values'):
                mean_abs_shap = np.mean(np.abs(shap_values.values), axis=0)
            else:
                mean_abs_shap = np.mean(np.abs(shap_values), axis=0)
            
            explanation = {
                'shap_values': shap_values,
                'feature_importance': mean_abs_shap,
                'feature_names': self.feature_names,
                'sample_size': len(X_sample)
            }
            
            return explanation
            
        except Exception as e:
            logger.error(f"Error in global explanation: {e}")
            return {}
    
    def plot_waterfall(self, explanation: Dict[str, Any], max_features: int = 10):
        """Plot SHAP waterfall chart."""
        if not explanation or 'shap_values' not in explanation:
            logger.error("No explanation data available")
            return
        
        try:
            shap_values = explanation['shap_values']
            feature_names = explanation['feature_names']
            
            # Sort features by absolute SHAP value
            if hasattr(shap_values, 'values'):
                values = shap_values.values[0]
            else:
                values = shap_values[0]
            
            feature_importance = np.abs(values)
            sorted_indices = np.argsort(feature_importance)[::-1][:max_features]
            
            # Create waterfall plot
            fig = go.Figure()
            
            cumulative = 0
            for i, idx in enumerate(sorted_indices):
                feature_name = feature_names[idx] if feature_names else f"Feature_{idx}"
                shap_value = values[idx]
                
                fig.add_trace(go.Bar(
                    x=[feature_name],
                    y=[shap_value],
                    name=feature_name,
                    text=f"{shap_value:.3f}",
                    textposition='auto'
                ))
            
            fig.update_layout(
                title="SHAP Waterfall Plot",
                xaxis_title="Features",
                yaxis_title="SHAP Value",
                showlegend=False
            )
            
            fig.show()
            
        except Exception as e:
            logger.error(f"Error plotting waterfall: {e}")


class LIMEExplainer(ModelExplainer):
    """LIME-based model explainer."""
    
    def __init__(self, model, feature_names: List[str] = None, training_data: np.ndarray = None):
        super().__init__(model, feature_names)
        self.training_data = training_data
        self._initialize_explainer()
    
    def _initialize_explainer(self):
        """Initialize LIME explainer."""
        try:
            if self.training_data is not None:
                self.explainer = lime.lime_tabular.LimeTabularExplainer(
                    self.training_data,
                    feature_names=self.feature_names,
                    mode='classification',
                    discretize_continuous=True
                )
            else:
                logger.warning("No training data provided for LIME explainer")
                self.explainer = None
        except Exception as e:
            logger.error(f"Could not initialize LIME explainer: {e}")
            self.explainer = None
    
    def explain_prediction(self, X: np.ndarray, instance_idx: int = 0) -> Dict[str, Any]:
        """Explain a single prediction using LIME."""
        if self.explainer is None:
            logger.error("LIME explainer not initialized")
            return {}
        
        try:
            # Get LIME explanation
            explanation = self.explainer.explain_instance(
                X[instance_idx],
                self.model.predict_proba if hasattr(self.model, 'predict_proba') else self.model.predict,
                num_features=10
            )
            
            # Extract explanation data
            explanation_data = {
                'instance_idx': instance_idx,
                'prediction': self.model.predict(X[instance_idx:instance_idx+1])[0],
                'prediction_proba': self.model.predict_proba(X[instance_idx:instance_idx+1])[0] if hasattr(self.model, 'predict_proba') else None,
                'feature_weights': explanation.as_list(),
                'feature_names': self.feature_names,
                'feature_values': X[instance_idx]
            }
            
            return explanation_data
            
        except Exception as e:
            logger.error(f"Error explaining prediction: {e}")
            return {}
    
    def plot_explanation(self, explanation: Dict[str, Any]):
        """Plot LIME explanation."""
        if not explanation or 'feature_weights' not in explanation:
            logger.error("No explanation data available")
            return
        
        try:
            feature_weights = explanation['feature_weights']
            
            # Separate positive and negative weights
            positive_weights = [(feat, weight) for feat, weight in feature_weights if weight > 0]
            negative_weights = [(feat, weight) for feat, weight in feature_weights if weight < 0]
            
            # Create subplot
            fig = make_subplots(
                rows=1, cols=2,
                subplot_titles=('Positive Features', 'Negative Features'),
                horizontal_spacing=0.1
            )
            
            # Plot positive weights
            if positive_weights:
                features_pos, weights_pos = zip(*positive_weights)
                fig.add_trace(
                    go.Bar(x=list(weights_pos), y=list(features_pos), orientation='h', name='Positive'),
                    row=1, col=1
                )
            
            # Plot negative weights
            if negative_weights:
                features_neg, weights_neg = zip(*negative_weights)
                fig.add_trace(
                    go.Bar(x=list(weights_neg), y=list(features_neg), orientation='h', name='Negative'),
                    row=1, col=2
                )
            
            fig.update_layout(
                title="LIME Feature Importance",
                showlegend=False,
                height=400
            )
            
            fig.show()
            
        except Exception as e:
            logger.error(f"Error plotting LIME explanation: {e}")


class FeatureImportanceAnalyzer:
    """Analyzer for feature importance across different methods."""
    
    def __init__(self, model, feature_names: List[str] = None):
        self.model = model
        self.feature_names = feature_names or [f"Feature_{i}" for i in range(100)]
    
    def get_permutation_importance(self, X: np.ndarray, y: np.ndarray, 
                                 n_repeats: int = 10) -> Dict[str, Any]:
        """Calculate permutation importance."""
        try:
            perm_importance = permutation_importance(
                self.model, X, y, n_repeats=n_repeats, random_state=42
            )
            
            importance_data = {
                'importances_mean': perm_importance.importances_mean,
                'importances_std': perm_importance.importances_std,
                'feature_names': self.feature_names[:len(perm_importance.importances_mean)]
            }
            
            return importance_data
            
        except Exception as e:
            logger.error(f"Error calculating permutation importance: {e}")
            return {}
    
    def get_tree_feature_importance(self) -> Dict[str, Any]:
        """Get feature importance from tree-based models."""
        try:
            if hasattr(self.model, 'feature_importances_'):
                importance_data = {
                    'importances': self.model.feature_importances_,
                    'feature_names': self.feature_names[:len(self.model.feature_importances_)]
                }
                return importance_data
            else:
                logger.warning("Model does not have feature_importances_ attribute")
                return {}
                
        except Exception as e:
            logger.error(f"Error getting tree feature importance: {e}")
            return {}
    
    def plot_feature_importance(self, importance_data: Dict[str, Any], 
                               top_n: int = 20, method_name: str = "Feature Importance"):
        """Plot feature importance."""
        if not importance_data or 'importances' not in importance_data:
            logger.error("No importance data available")
            return
        
        try:
            importances = importance_data['importances']
            feature_names = importance_data['feature_names']
            
            # Sort by importance
            sorted_indices = np.argsort(importances)[::-1][:top_n]
            sorted_importances = importances[sorted_indices]
            sorted_names = [feature_names[i] for i in sorted_indices]
            
            # Create plot
            fig = go.Figure()
            
            fig.add_trace(go.Bar(
                x=sorted_importances,
                y=sorted_names,
                orientation='h',
                text=[f"{imp:.3f}" for imp in sorted_importances],
                textposition='auto'
            ))
            
            fig.update_layout(
                title=f"{method_name} - Top {top_n} Features",
                xaxis_title="Importance Score",
                yaxis_title="Features",
                height=max(400, top_n * 20)
            )
            
            fig.show()
            
        except Exception as e:
            logger.error(f"Error plotting feature importance: {e}")


class ModelInterpretabilityDashboard:
    """Comprehensive dashboard for model interpretability."""
    
    def __init__(self, model, feature_names: List[str] = None, 
                 training_data: np.ndarray = None):
        self.model = model
        self.feature_names = feature_names
        self.training_data = training_data
        
        # Initialize explainers
        self.shap_explainer = SHAPExplainer(model, feature_names, training_data)
        self.lime_explainer = LIMEExplainer(model, feature_names, training_data)
        self.feature_analyzer = FeatureImportanceAnalyzer(model, feature_names)
    
    def generate_comprehensive_explanation(self, X: np.ndarray, y: np.ndarray = None,
                                        instance_idx: int = 0) -> Dict[str, Any]:
        """Generate comprehensive explanation using multiple methods."""
        logger.info(f"Generating comprehensive explanation for instance {instance_idx}")
        
        explanation = {
            'instance_idx': instance_idx,
            'feature_values': X[instance_idx],
            'feature_names': self.feature_names
        }
        
        # SHAP explanation
        try:
            shap_explanation = self.shap_explainer.explain_prediction(X, instance_idx)
            explanation['shap'] = shap_explanation
        except Exception as e:
            logger.error(f"SHAP explanation failed: {e}")
            explanation['shap'] = {}
        
        # LIME explanation
        try:
            lime_explanation = self.lime_explainer.explain_prediction(X, instance_idx)
            explanation['lime'] = lime_explanation
        except Exception as e:
            logger.error(f"LIME explanation failed: {e}")
            explanation['lime'] = {}
        
        # Feature importance
        try:
            if y is not None:
                perm_importance = self.feature_analyzer.get_permutation_importance(X, y)
                explanation['permutation_importance'] = perm_importance
            
            tree_importance = self.feature_analyzer.get_tree_feature_importance()
            explanation['tree_importance'] = tree_importance
        except Exception as e:
            logger.error(f"Feature importance analysis failed: {e}")
        
        return explanation
    
    def plot_comprehensive_dashboard(self, explanation: Dict[str, Any]):
        """Plot comprehensive interpretability dashboard."""
        try:
            # Create subplots
            fig = make_subplots(
                rows=2, cols=2,
                subplot_titles=('SHAP Values', 'LIME Weights', 'Feature Importance', 'Prediction Summary'),
                specs=[[{"type": "bar"}, {"type": "bar"}],
                       [{"type": "bar"}, {"type": "indicator"}]]
            )
            
            # SHAP plot
            if 'shap' in explanation and explanation['shap']:
                shap_values = explanation['shap'].get('shap_values', [])
                if len(shap_values) > 0:
                    top_features = np.argsort(np.abs(shap_values))[-10:][::-1]
                    fig.add_trace(
                        go.Bar(x=[self.feature_names[i] for i in top_features],
                              y=[shap_values[i] for i in top_features],
                              name='SHAP'),
                        row=1, col=1
                    )
            
            # LIME plot
            if 'lime' in explanation and explanation['lime']:
                lime_weights = explanation['lime'].get('feature_weights', [])
                if lime_weights:
                    features, weights = zip(*lime_weights[:10])
                    fig.add_trace(
                        go.Bar(x=list(weights), y=list(features), orientation='h', name='LIME'),
                        row=1, col=2
                    )
            
            # Feature importance plot
            if 'tree_importance' in explanation and explanation['tree_importance']:
                importances = explanation['tree_importance'].get('importances', [])
                if len(importances) > 0:
                    top_features = np.argsort(importances)[-10:][::-1]
                    fig.add_trace(
                        go.Bar(x=[self.feature_names[i] for i in top_features],
                              y=[importances[i] for i in top_features],
                              name='Tree Importance'),
                        row=2, col=1
                    )
            
            # Prediction summary
            prediction = explanation.get('shap', {}).get('prediction', 0)
            prediction_proba = explanation.get('shap', {}).get('prediction_proba', [0, 0])
            
            fig.add_trace(
                go.Indicator(
                    mode="gauge+number",
                    value=prediction_proba[1] if len(prediction_proba) > 1 else prediction,
                    title={'text': "Threat Probability"},
                    gauge={'axis': {'range': [None, 1]},
                          'bar': {'color': "darkblue"},
                          'steps': [{'range': [0, 0.5], 'color': "lightgray"},
                                   {'range': [0.5, 1], 'color': "gray"}],
                          'threshold': {'line': {'color': "red", 'width': 4},
                                      'thickness': 0.75, 'value': 0.8}}
                ),
                row=2, col=2
            )
            
            fig.update_layout(
                title="Model Interpretability Dashboard",
                height=800,
                showlegend=False
            )
            
            fig.show()
            
        except Exception as e:
            logger.error(f"Error plotting dashboard: {e}")
    
    def generate_explanation_report(self, explanation: Dict[str, Any]) -> str:
        """Generate a text report of the explanation."""
        report = []
        report.append("=== Model Explanation Report ===\n")
        
        # Instance information
        report.append(f"Instance Index: {explanation.get('instance_idx', 'N/A')}")
        report.append(f"Prediction: {explanation.get('shap', {}).get('prediction', 'N/A')}")
        
        prediction_proba = explanation.get('shap', {}).get('prediction_proba', [])
        if prediction_proba:
            report.append(f"Threat Probability: {prediction_proba[1]:.3f}")
        
        report.append("\n=== Feature Analysis ===\n")
        
        # SHAP analysis
        if 'shap' in explanation and explanation['shap']:
            shap_values = explanation['shap'].get('shap_values', [])
            if len(shap_values) > 0:
                top_shap_indices = np.argsort(np.abs(shap_values))[-5:][::-1]
                report.append("Top 5 Features by SHAP Value:")
                for i, idx in enumerate(top_shap_indices):
                    feature_name = self.feature_names[idx] if idx < len(self.feature_names) else f"Feature_{idx}"
                    report.append(f"  {i+1}. {feature_name}: {shap_values[idx]:.3f}")
        
        # LIME analysis
        if 'lime' in explanation and explanation['lime']:
            lime_weights = explanation['lime'].get('feature_weights', [])
            if lime_weights:
                report.append("\nTop 5 Features by LIME Weight:")
                for i, (feature, weight) in enumerate(lime_weights[:5]):
                    report.append(f"  {i+1}. {feature}: {weight:.3f}")
        
        # Feature importance
        if 'tree_importance' in explanation and explanation['tree_importance']:
            importances = explanation['tree_importance'].get('importances', [])
            if len(importances) > 0:
                top_importance_indices = np.argsort(importances)[-5:][::-1]
                report.append("\nTop 5 Features by Tree Importance:")
                for i, idx in enumerate(top_importance_indices):
                    feature_name = self.feature_names[idx] if idx < len(self.feature_names) else f"Feature_{idx}"
                    report.append(f"  {i+1}. {feature_name}: {importances[idx]:.3f}")
        
        return "\n".join(report)


def main():
    """Example usage of XAI components."""
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.datasets import make_classification
    
    # Generate synthetic data
    X, y = make_classification(n_samples=1000, n_features=20, n_classes=2, random_state=42)
    feature_names = [f"Feature_{i}" for i in range(20)]
    
    # Train a model
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X, y)
    
    # Create interpretability dashboard
    dashboard = ModelInterpretabilityDashboard(model, feature_names, X)
    
    # Generate explanation for a specific instance
    explanation = dashboard.generate_comprehensive_explanation(X, y, instance_idx=0)
    
    # Plot dashboard
    dashboard.plot_comprehensive_dashboard(explanation)
    
    # Generate report
    report = dashboard.generate_explanation_report(explanation)
    print(report)


if __name__ == "__main__":
    main()
