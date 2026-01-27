"""
SHAP Importance-Weighted Sampling
Implements Formula 4.1-4.2 from manuscript
"""

import numpy as np
import shap
from sklearn.ensemble import RandomForestClassifier
from typing import Dict, List, Optional, Tuple
import logging

logger = logging.getLogger(__name__)


class SHAPImportanceAnalyzer:
    """
    SHAP-based feature importance analysis for guided sampling

    Implements:
    - Eq. 19: φⱼ = (1/n) Σ |SHAPⱼ(xᵢ)|
    - Eq. 20: wⱼ = φⱼ / Σ φₖ (normalized weights)
    - Eq. 21: Shapley value calculation
    """

    def __init__(
        self,
        model_type: str = 'tree',
        n_estimators: int = 100,
        max_depth: int = 10,
        random_state: int = 42
    ):
        """
        Args:
            model_type: 'tree' or 'linear'
            n_estimators: For tree-based models
            max_depth: Tree depth
            random_state: Random seed
        """
        self.model_type = model_type
        self.random_state = random_state

        # Initialize model
        if model_type == 'tree':
            self.model = RandomForestClassifier(
                n_estimators=n_estimators,
                max_depth=max_depth,
                random_state=random_state,
                n_jobs=-1
            )
        else:
            from sklearn.linear_model import LogisticRegression
            self.model = LogisticRegression(
                max_iter=1000,
                random_state=random_state,
                n_jobs=-1
            )

        self.explainer = None
        self.shap_values_ = None
        self.feature_importance_ = None  # φⱼ (Eq. 19)
        self.sampling_weights_ = None     # wⱼ (Eq. 20)
        self.feature_names_ = None

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        feature_names: List[str] = None
    ):
        """
        Train model and compute SHAP importance

        Args:
            X: Feature matrix (n × d)
            y: Labels (diagnosis categories)
            feature_names: Feature names
        """
        logger.info(f"Training {self.model_type} model for SHAP analysis")
        logger.info(f"  X shape: {X.shape}")
        logger.info(f"  Unique labels: {len(np.unique(y))}")

        # Train model
        self.model.fit(X, y)
        train_score = self.model.score(X, y)
        logger.info(f"  Training accuracy: {train_score:.3f}")

        # Initialize SHAP explainer
        logger.info("Computing SHAP values...")
        if self.model_type == 'tree':
            self.explainer = shap.TreeExplainer(self.model)
        else:
            self.explainer = shap.LinearExplainer(
                self.model,
                X,
                feature_perturbation="interventional"
            )

        # Calculate SHAP values
        # For multiclass, average across classes
        shap_values = self.explainer.shap_values(X)

        if isinstance(shap_values, list):
            # Multiclass: average absolute SHAP across classes
            self.shap_values_ = np.mean(
                [np.abs(sv) for sv in shap_values],
                axis=0
            )
        else:
            self.shap_values_ = np.abs(shap_values)

        # Store feature names
        self.feature_names_ = feature_names

        # Compute feature importance (Eq. 19)
        self._compute_feature_importance()

        # Compute sampling weights (Eq. 20)
        self._compute_sampling_weights()

        logger.info("SHAP analysis complete")

        return self

    def _compute_feature_importance(self):
        """
        Global feature importance (Eq. 19)
        φⱼ = (1/n) Σᵢ |SHAPⱼ(xᵢ)|
        """
        # Mean absolute SHAP value per feature
        self.feature_importance_ = np.mean(self.shap_values_, axis=0)

        logger.info(
            f"Feature importance computed: {
                self.feature_importance_.shape}")
        logger.debug(
            f"  Top 5 features: {np.argsort(self.feature_importance_)[-5:][::-1]}")

    def _compute_sampling_weights(self):
        """
        Normalized sampling weights (Eq. 20)
        wⱼ = φⱼ / Σₖ φₖ
        """
        total_importance = np.sum(self.feature_importance_)

        if total_importance > 0:
            self.sampling_weights_ = self.feature_importance_ / total_importance
        else:
            # Uniform if all zero
            self.sampling_weights_ = np.ones_like(
                self.feature_importance_) / len(self.feature_importance_)

        logger.info(
            f"Sampling weights computed (sum={
                np.sum(
                    self.sampling_weights_):.4f})")

    def get_feature_importance(self, top_k: int = None) -> np.ndarray:
        """
        Get feature importance φⱼ

        Args:
            top_k: Return only top k features

        Returns:
            Feature importance array
        """
        if top_k is not None:
            indices = np.argsort(self.feature_importance_)[-top_k:][::-1]
            return self.feature_importance_[indices]

        return self.feature_importance_

    def get_sampling_weights(
            self,
            feature_indices: List[int] = None) -> np.ndarray:
        """
        Get sampling weights wⱼ for specific features

        Args:
            feature_indices: Indices of features to get weights for

        Returns:
            Sampling weights
        """
        if feature_indices is not None:
            return self.sampling_weights_[feature_indices]

        return self.sampling_weights_

    def get_top_features(
        self,
        top_k: int = 20
    ) -> List[Tuple[str, float]]:
        """
        Get top k most important features

        Returns:
            List of (feature_name, importance)
        """
        indices = np.argsort(self.feature_importance_)[-top_k:][::-1]

        if self.feature_names_:
            features = [(self.feature_names_[i], self.feature_importance_[i])
                        for i in indices]
        else:
            features = [(f"feature_{i}", self.feature_importance_[i])
                        for i in indices]

        return features

    def get_parameter_importance(
        self,
        parameter_feature_map: Dict[str, List[int]]
    ) -> Dict[str, float]:
        """
        Aggregate importance for parameters (may have multiple features)

        Args:
            parameter_feature_map: {param_name: [feature_indices]}

        Returns:
            {param_name: aggregated_importance}
        """
        param_importance = {}

        for param_name, feature_indices in parameter_feature_map.items():
            # Sum importance across all features of this parameter
            importance = np.sum(self.feature_importance_[feature_indices])
            param_importance[param_name] = importance

        return param_importance

    def explain_instance(
        self,
        x: np.ndarray,
        feature_names: List[str] = None
    ) -> Dict:
        """
        SHAP explanation for single instance

        Args:
            x: Single instance (d,)
            feature_names: Optional feature names

        Returns:
            Explanation dictionary
        """
        shap_values = self.explainer.shap_values(x.reshape(1, -1))

        if isinstance(shap_values, list):
            # Multiclass: use first class or average
            shap_values = shap_values[0][0]
        else:
            shap_values = shap_values[0]

        # Get top positive and negative contributors
        pos_indices = np.argsort(shap_values)[-5:][::-1]
        neg_indices = np.argsort(shap_values)[:5]

        names = feature_names or self.feature_names_ or [
            f"f{i}" for i in range(len(x))]

        return {
            'shap_values': shap_values,
            'top_positive': [(names[i], shap_values[i]) for i in pos_indices],
            'top_negative': [(names[i], shap_values[i]) for i in neg_indices]
        }

    def plot_importance(
        self,
        top_k: int = 20,
        save_path: str = None
    ):
        """
        Plot feature importance bar chart

        Args:
            top_k: Number of top features to show
            save_path: Optional path to save figure
        """
        import matplotlib.pyplot as plt

        top_features = self.get_top_features(top_k)
        names, importances = zip(*top_features)

        fig, ax = plt.subplots(figsize=(10, 8))
        y_pos = np.arange(len(names))

        ax.barh(y_pos, importances, color='steelblue')
        ax.set_yticks(y_pos)
        ax.set_yticklabels(names)
        ax.invert_yaxis()
        ax.set_xlabel('Mean |SHAP Value| (Feature Importance φⱼ)')
        ax.set_title(f'Top {top_k} Features by SHAP Importance')
        ax.grid(axis='x', alpha=0.3)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Saved importance plot to {save_path}")

        plt.show()

    def get_summary(self) -> Dict:
        """Get summary statistics"""
        top_features = self.get_top_features(10)

        return {
            'n_features': len(self.feature_importance_),
            'top_10_features': top_features,
            'total_importance': np.sum(self.feature_importance_),
            'weights_sum': np.sum(self.sampling_weights_),
            'model_type': self.model_type
        }
