"""
SHAP-Based Feature Reweighter for Synthetic Data Generation

Uses SHAP (SHapley Additive exPlanations) to compute feature importance
and reweight features during VAE training to emphasize clinically relevant patterns.

Implements Equations 19-21 from manuscript:
- Eq. 19: SHAP value computation φⱼ(xᵢ)
- Eq. 20: Global importance Φⱼ = (1/n) Σ |φⱼ(xᵢ)|
- Eq. 21: Normalized weights wⱼ = Φⱼ / Σ Φₖ

Author: Chatchai Tritham
Date: 2026-01-25
"""

import numpy as np
import xgboost as xgb
import shap
from typing import Dict, List, Tuple, Optional
import logging

logger = logging.getLogger(__name__)


class SHAPReweighter:
    """
    SHAP-based feature importance reweighting for synthetic data generation.

    Strategy:
    1. Train XGBoost diagnostic model on archetypes
    2. Compute SHAP values using TreeExplainer
    3. Aggregate to global feature importance (Eq. 20)
    4. Create importance-weighted sampling distribution (Eq. 21)
    5. Apply weights during VAE training

    Example:
        >>> reweighter = SHAPReweighter(model_type='xgboost')
        >>> reweighter.fit(X_archetypes, y_diagnosis)
        >>> weights = reweighter.get_sampling_weights()
        >>> X_weighted = reweighter.transform(X_synthetic)
    """

    def __init__(self,
                 model_type: str = 'xgboost',
                 background_samples: int = 100,
                 random_state: int = 42):
        """
        Initialize SHAP reweighter.

        Args:
            model_type: Type of model for SHAP analysis ('xgboost' only for now)
            background_samples: Number of background samples for SHAP explainer
            random_state: Random seed for reproducibility
        """
        self.model_type = model_type
        self.background_samples = background_samples
        self.random_state = random_state

        # Model and explainer
        self.model = None
        self.explainer = None

        # SHAP values and weights
        self.shap_values = None  # Raw SHAP values
        # Normalized global importance weights (Eq. 21)
        self.feature_weights = None

        logger.info(f"Initialized SHAPReweighter (model={model_type}, "
                    f"background_samples={background_samples})")

    def fit(self, X_archetypes: np.ndarray,
            y_diagnosis: np.ndarray) -> 'SHAPReweighter':
        """
        Train diagnostic model and compute SHAP feature importance.

        Args:
            X_archetypes: Feature matrix (n_archetypes, n_features)
            y_diagnosis: Diagnosis labels (n_archetypes,)

        Returns:
            self: Fitted reweighter
        """
        n_samples, n_features = X_archetypes.shape
        logger.info(
            f"Fitting SHAP reweighter on {n_samples} archetypes with {n_features} features...")

        # =====================================================================
        # STEP 1: Train diagnostic model
        # =====================================================================
        if self.model_type == 'xgboost':
            logger.info("Training XGBoost diagnostic classifier...")

            self.model = xgb.XGBClassifier(
                n_estimators=100,
                max_depth=6,
                learning_rate=0.1,
                random_state=self.random_state,
                eval_metric='mlogloss',
                use_label_encoder=False
            )
            self.model.fit(X_archetypes, y_diagnosis)

            train_accuracy = self.model.score(X_archetypes, y_diagnosis)
            logger.info(
                f"✓ Model trained. Train accuracy: {
                    train_accuracy:.3f}")

        else:
            raise ValueError(
                f"Unsupported model type: {
                    self.model_type}. Use 'xgboost'.")

        # =====================================================================
        # STEP 2: Create SHAP explainer
        # =====================================================================
        logger.info("Creating SHAP explainer...")

        # Sample background data for TreeExplainer
        if len(X_archetypes) > self.background_samples:
            background_indices = np.random.choice(
                len(X_archetypes),
                size=self.background_samples,
                replace=False
            )
            background = X_archetypes[background_indices]
        else:
            background = X_archetypes

        self.explainer = shap.TreeExplainer(self.model, background)
        logger.info(
            f"✓ SHAP explainer created with {
                len(background)} background samples")

        # =====================================================================
        # STEP 3: Compute SHAP values (Eq. 19)
        # =====================================================================
        logger.info("Computing SHAP values...")

        self.shap_values = self.explainer.shap_values(X_archetypes)

        logger.info(
            f"✓ SHAP values computed. Shape: {
                np.array(
                    self.shap_values).shape}")

        # =====================================================================
        # STEP 4: Aggregate to global importance (Eq. 20)
        # =====================================================================
        logger.info("Computing global feature importance...")

        # Handle multi-class case
        if isinstance(self.shap_values, list):
            # Multi-class: shap_values is list of arrays (one per class)
            # Take mean absolute SHAP across all classes
            abs_shap = np.mean([np.abs(sv) for sv in self.shap_values], axis=0)
        else:
            # Binary case
            abs_shap = np.abs(self.shap_values)

        # Eq. 20: Φⱼ = (1/n) Σᵢ |φⱼ(xᵢ)|
        global_importance = np.mean(abs_shap, axis=0)

        # =====================================================================
        # STEP 5: Normalize to weights (Eq. 21)
        # =====================================================================
        # Eq. 21: wⱼ = Φⱼ / Σₖ Φₖ
        self.feature_weights = global_importance / np.sum(global_importance)

        logger.info(
            f"✓ Feature weights computed. Sum: {
                np.sum(
                    self.feature_weights):.6f} " f"(should be 1.0)")

        # Log top features
        top_features = self.get_top_features(n=5)
        logger.info("Top 5 most important features:")
        for idx, (feat_idx, importance) in enumerate(top_features, 1):
            logger.info(f"  {idx}. Feature {feat_idx}: {importance:.6f}")

        return self

    def transform(self, X_synthetic: np.ndarray) -> np.ndarray:
        """
        Apply learned weights to synthetic data.

        In practice, this is used during VAE training to emphasize important features
        in the reconstruction loss.

        Args:
            X_synthetic: Synthetic data matrix (n_samples, n_features)

        Returns:
            X_weighted: Weighted synthetic data

        Note:
            Applies sqrt(weight) to preserve variance scaling:
            Var(w*X) = w²*Var(X), so use sqrt(w) to get Var(X_weighted) ≈ w*Var(X)
        """
        if self.feature_weights is None:
            raise ValueError("Must call fit() before transform()")

        # Apply sqrt of weights for variance scaling
        sqrt_weights = np.sqrt(self.feature_weights)

        # Element-wise multiplication (broadcasting)
        X_weighted = X_synthetic * sqrt_weights

        return X_weighted

    def get_top_features(self, n: int = 20) -> List[Tuple[int, float]]:
        """
        Return top-n most important features by SHAP importance.

        Args:
            n: Number of top features to return

        Returns:
            List of (feature_index, importance_weight) tuples, sorted descending
        """
        if self.feature_weights is None:
            raise ValueError("Must call fit() first")

        # Sort indices by importance (descending)
        top_indices = np.argsort(self.feature_weights)[::-1][:n]

        return [(int(idx), float(self.feature_weights[idx]))
                for idx in top_indices]

    def get_sampling_weights(self) -> np.ndarray:
        """
        Get feature sampling weights for importance-weighted phase (Phase 4).

        Returns:
            Normalized sampling weights (sum to 1)
        """
        if self.feature_weights is None:
            raise ValueError("Must call fit() first")

        return self.feature_weights.copy()

    def get_feature_ranking(self) -> np.ndarray:
        """
        Get feature indices sorted by importance (most important first).

        Returns:
            Array of feature indices sorted by importance
        """
        if self.feature_weights is None:
            raise ValueError("Must call fit() first")

        return np.argsort(self.feature_weights)[::-1]

    def plot_importance(self,
                        feature_names: Optional[List[str]] = None,
                        top_n: int = 20):
        """
        Generate SHAP summary plot for visualization (Figure 5).

        Args:
            feature_names: Optional list of feature names (defaults to Feature_0, Feature_1, ...)
            top_n: Number of top features to plot

        Returns:
            matplotlib Figure object
        """
        import matplotlib.pyplot as plt

        if self.feature_weights is None:
            raise ValueError("Must call fit() first")

        # Use default names if not provided
        if feature_names is None:
            feature_names = [f"Feature_{i}" for i in range(
                len(self.feature_weights))]

        # Get top features
        top_features = self.get_top_features(n=top_n)
        indices, weights = zip(*top_features)
        names = [feature_names[i] for i in indices]

        # Create figure
        fig, ax = plt.subplots(figsize=(10, 8))

        # Color gradient
        colors = plt.cm.viridis(np.linspace(0.3, 0.9, len(names)))

        # Horizontal bar chart
        y_pos = np.arange(len(names))
        ax.barh(
            y_pos,
            weights,
            color=colors,
            alpha=0.8,
            edgecolor='black',
            linewidth=0.5)

        # Formatting
        ax.set_yticks(y_pos)
        ax.set_yticklabels(names, fontsize=9)
        ax.set_xlabel('SHAP Importance Weight', fontsize=11, fontweight='bold')
        ax.set_title(
            f'Top {top_n} Features by SHAP Importance',
            fontsize=12,
            fontweight='bold')
        ax.grid(True, alpha=0.3, axis='x')

        # Invert y-axis to show highest at top
        ax.invert_yaxis()

        plt.tight_layout()
        return fig

    def summary(self) -> Dict:
        """
        Get summary statistics of SHAP reweighter.

        Returns:
            Dictionary with summary metrics
        """
        if self.feature_weights is None:
            return {'status': 'not_fitted'}

        return {
            'status': 'fitted',
            'model_type': self.model_type,
            'n_features': len(self.feature_weights),
            'top_10_features': self.get_top_features(n=10),
            'weight_statistics': {
                'mean': float(np.mean(self.feature_weights)),
                'std': float(np.std(self.feature_weights)),
                'min': float(np.min(self.feature_weights)),
                'max': float(np.max(self.feature_weights)),
                'sum': float(np.sum(self.feature_weights))
            }
        }

    def __repr__(self) -> str:
        """String representation"""
        if self.feature_weights is None:
            return f"SHAPReweighter(model_type='{
                self.model_type}', fitted=False)"
        else:
            return (f"SHAPReweighter(model_type='{self.model_type}', "
                    f"n_features={len(self.feature_weights)}, fitted=True)")


# Convenience function for quick SHAP analysis
def analyze_feature_importance(X: np.ndarray,
                               y: np.ndarray,
                               feature_names: Optional[List[str]] = None,
                               top_n: int = 20) -> Dict:
    """
    Quick SHAP feature importance analysis.

    Args:
        X: Feature matrix
        y: Target labels
        feature_names: Optional feature names
        top_n: Number of top features to return

    Returns:
        Dictionary with SHAP analysis results

    Example:
        >>> results = analyze_feature_importance(X_train, y_train)
        >>> print(results['top_features'])
    """
    reweighter = SHAPReweighter()
    reweighter.fit(X, y)

    if feature_names is None:
        feature_names = [f"Feature_{i}" for i in range(X.shape[1])]

    top_features = reweighter.get_top_features(n=top_n)

    return {
        'reweighter': reweighter,
        'top_features': [
            {
                'index': idx,
                'name': feature_names[idx],
                'importance': weight
            }
            for idx, weight in top_features
        ],
        'weights': reweighter.get_sampling_weights(),
        'summary': reweighter.summary()
    }


# Main demonstration
if __name__ == '__main__':
    import sys
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    logger.info("SHAPReweighter - Demo Mode")

    # Generate synthetic data for demonstration
    np.random.seed(42)
    n_samples = 1000
    n_features = 50

    # Create features with different importance levels
    X = np.random.randn(n_samples, n_features)

    # Make some features more important for diagnosis
    important_features = [0, 5, 10, 15, 20]
    for feat_idx in important_features:
        X[:, feat_idx] *= 3.0  # Amplify important features

    # Create diagnosis labels based on important features
    y = np.zeros(n_samples, dtype=int)
    for i in range(n_samples):
        score = np.sum(X[i, important_features])
        if score > 2:
            y[i] = 2  # High-risk
        elif score > 0:
            y[i] = 1  # Medium-risk
        else:
            y[i] = 0  # Low-risk

    logger.info(f"Generated {n_samples} samples with {n_features} features")
    logger.info(f"Diagnosis distribution: {np.bincount(y)}")

    # Fit SHAP reweighter
    reweighter = SHAPReweighter(background_samples=50)
    reweighter.fit(X, y)

    # Get results
    logger.info("\nTop 10 most important features:")
    for idx, (feat_idx, weight) in enumerate(
            reweighter.get_top_features(n=10), 1):
        marker = "★" if feat_idx in important_features else " "
        logger.info(
            f"  {
                idx:2d}. Feature {
                feat_idx:2d}: {
                weight:.6f} {marker}")

    # Apply transformation
    X_synthetic = np.random.randn(100, n_features)
    X_weighted = reweighter.transform(X_synthetic)

    logger.info(f"\nTransformation test:")
    logger.info(f"  Original shape: {X_synthetic.shape}")
    logger.info(f"  Weighted shape: {X_weighted.shape}")
    logger.info(f"  Original mean: {np.mean(X_synthetic):.3f}")
    logger.info(f"  Weighted mean: {np.mean(X_weighted):.3f}")

    # Summary
    summary = reweighter.summary()
    logger.info(f"\nSummary:")
    logger.info(f"  Status: {summary['status']}")
    logger.info(f"  N features: {summary['n_features']}")
    logger.info(
        f"  Weight stats: mean={
            summary['weight_statistics']['mean']:.6f}, " f"std={
            summary['weight_statistics']['std']:.6f}")

    logger.info("\n✓ Demo complete!")
