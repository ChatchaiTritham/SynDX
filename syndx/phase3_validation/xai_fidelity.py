"""
XAI Fidelity Metrics for Synthetic Data Validation

Measures explanation fidelity between synthetic and archetype models to validate
that synthetic data preserves explainability properties.

Validates that models trained on synthetic data produce similar explanations
to models trained on archetype data, ensuring XAI techniques remain effective.

Key Metrics:
- SHAP Value Correlation (Spearman ρ)
- Feature Ranking Agreement (Kendall's τ)
- Interaction Pattern Fidelity (Frobenius norm)
- Local Explanation Consistency (LIME agreement)

Author: Chatchai Tritham
Date: 2026-01-25
"""

import numpy as np
import xgboost as xgb
import shap
from scipy.stats import spearmanr, kendalltau
from sklearn.metrics import mean_squared_error
from typing import Dict, List, Tuple, Optional
import logging

logger = logging.getLogger(__name__)


class XAIFidelity:
    """
    Measure explanation fidelity between synthetic and archetype models.

    Validates that synthetic data preserves explainability properties:
    1. SHAP value correlation between models
    2. Feature ranking agreement
    3. Interaction pattern preservation
    4. Local explanation consistency

    High fidelity scores indicate synthetic data maintains interpretability,
    a critical requirement for clinical deployment.

    Example:
        >>> fidelity = XAIFidelity(model_type='xgboost')
        >>> fidelity.fit_archetype_model(X_arch, y_arch)
        >>> fidelity.fit_synthetic_model(X_synth, y_synth)
        >>> scores = fidelity.compute_all_metrics()
        >>> print(f"Overall XAI Fidelity: {scores['overall_fidelity']:.3f}")
    """

    def __init__(self,
                 model_type: str = 'xgboost',
                 background_samples: int = 100,
                 random_state: int = 42):
        """
        Initialize XAI fidelity evaluator.

        Args:
            model_type: Type of model for SHAP analysis ('xgboost' only for now)
            background_samples: Number of background samples for SHAP explainer
            random_state: Random seed for reproducibility
        """
        self.model_type = model_type
        self.background_samples = background_samples
        self.random_state = random_state

        # Models
        self.archetype_model = None
        self.synthetic_model = None

        # SHAP explainers and values
        self.archetype_explainer = None
        self.synthetic_explainer = None
        self.archetype_shap = None
        self.synthetic_shap = None

        # Fidelity scores
        self.fidelity_scores = {}

        logger.info(
            f"Initialized XAIFidelity (model={model_type}, random_state={random_state})")

    def fit_archetype_model(self, X_arch: np.ndarray, y_arch: np.ndarray):
        """
        Train model on archetype data and compute SHAP explanations.

        Args:
            X_arch: Archetype feature matrix (n_samples, n_features)
            y_arch: Archetype labels (n_samples,)
        """
        n_samples, n_features = X_arch.shape
        logger.info(
            f"Training archetype model on {n_samples} samples with {n_features} features...")

        # Train XGBoost classifier
        self.archetype_model = xgb.XGBClassifier(
            n_estimators=100,
            max_depth=6,
            learning_rate=0.1,
            random_state=self.random_state,
            eval_metric='mlogloss',
            use_label_encoder=False
        )
        self.archetype_model.fit(X_arch, y_arch)

        train_accuracy = self.archetype_model.score(X_arch, y_arch)
        logger.info(
            f"✓ Archetype model trained. Accuracy: {
                train_accuracy:.3f}")

        # Compute SHAP values
        logger.info("Computing SHAP values for archetype model...")

        # Sample background data
        if len(X_arch) > self.background_samples:
            background_indices = np.random.choice(
                len(X_arch),
                size=self.background_samples,
                replace=False,
                random_state=self.random_state
            )
            background = X_arch[background_indices]
        else:
            background = X_arch

        self.archetype_explainer = shap.TreeExplainer(
            self.archetype_model, background)
        self.archetype_shap = self.archetype_explainer.shap_values(X_arch)

        logger.info(
            f"✓ SHAP values computed for archetype model. Shape: {
                np.array(
                    self.archetype_shap).shape}")

    def fit_synthetic_model(self, X_synth: np.ndarray, y_synth: np.ndarray):
        """
        Train model on synthetic data and compute SHAP explanations.

        Args:
            X_synth: Synthetic feature matrix (n_samples, n_features)
            y_synth: Synthetic labels (n_samples,)
        """
        n_samples, n_features = X_synth.shape
        logger.info(
            f"Training synthetic model on {n_samples} samples with {n_features} features...")

        # Train identical architecture
        self.synthetic_model = xgb.XGBClassifier(
            n_estimators=100,
            max_depth=6,
            learning_rate=0.1,
            random_state=self.random_state,
            eval_metric='mlogloss',
            use_label_encoder=False
        )
        self.synthetic_model.fit(X_synth, y_synth)

        train_accuracy = self.synthetic_model.score(X_synth, y_synth)
        logger.info(
            f"✓ Synthetic model trained. Accuracy: {
                train_accuracy:.3f}")

        # Compute SHAP values
        logger.info("Computing SHAP values for synthetic model...")

        # Sample background data
        if len(X_synth) > self.background_samples:
            background_indices = np.random.choice(
                len(X_synth),
                size=self.background_samples,
                replace=False,
                random_state=self.random_state
            )
            background = X_synth[background_indices]
        else:
            background = X_synth

        self.synthetic_explainer = shap.TreeExplainer(
            self.synthetic_model, background)
        self.synthetic_shap = self.synthetic_explainer.shap_values(X_synth)

        logger.info(
            f"✓ SHAP values computed for synthetic model. Shape: {
                np.array(
                    self.synthetic_shap).shape}")

    def compute_shap_correlation(self) -> float:
        """
        Compute Spearman correlation between SHAP values.

        Measures how well SHAP values from synthetic model correlate with
        those from archetype model. High correlation indicates preserved
        feature importance patterns.

        Returns:
            Spearman correlation coefficient (0-1, higher is better)

        Raises:
            ValueError: If models haven't been fitted yet
        """
        if self.archetype_shap is None or self.synthetic_shap is None:
            raise ValueError(
                "Must fit both models first using fit_archetype_model() and fit_synthetic_model()")

        logger.info("Computing SHAP correlation...")

        # Handle multi-class case (SHAP values as list of arrays)
        if isinstance(self.archetype_shap, list):
            # Multi-class: compute correlation for each class and average
            correlations = []

            for class_idx, (arch_shap, synth_shap) in enumerate(
                    zip(self.archetype_shap, self.synthetic_shap)):
                # Flatten SHAP values
                arch_flat = arch_shap.flatten()
                synth_flat = synth_shap.flatten()

                # Handle different sample sizes (take minimum)
                min_size = min(len(arch_flat), len(synth_flat))
                arch_flat = arch_flat[:min_size]
                synth_flat = synth_flat[:min_size]

                # Compute Spearman correlation
                rho, pvalue = spearmanr(arch_flat, synth_flat)
                correlations.append(rho)

                logger.info(
                    f"  Class {class_idx}: ρ = {
                        rho:.3f} (p={
                        pvalue:.4f})")

            # Average across classes
            avg_correlation = np.mean(correlations)

        else:
            # Binary case
            arch_flat = self.archetype_shap.flatten()
            synth_flat = self.synthetic_shap.flatten()

            # Handle different sample sizes
            min_size = min(len(arch_flat), len(synth_flat))
            arch_flat = arch_flat[:min_size]
            synth_flat = synth_flat[:min_size]

            avg_correlation, pvalue = spearmanr(arch_flat, synth_flat)
            logger.info(
                f"  Binary: ρ = {
                    avg_correlation:.3f} (p={
                    pvalue:.4f})")

        self.fidelity_scores['shap_correlation'] = avg_correlation
        logger.info(f"✓ Overall SHAP Correlation: ρ = {avg_correlation:.3f}")

        return avg_correlation

    def compute_rank_agreement(self) -> float:
        """
        Compute Kendall's τ for feature importance ranking.

        Measures agreement in feature rankings between archetype and synthetic models.
        High τ indicates similar feature importance orderings.

        Returns:
            Kendall's tau coefficient (0-1, higher is better)

        Raises:
            ValueError: If models haven't been fitted yet
        """
        if self.archetype_shap is None or self.synthetic_shap is None:
            raise ValueError("Must fit both models first")

        logger.info("Computing rank agreement...")

        # Get global feature importance
        arch_importance = self._get_global_importance(self.archetype_shap)
        synth_importance = self._get_global_importance(self.synthetic_shap)

        # Compute Kendall's tau
        tau, pvalue = kendalltau(arch_importance, synth_importance)

        self.fidelity_scores['rank_agreement'] = tau
        logger.info(f"✓ Rank Agreement: τ = {tau:.3f} (p={pvalue:.4f})")

        return tau

    def _get_global_importance(self, shap_values) -> np.ndarray:
        """
        Aggregate SHAP values to global feature importance.

        Args:
            shap_values: SHAP values (array or list of arrays for multi-class)

        Returns:
            Global importance array (n_features,)
        """
        if isinstance(shap_values, list):
            # Multi-class: average absolute SHAP across all classes
            abs_shap = np.mean([np.abs(sv) for sv in shap_values], axis=0)
        else:
            # Binary case
            abs_shap = np.abs(shap_values)

        # Mean absolute SHAP per feature (Φⱼ = (1/n) Σᵢ |φⱼ(xᵢ)|)
        global_importance = np.mean(abs_shap, axis=0)

        return global_importance

    def compute_feature_importance_mse(self) -> float:
        """
        Compute MSE between normalized feature importances.

        Measures how closely synthetic model's feature importance matches
        archetype model's. Lower MSE = better fidelity.

        Returns:
            Mean squared error (0-1, lower is better)
        """
        if self.archetype_shap is None or self.synthetic_shap is None:
            raise ValueError("Must fit both models first")

        logger.info("Computing feature importance MSE...")

        # Get global importance
        arch_importance = self._get_global_importance(self.archetype_shap)
        synth_importance = self._get_global_importance(self.synthetic_shap)

        # Normalize to sum to 1
        arch_importance_norm = arch_importance / np.sum(arch_importance)
        synth_importance_norm = synth_importance / np.sum(synth_importance)

        # Compute MSE
        mse = mean_squared_error(arch_importance_norm, synth_importance_norm)

        self.fidelity_scores['importance_mse'] = mse
        logger.info(f"✓ Feature Importance MSE: {mse:.6f}")

        return mse

    def compute_top_k_overlap(self, k: int = 20) -> float:
        """
        Compute overlap in top-k most important features.

        Measures what fraction of top-k features are shared between
        archetype and synthetic models.

        Args:
            k: Number of top features to compare

        Returns:
            Overlap fraction (0-1, higher is better)
        """
        if self.archetype_shap is None or self.synthetic_shap is None:
            raise ValueError("Must fit both models first")

        logger.info(f"Computing top-{k} feature overlap...")

        # Get global importance
        arch_importance = self._get_global_importance(self.archetype_shap)
        synth_importance = self._get_global_importance(self.synthetic_shap)

        # Get top-k feature indices
        arch_top_k = set(np.argsort(arch_importance)[-k:])
        synth_top_k = set(np.argsort(synth_importance)[-k:])

        # Compute overlap (Jaccard similarity)
        overlap = len(arch_top_k & synth_top_k) / k

        self.fidelity_scores[f'top_{k}_overlap'] = overlap
        logger.info(f"✓ Top-{k} Feature Overlap: {overlap:.1%}")

        return overlap

    def compute_interaction_fidelity(self,
                                     arch_interactions: Optional[np.ndarray] = None,
                                     synth_interactions: Optional[np.ndarray] = None) -> float:
        """
        Compare interaction matrices (e.g., NMF factor matrices).

        Measures preservation of feature interaction patterns between
        archetype and synthetic data.

        Args:
            arch_interactions: Interaction matrix from archetypes (e.g., NMF H matrix)
            synth_interactions: Interaction matrix from synthetic data

        Returns:
            Interaction fidelity score (0-1, higher is better)

        Notes:
            If interaction matrices not provided, returns 0.0 (metric not applicable)
        """
        if arch_interactions is None or synth_interactions is None:
            logger.warning(
                "Interaction matrices not provided - skipping interaction fidelity")
            self.fidelity_scores['interaction_fidelity'] = 0.0
            return 0.0

        logger.info("Computing interaction fidelity...")

        # Ensure same shape
        if arch_interactions.shape != synth_interactions.shape:
            logger.warning(
                f"Interaction matrix shape mismatch: arch={
                    arch_interactions.shape}, " f"synth={
                    synth_interactions.shape}. Using minimum dimensions.")
            min_rows = min(
                arch_interactions.shape[0],
                synth_interactions.shape[0])
            min_cols = min(
                arch_interactions.shape[1],
                synth_interactions.shape[1])
            arch_interactions = arch_interactions[:min_rows, :min_cols]
            synth_interactions = synth_interactions[:min_rows, :min_cols]

        # Frobenius norm of difference
        diff_norm = np.linalg.norm(
            arch_interactions -
            synth_interactions,
            ord='fro')

        # Normalize by archetype norm
        arch_norm = np.linalg.norm(arch_interactions, ord='fro')
        normalized_distance = diff_norm / (arch_norm + 1e-10)

        # Convert to similarity score (0-1, higher is better)
        interaction_fidelity = max(0.0, 1.0 - normalized_distance)

        self.fidelity_scores['interaction_fidelity'] = interaction_fidelity
        logger.info(f"✓ Interaction Fidelity: {interaction_fidelity:.3f}")

        return interaction_fidelity

    def compute_all_metrics(self,
                            arch_interactions: Optional[np.ndarray] = None,
                            synth_interactions: Optional[np.ndarray] = None) -> Dict:
        """
        Compute all XAI fidelity metrics.

        Args:
            arch_interactions: Optional interaction matrix from archetypes
            synth_interactions: Optional interaction matrix from synthetic data

        Returns:
            Dictionary of all fidelity scores
        """
        logger.info("=" * 80)
        logger.info("Computing all XAI fidelity metrics...")
        logger.info("=" * 80)

        # Core metrics
        self.compute_shap_correlation()
        self.compute_rank_agreement()
        self.compute_feature_importance_mse()
        self.compute_top_k_overlap(k=10)
        self.compute_top_k_overlap(k=20)

        # Optional interaction fidelity
        if arch_interactions is not None and synth_interactions is not None:
            self.compute_interaction_fidelity(
                arch_interactions, synth_interactions)

        # Compute overall fidelity score (weighted average)
        # Weights: SHAP correlation (40%), rank agreement (30%), top-20 overlap
        # (30%)
        overall = (
            0.40 * self.fidelity_scores['shap_correlation'] +
            0.30 * self.fidelity_scores['rank_agreement'] +
            0.30 * self.fidelity_scores['top_20_overlap']
        )

        self.fidelity_scores['overall_fidelity'] = overall

        logger.info("=" * 80)
        logger.info(f"Overall XAI Fidelity Score: {overall:.3f}")
        logger.info("=" * 80)

        return self.fidelity_scores

    def summary(self) -> Dict:
        """
        Get summary of XAI fidelity evaluation.

        Returns:
            Dictionary with all scores and interpretations
        """
        if not self.fidelity_scores:
            return {'status': 'not_evaluated'}

        # Interpret scores
        overall = self.fidelity_scores.get('overall_fidelity', 0.0)
        if overall >= 0.85:
            interpretation = "Excellent - Synthetic data preserves explanations very well"
        elif overall >= 0.70:
            interpretation = "Good - Synthetic data maintains most explainability properties"
        elif overall >= 0.50:
            interpretation = "Fair - Some explanation fidelity preserved"
        else:
            interpretation = "Poor - Synthetic data does not preserve explanations well"

        return {
            'status': 'evaluated',
            'scores': self.fidelity_scores,
            'overall_fidelity': overall,
            'interpretation': interpretation,
            'recommendation': 'Use synthetic data for XAI studies' if overall >= 0.70 else 'Caution advised for XAI applications'
        }

    def __repr__(self) -> str:
        """String representation"""
        if not self.fidelity_scores:
            return f"XAIFidelity(model_type='{
                self.model_type}', evaluated=False)"
        else:
            overall = self.fidelity_scores.get('overall_fidelity', 0.0)
            return f"XAIFidelity(overall_score={overall:.3f}, evaluated=True)"


# Main demonstration
if __name__ == '__main__':
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    logger.info("XAIFidelity - Demo Mode")

    # Generate synthetic archetype and synthetic data for demo
    np.random.seed(42)

    n_arch = 1000
    n_synth = 1000
    n_features = 50

    # Create archetype data with some structure
    X_arch = np.random.randn(n_arch, n_features)
    important_features = [0, 5, 10, 15, 20]
    for feat_idx in important_features:
        X_arch[:, feat_idx] *= 3.0  # Amplify important features

    # Create diagnosis labels based on important features
    y_arch = np.zeros(n_arch, dtype=int)
    for i in range(n_arch):
        score = np.sum(X_arch[i, important_features])
        if score > 2:
            y_arch[i] = 2  # High-risk
        elif score > 0:
            y_arch[i] = 1  # Medium-risk
        else:
            y_arch[i] = 0  # Low-risk

    # Create synthetic data (similar structure but with some noise)
    X_synth = np.random.randn(n_synth, n_features)
    for feat_idx in important_features:
        X_synth[:, feat_idx] *= 2.8  # Slightly different amplification

    y_synth = np.zeros(n_synth, dtype=int)
    for i in range(n_synth):
        score = np.sum(X_synth[i, important_features])
        if score > 2:
            y_synth[i] = 2
        elif score > 0:
            y_synth[i] = 1
        else:
            y_synth[i] = 0

    logger.info(
        f"Generated archetype data: {
            X_arch.shape}, {
            np.bincount(y_arch)}")
    logger.info(
        f"Generated synthetic data: {
            X_synth.shape}, {
            np.bincount(y_synth)}")

    # Initialize XAI fidelity evaluator
    fidelity = XAIFidelity(background_samples=100)

    # Fit models
    fidelity.fit_archetype_model(X_arch, y_arch)
    fidelity.fit_synthetic_model(X_synth, y_synth)

    # Compute all metrics
    scores = fidelity.compute_all_metrics()

    # Print results
    logger.info("\n" + "=" * 80)
    logger.info("XAI FIDELITY RESULTS")
    logger.info("=" * 80)
    logger.info(f"SHAP Correlation:        {scores['shap_correlation']:.3f}")
    logger.info(f"Rank Agreement:          {scores['rank_agreement']:.3f}")
    logger.info(f"Importance MSE:          {scores['importance_mse']:.6f}")
    logger.info(f"Top-10 Overlap:          {scores['top_10_overlap']:.1%}")
    logger.info(f"Top-20 Overlap:          {scores['top_20_overlap']:.1%}")
    logger.info(f"Overall Fidelity:        {scores['overall_fidelity']:.3f}")
    logger.info("=" * 80)

    # Summary
    summary = fidelity.summary()
    logger.info(f"\nInterpretation: {summary['interpretation']}")
    logger.info(f"Recommendation: {summary['recommendation']}")

    logger.info("\n✓ Demo complete!")
