"""
Empirical Validation Framework for SynDX XAI Methods
=====================================================

This script provides comprehensive empirical validation for the three core
XAI methods in SynDX:
1. SHAP Values - Feature importance validation
2. Counterfactual Explanations - Plausibility and effectiveness validation
3. NMF Interpretability - Phenotype stability and coherence validation

Author: Chatchai Tritham
Date: 2026-01-25
Version: 1.0.0
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    silhouette_score, davies_bouldin_score, calinski_harabasz_score
)
from sklearn.model_selection import KFold
from sklearn.decomposition import NMF, PCA, FastICA
from sklearn.cluster import KMeans
from scipy.stats import spearmanr, pearsonr, kendalltau
from scipy.spatial.distance import euclidean, cosine
import shap
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Any
import json

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class XAIValidator:
    """
    Comprehensive validation framework for XAI methods.

    Validates:
    - SHAP: Consistency, correlation with baselines, clinical coherence
    - Counterfactual: Plausibility, sparsity, diversity, success rate
    - NMF: Stability, coherence, comparison with alternatives
    """

    def __init__(self, output_dir: str = "outputs/validation"):
        """
        Initialize XAI validator.

        Args:
            output_dir: Directory to save validation results
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Create subdirectories
        (self.output_dir / 'shap').mkdir(exist_ok=True)
        (self.output_dir / 'counterfactual').mkdir(exist_ok=True)
        (self.output_dir / 'nmf').mkdir(exist_ok=True)
        (self.output_dir / 'reports').mkdir(exist_ok=True)

        logger.info(f"XAI Validator initialized. Output: {self.output_dir}")

    # =========================================================================
    # SHAP VALIDATION
    # =========================================================================

    def validate_shap(
        self,
        shap_values: np.ndarray,
        X_data: np.ndarray,
        model,
        feature_names: List[str],
        n_bootstrap: int = 100
    ) -> Dict[str, Any]:
        """
        Comprehensive SHAP validation.

        Metrics:
        1. Consistency (bootstrap stability)
        2. Correlation with model feature importance
        3. Correlation with permutation importance
        4. Rank stability

        Args:
            shap_values: SHAP values array
            X_data: Feature data
            model: Trained model
            feature_names: Feature names
            n_bootstrap: Number of bootstrap samples

        Returns:
            Dictionary with validation metrics
        """
        logger.info("=" * 80)
        logger.info("SHAP VALIDATION")
        logger.info("=" * 80)

        results = {}

        # 1. Bootstrap Consistency
        logger.info("Computing bootstrap consistency...")
        consistency = self._shap_bootstrap_consistency(
            shap_values, X_data, model, n_bootstrap
        )
        results['bootstrap_consistency'] = consistency
        logger.info(
            f"Bootstrap consistency: {
                consistency['mean_rank_correlation']:.4f}")

        # 2. Correlation with model feature importance
        logger.info("Comparing with model feature importance...")
        model_correlation = self._shap_vs_model_importance(
            shap_values, model, feature_names
        )
        results['model_correlation'] = model_correlation
        logger.info(
            f"Correlation with model importance: {
                model_correlation['spearman_r']:.4f}")

        # 3. Permutation importance comparison
        logger.info("Comparing with permutation importance...")
        perm_correlation = self._shap_vs_permutation_importance(
            shap_values, X_data, model, feature_names
        )
        results['permutation_correlation'] = perm_correlation
        logger.info(
            f"Correlation with permutation: {
                perm_correlation['spearman_r']:.4f}")

        # 4. Rank stability across different samples
        logger.info("Computing rank stability...")
        rank_stability = self._shap_rank_stability(shap_values)
        results['rank_stability'] = rank_stability
        logger.info(
            f"Rank stability (Kendall tau): {
                rank_stability['mean_kendall_tau']:.4f}")

        # 5. Save results
        self._save_shap_validation_results(results, feature_names)

        logger.info("SHAP validation complete!")
        return results

    def _shap_bootstrap_consistency(
        self,
        shap_values: np.ndarray,
        X_data: np.ndarray,
        model,
        n_bootstrap: int
    ) -> Dict[str, float]:
        """Bootstrap sampling to test SHAP stability."""

        # Handle multi-class SHAP values
        if isinstance(shap_values, list):
            # Take mean across classes
            shap_array = np.mean([np.abs(sv) for sv in shap_values], axis=0)
        else:
            shap_array = np.abs(shap_values)

        # Compute mean SHAP importance
        mean_shap = np.mean(shap_array, axis=0)

        # Bootstrap sampling
        n_samples = len(X_data)
        n_features = shap_array.shape[1]

        bootstrap_importance = []

        for i in range(n_bootstrap):
            # Random sample with replacement
            idx = np.random.choice(n_samples, size=n_samples, replace=True)
            bootstrap_shap = shap_array[idx]
            bootstrap_mean = np.mean(bootstrap_shap, axis=0)
            bootstrap_importance.append(bootstrap_mean)

        bootstrap_importance = np.array(bootstrap_importance)

        # Compute rank correlation between original and bootstrap
        correlations = []
        for boot_imp in bootstrap_importance:
            corr, _ = spearmanr(mean_shap, boot_imp)
            correlations.append(corr)

        return {
            'mean_rank_correlation': np.mean(correlations),
            'std_rank_correlation': np.std(correlations),
            'min_rank_correlation': np.min(correlations),
            'max_rank_correlation': np.max(correlations)
        }

    def _shap_vs_model_importance(
        self,
        shap_values: np.ndarray,
        model,
        feature_names: List[str]
    ) -> Dict[str, float]:
        """Compare SHAP with model's built-in feature importance."""

        # Handle multi-class SHAP values
        if isinstance(shap_values, list):
            shap_array = np.mean([np.abs(sv) for sv in shap_values], axis=0)
        else:
            shap_array = np.abs(shap_values)

        mean_shap = np.mean(shap_array, axis=0)

        # Get model feature importance (XGBoost)
        try:
            if hasattr(model, 'feature_importances_'):
                model_importance = model.feature_importances_
            else:
                logger.warning(
                    "Model doesn't have feature_importances_, using zeros")
                model_importance = np.zeros(len(feature_names))
        except BaseException:
            logger.warning("Could not extract model importance")
            model_importance = np.zeros(len(feature_names))

        # Correlations
        spearman_r, spearman_p = spearmanr(mean_shap, model_importance)
        pearson_r, pearson_p = pearsonr(mean_shap, model_importance)
        kendall_tau, kendall_p = kendalltau(mean_shap, model_importance)

        return {
            'spearman_r': spearman_r,
            'spearman_p': spearman_p,
            'pearson_r': pearson_r,
            'pearson_p': pearson_p,
            'kendall_tau': kendall_tau,
            'kendall_p': kendall_p
        }

    def _shap_vs_permutation_importance(
        self,
        shap_values: np.ndarray,
        X_data: np.ndarray,
        model,
        feature_names: List[str],
        n_repeats: int = 10
    ) -> Dict[str, float]:
        """Compare SHAP with permutation importance."""
        from sklearn.inspection import permutation_importance

        # Handle multi-class SHAP values
        if isinstance(shap_values, list):
            shap_array = np.mean([np.abs(sv) for sv in shap_values], axis=0)
        else:
            shap_array = np.abs(shap_values)

        mean_shap = np.mean(shap_array, axis=0)

        # Compute permutation importance
        # Note: Need y_data for permutation importance
        # For now, use model predictions as proxy
        try:
            y_pred = model.predict(X_data)
            perm_result = permutation_importance(
                model, X_data, y_pred, n_repeats=n_repeats, random_state=42
            )
            perm_importance = perm_result.importances_mean
        except Exception as e:
            logger.warning(f"Could not compute permutation importance: {e}")
            perm_importance = np.zeros(len(feature_names))

        # Correlations
        spearman_r, spearman_p = spearmanr(mean_shap, perm_importance)
        pearson_r, pearson_p = pearsonr(mean_shap, perm_importance)

        return {
            'spearman_r': spearman_r,
            'spearman_p': spearman_p,
            'pearson_r': pearson_r,
            'pearson_p': pearson_p
        }

    def _shap_rank_stability(
            self, shap_values: np.ndarray) -> Dict[str, float]:
        """Test rank stability across different samples."""

        # Handle multi-class SHAP values
        if isinstance(shap_values, list):
            shap_array = np.mean([np.abs(sv) for sv in shap_values], axis=0)
        else:
            shap_array = np.abs(shap_values)

        n_samples = shap_array.shape[0]

        # Split into 10 groups
        n_groups = 10
        group_size = n_samples // n_groups

        group_rankings = []
        for i in range(n_groups):
            start_idx = i * group_size
            end_idx = (i + 1) * group_size if i < n_groups - 1 else n_samples
            group_shap = shap_array[start_idx:end_idx]
            group_mean = np.mean(group_shap, axis=0)
            group_rankings.append(group_mean)

        # Compute pairwise Kendall tau
        tau_values = []
        for i in range(n_groups):
            for j in range(i + 1, n_groups):
                tau, _ = kendalltau(group_rankings[i], group_rankings[j])
                tau_values.append(tau)

        return {
            'mean_kendall_tau': np.mean(tau_values),
            'std_kendall_tau': np.std(tau_values),
            'min_kendall_tau': np.min(tau_values),
            'max_kendall_tau': np.max(tau_values)
        }

    def _save_shap_validation_results(
        self,
        results: Dict[str, Any],
        feature_names: List[str]
    ):
        """Save SHAP validation results."""

        # Save JSON
        json_path = self.output_dir / 'shap' / 'validation_metrics.json'

        # Convert numpy types to Python types for JSON serialization
        json_results = {}
        for key, value in results.items():
            if isinstance(value, dict):
                json_results[key] = {
                    k: float(v) if isinstance(v, (np.floating, np.integer)) else v
                    for k, v in value.items()
                }
            else:
                json_results[key] = value

        with open(json_path, 'w') as f:
            json.dump(json_results, f, indent=2)

        logger.info(f"SHAP validation results saved to {json_path}")

        # Create summary report
        report_path = self.output_dir / 'reports' / 'shap_validation_report.txt'
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write("=" * 80 + "\n")
            f.write("SHAP VALIDATION REPORT\n")
            f.write("=" * 80 + "\n\n")

            f.write("1. BOOTSTRAP CONSISTENCY\n")
            bc = results['bootstrap_consistency']
            f.write(
                f"   Mean rank correlation: {
                    bc['mean_rank_correlation']:.4f}\n")
            f.write(
                f"   Std rank correlation: {
                    bc['std_rank_correlation']:.4f}\n")
            f.write(
                f"   Range: [{
                    bc['min_rank_correlation']:.4f}, {
                    bc['max_rank_correlation']:.4f}]\n\n")

            f.write("2. CORRELATION WITH MODEL IMPORTANCE\n")
            mc = results['model_correlation']
            f.write(
                f"   Spearman r: {
                    mc['spearman_r']:.4f} (p={
                    mc['spearman_p']:.4e})\n")
            f.write(
                f"   Pearson r: {
                    mc['pearson_r']:.4f} (p={
                    mc['pearson_p']:.4e})\n")
            f.write(
                f"   Kendall tau: {
                    mc['kendall_tau']:.4f} (p={
                    mc['kendall_p']:.4e})\n\n")

            f.write("3. CORRELATION WITH PERMUTATION IMPORTANCE\n")
            pc = results['permutation_correlation']
            f.write(
                f"   Spearman r: {
                    pc['spearman_r']:.4f} (p={
                    pc['spearman_p']:.4e})\n")
            f.write(
                f"   Pearson r: {
                    pc['pearson_r']:.4f} (p={
                    pc['pearson_p']:.4e})\n\n")

            f.write("4. RANK STABILITY\n")
            rs = results['rank_stability']
            f.write(f"   Mean Kendall tau: {rs['mean_kendall_tau']:.4f}\n")
            f.write(f"   Std: {rs['std_kendall_tau']:.4f}\n")
            f.write(
                f"   Range: [{
                    rs['min_kendall_tau']:.4f}, {
                    rs['max_kendall_tau']:.4f}]\n\n")

            f.write("INTERPRETATION:\n")
            f.write("- Bootstrap consistency > 0.9 → Excellent stability\n")
            f.write("- Correlation with baselines > 0.7 → Strong agreement\n")
            f.write("- Rank stability > 0.8 → Robust feature ranking\n")

        logger.info(f"SHAP validation report saved to {report_path}")

    # =========================================================================
    # COUNTERFACTUAL VALIDATION
    # =========================================================================

    def validate_counterfactuals(
        self,
        counterfactuals: List[Dict[str, Any]],
        original_data: np.ndarray,
        feature_names: List[str],
        clinical_plausibility_scores: List[float] = None
    ) -> Dict[str, Any]:
        """
        Comprehensive counterfactual validation.

        Metrics:
        1. Success rate (% counterfactuals generated)
        2. Sparsity (average # features changed)
        3. Proximity (distance from original)
        4. Diversity (variation across counterfactuals)
        5. Clinical plausibility (if expert scores provided)

        Args:
            counterfactuals: List of counterfactual results
            original_data: Original patient data
            feature_names: Feature names
            clinical_plausibility_scores: Expert ratings (1-5 scale)

        Returns:
            Dictionary with validation metrics
        """
        logger.info("=" * 80)
        logger.info("COUNTERFACTUAL VALIDATION")
        logger.info("=" * 80)

        results = {}

        # 1. Success Rate
        logger.info("Computing success rate...")
        success_rate = self._cf_success_rate(counterfactuals)
        results['success_rate'] = success_rate
        logger.info(f"Success rate: {success_rate:.2%}")

        # 2. Sparsity
        logger.info("Computing sparsity...")
        sparsity = self._cf_sparsity(counterfactuals)
        results['sparsity'] = sparsity
        logger.info(f"Average features changed: {sparsity['mean']:.2f}")

        # 3. Proximity
        logger.info("Computing proximity...")
        proximity = self._cf_proximity(counterfactuals, original_data)
        results['proximity'] = proximity
        logger.info(f"Average L2 distance: {proximity['mean_l2']:.4f}")

        # 4. Diversity
        logger.info("Computing diversity...")
        diversity = self._cf_diversity(counterfactuals)
        results['diversity'] = diversity
        logger.info(
            f"Diversity score: {
                diversity['mean_pairwise_distance']:.4f}")

        # 5. Clinical Plausibility
        if clinical_plausibility_scores is not None:
            logger.info("Computing clinical plausibility...")
            plausibility = self._cf_clinical_plausibility(
                clinical_plausibility_scores)
            results['clinical_plausibility'] = plausibility
            logger.info(f"Average plausibility: {plausibility['mean']:.2f}/5")

        # 6. Save results
        self._save_cf_validation_results(results, feature_names)

        logger.info("Counterfactual validation complete!")
        return results

    def _cf_success_rate(self, counterfactuals: List[Dict[str, Any]]) -> float:
        """Compute success rate of counterfactual generation."""
        successful = sum(
            1 for cf in counterfactuals if cf.get(
                'n_counterfactuals', 0) > 0)
        total = len(counterfactuals)
        return successful / total if total > 0 else 0.0

    def _cf_sparsity(
            self, counterfactuals: List[Dict[str, Any]]) -> Dict[str, float]:
        """Compute sparsity (number of features changed)."""
        n_changes = []

        for cf_result in counterfactuals:
            if cf_result.get('n_counterfactuals', 0) > 0:
                for cf in cf_result['counterfactuals']:
                    n_changes.append(len(cf.get('changed_features', {})))

        if len(n_changes) == 0:
            return {'mean': 0, 'std': 0, 'min': 0, 'max': 0}

        return {
            'mean': np.mean(n_changes),
            'std': np.std(n_changes),
            'min': np.min(n_changes),
            'max': np.max(n_changes),
            'median': np.median(n_changes)
        }

    def _cf_proximity(
        self,
        counterfactuals: List[Dict[str, Any]],
        original_data: np.ndarray
    ) -> Dict[str, float]:
        """Compute proximity (distance from original)."""
        l2_distances = []
        l1_distances = []

        for i, cf_result in enumerate(counterfactuals):
            if cf_result.get('n_counterfactuals', 0) > 0:
                original = original_data[i] if i < len(
                    original_data) else original_data[0]

                for cf in cf_result['counterfactuals']:
                    cf_data = cf.get('counterfactual', original)
                    if isinstance(cf_data, np.ndarray):
                        l2_dist = euclidean(original, cf_data.flatten())
                        l1_dist = np.sum(np.abs(original - cf_data.flatten()))
                        l2_distances.append(l2_dist)
                        l1_distances.append(l1_dist)

        if len(l2_distances) == 0:
            return {'mean_l2': 0, 'mean_l1': 0}

        return {
            'mean_l2': np.mean(l2_distances),
            'std_l2': np.std(l2_distances),
            'mean_l1': np.mean(l1_distances),
            'std_l1': np.std(l1_distances)
        }

    def _cf_diversity(
            self, counterfactuals: List[Dict[str, Any]]) -> Dict[str, float]:
        """Compute diversity across counterfactuals."""
        all_cfs = []

        for cf_result in counterfactuals:
            if cf_result.get('n_counterfactuals', 0) > 0:
                for cf in cf_result['counterfactuals']:
                    cf_data = cf.get('counterfactual')
                    if isinstance(cf_data, np.ndarray):
                        all_cfs.append(cf_data.flatten())

        if len(all_cfs) < 2:
            return {'mean_pairwise_distance': 0}

        # Compute pairwise distances
        pairwise_distances = []
        for i in range(len(all_cfs)):
            for j in range(i + 1, len(all_cfs)):
                dist = euclidean(all_cfs[i], all_cfs[j])
                pairwise_distances.append(dist)

        return {
            'mean_pairwise_distance': np.mean(pairwise_distances),
            'std_pairwise_distance': np.std(pairwise_distances)
        }

    def _cf_clinical_plausibility(
        self,
        scores: List[float]
    ) -> Dict[str, float]:
        """Analyze clinical plausibility scores (expert ratings)."""
        return {
            'mean': np.mean(scores),
            'std': np.std(scores),
            'median': np.median(scores),
            'min': np.min(scores),
            'max': np.max(scores),
            # 3+ = plausible
            'percent_plausible': np.mean(np.array(scores) >= 3) * 100
        }

    def _save_cf_validation_results(
        self,
        results: Dict[str, Any],
        feature_names: List[str]
    ):
        """Save counterfactual validation results."""

        # Save JSON
        json_path = self.output_dir / 'counterfactual' / 'validation_metrics.json'

        with open(json_path, 'w') as f:
            json.dump(results, f, indent=2, default=float)

        logger.info(f"Counterfactual validation results saved to {json_path}")

        # Create summary report
        report_path = self.output_dir / 'reports' / \
            'counterfactual_validation_report.txt'
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write("=" * 80 + "\n")
            f.write("COUNTERFACTUAL VALIDATION REPORT\n")
            f.write("=" * 80 + "\n\n")

            f.write(f"1. SUCCESS RATE\n")
            f.write(
                f"   {
                    results['success_rate']:.2%} of attempts generated counterfactuals\n\n")

            f.write("2. SPARSITY (Features Changed)\n")
            sp = results['sparsity']
            f.write(f"   Mean: {sp['mean']:.2f} features\n")
            f.write(f"   Median: {sp.get('median', 0):.2f} features\n")
            f.write(f"   Range: [{sp['min']:.0f}, {sp['max']:.0f}]\n\n")

            f.write("3. PROXIMITY (Distance from Original)\n")
            pr = results['proximity']
            f.write(f"   Mean L2 distance: {pr['mean_l2']:.4f}\n")
            f.write(f"   Mean L1 distance: {pr['mean_l1']:.4f}\n\n")

            f.write("4. DIVERSITY\n")
            div = results['diversity']
            f.write(
                f"   Mean pairwise distance: {
                    div['mean_pairwise_distance']:.4f}\n\n")

            if 'clinical_plausibility' in results:
                f.write("5. CLINICAL PLAUSIBILITY (Expert Ratings)\n")
                cp = results['clinical_plausibility']
                f.write(f"   Mean score: {cp['mean']:.2f}/5\n")
                f.write(
                    f"   Percent plausible (≥3): {
                        cp['percent_plausible']:.1f}%\n\n")

            f.write("INTERPRETATION:\n")
            f.write("- Success rate > 70% → Good coverage\n")
            f.write("- Sparsity < 5 features → Minimal changes (desirable)\n")
            f.write("- Plausibility ≥ 3/5 → Clinically acceptable\n")

        logger.info(f"Counterfactual validation report saved to {report_path}")

    # =========================================================================
    # NMF VALIDATION
    # =========================================================================

    def validate_nmf(
        self,
        W_matrix: np.ndarray,
        H_matrix: np.ndarray,
        X_data: np.ndarray,
        n_factors: int = 20,
        n_bootstrap: int = 50
    ) -> Dict[str, Any]:
        """
        Comprehensive NMF validation.

        Metrics:
        1. Stability (bootstrap resampling)
        2. Reconstruction error
        3. Comparison with PCA
        4. Comparison with ICA
        5. Comparison with K-means
        6. Clustering metrics (Silhouette, Davies-Bouldin)

        Args:
            W_matrix: Patient-Factor matrix
            H_matrix: Factor-Feature matrix
            X_data: Original data
            n_factors: Number of factors
            n_bootstrap: Number of bootstrap samples

        Returns:
            Dictionary with validation metrics
        """
        logger.info("=" * 80)
        logger.info("NMF VALIDATION")
        logger.info("=" * 80)

        results = {}

        # 1. Bootstrap Stability
        logger.info("Computing bootstrap stability...")
        stability = self._nmf_bootstrap_stability(
            X_data, n_factors, n_bootstrap
        )
        results['stability'] = stability
        logger.info(
            f"Factor stability (mean correlation): {
                stability['mean_correlation']:.4f}")

        # 2. Reconstruction Error
        logger.info("Computing reconstruction error...")
        reconstruction = self._nmf_reconstruction_error(
            W_matrix, H_matrix, X_data)
        results['reconstruction'] = reconstruction
        logger.info(f"Reconstruction RMSE: {reconstruction['rmse']:.4f}")

        # 3. Comparison with PCA
        logger.info("Comparing with PCA...")
        pca_comparison = self._nmf_vs_pca(X_data, n_factors)
        results['pca_comparison'] = pca_comparison
        logger.info(
            f"Explained variance: NMF={
                pca_comparison['nmf_variance']:.2%}, PCA={
                pca_comparison['pca_variance']:.2%}")

        # 4. Comparison with ICA
        logger.info("Comparing with ICA...")
        ica_comparison = self._nmf_vs_ica(X_data, n_factors)
        results['ica_comparison'] = ica_comparison

        # 5. Clustering Metrics
        logger.info("Computing clustering metrics...")
        clustering = self._nmf_clustering_metrics(W_matrix, X_data)
        results['clustering'] = clustering
        logger.info(f"Silhouette score: {clustering['silhouette']:.4f}")

        # 6. Save results
        self._save_nmf_validation_results(results)

        logger.info("NMF validation complete!")
        return results

    def _nmf_bootstrap_stability(
        self,
        X_data: np.ndarray,
        n_factors: int,
        n_bootstrap: int
    ) -> Dict[str, float]:
        """Test NMF stability with bootstrap resampling."""

        n_samples = X_data.shape[0]

        # Fit original NMF
        nmf_original = NMF(
            n_components=n_factors,
            init='nndsvd',
            random_state=42)
        W_original = nmf_original.fit_transform(X_data)
        H_original = nmf_original.components_

        # Bootstrap
        correlations = []

        for i in range(n_bootstrap):
            # Random sample with replacement
            idx = np.random.choice(n_samples, size=n_samples, replace=True)
            X_bootstrap = X_data[idx]

            # Fit NMF
            nmf_boot = NMF(
                n_components=n_factors,
                init='nndsvd',
                random_state=i)
            try:
                W_boot = nmf_boot.fit_transform(X_bootstrap)
                H_boot = nmf_boot.components_

                # Compute correlation between H matrices
                corr = np.corrcoef(
                    H_original.flatten(),
                    H_boot.flatten())[
                    0,
                    1]
                correlations.append(corr)
            except BaseException:
                logger.warning(f"Bootstrap {i} failed")
                continue

        return {
            'mean_correlation': np.mean(correlations),
            'std_correlation': np.std(correlations),
            'min_correlation': np.min(correlations),
            'max_correlation': np.max(correlations)
        }

    def _nmf_reconstruction_error(
        self,
        W_matrix: np.ndarray,
        H_matrix: np.ndarray,
        X_data: np.ndarray
    ) -> Dict[str, float]:
        """Compute reconstruction error."""

        X_reconstructed = np.dot(W_matrix, H_matrix)

        # RMSE
        rmse = np.sqrt(np.mean((X_data - X_reconstructed) ** 2))

        # MAE
        mae = np.mean(np.abs(X_data - X_reconstructed))

        # Frobenius norm
        frobenius = np.linalg.norm(X_data - X_reconstructed, 'fro')

        return {
            'rmse': rmse,
            'mae': mae,
            'frobenius': frobenius
        }

    def _nmf_vs_pca(
        self,
        X_data: np.ndarray,
        n_factors: int
    ) -> Dict[str, float]:
        """Compare NMF with PCA."""

        # Fit NMF
        nmf = NMF(n_components=n_factors, init='nndsvd', random_state=42)
        W_nmf = nmf.fit_transform(X_data)
        H_nmf = nmf.components_
        X_nmf_recon = np.dot(W_nmf, H_nmf)

        # Fit PCA
        pca = PCA(n_components=n_factors, random_state=42)
        W_pca = pca.fit_transform(X_data)
        X_pca_recon = pca.inverse_transform(W_pca)

        # Explained variance
        nmf_variance = 1 - (np.var(X_data - X_nmf_recon) / np.var(X_data))
        pca_variance = np.sum(pca.explained_variance_ratio_)

        # Reconstruction error
        nmf_rmse = np.sqrt(np.mean((X_data - X_nmf_recon) ** 2))
        pca_rmse = np.sqrt(np.mean((X_data - X_pca_recon) ** 2))

        return {
            'nmf_variance': nmf_variance,
            'pca_variance': pca_variance,
            'nmf_rmse': nmf_rmse,
            'pca_rmse': pca_rmse
        }

    def _nmf_vs_ica(
        self,
        X_data: np.ndarray,
        n_factors: int
    ) -> Dict[str, float]:
        """Compare NMF with ICA."""

        # Fit NMF
        nmf = NMF(n_components=n_factors, init='nndsvd', random_state=42)
        W_nmf = nmf.fit_transform(X_data)

        # Fit ICA
        ica = FastICA(n_components=n_factors, random_state=42)
        W_ica = ica.fit_transform(X_data)

        # Compute clustering quality for both
        from sklearn.metrics import silhouette_score

        # Assign to dominant factor
        nmf_labels = np.argmax(W_nmf, axis=1)
        ica_labels = np.argmax(np.abs(W_ica), axis=1)

        try:
            nmf_silhouette = silhouette_score(X_data, nmf_labels)
        except BaseException:
            nmf_silhouette = 0.0

        try:
            ica_silhouette = silhouette_score(X_data, ica_labels)
        except BaseException:
            ica_silhouette = 0.0

        return {
            'nmf_silhouette': nmf_silhouette,
            'ica_silhouette': ica_silhouette
        }

    def _nmf_clustering_metrics(
        self,
        W_matrix: np.ndarray,
        X_data: np.ndarray
    ) -> Dict[str, float]:
        """Compute clustering quality metrics."""

        # Assign each patient to dominant factor
        labels = np.argmax(W_matrix, axis=1)

        # Ensure at least 2 unique labels
        if len(np.unique(labels)) < 2:
            logger.warning("Less than 2 clusters found")
            return {
                'silhouette': 0.0,
                'davies_bouldin': 0.0,
                'calinski_harabasz': 0.0
            }

        # Silhouette score
        try:
            silhouette = silhouette_score(X_data, labels)
        except BaseException:
            silhouette = 0.0

        # Davies-Bouldin index (lower is better)
        try:
            davies_bouldin = davies_bouldin_score(X_data, labels)
        except BaseException:
            davies_bouldin = 0.0

        # Calinski-Harabasz index (higher is better)
        try:
            calinski_harabasz = calinski_harabasz_score(X_data, labels)
        except BaseException:
            calinski_harabasz = 0.0

        return {
            'silhouette': silhouette,
            'davies_bouldin': davies_bouldin,
            'calinski_harabasz': calinski_harabasz
        }

    def _save_nmf_validation_results(self, results: Dict[str, Any]):
        """Save NMF validation results."""

        # Save JSON
        json_path = self.output_dir / 'nmf' / 'validation_metrics.json'

        with open(json_path, 'w') as f:
            json.dump(results, f, indent=2, default=float)

        logger.info(f"NMF validation results saved to {json_path}")

        # Create summary report
        report_path = self.output_dir / 'reports' / 'nmf_validation_report.txt'
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write("=" * 80 + "\n")
            f.write("NMF VALIDATION REPORT\n")
            f.write("=" * 80 + "\n\n")

            f.write("1. BOOTSTRAP STABILITY\n")
            st = results['stability']
            f.write(f"   Mean correlation: {st['mean_correlation']:.4f}\n")
            f.write(f"   Std: {st['std_correlation']:.4f}\n")
            f.write(
                f"   Range: [{
                    st['min_correlation']:.4f}, {
                    st['max_correlation']:.4f}]\n\n")

            f.write("2. RECONSTRUCTION ERROR\n")
            re = results['reconstruction']
            f.write(f"   RMSE: {re['rmse']:.4f}\n")
            f.write(f"   MAE: {re['mae']:.4f}\n")
            f.write(f"   Frobenius norm: {re['frobenius']:.4f}\n\n")

            f.write("3. COMPARISON WITH PCA\n")
            pca = results['pca_comparison']
            f.write(f"   NMF explained variance: {pca['nmf_variance']:.2%}\n")
            f.write(f"   PCA explained variance: {pca['pca_variance']:.2%}\n")
            f.write(f"   NMF RMSE: {pca['nmf_rmse']:.4f}\n")
            f.write(f"   PCA RMSE: {pca['pca_rmse']:.4f}\n\n")

            f.write("4. COMPARISON WITH ICA\n")
            ica = results['ica_comparison']
            f.write(f"   NMF silhouette: {ica['nmf_silhouette']:.4f}\n")
            f.write(f"   ICA silhouette: {ica['ica_silhouette']:.4f}\n\n")

            f.write("5. CLUSTERING METRICS\n")
            cl = results['clustering']
            f.write(f"   Silhouette score: {cl['silhouette']:.4f}\n")
            f.write(f"   Davies-Bouldin index: {cl['davies_bouldin']:.4f}\n")
            f.write(
                f"   Calinski-Harabasz index: {cl['calinski_harabasz']:.2f}\n\n")

            f.write("INTERPRETATION:\n")
            f.write("- Stability > 0.8 → Robust factors\n")
            f.write("- Silhouette > 0.5 → Well-separated clusters\n")
            f.write("- NMF > PCA variance → Better for non-negative data\n")

        logger.info(f"NMF validation report saved to {report_path}")


# =============================================================================
# DEMONSTRATION
# =============================================================================

def demonstrate_validation():
    """Demonstrate validation framework."""
    import xgboost as xgb
    from sklearn.model_selection import train_test_split

    logger.info("=" * 80)
    logger.info("XAI VALIDATION FRAMEWORK DEMONSTRATION")
    logger.info("=" * 80)

    # Generate synthetic data
    np.random.seed(42)
    n_samples = 1000
    n_features = 30

    X = np.random.randn(n_samples, n_features)
    # Use binary classification to avoid SHAP multi-class issues
    y = np.random.randint(0, 2, n_samples)

    feature_names = [f"feature_{i}" for i in range(n_features)]

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Train model
    model = xgb.XGBClassifier(
        n_estimators=100,
        max_depth=6,
        random_state=42,
        use_label_encoder=False,
        eval_metric='logloss'
    )
    model.fit(X_train, y_train)

    # Compute SHAP values
    import shap
    try:
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_test[:100])
    except Exception as e:
        logger.warning(f"SHAP computation failed: {e}")
        # Use dummy SHAP values for demonstration
        shap_values = np.random.randn(100, n_features)

    # Initialize validator
    validator = XAIValidator(output_dir='outputs/validation_demo')

    # 1. Validate SHAP
    logger.info("\n" + "=" * 80)
    logger.info("VALIDATING SHAP VALUES")
    logger.info("=" * 80)
    shap_results = validator.validate_shap(
        shap_values=shap_values,
        X_data=X_test[:100],
        model=model,
        feature_names=feature_names,
        n_bootstrap=50
    )

    # 2. Validate Counterfactuals (simulated)
    logger.info("\n" + "=" * 80)
    logger.info("VALIDATING COUNTERFACTUALS")
    logger.info("=" * 80)

    # Simulate counterfactual results
    cf_results = []
    for i in range(20):
        cf_results.append({
            'n_counterfactuals': np.random.choice([0, 3, 5]),
            'counterfactuals': [
                {
                    'changed_features': {f'feature_{j}': {} for j in range(np.random.randint(2, 6))},
                    'counterfactual': np.random.randn(n_features)
                }
                for _ in range(3)
            ]
        })

    cf_validation = validator.validate_counterfactuals(
        counterfactuals=cf_results,
        original_data=X_test[:20],
        feature_names=feature_names,
        clinical_plausibility_scores=np.random.uniform(2.5, 4.5, 20)
    )

    # 3. Validate NMF
    logger.info("\n" + "=" * 80)
    logger.info("VALIDATING NMF")
    logger.info("=" * 80)

    # Fit NMF
    nmf = NMF(n_components=10, init='nndsvd', random_state=42)
    X_positive = np.abs(X_train)  # Ensure non-negative
    W_matrix = nmf.fit_transform(X_positive)
    H_matrix = nmf.components_

    nmf_validation = validator.validate_nmf(
        W_matrix=W_matrix,
        H_matrix=H_matrix,
        X_data=X_positive,
        n_factors=10,
        n_bootstrap=30
    )

    logger.info("\n" + "=" * 80)
    logger.info("VALIDATION COMPLETE!")
    logger.info("=" * 80)
    logger.info(f"Results saved to: outputs/validation_demo/")
    logger.info("\nGenerated reports:")
    logger.info("  - SHAP validation report")
    logger.info("  - Counterfactual validation report")
    logger.info("  - NMF validation report")


if __name__ == "__main__":
    demonstrate_validation()
