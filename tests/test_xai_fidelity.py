"""
Unit tests for XAIFidelity module.

Tests XAI fidelity metrics for synthetic data validation.

Author: Chatchai Tritham
Date: 2026-01-25
"""

import pytest
import numpy as np
from syndx.phase3_validation.xai_fidelity import XAIFidelity


@pytest.mark.unit
@pytest.mark.fidelity
class TestXAIFidelityInit:
    """Test XAIFidelity initialization."""

    def test_default_initialization(self):
        """Test initialization with default parameters."""
        fidelity = XAIFidelity()

        assert fidelity.model_type == 'xgboost'
        assert fidelity.background_samples == 100
        assert fidelity.random_state == 42
        assert fidelity.archetype_model is None
        assert fidelity.synthetic_model is None
        assert fidelity.fidelity_scores == {}

    def test_custom_initialization(self):
        """Test initialization with custom parameters."""
        fidelity = XAIFidelity(
            model_type='xgboost',
            background_samples=50,
            random_state=123
        )

        assert fidelity.background_samples == 50
        assert fidelity.random_state == 123

    def test_repr_before_evaluation(self):
        """Test string representation before evaluation."""
        fidelity = XAIFidelity()
        repr_str = repr(fidelity)

        assert "XAIFidelity" in repr_str
        assert "evaluated=False" in repr_str

    def test_repr_after_evaluation(self, mock_archetype_data, mock_synthetic_data):
        """Test string representation after evaluation."""
        X_arch, y_arch = mock_archetype_data
        X_synth, y_synth = mock_synthetic_data

        fidelity = XAIFidelity()
        fidelity.fit_archetype_model(X_arch, y_arch)
        fidelity.fit_synthetic_model(X_synth, y_synth)
        fidelity.compute_all_metrics()

        repr_str = repr(fidelity)

        assert "XAIFidelity" in repr_str
        assert "overall_score" in repr_str


@pytest.mark.unit
@pytest.mark.fidelity
class TestFitModels:
    """Test model fitting functionality."""

    def test_fit_archetype_model(self, mock_archetype_data):
        """Test fitting archetype model."""
        X, y = mock_archetype_data
        fidelity = XAIFidelity()

        fidelity.fit_archetype_model(X, y)

        assert fidelity.archetype_model is not None
        assert fidelity.archetype_explainer is not None
        assert fidelity.archetype_shap is not None

    def test_fit_synthetic_model(self, mock_synthetic_data):
        """Test fitting synthetic model."""
        X, y = mock_synthetic_data
        fidelity = XAIFidelity()

        fidelity.fit_synthetic_model(X, y)

        assert fidelity.synthetic_model is not None
        assert fidelity.synthetic_explainer is not None
        assert fidelity.synthetic_shap is not None

    def test_fit_both_models(self, mock_archetype_data, mock_synthetic_data):
        """Test fitting both models."""
        X_arch, y_arch = mock_archetype_data
        X_synth, y_synth = mock_synthetic_data

        fidelity = XAIFidelity()
        fidelity.fit_archetype_model(X_arch, y_arch)
        fidelity.fit_synthetic_model(X_synth, y_synth)

        assert fidelity.archetype_model is not None
        assert fidelity.synthetic_model is not None

    def test_shap_values_shape(self, mock_archetype_data):
        """Test SHAP values have correct shape."""
        X, y = mock_archetype_data
        fidelity = XAIFidelity()
        fidelity.fit_archetype_model(X, y)

        # For multi-class, SHAP values are a list
        if isinstance(fidelity.archetype_shap, list):
            for shap_vals in fidelity.archetype_shap:
                assert shap_vals.shape[0] == X.shape[0]
                assert shap_vals.shape[1] == X.shape[1]
        else:
            # Binary case
            assert fidelity.archetype_shap.shape == X.shape


@pytest.mark.unit
@pytest.mark.fidelity
class TestSHAPCorrelation:
    """Test SHAP correlation computation."""

    def test_compute_shap_correlation_basic(self, mock_archetype_data, mock_synthetic_data):
        """Test basic SHAP correlation computation."""
        X_arch, y_arch = mock_archetype_data
        X_synth, y_synth = mock_synthetic_data

        fidelity = XAIFidelity()
        fidelity.fit_archetype_model(X_arch, y_arch)
        fidelity.fit_synthetic_model(X_synth, y_synth)

        correlation = fidelity.compute_shap_correlation()

        assert -1.0 <= correlation <= 1.0
        assert 'shap_correlation' in fidelity.fidelity_scores

    def test_compute_shap_correlation_without_models_raises_error(self):
        """Test that computing correlation without models raises error."""
        fidelity = XAIFidelity()

        with pytest.raises(ValueError, match="Must fit both models"):
            fidelity.compute_shap_correlation()

    def test_shap_correlation_high_for_similar_models(self, mock_archetype_data):
        """Test that SHAP correlation is high for similar models."""
        X, y = mock_archetype_data

        fidelity = XAIFidelity()
        fidelity.fit_archetype_model(X, y)

        # Use same data for both (should give high correlation)
        fidelity.fit_synthetic_model(X, y)

        correlation = fidelity.compute_shap_correlation()

        # Should be very high for identical data
        assert correlation > 0.8


@pytest.mark.unit
@pytest.mark.fidelity
class TestRankAgreement:
    """Test feature ranking agreement."""

    def test_compute_rank_agreement_basic(self, mock_archetype_data, mock_synthetic_data):
        """Test basic rank agreement computation."""
        X_arch, y_arch = mock_archetype_data
        X_synth, y_synth = mock_synthetic_data

        fidelity = XAIFidelity()
        fidelity.fit_archetype_model(X_arch, y_arch)
        fidelity.fit_synthetic_model(X_synth, y_synth)

        tau = fidelity.compute_rank_agreement()

        assert -1.0 <= tau <= 1.0
        assert 'rank_agreement' in fidelity.fidelity_scores

    def test_rank_agreement_without_models_raises_error(self):
        """Test that computing rank agreement without models raises error."""
        fidelity = XAIFidelity()

        with pytest.raises(ValueError, match="Must fit both models"):
            fidelity.compute_rank_agreement()

    def test_rank_agreement_high_for_identical_data(self, mock_archetype_data):
        """Test that rank agreement is high for identical data."""
        X, y = mock_archetype_data

        fidelity = XAIFidelity()
        fidelity.fit_archetype_model(X, y)
        fidelity.fit_synthetic_model(X, y)

        tau = fidelity.compute_rank_agreement()

        # Should be very high for identical data
        assert tau > 0.7


@pytest.mark.unit
@pytest.mark.fidelity
class TestFeatureImportanceMSE:
    """Test feature importance MSE computation."""

    def test_compute_importance_mse_basic(self, mock_archetype_data, mock_synthetic_data):
        """Test basic importance MSE computation."""
        X_arch, y_arch = mock_archetype_data
        X_synth, y_synth = mock_synthetic_data

        fidelity = XAIFidelity()
        fidelity.fit_archetype_model(X_arch, y_arch)
        fidelity.fit_synthetic_model(X_synth, y_synth)

        mse = fidelity.compute_feature_importance_mse()

        assert mse >= 0
        assert 'importance_mse' in fidelity.fidelity_scores

    def test_importance_mse_low_for_identical_data(self, mock_archetype_data):
        """Test that MSE is low for identical data."""
        X, y = mock_archetype_data

        fidelity = XAIFidelity()
        fidelity.fit_archetype_model(X, y)
        fidelity.fit_synthetic_model(X, y)

        mse = fidelity.compute_feature_importance_mse()

        # Should be very low for identical data
        assert mse < 0.01

    def test_importance_mse_without_models_raises_error(self):
        """Test that computing MSE without models raises error."""
        fidelity = XAIFidelity()

        with pytest.raises(ValueError, match="Must fit both models"):
            fidelity.compute_feature_importance_mse()


@pytest.mark.unit
@pytest.mark.fidelity
class TestTopKOverlap:
    """Test top-k feature overlap computation."""

    def test_compute_top_k_overlap_basic(self, mock_archetype_data, mock_synthetic_data):
        """Test basic top-k overlap computation."""
        X_arch, y_arch = mock_archetype_data
        X_synth, y_synth = mock_synthetic_data

        fidelity = XAIFidelity()
        fidelity.fit_archetype_model(X_arch, y_arch)
        fidelity.fit_synthetic_model(X_synth, y_synth)

        overlap = fidelity.compute_top_k_overlap(k=10)

        assert 0.0 <= overlap <= 1.0
        assert 'top_10_overlap' in fidelity.fidelity_scores

    def test_top_k_overlap_different_k_values(self, mock_archetype_data, mock_synthetic_data):
        """Test top-k overlap with different k values."""
        X_arch, y_arch = mock_archetype_data
        X_synth, y_synth = mock_synthetic_data

        fidelity = XAIFidelity()
        fidelity.fit_archetype_model(X_arch, y_arch)
        fidelity.fit_synthetic_model(X_synth, y_synth)

        overlap_5 = fidelity.compute_top_k_overlap(k=5)
        overlap_10 = fidelity.compute_top_k_overlap(k=10)
        overlap_20 = fidelity.compute_top_k_overlap(k=20)

        assert all(0.0 <= o <= 1.0 for o in [overlap_5, overlap_10, overlap_20])
        assert 'top_5_overlap' in fidelity.fidelity_scores
        assert 'top_10_overlap' in fidelity.fidelity_scores
        assert 'top_20_overlap' in fidelity.fidelity_scores

    def test_top_k_overlap_perfect_for_identical_data(self, mock_archetype_data):
        """Test that overlap is 1.0 for identical data."""
        X, y = mock_archetype_data

        fidelity = XAIFidelity()
        fidelity.fit_archetype_model(X, y)
        fidelity.fit_synthetic_model(X, y)

        overlap = fidelity.compute_top_k_overlap(k=10)

        # Should be perfect for identical data
        assert overlap == pytest.approx(1.0)


@pytest.mark.unit
@pytest.mark.fidelity
class TestInteractionFidelity:
    """Test interaction pattern fidelity."""

    def test_compute_interaction_fidelity_with_matrices(self, mock_nmf_matrices):
        """Test interaction fidelity with provided matrices."""
        (W_arch, H_arch), (W_synth, H_synth) = mock_nmf_matrices

        fidelity = XAIFidelity()
        fid_score = fidelity.compute_interaction_fidelity(H_arch, H_synth)

        assert 0.0 <= fid_score <= 1.0
        assert 'interaction_fidelity' in fidelity.fidelity_scores

    def test_compute_interaction_fidelity_without_matrices(self):
        """Test interaction fidelity without matrices."""
        fidelity = XAIFidelity()
        fid_score = fidelity.compute_interaction_fidelity(None, None)

        assert fid_score == 0.0
        assert fidelity.fidelity_scores['interaction_fidelity'] == 0.0

    def test_interaction_fidelity_high_for_similar_matrices(self):
        """Test that fidelity is high for similar matrices."""
        # Create nearly identical matrices
        np.random.seed(42)
        matrix1 = np.random.rand(20, 50)
        matrix2 = matrix1 + np.random.randn(20, 50) * 0.01  # Small noise

        fidelity = XAIFidelity()
        fid_score = fidelity.compute_interaction_fidelity(matrix1, matrix2)

        assert fid_score > 0.9


@pytest.mark.unit
@pytest.mark.fidelity
class TestComputeAllMetrics:
    """Test comprehensive metric computation."""

    def test_compute_all_metrics_basic(self, mock_archetype_data, mock_synthetic_data):
        """Test computing all metrics."""
        X_arch, y_arch = mock_archetype_data
        X_synth, y_synth = mock_synthetic_data

        fidelity = XAIFidelity()
        fidelity.fit_archetype_model(X_arch, y_arch)
        fidelity.fit_synthetic_model(X_synth, y_synth)

        scores = fidelity.compute_all_metrics()

        # Check all metrics are present
        assert 'shap_correlation' in scores
        assert 'rank_agreement' in scores
        assert 'importance_mse' in scores
        assert 'top_10_overlap' in scores
        assert 'top_20_overlap' in scores
        assert 'overall_fidelity' in scores

    def test_compute_all_metrics_with_interactions(self, mock_archetype_data,
                                                   mock_synthetic_data, mock_nmf_matrices):
        """Test computing all metrics including interactions."""
        X_arch, y_arch = mock_archetype_data
        X_synth, y_synth = mock_synthetic_data
        (W_arch, H_arch), (W_synth, H_synth) = mock_nmf_matrices

        fidelity = XAIFidelity()
        fidelity.fit_archetype_model(X_arch, y_arch)
        fidelity.fit_synthetic_model(X_synth, y_synth)

        scores = fidelity.compute_all_metrics(H_arch, H_synth)

        assert 'interaction_fidelity' in scores
        assert scores['interaction_fidelity'] > 0

    def test_overall_fidelity_calculation(self, mock_archetype_data, mock_synthetic_data):
        """Test overall fidelity score calculation."""
        X_arch, y_arch = mock_archetype_data
        X_synth, y_synth = mock_synthetic_data

        fidelity = XAIFidelity()
        fidelity.fit_archetype_model(X_arch, y_arch)
        fidelity.fit_synthetic_model(X_synth, y_synth)

        scores = fidelity.compute_all_metrics()

        # Overall fidelity should be weighted average
        expected = (
            0.40 * scores['shap_correlation'] +
            0.30 * scores['rank_agreement'] +
            0.30 * scores['top_20_overlap']
        )

        assert scores['overall_fidelity'] == pytest.approx(expected)

    def test_overall_fidelity_in_valid_range(self, mock_archetype_data, mock_synthetic_data):
        """Test that overall fidelity is in [0, 1]."""
        X_arch, y_arch = mock_archetype_data
        X_synth, y_synth = mock_synthetic_data

        fidelity = XAIFidelity()
        fidelity.fit_archetype_model(X_arch, y_arch)
        fidelity.fit_synthetic_model(X_synth, y_synth)

        scores = fidelity.compute_all_metrics()

        assert 0.0 <= scores['overall_fidelity'] <= 1.0


@pytest.mark.unit
@pytest.mark.fidelity
class TestSummary:
    """Test XAIFidelity summary method."""

    def test_summary_before_evaluation(self):
        """Test summary before evaluation."""
        fidelity = XAIFidelity()
        summary = fidelity.summary()

        assert summary['status'] == 'not_evaluated'

    def test_summary_after_evaluation(self, mock_archetype_data, mock_synthetic_data):
        """Test summary after evaluation."""
        X_arch, y_arch = mock_archetype_data
        X_synth, y_synth = mock_synthetic_data

        fidelity = XAIFidelity()
        fidelity.fit_archetype_model(X_arch, y_arch)
        fidelity.fit_synthetic_model(X_synth, y_synth)
        fidelity.compute_all_metrics()

        summary = fidelity.summary()

        assert summary['status'] == 'evaluated'
        assert 'scores' in summary
        assert 'overall_fidelity' in summary
        assert 'interpretation' in summary
        assert 'recommendation' in summary

    def test_summary_interpretation_excellent(self):
        """Test interpretation for excellent fidelity."""
        # Create perfect fidelity scenario
        fidelity = XAIFidelity()
        fidelity.fidelity_scores = {'overall_fidelity': 0.95}

        summary = fidelity.summary()

        assert "Excellent" in summary['interpretation']
        assert "Use synthetic data" in summary['recommendation']

    def test_summary_interpretation_good(self):
        """Test interpretation for good fidelity."""
        fidelity = XAIFidelity()
        fidelity.fidelity_scores = {'overall_fidelity': 0.75}

        summary = fidelity.summary()

        assert "Good" in summary['interpretation']

    def test_summary_interpretation_fair(self):
        """Test interpretation for fair fidelity."""
        fidelity = XAIFidelity()
        fidelity.fidelity_scores = {'overall_fidelity': 0.60}

        summary = fidelity.summary()

        assert "Fair" in summary['interpretation']

    def test_summary_interpretation_poor(self):
        """Test interpretation for poor fidelity."""
        fidelity = XAIFidelity()
        fidelity.fidelity_scores = {'overall_fidelity': 0.40}

        summary = fidelity.summary()

        assert "Poor" in summary['interpretation']
        assert "Caution" in summary['recommendation']


@pytest.mark.unit
@pytest.mark.fidelity
class TestXAIFidelityIntegration:
    """Integration tests for XAIFidelity."""

    def test_full_evaluation_pipeline(self, mock_archetype_data, mock_synthetic_data):
        """Test complete evaluation pipeline."""
        X_arch, y_arch = mock_archetype_data
        X_synth, y_synth = mock_synthetic_data

        fidelity = XAIFidelity()

        # Fit models
        fidelity.fit_archetype_model(X_arch, y_arch)
        fidelity.fit_synthetic_model(X_synth, y_synth)

        # Compute metrics
        scores = fidelity.compute_all_metrics()

        # Get summary
        summary = fidelity.summary()

        # Verify complete workflow
        assert len(scores) >= 6  # All core metrics
        assert summary['status'] == 'evaluated'
        assert 0.0 <= scores['overall_fidelity'] <= 1.0

    def test_reproducibility_with_random_state(self, mock_archetype_data, mock_synthetic_data):
        """Test reproducibility with fixed random state."""
        X_arch, y_arch = mock_archetype_data
        X_synth, y_synth = mock_synthetic_data

        # Run 1
        fidelity1 = XAIFidelity(random_state=42)
        fidelity1.fit_archetype_model(X_arch, y_arch)
        fidelity1.fit_synthetic_model(X_synth, y_synth)
        scores1 = fidelity1.compute_all_metrics()

        # Run 2
        fidelity2 = XAIFidelity(random_state=42)
        fidelity2.fit_archetype_model(X_arch, y_arch)
        fidelity2.fit_synthetic_model(X_synth, y_synth)
        scores2 = fidelity2.compute_all_metrics()

        # Should be identical
        assert scores1['overall_fidelity'] == pytest.approx(scores2['overall_fidelity'])

    def test_high_fidelity_for_identical_data(self, mock_archetype_data):
        """Test that identical data gives very high fidelity."""
        X, y = mock_archetype_data

        fidelity = XAIFidelity()
        fidelity.fit_archetype_model(X, y)
        fidelity.fit_synthetic_model(X, y)  # Same data

        scores = fidelity.compute_all_metrics()

        # Should be near-perfect
        assert scores['overall_fidelity'] > 0.9
        assert scores['shap_correlation'] > 0.9
        assert scores['top_20_overlap'] == 1.0
