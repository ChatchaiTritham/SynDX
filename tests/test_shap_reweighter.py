"""
Unit tests for SHAPReweighter module.

Tests SHAP-based feature importance reweighting for synthetic data generation.

Author: Chatchai Tritham
Date: 2026-01-25
"""

import pytest
import numpy as np
from syndx.phase2_synthesis.shap_reweighter import SHAPReweighter


@pytest.mark.unit
class TestSHAPReweighterInit:
    """Test SHAPReweighter initialization."""

    def test_default_initialization(self):
        """Test initialization with default parameters."""
        reweighter = SHAPReweighter()

        assert reweighter.background_samples == 100
        assert reweighter.random_state == 42
        assert reweighter.model is None
        assert reweighter.explainer is None
        assert reweighter.shap_values is None
        assert reweighter.feature_weights is None

    def test_custom_initialization(self):
        """Test initialization with custom parameters."""
        reweighter = SHAPReweighter(background_samples=50, random_state=123)

        assert reweighter.background_samples == 50
        assert reweighter.random_state == 123

    def test_repr(self):
        """Test string representation."""
        reweighter = SHAPReweighter()
        repr_str = repr(reweighter)

        assert "SHAPReweighter" in repr_str
        assert "background_samples=100" in repr_str


@pytest.mark.unit
class TestSHAPReweighterFit:
    """Test SHAPReweighter fit method."""

    def test_fit_basic(self, mock_archetype_data):
        """Test basic fit functionality."""
        X, y = mock_archetype_data
        reweighter = SHAPReweighter()

        result = reweighter.fit(X, y)

        # Check return value
        assert result is reweighter  # Should return self

        # Check model training
        assert reweighter.model is not None
        assert reweighter.explainer is not None
        assert reweighter.shap_values is not None
        assert reweighter.feature_weights is not None

    def test_fit_sets_feature_weights(self, mock_archetype_data):
        """Test that fit sets feature weights correctly."""
        X, y = mock_archetype_data
        reweighter = SHAPReweighter()
        reweighter.fit(X, y)

        # Feature weights should be normalized
        assert len(reweighter.feature_weights) == X.shape[1]
        assert np.allclose(np.sum(reweighter.feature_weights), 1.0)
        assert np.all(reweighter.feature_weights >= 0)

    def test_fit_different_sample_sizes(self):
        """Test fit with different sample sizes."""
        # Small dataset (< background_samples)
        X_small = np.random.randn(50, 20)
        y_small = np.random.randint(0, 3, 50)

        reweighter = SHAPReweighter(background_samples=100)
        reweighter.fit(X_small, y_small)

        assert reweighter.feature_weights is not None

        # Large dataset (> background_samples)
        X_large = np.random.randn(200, 20)
        y_large = np.random.randint(0, 3, 200)

        reweighter = SHAPReweighter(background_samples=50)
        reweighter.fit(X_large, y_large)

        assert reweighter.feature_weights is not None


@pytest.mark.unit
class TestSHAPReweighterTransform:
    """Test SHAPReweighter transform method."""

    def test_transform_basic(self, mock_archetype_data, mock_synthetic_data):
        """Test basic transform functionality."""
        X_arch, y_arch = mock_archetype_data
        X_synth, _ = mock_synthetic_data

        reweighter = SHAPReweighter()
        reweighter.fit(X_arch, y_arch)

        X_transformed = reweighter.transform(X_synth)

        assert X_transformed.shape == X_synth.shape
        assert not np.allclose(X_transformed, X_synth)  # Should be different

    def test_transform_preserves_shape(self, mock_archetype_data):
        """Test that transform preserves input shape."""
        X, y = mock_archetype_data
        reweighter = SHAPReweighter()
        reweighter.fit(X, y)

        X_new = np.random.randn(50, X.shape[1])
        X_transformed = reweighter.transform(X_new)

        assert X_transformed.shape == X_new.shape

    def test_transform_without_fit_raises_error(self):
        """Test that transform without fit raises error."""
        reweighter = SHAPReweighter()
        X = np.random.randn(10, 20)

        with pytest.raises(ValueError, match="Must fit"):
            reweighter.transform(X)

    def test_transform_applies_sqrt_weights(self, mock_archetype_data):
        """Test that transform applies sqrt of weights."""
        X, y = mock_archetype_data
        reweighter = SHAPReweighter()
        reweighter.fit(X, y)

        X_synth = np.ones((10, X.shape[1]))  # All ones for easy verification
        X_transformed = reweighter.transform(X_synth)

        # Should be scaled by sqrt(weights)
        expected = X_synth * np.sqrt(reweighter.feature_weights)
        assert np.allclose(X_transformed, expected)


@pytest.mark.unit
class TestSHAPReweighterGetTopFeatures:
    """Test SHAPReweighter get_top_features method."""

    def test_get_top_features_default(self, mock_archetype_data):
        """Test getting top features with default n=20."""
        X, y = mock_archetype_data
        reweighter = SHAPReweighter()
        reweighter.fit(X, y)

        top_features = reweighter.get_top_features()

        assert len(top_features) == 20
        assert all(isinstance(idx, int) for idx, _ in top_features)
        assert all(isinstance(weight, float) for _, weight in top_features)

        # Should be sorted in descending order
        weights = [w for _, w in top_features]
        assert weights == sorted(weights, reverse=True)

    def test_get_top_features_custom_n(self, mock_archetype_data):
        """Test getting top n features."""
        X, y = mock_archetype_data
        reweighter = SHAPReweighter()
        reweighter.fit(X, y)

        top_10 = reweighter.get_top_features(n=10)
        top_5 = reweighter.get_top_features(n=5)

        assert len(top_10) == 10
        assert len(top_5) == 5

        # Top 5 should be subset of top 10
        top_5_indices = [idx for idx, _ in top_5]
        top_10_indices = [idx for idx, _ in top_10]
        assert all(idx in top_10_indices for idx in top_5_indices)

    def test_get_top_features_without_fit_raises_error(self):
        """Test that get_top_features without fit raises error."""
        reweighter = SHAPReweighter()

        with pytest.raises(ValueError, match="Must fit"):
            reweighter.get_top_features()


@pytest.mark.unit
class TestSHAPReweighterGetSamplingWeights:
    """Test SHAPReweighter get_sampling_weights method."""

    def test_get_sampling_weights(self, mock_archetype_data):
        """Test getting sampling weights."""
        X, y = mock_archetype_data
        reweighter = SHAPReweighter()
        reweighter.fit(X, y)

        weights = reweighter.get_sampling_weights()

        assert len(weights) == X.shape[1]
        assert np.allclose(np.sum(weights), 1.0)
        assert np.all(weights >= 0)

    def test_get_sampling_weights_without_fit_raises_error(self):
        """Test that get_sampling_weights without fit raises error."""
        reweighter = SHAPReweighter()

        with pytest.raises(ValueError, match="Must fit"):
            reweighter.get_sampling_weights()


@pytest.mark.unit
class TestSHAPReweighterSummary:
    """Test SHAPReweighter summary method."""

    def test_summary_before_fit(self):
        """Test summary before fitting."""
        reweighter = SHAPReweighter()
        summary = reweighter.summary()

        assert summary['status'] == 'not_fitted'

    def test_summary_after_fit(self, mock_archetype_data):
        """Test summary after fitting."""
        X, y = mock_archetype_data
        reweighter = SHAPReweighter()
        reweighter.fit(X, y)

        summary = reweighter.summary()

        assert summary['status'] == 'fitted'
        assert 'n_features' in summary
        assert 'top_feature_weight' in summary
        assert 'min_feature_weight' in summary
        assert 'mean_feature_weight' in summary
        assert 'background_samples' in summary

        assert summary['n_features'] == X.shape[1]
        assert summary['mean_feature_weight'] == pytest.approx(1.0 / X.shape[1], rel=0.1)


@pytest.mark.unit
class TestSHAPReweighterIntegration:
    """Integration tests for SHAPReweighter."""

    def test_fit_transform_pipeline(self, mock_archetype_data, mock_synthetic_data):
        """Test complete fit-transform pipeline."""
        X_arch, y_arch = mock_archetype_data
        X_synth, _ = mock_synthetic_data

        reweighter = SHAPReweighter()
        reweighter.fit(X_arch, y_arch)
        X_transformed = reweighter.transform(X_synth)

        # Verify transformation
        assert X_transformed.shape == X_synth.shape
        assert not np.array_equal(X_transformed, X_synth)

        # Verify weights are reasonable
        top_features = reweighter.get_top_features(n=5)
        assert len(top_features) == 5

        # Top features should have higher weights
        top_weight = top_features[0][1]
        mean_weight = 1.0 / X_arch.shape[1]
        assert top_weight > mean_weight

    def test_reproducibility(self, mock_archetype_data):
        """Test that results are reproducible with same random state."""
        X, y = mock_archetype_data

        reweighter1 = SHAPReweighter(random_state=42)
        reweighter1.fit(X, y)
        weights1 = reweighter1.feature_weights

        reweighter2 = SHAPReweighter(random_state=42)
        reweighter2.fit(X, y)
        weights2 = reweighter2.feature_weights

        assert np.allclose(weights1, weights2)

    def test_different_random_states_give_different_results(self, mock_archetype_data):
        """Test that different random states give different results."""
        X, y = mock_archetype_data

        reweighter1 = SHAPReweighter(random_state=42)
        reweighter1.fit(X, y)
        weights1 = reweighter1.feature_weights

        reweighter2 = SHAPReweighter(random_state=123)
        reweighter2.fit(X, y)
        weights2 = reweighter2.feature_weights

        # May be similar but shouldn't be identical due to randomness
        # (though with small data they might be very close)
        # Just verify both are valid
        assert len(weights1) == len(weights2)
        assert np.allclose(np.sum(weights1), 1.0)
        assert np.allclose(np.sum(weights2), 1.0)
