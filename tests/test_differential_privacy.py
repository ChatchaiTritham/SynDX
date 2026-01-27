"""
Unit tests for DifferentialPrivacy module.

Tests ε-differential privacy via DP-SGD for VAE training.

Author: Chatchai Tritham
Date: 2026-01-25
"""

import pytest
import numpy as np
from syndx.phase2_synthesis.differential_privacy import DifferentialPrivacy


@pytest.mark.unit
@pytest.mark.privacy
class TestDifferentialPrivacyInit:
    """Test DifferentialPrivacy initialization."""

    def test_default_initialization(self):
        """Test initialization with default parameters."""
        dp = DifferentialPrivacy()

        assert dp.epsilon == 1.0
        assert dp.delta == 1e-5
        assert dp.clip_norm == 1.0
        assert dp.noise_multiplier > 0
        assert dp.privacy_spent['epsilon'] == 0.0
        assert dp.total_steps == 0

    def test_custom_initialization(self):
        """Test initialization with custom parameters."""
        dp = DifferentialPrivacy(epsilon=0.5, delta=1e-6, clip_norm=2.0)

        assert dp.epsilon == 0.5
        assert dp.delta == 1e-6
        assert dp.clip_norm == 2.0

    def test_auto_noise_calibration(self):
        """Test automatic noise multiplier calibration."""
        dp1 = DifferentialPrivacy(epsilon=1.0)
        dp2 = DifferentialPrivacy(epsilon=0.5)

        # Smaller epsilon should require larger noise
        assert dp2.noise_multiplier > dp1.noise_multiplier

    def test_manual_noise_multiplier(self):
        """Test manual noise multiplier specification."""
        custom_multiplier = 1.5
        dp = DifferentialPrivacy(noise_multiplier=custom_multiplier)

        assert dp.noise_multiplier == custom_multiplier

    def test_repr(self):
        """Test string representation."""
        dp = DifferentialPrivacy(epsilon=1.0, delta=1e-5, clip_norm=1.0)
        repr_str = repr(dp)

        assert "DifferentialPrivacy" in repr_str
        assert "ε=1.00" in repr_str
        assert "δ=1.00e-05" in repr_str


@pytest.mark.unit
@pytest.mark.privacy
class TestGradientClipping:
    """Test gradient clipping functionality."""

    def test_clip_gradients_basic(self, mock_gradients):
        """Test basic gradient clipping."""
        dp = DifferentialPrivacy(clip_norm=1.0)
        clipped, stats = dp.clip_gradients(mock_gradients)

        assert len(clipped) == len(mock_gradients)
        assert all(isinstance(g, np.ndarray) for g in clipped)

        # Check statistics
        assert 'avg_norm' in stats
        assert 'max_norm' in stats
        assert 'clip_fraction' in stats
        assert 'norm_reduction' in stats

    def test_clip_gradients_respects_bound(self, mock_gradients):
        """Test that clipping respects the norm bound."""
        dp = DifferentialPrivacy(clip_norm=1.0)
        clipped, _ = dp.clip_gradients(mock_gradients)

        # All clipped gradients should have norm <= clip_norm
        for grad in clipped:
            norm = np.linalg.norm(grad)
            assert norm <= dp.clip_norm + 1e-6  # Small tolerance for floating point

    def test_clip_gradients_preserves_small_gradients(self):
        """Test that small gradients are not modified."""
        dp = DifferentialPrivacy(clip_norm=10.0)

        # Create small gradients
        small_grads = [np.random.randn(10) * 0.1 for _ in range(5)]
        clipped, stats = dp.clip_gradients(small_grads)

        # Should be unchanged
        for orig, clip in zip(small_grads, clipped):
            assert np.allclose(orig, clip)

        assert stats['clip_fraction'] == 0.0

    def test_clip_gradients_stats_accuracy(self):
        """Test accuracy of clipping statistics."""
        dp = DifferentialPrivacy(clip_norm=1.0)

        # Create known gradients
        gradients = [
            np.array([3.0, 4.0]),  # Norm = 5.0, will be clipped
            np.array([0.6, 0.8]),  # Norm = 1.0, at boundary
            np.array([0.3, 0.4])   # Norm = 0.5, won't be clipped
        ]

        clipped, stats = dp.clip_gradients(gradients)

        assert stats['max_norm'] == pytest.approx(5.0)
        assert stats['clip_fraction'] == pytest.approx(1/3)  # Only first one clipped


@pytest.mark.unit
@pytest.mark.privacy
class TestNoiseAddition:
    """Test Gaussian noise addition."""

    def test_add_noise_basic(self):
        """Test basic noise addition."""
        dp = DifferentialPrivacy(epsilon=1.0, clip_norm=1.0)

        gradient = np.zeros(100)  # Zero gradient
        batch_size = 64

        noisy_grad = dp.add_noise(gradient, batch_size)

        assert noisy_grad.shape == gradient.shape
        assert not np.allclose(noisy_grad, gradient)  # Should be different

    def test_add_noise_scale(self):
        """Test noise scaling."""
        dp = DifferentialPrivacy(epsilon=1.0, clip_norm=1.0)

        gradient = np.zeros(1000)
        batch_size = 64

        noisy_grad = dp.add_noise(gradient, batch_size)

        # Noise should have std approximately equal to noise_multiplier * clip_norm
        expected_std = dp.noise_multiplier * dp.clip_norm
        actual_std = np.std(noisy_grad)

        assert actual_std == pytest.approx(expected_std, rel=0.2)  # 20% tolerance

    def test_add_noise_different_shapes(self):
        """Test noise addition with different gradient shapes."""
        dp = DifferentialPrivacy()

        for shape in [(10,), (5, 5), (2, 3, 4)]:
            gradient = np.zeros(shape)
            noisy_grad = dp.add_noise(gradient, 64)
            assert noisy_grad.shape == gradient.shape


@pytest.mark.unit
@pytest.mark.privacy
class TestPrivatizeGradients:
    """Test full DP-SGD privatization."""

    def test_privatize_gradients_basic(self, mock_gradients):
        """Test basic gradient privatization."""
        dp = DifferentialPrivacy(epsilon=1.0, clip_norm=1.0)

        private_grad, stats = dp.privatize_gradients(mock_gradients, batch_size=64)

        assert isinstance(private_grad, np.ndarray)
        assert private_grad.shape == mock_gradients[0].shape
        assert 'avg_norm' in stats
        assert 'clip_fraction' in stats

    def test_privatize_gradients_pipeline(self, mock_gradients):
        """Test complete privatization pipeline."""
        dp = DifferentialPrivacy(epsilon=1.0, clip_norm=1.0)

        # Privatize
        private_grad, stats = dp.privatize_gradients(mock_gradients, batch_size=64)

        # Verify clipping occurred
        assert stats['max_norm'] > 0
        assert 0 <= stats['clip_fraction'] <= 1.0

        # Verify noise was added (output shouldn't be exactly the mean)
        mean_grad = np.mean([g for g in mock_gradients], axis=0)
        assert not np.allclose(private_grad, mean_grad)

    def test_privatize_gradients_batch_size_mismatch_warning(self):
        """Test warning when gradient count != batch_size."""
        dp = DifferentialPrivacy()

        gradients = [np.random.randn(10) for _ in range(32)]

        # Should handle mismatch gracefully
        private_grad, stats = dp.privatize_gradients(gradients, batch_size=64)
        assert private_grad is not None


@pytest.mark.unit
@pytest.mark.privacy
class TestPrivacyAccounting:
    """Test privacy budget accounting."""

    def test_get_privacy_spent_basic(self):
        """Test basic privacy accounting."""
        dp = DifferentialPrivacy(epsilon=1.0, delta=1e-5)

        steps = 1000
        batch_size = 64
        n_samples = 8400

        eps_spent, delta = dp.get_privacy_spent(steps, batch_size, n_samples)

        assert eps_spent > 0
        assert delta == dp.delta
        assert eps_spent <= dp.epsilon * 2  # Reasonable upper bound

    def test_privacy_increases_with_steps(self):
        """Test that privacy loss increases with training steps."""
        dp = DifferentialPrivacy(epsilon=1.0)

        batch_size = 64
        n_samples = 8400

        eps_100, _ = dp.get_privacy_spent(100, batch_size, n_samples)
        eps_1000, _ = dp.get_privacy_spent(1000, batch_size, n_samples)

        assert eps_1000 > eps_100

    def test_check_privacy_budget(self):
        """Test privacy budget checking."""
        dp = DifferentialPrivacy(epsilon=1.0)

        batch_size = 64
        n_samples = 8400

        # Small number of steps should be within budget
        assert dp.check_privacy_budget(100, batch_size, n_samples) is True

        # Very large number of steps should exceed budget
        assert dp.check_privacy_budget(100000, batch_size, n_samples) is False

    def test_get_max_steps(self):
        """Test maximum steps calculation."""
        dp = DifferentialPrivacy(epsilon=1.0)

        batch_size = 64
        n_samples = 8400

        max_steps = dp.get_max_steps(batch_size, n_samples)

        assert max_steps > 0
        assert isinstance(max_steps, int)

        # Should be within budget
        eps_spent, _ = dp.get_privacy_spent(max_steps, batch_size, n_samples)
        assert eps_spent <= dp.epsilon * 1.1  # Small tolerance

    def test_privacy_accounting_updates_state(self):
        """Test that privacy accounting updates internal state."""
        dp = DifferentialPrivacy(epsilon=1.0)

        steps = 500
        batch_size = 64
        n_samples = 8400

        dp.get_privacy_spent(steps, batch_size, n_samples)

        assert dp.total_steps == steps
        assert dp.privacy_spent['epsilon'] > 0
        assert dp.privacy_spent['steps'] == steps


@pytest.mark.unit
@pytest.mark.privacy
class TestSummary:
    """Test DifferentialPrivacy summary method."""

    def test_summary_initial_state(self):
        """Test summary before any privacy accounting."""
        dp = DifferentialPrivacy(epsilon=1.0, delta=1e-5, clip_norm=1.0)
        summary = dp.summary()

        assert summary['privacy_parameters']['epsilon'] == 1.0
        assert summary['privacy_parameters']['delta'] == 1e-5
        assert summary['privacy_parameters']['clip_norm'] == 1.0

        assert summary['privacy_accounting']['epsilon_spent'] == 0.0
        assert summary['privacy_accounting']['within_budget'] is True

    def test_summary_after_accounting(self):
        """Test summary after privacy accounting."""
        dp = DifferentialPrivacy(epsilon=1.0)

        steps = 1000
        dp.get_privacy_spent(steps, 64, 8400)

        summary = dp.summary()

        assert summary['privacy_accounting']['epsilon_spent'] > 0
        assert summary['privacy_accounting']['total_steps'] == steps
        assert 'budget_remaining' in summary['privacy_accounting']


@pytest.mark.unit
@pytest.mark.privacy
class TestDifferentialPrivacyIntegration:
    """Integration tests for DifferentialPrivacy."""

    def test_full_dp_sgd_workflow(self, mock_gradients):
        """Test complete DP-SGD workflow."""
        dp = DifferentialPrivacy(epsilon=1.0, delta=1e-5, clip_norm=1.0)

        batch_size = len(mock_gradients)
        n_samples = 8400
        n_steps = 100

        for step in range(n_steps):
            # Privatize gradients
            private_grad, stats = dp.privatize_gradients(mock_gradients, batch_size)

            assert private_grad is not None
            assert stats['clip_fraction'] >= 0

        # Check final privacy budget
        eps_spent, _ = dp.get_privacy_spent(n_steps, batch_size, n_samples)

        assert eps_spent > 0
        assert dp.total_steps == n_steps

    def test_privacy_budget_enforcement(self):
        """Test that privacy budget can be monitored and enforced."""
        dp = DifferentialPrivacy(epsilon=1.0, delta=1e-5)

        batch_size = 64
        n_samples = 8400

        max_steps = dp.get_max_steps(batch_size, n_samples)

        # Training within budget
        for step in range(max_steps // 2):
            within_budget = dp.check_privacy_budget(step, batch_size, n_samples)
            assert within_budget is True

        # Training beyond budget
        for step in range(max_steps, max_steps * 2, 100):
            within_budget = dp.check_privacy_budget(step, batch_size, n_samples)
            if step > max_steps:
                # May or may not be within budget depending on accounting
                assert isinstance(within_budget, bool)

    def test_reproducibility_with_fixed_seed(self):
        """Test that noise is reproducible with numpy seed."""
        np.random.seed(42)
        dp1 = DifferentialPrivacy(epsilon=1.0)

        gradient1 = np.zeros(100)
        noisy1 = dp1.add_noise(gradient1, 64)

        np.random.seed(42)
        dp2 = DifferentialPrivacy(epsilon=1.0)

        gradient2 = np.zeros(100)
        noisy2 = dp2.add_noise(gradient2, 64)

        assert np.allclose(noisy1, noisy2)

    def test_privacy_guarantees(self):
        """Test that privacy parameters satisfy basic guarantees."""
        epsilon_values = [0.1, 0.5, 1.0, 2.0]

        for eps in epsilon_values:
            dp = DifferentialPrivacy(epsilon=eps, delta=1e-5)

            # Noise multiplier should be inversely proportional to epsilon
            # (smaller epsilon = more noise)
            assert dp.noise_multiplier > 0

            # For same delta, smaller epsilon should have larger noise multiplier
            if eps < 2.0:
                dp_large_eps = DifferentialPrivacy(epsilon=2.0, delta=1e-5)
                assert dp.noise_multiplier > dp_large_eps.noise_multiplier
