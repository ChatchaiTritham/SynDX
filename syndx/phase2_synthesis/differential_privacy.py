"""
Differential Privacy via DP-SGD for VAE Training

Implements ε-differential privacy using the Gaussian Mechanism with per-sample
gradient clipping and noise addition during VAE training.

Based on: Abadi et al. (2016) "Deep Learning with Differential Privacy"
          https://arxiv.org/abs/1607.00133

Key Concepts:
- Per-sample gradient clipping bounds sensitivity
- Gaussian noise calibrated to (ε, δ)-DP guarantee
- Moments accountant tracks cumulative privacy loss
- Privacy budget: ε = 1.0, δ = 10⁻⁵ (strong privacy)

Author: Chatchai Tritham
Date: 2026-01-25
"""

import numpy as np
from typing import List, Tuple, Dict, Optional
import logging

logger = logging.getLogger(__name__)


class DifferentialPrivacy:
    """
    ε-differential privacy via Gaussian mechanism for VAE training.

    Implements DP-SGD (Differentially Private Stochastic Gradient Descent):
    1. Per-sample gradient clipping (L2 norm bound C)
    2. Gaussian noise addition σ ~ N(0, (σ·C)²)
    3. Privacy accounting via moments accountant

    Privacy Guarantee:
    For any two neighboring datasets D and D' differing in one sample,
    and for all sets S of outputs:

    P[M(D) ∈ S] ≤ exp(ε) · P[M(D') ∈ S] + δ

    where M is the DP-SGD mechanism, ε is privacy budget, δ is failure probability.

    Example:
        >>> dp = DifferentialPrivacy(epsilon=1.0, delta=1e-5, clip_norm=1.0)
        >>> private_grad = dp.privatize_gradients(per_sample_grads, batch_size=64)
        >>> eps_spent, _ = dp.get_privacy_spent(steps=1000, batch_size=64, n_samples=8400)
    """

    def __init__(self,
                 epsilon: float = 1.0,
                 delta: float = 1e-5,
                 clip_norm: float = 1.0,
                 noise_multiplier: Optional[float] = None):
        """
        Initialize differential privacy mechanism.

        Args:
            epsilon: Privacy budget (smaller = more private, typically 0.1-10)
            delta: Probability of privacy breach (typically 10⁻⁵ to 10⁻⁶)
            clip_norm: L2 norm bound C for gradient clipping (sensitivity bound)
            noise_multiplier: Noise scale σ/C (auto-calibrated if None)

        Notes:
            - ε = 1.0 is considered "strong privacy" in practice
            - δ should be << 1/n where n is dataset size
            - clip_norm typically 1.0-5.0 depending on model architecture
        """
        self.epsilon = epsilon
        self.delta = delta
        self.clip_norm = clip_norm

        # Calibrate noise to achieve (ε, δ)-DP
        if noise_multiplier is None:
            self.noise_multiplier = self._compute_noise_multiplier()
        else:
            self.noise_multiplier = noise_multiplier

        # Privacy accounting
        self.privacy_spent = {'epsilon': 0.0, 'delta': 0.0}
        self.total_steps = 0

        logger.info(
            f"Initialized DP with ε={epsilon:.2f}, δ={delta:.2e}, "
            f"C={clip_norm:.2f}, σ={self.noise_multiplier:.3f}"
        )

    def _compute_noise_multiplier(self) -> float:
        """
        Calibrate Gaussian noise to achieve (ε, δ)-DP.

        Uses the analytic Gaussian mechanism formula:
        σ ≥ C · √(2 ln(1.25/δ)) / ε

        where:
        - σ is noise standard deviation
        - C is gradient clipping norm
        - ε is privacy budget
        - δ is failure probability

        Returns:
            Noise multiplier σ/C
        """
        # Analytic Gaussian mechanism (conservative bound)
        sigma = self.clip_norm * \
            np.sqrt(2 * np.log(1.25 / self.delta)) / self.epsilon
        noise_multiplier = sigma / self.clip_norm

        logger.info(
            f"Calibrated noise: σ = {
                sigma:.3f}, multiplier = {
                noise_multiplier:.3f}")

        return noise_multiplier

    def clip_gradients(
            self, gradients: List[np.ndarray]) -> Tuple[List[np.ndarray], Dict]:
        """
        Clip per-sample gradients to bound sensitivity.

        For each sample's gradient g, clips to:
        g_clipped = g · min(1, C / ||g||₂)

        where C is the clipping norm bound.

        Args:
            gradients: List of gradient arrays (one per sample in batch)

        Returns:
            clipped_gradients: List of clipped gradient arrays
            stats: Dictionary with clipping statistics
                - avg_norm: Average gradient L2 norm before clipping
                - max_norm: Maximum gradient L2 norm before clipping
                - clip_fraction: Fraction of gradients that were clipped
                - norm_reduction: Average norm reduction (0-1)

        Example:
            >>> per_sample_grads = [np.random.randn(100) for _ in range(64)]
            >>> clipped, stats = dp.clip_gradients(per_sample_grads)
            >>> print(f"Clipped {stats['clip_fraction']:.1%} of gradients")
        """
        clipped = []
        norms = []

        for grad in gradients:
            # Compute L2 norm
            grad_norm = np.linalg.norm(grad)
            norms.append(grad_norm)

            # Clip if exceeds bound
            if grad_norm > self.clip_norm:
                clipped_grad = grad * (self.clip_norm / grad_norm)
            else:
                clipped_grad = grad.copy()

            clipped.append(clipped_grad)

        # Compute statistics
        norms = np.array(norms)
        avg_norm = np.mean(norms)
        max_norm = np.max(norms)
        clip_fraction = np.mean(norms > self.clip_norm)

        # Average norm reduction
        clipped_norms = np.array([np.linalg.norm(g) for g in clipped])
        norm_reduction = 1.0 - np.mean(clipped_norms / (norms + 1e-10))

        stats = {
            'avg_norm': float(avg_norm),
            'max_norm': float(max_norm),
            'clip_fraction': float(clip_fraction),
            'norm_reduction': float(norm_reduction)
        }

        # Warn if too much clipping
        if clip_fraction > 0.5:
            logger.warning(
                f"Clipping {clip_fraction:.1%} of gradients - "
                f"consider increasing clip_norm (current: {self.clip_norm})"
            )

        return clipped, stats

    def add_noise(
            self,
            aggregated_gradient: np.ndarray,
            batch_size: int) -> np.ndarray:
        """
        Add calibrated Gaussian noise to aggregated gradient.

        Noise scale: σ = noise_multiplier · clip_norm
        Noise: N(0, σ²I) where I is identity matrix

        Args:
            aggregated_gradient: Sum or mean of clipped gradients
            batch_size: Number of samples in batch

        Returns:
            noisy_gradient: Gradient with privacy-preserving noise added

        Notes:
            - Noise is sampled independently for each parameter
            - Noise scale does NOT depend on batch size (already calibrated)
            - Larger noise_multiplier = more privacy but lower utility
        """
        # Noise scale: σ = noise_multiplier * clip_norm
        noise_scale = self.noise_multiplier * self.clip_norm

        # Sample Gaussian noise with same shape as gradient
        noise = np.random.normal(
            0, noise_scale, size=aggregated_gradient.shape)

        # Add noise to gradient
        noisy_gradient = aggregated_gradient + noise

        return noisy_gradient

    def privatize_gradients(self,
                            gradients: List[np.ndarray],
                            batch_size: int) -> Tuple[np.ndarray, Dict]:
        """
        Full DP-SGD step: clip + aggregate + add noise.

        This is the main method to use during training:
        1. Clip each per-sample gradient to bound sensitivity
        2. Aggregate (average) the clipped gradients
        3. Add calibrated Gaussian noise

        Args:
            gradients: List of per-sample gradients (one per sample in batch)
            batch_size: Batch size (should equal len(gradients))

        Returns:
            private_gradient: Differentially private gradient for model update
            stats: Clipping statistics from clip_gradients()

        Example:
            >>> # During VAE training
            >>> per_sample_grads = compute_per_sample_gradients(batch)
            >>> private_grad, stats = dp.privatize_gradients(per_sample_grads, batch_size=64)
            >>> optimizer.apply_gradients(private_grad)
        """
        if len(gradients) != batch_size:
            logger.warning(
                f"Gradient count ({len(gradients)}) != batch_size ({batch_size}). "
                f"Using actual count."
            )
            batch_size = len(gradients)

        # Step 1: Clip per-sample gradients
        clipped_grads, stats = self.clip_gradients(gradients)

        # Step 2: Aggregate (average)
        aggregated = np.mean(clipped_grads, axis=0)

        # Step 3: Add noise
        private_grad = self.add_noise(aggregated, batch_size)

        return private_grad, stats

    def get_privacy_spent(self,
                          steps: int,
                          batch_size: int,
                          n_samples: int) -> Tuple[float, float]:
        """
        Compute cumulative privacy loss via moments accountant.

        Uses simplified composition (conservative upper bound):
        ε_total ≈ q · √(2T ln(1/δ)) / σ

        where:
        - q = batch_size / n_samples (sampling rate)
        - T = steps (number of training iterations)
        - σ = noise_multiplier (noise scale)

        For exact accounting, use opacus library's RDP accountant.

        Args:
            steps: Number of training steps completed
            batch_size: Mini-batch size
            n_samples: Total dataset size

        Returns:
            epsilon_spent: Cumulative privacy budget spent
            delta_spent: Cumulative failure probability (unchanged)

        Example:
            >>> # After 1000 training steps
            >>> eps_spent, delta = dp.get_privacy_spent(
            ...     steps=1000, batch_size=64, n_samples=8400
            ... )
            >>> print(f"Privacy spent: ε={eps_spent:.2f}/{dp.epsilon}")
        """
        q = batch_size / n_samples  # Sampling rate
        T = steps  # Number of steps

        # Moments accountant formula (simplified)
        # This is a conservative upper bound; exact RDP gives tighter bounds
        epsilon_spent = (q * np.sqrt(2 * T * np.log(1 / self.delta)) /
                         self.noise_multiplier)

        # Update tracking
        self.privacy_spent = {
            'epsilon': epsilon_spent,
            'delta': self.delta,
            'steps': steps,
            'sampling_rate': q
        }
        self.total_steps = steps

        # Warn if budget exceeded
        if epsilon_spent > self.epsilon:
            logger.warning(
                f"Privacy budget EXCEEDED: ε={
                    epsilon_spent:.2f} > {
                    self.epsilon:.2f}. " f"Consider stopping training or increasing ε.")
        else:
            logger.info(
                f"Privacy spent: ε={epsilon_spent:.2f}/{self.epsilon:.2f}, "
                f"δ={self.delta:.2e} after {steps} steps"
            )

        return epsilon_spent, self.delta

    def check_privacy_budget(
            self,
            steps: int,
            batch_size: int,
            n_samples: int) -> bool:
        """
        Check if privacy budget will be exceeded.

        Args:
            steps: Number of training steps
            batch_size: Batch size
            n_samples: Dataset size

        Returns:
            True if within budget, False if exceeded
        """
        eps_spent, _ = self.get_privacy_spent(steps, batch_size, n_samples)
        return eps_spent <= self.epsilon

    def get_max_steps(self, batch_size: int, n_samples: int) -> int:
        """
        Compute maximum training steps before privacy budget is exhausted.

        Solves for T in: ε = q · √(2T ln(1/δ)) / σ

        Args:
            batch_size: Batch size
            n_samples: Dataset size

        Returns:
            max_steps: Maximum number of training steps within budget
        """
        q = batch_size / n_samples
        sigma = self.noise_multiplier

        # Solve for T: ε = q · √(2T ln(1/δ)) / σ
        # T = (ε · σ / q)² / (2 ln(1/δ))
        max_steps = int(
            (self.epsilon * sigma / q) ** 2 / (2 * np.log(1 / self.delta))
        )

        logger.info(
            f"Maximum training steps within ε={self.epsilon}: {max_steps} "
            f"(batch_size={batch_size}, n_samples={n_samples})"
        )

        return max_steps

    def summary(self) -> Dict:
        """
        Get summary of differential privacy configuration and current state.

        Returns:
            Dictionary with DP parameters and privacy accounting
        """
        return {
            'privacy_parameters': {
                'epsilon': self.epsilon,
                'delta': self.delta,
                'clip_norm': self.clip_norm,
                'noise_multiplier': self.noise_multiplier,
                'noise_stddev': self.noise_multiplier *
                self.clip_norm},
            'privacy_accounting': {
                'epsilon_spent': self.privacy_spent.get(
                    'epsilon',
                    0.0),
                'delta_spent': self.privacy_spent.get(
                    'delta',
                    0.0),
                'total_steps': self.total_steps,
                'budget_remaining': max(
                    0,
                    self.epsilon -
                    self.privacy_spent.get(
                        'epsilon',
                        0.0)),
                'within_budget': self.privacy_spent.get(
                    'epsilon',
                    0.0) <= self.epsilon}}

    def __repr__(self) -> str:
        """String representation"""
        return (
            f"DifferentialPrivacy(ε={
                self.epsilon:.2f}, δ={
                self.delta:.2e}, " f"C={
                self.clip_norm:.2f}, σ={
                    self.noise_multiplier:.3f})")


# Main demonstration
if __name__ == '__main__':
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    logger.info("DifferentialPrivacy - Demo Mode")

    # Initialize DP mechanism with strong privacy (ε=1.0)
    dp = DifferentialPrivacy(epsilon=1.0, delta=1e-5, clip_norm=1.0)

    logger.info(f"\n{dp}")
    logger.info(f"Summary: {dp.summary()}")

    # Simulate per-sample gradients
    logger.info("\n" + "=" * 80)
    logger.info("Simulating DP-SGD training step...")

    batch_size = 64
    grad_dim = 100

    # Generate random per-sample gradients (some with large norms)
    np.random.seed(42)
    per_sample_grads = []
    for i in range(batch_size):
        grad = np.random.randn(grad_dim)
        # Make some gradients large to trigger clipping
        if i % 10 == 0:
            grad *= 5.0
        per_sample_grads.append(grad)

    logger.info(
        f"Generated {batch_size} per-sample gradients (dim={grad_dim})")

    # Apply DP-SGD
    private_grad, stats = dp.privatize_gradients(per_sample_grads, batch_size)

    logger.info("\nClipping Statistics:")
    logger.info(f"  Average norm (before): {stats['avg_norm']:.3f}")
    logger.info(f"  Max norm (before): {stats['max_norm']:.3f}")
    logger.info(f"  Clipping fraction: {stats['clip_fraction']:.1%}")
    logger.info(f"  Norm reduction: {stats['norm_reduction']:.1%}")

    logger.info(f"\nPrivate gradient shape: {private_grad.shape}")
    logger.info(f"Private gradient norm: {np.linalg.norm(private_grad):.3f}")

    # Privacy accounting
    logger.info("\n" + "=" * 80)
    logger.info("Privacy Accounting Example...")

    n_samples = 8400  # SynDX archetype dataset size
    steps_per_epoch = n_samples // batch_size
    n_epochs = 100

    total_steps = steps_per_epoch * n_epochs

    logger.info(f"Dataset size: {n_samples}")
    logger.info(f"Batch size: {batch_size}")
    logger.info(f"Steps per epoch: {steps_per_epoch}")
    logger.info(f"Training for {n_epochs} epochs = {total_steps} steps")

    # Check maximum steps
    max_steps = dp.get_max_steps(batch_size, n_samples)

    # Compute privacy spent
    eps_spent, delta = dp.get_privacy_spent(total_steps, batch_size, n_samples)

    logger.info(f"\nPrivacy Budget Status:")
    logger.info(f"  ε spent: {eps_spent:.2f} / {dp.epsilon:.2f}")
    logger.info(f"  δ: {delta:.2e}")
    logger.info(f"  Within budget: {eps_spent <= dp.epsilon}")

    if eps_spent > dp.epsilon:
        recommended_epochs = int(max_steps / steps_per_epoch)
        logger.warning(
            f"\n⚠️  Budget exceeded! Reduce to {recommended_epochs} epochs "
            f"to stay within ε={dp.epsilon}"
        )
    else:
        logger.info(f"\n✓ Privacy budget satisfied for {n_epochs} epochs")

    logger.info("\n✓ Demo complete!")
