"""
NMF Latent Archetype Extractor

Takes our 8,400 guideline archetypes and compresses them down to 20 latent
patterns using Non-negative Matrix Factorization.

Rationale: Because working with 8,400 dimensions is computationally silly when
most of the variation can be captured in 20 latent components.

Reference: Equations (3-4) in the paper
"""

import numpy as np
from sklearn.decomposition import NMF
from typing import Tuple, Dict
import logging

logger = logging.getLogger(__name__)


class NMFExtractor:
    """
    Extract latent archetypes using NMF decomposition.

    Decomposes archetype matrix V ≈ WH where:
    - V ∈ R^(8400 × 150): archetype-feature matrix
    - W ∈ R^(8400 × r): archetype-to-latent weights
    - H ∈ R^(r × 150): latent-pattern-to-feature basis
    - r = 20: number of latent components
    """

    def __init__(self, n_components: int = 20, random_state: int = 42):
        """
        Initialize NMF extractor.

        Args:
            n_components: Number of latent archetypes (r)
            random_state: Random seed for reproducibility
        """
        self.n_components = n_components
        self.random_state = random_state
        self.model = None
        self.W = None  # Archetype-to-latent weights
        self.H = None  # Latent-pattern-to-feature basis
        self.reconstruction_error_ = None  # Frobenius reconstruction error

    def fit(self, archetype_matrix: np.ndarray) -> "NMFExtractor":
        """
        Fit NMF model to archetype matrix.

        Solves: min_{W≥0, H≥0} ||V - WH||_F^2 + λ(||W||_F^2 + ||H||_F^2)

        Args:
        archetype_matrix: V ∈ R^(n_archetypes × n_features)

        Returns:
        Self for chaining
        """
        logger.info(f"Fitting NMF with {self.n_components} components...")
        logger.info(f"Input shape: {archetype_matrix.shape}")

        # Initialize NMF with Frobenius norm
        self.model = NMF(
            n_components=self.n_components,
            init='nndsvd',  # Non-negative SVD initialization
            solver='mu',  # Multiplicative update
            beta_loss='frobenius',
            max_iter=1000,
            random_state=self.random_state,
            alpha_W=0.01,  # Regularization λ
            alpha_H=0.01,
            l1_ratio=0.0,  # L2 regularization only
            verbose=1
        )

        # Fit and transform
        self.W = self.model.fit_transform(archetype_matrix)
        self.H = self.model.components_

        # Log reconstruction error
        reconstruction = self.W @ self.H
        frobenius_error = np.linalg.norm(
            archetype_matrix - reconstruction, 'fro')
        relative_error = frobenius_error / \
            np.linalg.norm(archetype_matrix, 'fro')

        # Store reconstruction error
        self.reconstruction_error_ = relative_error

        logger.info(f"Frobenius reconstruction error: {frobenius_error:.4f}")
        logger.info(f"Relative error: {relative_error:.4f}")
        logger.info(f"W shape: {self.W.shape}, H shape: {self.H.shape}")

        return self

    def transform(self, archetype_matrix: np.ndarray) -> np.ndarray:
        """
        Transform archetypes to latent representation.

        Args:
        archetype_matrix: V ∈ R^(n × 150)

        Returns:
        W ∈ R^(n × r): latent weights
        """
        if self.model is None:
            raise ValueError("Model not fitted. Call fit() first.")

        return self.model.transform(archetype_matrix)

    def fit_transform(
            self, archetype_matrix: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Fit NMF model and transform in one step.

        Convenience method that combines fit() and returns both W and H matrices.

        Args:
        archetype_matrix: V ∈ R^(n_archetypes × n_features)

        Returns:
        Tuple of (W, H):
        - W ∈ R^(n_archetypes × r): Archetype-to-latent weights
        - H ∈ R^(r × n_features): Latent-pattern-to-feature basis

        Example:
        >>> nmf = NMFExtractor(n_components=20)
        >>> W, H = nmf.fit_transform(archetype_matrix)
        >>> print(f"W shape: {W.shape}, H shape: {H.shape}")
        """
        self.fit(archetype_matrix)
        return self.W, self.H

    def inverse_transform(self, latent_weights: np.ndarray) -> np.ndarray:
        """
        Reconstruct archetypes from latent weights.

        Args:
        latent_weights: W ∈ R^(n × r)

        Returns:
        Reconstructed archetypes V_hat ∈ R^(n × 150)
        """
        if self.H is None:
            raise ValueError("Model not fitted. Call fit() first.")

        return latent_weights @ self.H

    def get_latent_archetypes(self) -> np.ndarray:
        """
        Get the r=20 latent archetype basis vectors.

        Returns:
        H ∈ R^(r × 150): latent archetype basis
        """
        return self.H

    def interpret_components(self, feature_names: list = None) -> Dict:
        """
        Interpret latent components by top features.

        Args:
        feature_names: List of feature names (length 150)

        Returns:
        Dictionary mapping component index to top features
        """
        if self.H is None:
            raise ValueError("Model not fitted. Call fit() first.")

        interpretation = {}

        for i in range(self.n_components):
            component_weights = self.H[i, :]
            top_indices = np.argsort(component_weights)[-10:][::-1]

            if feature_names:
                top_features = [(feature_names[idx], component_weights[idx])
                                for idx in top_indices]
            else:
                top_features = [(f"feature_{idx}", component_weights[idx])
                                for idx in top_indices]

            interpretation[f"component_{i}"] = top_features

        return interpretation

    def get_statistics(self) -> Dict:
        """Get NMF statistics"""
        if self.W is None or self.H is None:
            return {}

        return {
            "n_components": self.n_components,
            "n_archetypes": self.W.shape[0],
            "n_features": self.H.shape[1],
            "reconstruction_error": self.model.reconstruction_err_,
            "n_iterations": self.model.n_iter_,
            "W_sparsity": np.mean(self.W == 0),
            "H_sparsity": np.mean(self.H == 0),
        }


if __name__ == "__main__":
    # Test NMF extractor
    logging.basicConfig(level=logging.INFO)

    # Generate test data (8400 × 150)
    np.random.seed(42)
    n_archetypes = 8400
    n_features = 150
    archetype_matrix = np.abs(np.random.randn(n_archetypes, n_features))

    # Fit NMF
    extractor = NMFExtractor(n_components=20, random_state=42)
    extractor.fit(archetype_matrix)

    # Get statistics
    stats = extractor.get_statistics()
    print("\nNMF Statistics:")
    for key, val in stats.items():
        print(f" {key}: {val}")

    # Test transform
    latent_repr = extractor.transform(archetype_matrix[:10])
    print(f"\nLatent representation shape: {latent_repr.shape}")

    # Test inverse transform
    reconstructed = extractor.inverse_transform(latent_repr)
    print(f"Reconstructed shape: {reconstructed.shape}")

    print("\nNMF test completed!")
