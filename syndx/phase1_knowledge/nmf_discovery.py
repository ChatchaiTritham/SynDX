"""
NMF Latent Factor Discovery
Implements Formula 3.1-3.2 from manuscript
"""

import numpy as np
from sklearn.decomposition import NMF
from typing import List, Dict, Tuple, Optional
import logging

logger = logging.getLogger(__name__)


class NMFFactorDiscovery:
    """
    Non-negative Matrix Factorization for clinical pattern discovery

    Implements:
    - Eq. 16: r_clinical = ⌈log₂(|D|) + √(m/10)⌉
    - Eq. 17: X ≈ WH
    - Eq. 18: Optimization with L1 regularization
    """

    def __init__(
        self,
        n_components: int = None,
        n_diagnoses: int = None,
        n_parameters: int = None,
        alpha_W: float = 0.1,
        alpha_H: float = 0.1,
        l1_ratio: float = 0.5,
        max_iter: int = 500,
        random_state: int = 42
    ):
        """
        Args:
            n_components: Number of NMF factors (r). If None, calculated from Eq. 16
            n_diagnoses: |D| for calculating r_clinical
            n_parameters: m for calculating r_clinical
            alpha_W: L1 regularization for W (Eq. 18)
            alpha_H: L1 regularization for H (Eq. 18)
            l1_ratio: Balance between L1 and L2 (1.0 = pure L1)
            max_iter: Maximum iterations
            random_state: Random seed
        """
        # Calculate optimal r if not provided (Eq. 16)
        if n_components is None:
            if n_diagnoses is None or n_parameters is None:
                raise ValueError(
                    "Must provide either n_components or (n_diagnoses, n_parameters)")
            n_components = self._calculate_r_clinical(
                n_diagnoses, n_parameters)

        self.n_components = n_components
        self.alpha_W = alpha_W
        self.alpha_H = alpha_H

        # Initialize NMF model (Eq. 18)
        self.model = NMF(
            n_components=n_components,
            init='nndsvd',  # Non-negative double SVD initialization
            solver='cd',  # Coordinate Descent (supports L1)
            beta_loss='frobenius',  # ||X - WH||²_F
            alpha_W=alpha_W,
            alpha_H=alpha_H,
            l1_ratio=l1_ratio,
            max_iter=max_iter,
            random_state=random_state,
            verbose=0
        )

        self.W_ = None  # Archetype-to-factor loadings (n × r)
        self.H_ = None  # Factor-to-feature weights (r × d)
        self.reconstruction_error_ = None
        self.feature_names_ = None
        self.factor_interpretations_ = None

    @staticmethod
    def _calculate_r_clinical(n_diagnoses: int, n_parameters: int) -> int:
        """
        Clinical heuristic for optimal NMF factors (Eq. 16)
        r_clinical = ⌈log₂(|D|) + √(m/10)⌉
        """
        r = int(np.ceil(
            np.log2(n_diagnoses) + np.sqrt(n_parameters / 10)
        ))
        logger.info(
            f"Calculated r_clinical = {r} (|D|={n_diagnoses}, m={n_parameters})")
        return r

    def fit(self, X: np.ndarray, feature_names: List[str] = None):
        """
        Fit NMF model (Eq. 17-18)

        Args:
            X: Archetype feature matrix (n × d)
            feature_names: Names of d features
        """
        logger.info(f"Fitting NMF with r={self.n_components} components")
        logger.info(f"Input matrix shape: {X.shape}")

        # Ensure non-negative
        X = np.maximum(X, 0)

        # Fit model
        self.W_ = self.model.fit_transform(X)
        self.H_ = self.model.components_

        # Store reconstruction error
        self.reconstruction_error_ = self.model.reconstruction_err_

        # Store feature names
        self.feature_names_ = feature_names

        logger.info(f"NMF fitting complete")
        logger.info(f"  W shape: {self.W_.shape} (archetypes × factors)")
        logger.info(f"  H shape: {self.H_.shape} (factors × features)")
        logger.info(
            f"  Reconstruction error: {
                self.reconstruction_error_:.4f}")

        # Interpret factors
        self.factor_interpretations_ = self._interpret_factors()

        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        """
        Transform new data to factor space

        Args:
            X: New archetype features (n × d)

        Returns:
            W: Factor loadings (n × r)
        """
        X = np.maximum(X, 0)
        return self.model.transform(X)

    def _interpret_factors(self, top_k: int = 10) -> List[Dict]:
        """
        Interpret each NMF factor by top features

        Returns:
            List of factor interpretations
        """
        interpretations = []

        for factor_idx in range(self.n_components):
            # Get factor weights
            factor_weights = self.H_[factor_idx]

            # Top k features
            top_indices = np.argsort(factor_weights)[-top_k:][::-1]
            top_weights = factor_weights[top_indices]

            if self.feature_names_:
                top_features = [self.feature_names_[i] for i in top_indices]
            else:
                top_features = [f"feature_{i}" for i in top_indices]

            interpretation = {
                'factor_id': factor_idx,
                'top_features': list(zip(top_features, top_weights)),
                'clinical_pattern': self._infer_clinical_pattern(top_features, top_weights)
            }

            interpretations.append(interpretation)

            logger.info(
                f"Factor {factor_idx}: {
                    interpretation['clinical_pattern']}")
            logger.debug(f"  Top features: {top_features[:5]}")

        return interpretations

    def _infer_clinical_pattern(
        self,
        features: List[str],
        weights: np.ndarray
    ) -> str:
        """
        Infer clinical pattern from top features

        Simple heuristic-based naming
        """
        # Convert to lowercase for matching
        features_lower = [f.lower() for f in features[:5]]

        # Pattern matching
        if 'stroke' in str(features_lower) or 'central' in str(features_lower):
            return "Stroke risk pattern"
        elif 'bppv' in str(features_lower) or 'positional' in str(features_lower):
            return "BPPV characteristic pattern"
        elif 'migraine' in str(features_lower):
            return "Vestibular migraine pattern"
        elif 'peripheral' in str(features_lower):
            return "Peripheral vestibular pattern"
        elif 'age' in str(features_lower) and weights[0] > 0.5:
            return "Age-related pattern"
        else:
            return f"Clinical pattern {features[0]}"

    def get_factor_loadings(self, archetype_idx: int) -> np.ndarray:
        """
        Get factor loadings for specific archetype

        Args:
            archetype_idx: Index of archetype

        Returns:
            Factor loadings (r,)
        """
        return self.W_[archetype_idx]

    def get_dominant_factor(self, archetype_idx: int) -> Tuple[int, float]:
        """
        Get dominant factor for archetype

        Returns:
            (factor_id, loading)
        """
        loadings = self.W_[archetype_idx]
        dominant_idx = np.argmax(loadings)
        return dominant_idx, loadings[dominant_idx]

    def reconstruct(self, X: np.ndarray) -> np.ndarray:
        """
        Reconstruct X from W and H

        Returns:
            X_reconstructed ≈ WH
        """
        W = self.transform(X)
        return W @ self.H_

    def get_reconstruction_error(self, X: np.ndarray) -> float:
        """
        Calculate reconstruction error ||X - WH||²_F
        """
        X_recon = self.reconstruct(X)
        return np.linalg.norm(X - X_recon, 'fro') ** 2

    def elbow_analysis(
        self,
        X: np.ndarray,
        r_range: range = range(5, 31, 5)
    ) -> Dict[int, float]:
        """
        Elbow method for optimal r (Eq. 15)

        Args:
            X: Feature matrix
            r_range: Range of r values to test

        Returns:
            {r: reconstruction_error}
        """
        errors = {}

        for r in r_range:
            model = NMF(
                n_components=r,
                init='nndsvd',
                max_iter=500,
                random_state=42
            )
            model.fit(X)
            errors[r] = model.reconstruction_err_
            logger.info(f"r={r}: error={errors[r]:.4f}")

        return errors

    def get_summary(self) -> Dict:
        """Get summary statistics"""
        return {
            'n_components': self.n_components,
            'reconstruction_error': self.reconstruction_error_,
            'W_shape': self.W_.shape if self.W_ is not None else None,
            'H_shape': self.H_.shape if self.H_ is not None else None,
            'factor_interpretations': self.factor_interpretations_
        }
