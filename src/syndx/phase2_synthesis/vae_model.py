"""Small VAE placeholders used for package stabilization."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict

import numpy as np


@dataclass
class VAEModel:
    """Minimal VAE-like container with deterministic sampling."""

    latent_dim: int = 8
    random_state: int = 42

    def fit(self, data: np.ndarray) -> "VAEModel":
        self.feature_count_ = data.shape[1] if data.ndim > 1 else 1
        return self

    def sample(self, n_samples: int) -> np.ndarray:
        rng = np.random.default_rng(self.random_state)
        feature_count = getattr(self, "feature_count_", self.latent_dim)
        return rng.normal(size=(n_samples, feature_count))


def train_vae(data: np.ndarray, latent_dim: int = 8, random_state: int = 42) -> VAEModel:
    """Train and return a minimal VAE model."""
    return VAEModel(latent_dim=latent_dim, random_state=random_state).fit(data)


def sample_from_vae(model: VAEModel, n_samples: int) -> np.ndarray:
    """Sample synthetic vectors from a trained placeholder VAE."""
    return model.sample(n_samples)


def evaluate_vae_reconstruction(model: VAEModel, data: np.ndarray) -> Dict[str, Any]:
    """Return a trivial reconstruction summary for compatibility."""
    return {"reconstruction_error": 0.0, "n_samples": int(len(data))}
