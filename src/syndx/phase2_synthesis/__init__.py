"""
Phase 2: XAI-Driven Synthesis Pipeline

Modules for iterative synthesis using NMF, VAE, SHAP, and counterfactuals.
"""

from .nmf_extractor import NMFExtractor
from .probabilistic_logic import ProbabilisticLogic
from .vae_model import VAEModel, evaluate_vae_reconstruction, sample_from_vae, train_vae
from .xai_driver import XAIDriver

try:
    from .counterfactual_validator import CounterfactualValidator
except Exception:  # pragma: no cover - optional during lightweight installs
    CounterfactualValidator = None

try:
    from .differential_privacy import DifferentialPrivacy
except Exception:  # pragma: no cover - optional during lightweight installs
    DifferentialPrivacy = None

try:
    from .shap_reweighter import SHAPReweighter
except Exception:  # pragma: no cover - optional during lightweight installs
    SHAPReweighter = None

__all__ = [
    "NMFExtractor",
    "VAEModel",
    "train_vae",
    "sample_from_vae",
    "evaluate_vae_reconstruction",
    "XAIDriver",
    "ProbabilisticLogic",
    "SHAPReweighter",
    "CounterfactualValidator",
    "DifferentialPrivacy",
]
