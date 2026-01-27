"""
Phase 2: XAI-Driven Synthesis Pipeline

Modules for iterative synthesis using NMF, VAE, SHAP, and counterfactuals.
"""

from .nmf_extractor import NMFExtractor
from .vae_model import VAEModel, train_vae, sample_from_vae, evaluate_vae_reconstruction
from .xai_driver import XAIDriver
from .probabilistic_logic import ProbabilisticLogic

# Additional modules available in repository:
# - shap_reweighter: SHAP-based feature reweighting
# - counterfactual_validator: Clinical plausibility validation
# - differential_privacy: Privacy-preserving mechanisms

__all__ = [
    "NMFExtractor",
    "VAEModel",
    "train_vae",
    "sample_from_vae",
    "evaluate_vae_reconstruction",
    "XAIDriver",
    "ProbabilisticLogic",
    # "SHAPReweighter",
    # "CounterfactualValidator",
    # "DifferentialPrivacy",
]
