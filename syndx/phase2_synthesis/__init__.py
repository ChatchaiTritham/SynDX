"""
Phase 2: XAI-Driven Synthesis Pipeline

Modules for iterative synthesis using NMF, VAE, SHAP, and counterfactuals.
"""

from .nmf_extractor import NMFExtractor
from .vae_model import VAEModel, train_vae, sample_from_vae, evaluate_vae_reconstruction
from .xai_driver import XAIDriver
from .probabilistic_logic import ProbabilisticLogic

# Placeholder modules (to be implemented)
# from .shap_reweighter import SHAPReweighter
# from .counterfactual_validator import CounterfactualValidator
# from .differential_privacy import DifferentialPrivacy

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
