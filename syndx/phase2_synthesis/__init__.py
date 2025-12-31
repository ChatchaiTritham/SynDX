"""
Phase 2: XAI-Driven Synthesis Pipeline

Modules for iterative synthesis using NMF, VAE, SHAP, and counterfactuals.
"""

from .nmf_extractor import NMFExtractor
from .vae_model import VAEModel, VAEEncoder, VAEDecoder
from .shap_reweighter import SHAPReweighter
from .counterfactual_validator import CounterfactualValidator
from .differential_privacy import DifferentialPrivacy

__all__ = [
    "NMFExtractor",
    "VAEModel",
    "VAEEncoder",
    "VAEDecoder",
    "SHAPReweighter",
    "CounterfactualValidator",
    "DifferentialPrivacy",
]
