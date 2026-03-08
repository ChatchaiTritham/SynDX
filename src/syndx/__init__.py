"""
SynDX: Synthetic Data Generation for Clinical Decision Support

A comprehensive framework for generating privacy-preserving synthetic medical data
using XAI-driven synthesis for vestibular disorder archetypes.

Three-Phase Architecture:
- Phase 1: Clinical Knowledge Extraction (TiTrATE, archetypes)
- Phase 2: XAI-Driven Synthesis (NMF, VAE, SHAP)
- Phase 3: Multi-Level Validation (statistical, diagnostic, XAI fidelity)

Author: Chatchai Tritham
Institution: Naresuan University
"""

from syndx.constants import FRAMEWORK_NAME, PACKAGE_VERSION

__version__ = PACKAGE_VERSION
__author__ = "Chatchai Tritham"
__email__ = "chatchait66@nu.ac.th"

from syndx.phase1_knowledge import ArchetypeGenerator, TiTrATEFormalizer
from syndx.phase2_synthesis import NMFExtractor
from syndx.phase3_validation import StatisticalMetrics, TriateClassifier
from syndx.pipeline import SynDXPipeline

__all__ = [
    "__version__",
    "__author__",
    "__email__",
    "SynDXPipeline",
    "TiTrATEFormalizer",
    "ArchetypeGenerator",
    "NMFExtractor",
    "StatisticalMetrics",
    "TriateClassifier",
    "FRAMEWORK_NAME",
]
