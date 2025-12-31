"""
SynDX: Explainable AI-Driven Synthetic Data Generation
for Privacy-Preserving Differential Diagnosis of Vestibular Disorders

This is preliminary work without clinical validation.
All validation uses synthetic data only.

Authors: Chatchai Tritham, Chakkrit Snae Namahoot
Institution: Naresuan University, Thailand
License: MIT
"""

__version__ = "0.1.0"
__author__ = "Chatchai Tritham, Chakkrit Snae Namahoot"
__email__ = "chatchai.tritham@nu.ac.th"

# Import main pipeline
from syndx.pipeline import SynDXPipeline

# Import Phase 1 modules
from syndx.phase1_knowledge.titrate_formalizer import TiTrATEFormalizer
from syndx.phase1_knowledge.archetype_generator import ArchetypeGenerator
from syndx.phase1_knowledge.standards_mapper import StandardsMapper

# Import Phase 2 modules
from syndx.phase2_synthesis.nmf_extractor import NMFExtractor
from syndx.phase2_synthesis.vae_model import VAEModel, VAEEncoder, VAEDecoder
from syndx.phase2_synthesis.shap_reweighter import SHAPReweighter
from syndx.phase2_synthesis.counterfactual_validator import CounterfactualValidator
from syndx.phase2_synthesis.differential_privacy import DifferentialPrivacy

# Import Phase 3 modules
from syndx.phase3_validation.statistical_metrics import StatisticalMetrics
from syndx.phase3_validation.diagnostic_evaluator import DiagnosticEvaluator
from syndx.phase3_validation.xai_fidelity import XAIFidelity

# Import utilities
from syndx.utils.fhir_exporter import FHIRExporter
from syndx.utils.snomed_mapper import SNOMEDMapper
from syndx.utils.data_loader import DataLoader

__all__ = [
    "SynDXPipeline",
    # Phase 1
    "TiTrATEFormalizer",
    "ArchetypeGenerator",
    "StandardsMapper",
    # Phase 2
    "NMFExtractor",
    "VAEModel",
    "VAEEncoder",
    "VAEDecoder",
    "SHAPReweighter",
    "CounterfactualValidator",
    "DifferentialPrivacy",
    # Phase 3
    "StatisticalMetrics",
    "DiagnosticEvaluator",
    "XAIFidelity",
    # Utils
    "FHIRExporter",
    "SNOMEDMapper",
    "DataLoader",
]

# Configuration
RANDOM_SEED = 42
N_ARCHETYPES = 8400
NMF_COMPONENTS = 20
VAE_LATENT_DIM = 50
EPSILON = 1.0  # Differential privacy budget
CONVERGENCE_THRESHOLD = 0.05

# Warning message
import warnings
warnings.warn(
    "\n⚠️  IMPORTANT: This is preliminary work without clinical validation.\n"
    "   Do NOT use for clinical decision-making.\n"
    "   All metrics are based on synthetic-to-synthetic validation only.\n",
    UserWarning,
    stacklevel=2
)
