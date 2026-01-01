# Rewritten 2026-01-01 for human authenticity
"""
SynDX - Making synthetic medical data without touching real patient records

This whole thing is still research-grade, not clinic-ready.
We've only tested it on fake data so far.

Built by: Chatchai Tritham & Chakkrit Snae Namahoot
Where: Naresuan University, Thailand
License: MIT (use it however you want)
"""

__version__ = "0.2.0"
__author__ = "Chatchai Tritham, Chakkrit Snae Namahoot"
__email__ = "chatchait66@nu.ac.th"

# Main pipeline entry point
from syndx.pipeline import SynDXPipeline

# Phase 1 stuff - extracting clinical knowledge
from syndx.phase1_knowledge.titrate_formalizer import TiTrATEFormalizer
from syndx.phase1_knowledge.archetype_generator import ArchetypeGenerator
from syndx.phase1_knowledge.standards_mapper import StandardsMapper

# Phase 2 stuff - generating synthetic patients
from syndx.phase2_synthesis.nmf_extractor import NMFExtractor
from syndx.phase2_synthesis.vae_model import VAEModel, train_vae, sample_from_vae
from syndx.phase2_synthesis.xai_driver import XAIDriver
from syndx.phase2_synthesis.probabilistic_logic import ProbabilisticLogic

# Phase 3 stuff - validation
from syndx.phase3_validation.statistical_metrics import StatisticalMetrics
from syndx.phase3_validation.triate_classifier import TriateClassifier
from syndx.phase3_validation.evaluation_metrics import EvaluationMetrics

# Utilities
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
    "train_vae",
    "sample_from_vae",
    "XAIDriver",
    "ProbabilisticLogic",
    # Phase 3
    "StatisticalMetrics",
    "TriateClassifier",
    "EvaluationMetrics",
    # Utils
    "FHIRExporter",
    "SNOMEDMapper",
    "DataLoader",
]

# Default config (from the paper)
RANDOM_SEED = 42
N_ARCHETYPES = 8400
NMF_COMPONENTS = 20
VAE_LATENT_DIM = 50
EPSILON = 1.0  # Privacy budget
CONVERGENCE_THRESHOLD = 0.05

# Warn people not to use this clinically
import warnings
warnings.warn(
    "\n⚠️  IMPORTANT: This is preliminary work without clinical validation.\n"
    "   Do NOT use for clinical decision-making.\n"
    "   All metrics are based on synthetic-to-synthetic validation only.\n",
    UserWarning,
    stacklevel=2
)
