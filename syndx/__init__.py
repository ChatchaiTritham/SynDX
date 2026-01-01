"""
SynDX: Explainable AI-Driven Synthetic Data Generation Framework

A research framework for generating privacy-preserving synthetic medical data
for vestibular disorder differential diagnosis without requiring real patient records.

This implementation integrates clinical guidelines (TiTrATE framework, Bárány Society
ICVD 2025) with explainable AI techniques including SHAP-guided feature importance,
non-negative matrix factorization, and counterfactual validation.

IMPORTANT: This is preliminary research without clinical validation. Do NOT use
for actual clinical decision-making. Clinical trials are required before deployment.

Authors:
    Chatchai Tritham (PhD Student)
    Assoc. Prof. Dr. Chakkrit Snae Namahoot (Advisor)

Institution:
    Department of Computer Science and Information Technology
    Faculty of Science, Naresuan University
    Phitsanulok 65000, Thailand

License: MIT
"""

__version__ = "0.1.0"
__author__ = "Chatchai Tritham, Chakkrit Snae Namahoot"
__email__ = "chatchait66@nu.ac.th"

# Main pipeline orchestrator
from syndx.pipeline import SynDXPipeline

# Phase 1: Clinical Knowledge Extraction
from syndx.phase1_knowledge.titrate_formalizer import TiTrATEFormalizer
from syndx.phase1_knowledge.archetype_generator import ArchetypeGenerator
from syndx.phase1_knowledge.standards_mapper import StandardsMapper

# Phase 2: XAI-Driven Synthesis
from syndx.phase2_synthesis.nmf_extractor import NMFExtractor
from syndx.phase2_synthesis.vae_model import VAEModel, train_vae, sample_from_vae
from syndx.phase2_synthesis.xai_driver import XAIDriver
from syndx.phase2_synthesis.probabilistic_logic import ProbabilisticLogic

# Phase 3: Multi-Level Validation
from syndx.phase3_validation.statistical_metrics import StatisticalMetrics
from syndx.phase3_validation.triate_classifier import TriateClassifier
from syndx.phase3_validation.evaluation_metrics import EvaluationMetrics

# Utility modules
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

# Default configuration parameters (as specified in the research paper)
RANDOM_SEED = 42
N_ARCHETYPES = 8400
NMF_COMPONENTS = 20
VAE_LATENT_DIM = 50
EPSILON = 1.0  # Differential privacy budget
CONVERGENCE_THRESHOLD = 0.05

# Clinical validation warning
import warnings
warnings.warn(
    "\n⚠️  IMPORTANT: Preliminary research without clinical validation.\n"
    "   This framework is not approved for clinical decision-making.\n"
    "   All reported metrics are based on synthetic-to-synthetic validation.\n"
    "   Clinical trials with real patient data are required before deployment.\n",
    UserWarning,
    stacklevel=2
)
