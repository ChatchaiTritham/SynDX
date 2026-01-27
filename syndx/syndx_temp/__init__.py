"""
SynDX-Hybrid: Five-Layer Explainable Framework for Clinically-Grounded
Synthetic Medical Data Generation

Top-level package initialization for the complete SynDX-Hybrid framework.
"""

import warnings
__version__ = "1.0.0"
__author__ = "Chatchai Tritham, Chakkrit Snae Namahoot"
__email__ = "chatchait66@nu.ac.th"

# Import main components
from .pipeline import SynDXHybridPipeline

# Import layer components
from .layer1_combinatorial.archetype_generator import ArchetypeGenerator, ClinicalArchetype, TimingPattern, TriggerType, DiagnosisCategory
from .layer2_bayesian.bayesian_network import BayesianNetworkGenerator
from .layer3_rules.rule_engine import RuleBasedExpertSystem
from .layer4_xai.provenance_tracker import ProvenanceTracker
from .layer5_counterfactual.perturbation_engine import PerturbationEngine
from .ensemble_integration.weighted_merger import WeightedEnsembleMerger

# Import dataset generator
from .dataset_generator import SynDXDatasetGenerator

# Import visualization system
from .visualization_system import SynDXVisualizer

__all__ = [
    # Main pipeline
    'SynDXHybridPipeline',

    # Dataset generator
    'SynDXDatasetGenerator',

    # Visualization system
    'SynDXVisualizer',

    # Layer 1
    'ArchetypeGenerator', 'ClinicalArchetype', 'TimingPattern', 'TriggerType', 'DiagnosisCategory',

    # Layer 2
    'BayesianNetworkGenerator',

    # Layer 3
    'RuleBasedExpertSystem',

    # Layer 4
    'ProvenanceTracker',

    # Layer 5
    'PerturbationEngine',

    # Ensemble
    'WeightedEnsembleMerger',
]

# Default configuration parameters (as specified in the research paper)
RANDOM_SEED = 42
N_ARCHETYPES = 8400
BAYESIAN_NODES = 45
RULE_BASE_SIZE = 247
# [Comb, Bayes, Rules, XAI, CF]
ENSEMBLE_WEIGHTS = [0.25, 0.20, 0.25, 0.15, 0.15]

# Clinical validation warning
warnings.warn(
    "\\n⚠️ IMPORTANT: This is preliminary research without clinical validation.\\n"
    " This framework is not approved for clinical decision-making.\\n"
    " All reported metrics are based on synthetic-to-synthetic validation.\\n"
    " Clinical trials with real patient data are required before deployment.\\n",
    UserWarning,
    stacklevel=2)
