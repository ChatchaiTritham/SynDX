"""
Phase 3: Multi-Level Validation

Validates the quality and clinical plausibility of generated synthetic data.
Statistical metrics, diagnostic performance, XAI fidelity - the whole shebang.
Comprehensive validation ensures data utility for downstream applications."
"""

from .statistical_metrics import StatisticalMetrics
from .triate_classifier import TriateClassifier
from .evaluation_metrics import EvaluationMetrics

# Placeholder modules (to be implemented)
# from .diagnostic_evaluator import DiagnosticEvaluator
# from .xai_fidelity import XAIFidelity

__all__ = [
    "StatisticalMetrics",
    "TriateClassifier",
    "EvaluationMetrics",
    # "DiagnosticEvaluator",
    # "XAIFidelity",
]
