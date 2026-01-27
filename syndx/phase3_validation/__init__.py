"""
Phase 3: Multi-Level Validation

Validates the quality and clinical plausibility of generated synthetic data.
Statistical metrics, diagnostic performance, XAI fidelity - the whole shebang.
Comprehensive validation ensures data utility for downstream applications."
"""

from .statistical_metrics import StatisticalMetrics
from .triate_classifier import TriateClassifier
from .evaluation_metrics import EvaluationMetrics

# Additional modules available in repository:
# - diagnostic_evaluator: Clinical diagnostic performance evaluation
# - xai_fidelity: XAI explanation consistency validation

__all__ = [
    "StatisticalMetrics",
    "TriateClassifier",
    "EvaluationMetrics",
    # "DiagnosticEvaluator",
    # "XAIFidelity",
]
