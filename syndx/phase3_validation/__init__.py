# Rewritten 2026-01-01 for human authenticity
"""
Phase 3: Multi-Level Validation

Where we prove the synthetic data is actually good.
Statistical metrics, diagnostic performance, XAI fidelity - the whole shebang.
This is what separates "we generated some data" from "we generated useful data."
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
