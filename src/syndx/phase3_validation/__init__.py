"""
Phase 3: Multi-Level Validation

Validates the quality and clinical plausibility of generated synthetic data.
Statistical metrics, diagnostic performance, XAI fidelity - the whole shebang.
Comprehensive validation ensures data utility for downstream applications."
"""

from .evaluation_metrics import EvaluationMetrics
from .statistical_metrics import StatisticalMetrics
from .triate_classifier import TriateClassifier

try:
    from .diagnostic_evaluator import DiagnosticEvaluator
except Exception:  # pragma: no cover - optional during lightweight installs
    DiagnosticEvaluator = None

try:
    from .xai_fidelity import XAIFidelity
except Exception:  # pragma: no cover - optional during lightweight installs
    XAIFidelity = None

__all__ = [
    "StatisticalMetrics",
    "TriateClassifier",
    "EvaluationMetrics",
    "DiagnosticEvaluator",
    "XAIFidelity",
]
