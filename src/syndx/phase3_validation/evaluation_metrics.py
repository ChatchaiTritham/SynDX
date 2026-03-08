"""Compatibility metrics wrapper for phase 3 validation."""

from __future__ import annotations

from typing import Dict

import numpy as np

from syndx.phase3_validation.statistical_metrics import StatisticalMetrics


class EvaluationMetrics:
    """Aggregate a small summary of statistical validation metrics."""

    def compute(self, archetype_data: np.ndarray, synthetic_data: np.ndarray) -> Dict[str, float]:
        results = StatisticalMetrics.compute_all_metrics(archetype_data, synthetic_data)
        return {
            "mean_kl": float(results["summary"]["mean_kl"]),
            "mean_js": float(results["summary"]["mean_js"]),
            "mean_wasserstein": float(results["summary"]["mean_wasserstein"]),
        }
