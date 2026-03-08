"""Simple probabilistic logic utilities for SynDX."""

from __future__ import annotations

from typing import Iterable

import numpy as np


class ProbabilisticLogic:
    """Small helper for combining probabilities conservatively."""

    def combine(self, probabilities: Iterable[float]) -> float:
        values = np.clip(np.asarray(list(probabilities), dtype=float), 0.0, 1.0)
        if values.size == 0:
            return 0.0
        return float(values.mean())
