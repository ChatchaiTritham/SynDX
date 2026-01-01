"""
Diagnostic Performance Evaluator - placeholder for now

This should evaluate how well models trained on synthetic data perform
diagnostic tasks compared to the archetype baseline.

Implementation concept: train a diagnostic classifier on synthetic data, test it on
archetypes, see if the performance is acceptable. If not, the synthetic
data isn't good enough.

Not implemented yet - on the roadmap.
"""

import logging

logger = logging.getLogger(__name__)


class DiagnosticEvaluator:
    """
    Diagnostic performance evaluation (not implemented)

    Will compare diagnostic model performance when trained on synthetic
    vs archetype data.
    """
    def __init__(self):
        logger.warning("Using diagnostic evaluator placeholder - full implementation pending")
