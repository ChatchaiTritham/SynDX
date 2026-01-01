"""
XAI Fidelity Metrics - placeholder implementation

This should measure whether SHAP explanations from models trained on
synthetic data match the explanations from archetype-trained models.

Objective: if the feature importances are different, the synthetic data
is teaching models the wrong patterns. XAI fidelity ensures the synthetic
data preserves the causal structure, not just the correlations.

Not implemented yet but important for validation.
"""

import logging

logger = logging.getLogger(__name__)


class XAIFidelity:
    """
    XAI fidelity metrics (not implemented)

    Will compare SHAP explanations between models trained on synthetic
    vs archetype data to ensure causal structure preservation.
    """
    def __init__(self):
        logger.warning("Using XAI fidelity placeholder - full implementation pending")
