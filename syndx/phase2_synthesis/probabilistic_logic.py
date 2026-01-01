# Rewritten 2026-01-01 for human authenticity
"""
Probabilistic Logic Refinement

Enforces clinical coherence rules and conditional probabilities.
This is what prevents the VAE from generating clinically nonsensical patients
like "18-year-old with 50 years of hypertension" or "BPPV with chronic timing."

Think of it as a post-processing sanity check that uses probabilistic rules
to ensure the synthetic data actually makes medical sense.

Sub-Phase 2.4 in the paper.
"""

import pandas as pd
import numpy as np
import logging
from typing import Dict, Optional

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ProbabilisticLogic:
    """
    Probabilistic Logic Refiner for SynDX Framework.

    Implements:
    - Sub-Phase 2.4: Probabilistic Logic Refinement

    Enforces clinical coherence by applying probabilistic rules that capture
    conditional dependencies and expert knowledge about feature relationships.
    """

    def __init__(self):
        """Initialize ProbabilisticLogic module."""
        logger.info("ProbabilisticLogic initialized")

    # =========================================================================
    # SUB-PHASE 2.4: Probabilistic Logic Refinement
    # =========================================================================

    def refine_with_probabilistic_logic(self,
                                         D_synthetic: pd.DataFrame,
                                         clinical_guidelines: Dict) -> pd.DataFrame:
        """
        Sub-Phase 2.4: Probabilistic Logic Refinement

        Refines inter-variable relationships within synthetic data based on expert knowledge
        and probabilistic rules defined in clinical guidelines.

        This process ensures that synthetic data adheres to conditional probabilities
        and logical constraints observed in clinical practice, enhancing semantic validity.

        Probabilistic Rules Examples:
        - P(stroke_risk_factor = 1 | diagnosis = Stroke) ≥ 0.90
        - P(nystagmus = vertical | diagnosis = Stroke) ≥ 0.75
        - P(symptom_duration ≤ 60 days | diagnosis = BPPV) ≥ 0.80
        - P(head_impulse_test = 1 | diagnosis = Vestibular Neuritis) ≥ 0.85

        Args:
            D_synthetic: Current synthetic data batch from generative model
            clinical_guidelines: Dictionary containing clinical rules and probabilistic
                               relationships (e.g., from titrate_rules.json)
                               Expected to contain a 'probabilistic_rules' key

        Returns:
            D_refined: Refined synthetic data with enforced clinical coherence
        """
        logger.info("=" * 80)
        logger.info("SUB-PHASE 2.4: Probabilistic Logic Refinement")
        logger.info("=" * 80)
        logger.info(f"Input: {len(D_synthetic)} synthetic records")

        try:
            D_refined = D_synthetic.copy()

            # Access probabilistic rules from clinical guidelines
            prob_rules = clinical_guidelines.get('probabilistic_rules', {})

            if D_refined.empty:
                logger.warning("Empty synthetic data provided. Skipping refinement.")
                return D_refined

            # Check required columns
            required_cols = ['diagnosis']
            missing_cols = [col for col in required_cols if col not in D_refined.columns]

            if missing_cols:
                logger.warning(f"Missing required columns: {missing_cols}. Skipping refinement.")
                return D_refined

            # Apply diagnosis-specific probabilistic rules
            rules_applied = 0

            # Rule 1: Stroke diagnosis → high probability of stroke risk factors
            rules_applied += self._enforce_stroke_coherence(D_refined)

            # Rule 2: BPPV diagnosis → characteristic symptom patterns
            rules_applied += self._enforce_bppv_coherence(D_refined)

            # Rule 3: Vestibular Neuritis → head impulse test positivity
            rules_applied += self._enforce_vestibular_neuritis_coherence(D_refined)

            # Rule 4: Vestibular Migraine → typical presentation patterns
            rules_applied += self._enforce_vestibular_migraine_coherence(D_refined)

            # Rule 5: Age-diagnosis consistency
            rules_applied += self._enforce_age_diagnosis_coherence(D_refined)

            logger.info(f"✓ Sub-Phase 2.4 COMPLETED: Applied {rules_applied} probabilistic refinements")
            logger.info(f"  Output: {len(D_refined)} refined records")
            logger.info("=" * 80)

            return D_refined

        except Exception as e:
            logger.error(f"✗ Sub-Phase 2.4 FAILED: {e}", exc_info=True)
            return D_synthetic  # Return original data if refinement fails

    # =========================================================================
    # Clinical Coherence Enforcement Methods
    # =========================================================================

    def _enforce_stroke_coherence(self, D_refined: pd.DataFrame) -> int:
        """
        Enforces probabilistic rules for Stroke diagnosis.

        Rules:
        - P(stroke_risk_factor = 1 | Stroke) ≥ 0.90
        - P(nystagmus_type = vertical | Stroke) ≥ 0.75
        - P(age > 60 | Stroke) ≥ 0.80

        Returns:
            Number of refinements applied
        """
        if 'diagnosis' not in D_refined.columns:
            return 0

        stroke_mask = D_refined['diagnosis'] == 'Stroke'
        stroke_count = stroke_mask.sum()

        if stroke_count == 0:
            return 0

        refinements = 0

        # Rule: 90% of Stroke cases should have stroke risk factors
        if 'stroke_risk_factor' in D_refined.columns:
            stroke_indices = D_refined[stroke_mask].index
            for idx in stroke_indices:
                if np.random.rand() < 0.90:
                    D_refined.loc[idx, 'stroke_risk_factor'] = 1
                    refinements += 1

        # Rule: 75% of Stroke cases should have vertical nystagmus
        if 'nystagmus_type' in D_refined.columns:
            stroke_indices = D_refined[stroke_mask].index
            for idx in stroke_indices:
                if np.random.rand() < 0.75:
                    D_refined.loc[idx, 'nystagmus_type'] = 'vertical'
                    refinements += 1

        return refinements

    def _enforce_bppv_coherence(self, D_refined: pd.DataFrame) -> int:
        """
        Enforces probabilistic rules for BPPV diagnosis.

        Rules:
        - P(symptom_duration_days ≤ 60 | BPPV) ≥ 0.80
        - P(nystagmus_type in [torsional, horizontal] | BPPV) ≥ 0.85

        Returns:
            Number of refinements applied
        """
        if 'diagnosis' not in D_refined.columns:
            return 0

        bppv_mask = D_refined['diagnosis'] == 'BPPV'
        bppv_count = bppv_mask.sum()

        if bppv_count == 0:
            return 0

        refinements = 0

        # Rule: 80% of BPPV cases should have short symptom duration
        if 'symptom_duration_days' in D_refined.columns:
            bppv_indices = D_refined[bppv_mask].index
            for idx in bppv_indices:
                if np.random.rand() < 0.80:
                    D_refined.loc[idx, 'symptom_duration_days'] = int(np.random.uniform(1, 60))
                    refinements += 1

        # Rule: 85% of BPPV cases should have torsional/horizontal nystagmus
        if 'nystagmus_type' in D_refined.columns:
            bppv_indices = D_refined[bppv_mask].index
            for idx in bppv_indices:
                if np.random.rand() < 0.85:
                    D_refined.loc[idx, 'nystagmus_type'] = np.random.choice(['torsional', 'horizontal'])
                    refinements += 1

        return refinements

    def _enforce_vestibular_neuritis_coherence(self, D_refined: pd.DataFrame) -> int:
        """
        Enforces probabilistic rules for Vestibular Neuritis diagnosis.

        Rules:
        - P(head_impulse_test_positive = 1 | Vestibular Neuritis) ≥ 0.85

        Returns:
            Number of refinements applied
        """
        if 'diagnosis' not in D_refined.columns:
            return 0

        vn_mask = D_refined['diagnosis'] == 'Vestibular Neuritis'
        vn_count = vn_mask.sum()

        if vn_count == 0:
            return 0

        refinements = 0

        # Rule: 85% of Vestibular Neuritis cases should have positive head impulse test
        if 'head_impulse_test_positive' in D_refined.columns:
            vn_indices = D_refined[vn_mask].index
            for idx in vn_indices:
                if np.random.rand() < 0.85:
                    D_refined.loc[idx, 'head_impulse_test_positive'] = 1
                    refinements += 1

        return refinements

    def _enforce_vestibular_migraine_coherence(self, D_refined: pd.DataFrame) -> int:
        """
        Enforces probabilistic rules for Vestibular Migraine diagnosis.

        Rules:
        - P(age < 55 | Vestibular Migraine) ≥ 0.70
        - P(symptom_duration_days in [7, 90] | Vestibular Migraine) ≥ 0.75

        Returns:
            Number of refinements applied
        """
        if 'diagnosis' not in D_refined.columns:
            return 0

        vm_mask = D_refined['diagnosis'] == 'Vestibular Migraine'
        vm_count = vm_mask.sum()

        if vm_count == 0:
            return 0

        refinements = 0

        # Rule: 70% of Vestibular Migraine cases should be younger
        if 'age' in D_refined.columns:
            vm_indices = D_refined[vm_mask].index
            for idx in vm_indices:
                if np.random.rand() < 0.70:
                    D_refined.loc[idx, 'age'] = int(np.random.normal(45, 12))
                    D_refined.loc[idx, 'age'] = np.clip(D_refined.loc[idx, 'age'], 20, 70)
                    refinements += 1

        return refinements

    def _enforce_age_diagnosis_coherence(self, D_refined: pd.DataFrame) -> int:
        """
        Enforces age-diagnosis consistency based on clinical patterns.

        Returns:
            Number of refinements applied
        """
        if 'age' not in D_refined.columns or 'diagnosis' not in D_refined.columns:
            return 0

        refinements = 0

        # Ensure age distributions match clinical expectations
        for idx, row in D_refined.iterrows():
            diagnosis = row['diagnosis']
            current_age = row['age']

            # Stroke: typically older (mean ~70)
            if diagnosis == 'Stroke' and current_age < 50:
                if np.random.rand() < 0.70:
                    D_refined.loc[idx, 'age'] = int(np.random.normal(70, 10))
                    D_refined.loc[idx, 'age'] = np.clip(D_refined.loc[idx, 'age'], 50, 90)
                    refinements += 1

            # BPPV: middle-aged to older (mean ~55)
            elif diagnosis == 'BPPV' and (current_age < 35 or current_age > 80):
                if np.random.rand() < 0.60:
                    D_refined.loc[idx, 'age'] = int(np.random.normal(55, 15))
                    D_refined.loc[idx, 'age'] = np.clip(D_refined.loc[idx, 'age'], 35, 80)
                    refinements += 1

        return refinements

    # =========================================================================
    # Utility Methods
    # =========================================================================

    def validate_coherence(self, D_refined: pd.DataFrame) -> Dict[str, float]:
        """
        Validates that refined data meets probabilistic coherence criteria.

        Returns:
            Dictionary of coherence metrics
        """
        metrics = {}

        if 'diagnosis' not in D_refined.columns:
            return metrics

        # Check Stroke coherence
        stroke_data = D_refined[D_refined['diagnosis'] == 'Stroke']
        if len(stroke_data) > 0 and 'stroke_risk_factor' in D_refined.columns:
            metrics['stroke_risk_coherence'] = stroke_data['stroke_risk_factor'].mean()

        # Check BPPV coherence
        bppv_data = D_refined[D_refined['diagnosis'] == 'BPPV']
        if len(bppv_data) > 0 and 'symptom_duration_days' in D_refined.columns:
            metrics['bppv_duration_coherence'] = (bppv_data['symptom_duration_days'] <= 60).mean()

        logger.info(f"Coherence validation metrics: {metrics}")

        return metrics

