"""
TiTrATE-Constrained Counterfactual Validator

Generates counterfactual examples to validate synthetic data plausibility
and test TiTrATE constraint boundaries.

Counterfactuals answer "what-if" questions:
- "What if we change trigger from positional to spontaneous?"
- "What if nystagmus direction becomes bidirectional?"
- "What minimal changes flip the diagnosis from BPPV to VM?"

Ensures generated counterfactuals respect clinical constraints, providing
validation of synthetic data generation quality.

Author: Chatchai Tritham
Date: 2026-01-25
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Callable
import logging

logger = logging.getLogger(__name__)


class CounterfactualValidator:
    """
    TiTrATE-constrained counterfactual generation for validation.

    Generates clinically plausible "what-if" scenarios by minimally modifying
    patient features while maintaining TiTrATE constraint satisfaction.

    Strategy:
    1. Given patient P with diagnosis D
    2. Generate counterfactual P' with diagnosis D' ≠ D
    3. Ensure P' passes all TiTrATE constraints
    4. Minimize feature distance ||P - P'||
    5. Validate clinical plausibility

    Example:
        >>> validator = CounterfactualValidator(constraint_checker)
        >>> counterfactuals = validator.generate_counterfactuals(
        ...     patient=archetype_features,
        ...     target_diagnosis='VM',
        ...     n_counterfactuals=5
        ... )
        >>> for cf in counterfactuals:
        ...     print(f"Distance: {cf['distance']:.2f}, Valid: {cf['is_valid']}")
    """

    def __init__(self,
                 constraint_checker,
                 max_iterations: int = 100,
                 distance_metric: str = 'l2',
                 random_state: int = 42):
        """
        Initialize counterfactual validator.

        Args:
            constraint_checker: TiTrATEFormalizer instance for validation
            max_iterations: Maximum iterations for counterfactual search
            distance_metric: Distance metric ('l1' or 'l2')
            random_state: Random seed
        """
        self.constraint_checker = constraint_checker
        self.max_iterations = max_iterations
        self.distance_metric = distance_metric
        self.random_state = random_state

        np.random.seed(random_state)

        # Store generated counterfactuals
        self.counterfactuals = []

        logger.info(
            f"Initialized CounterfactualValidator "
            f"(max_iter={max_iterations}, metric={distance_metric})"
        )

    def generate_counterfactual(self,
                                patient: np.ndarray,
                                target_diagnosis: str,
                                feature_names: Optional[List[str]] = None,
                                modifiable_features: Optional[List[int]] = None) -> Optional[Dict]:
        """
        Generate a single counterfactual for the patient.

        Args:
            patient: Original patient feature vector
            target_diagnosis: Desired diagnosis for counterfactual
            feature_names: Optional feature names for interpretability
            modifiable_features: Indices of features that can be modified
                                (None = all features modifiable)

        Returns:
            Counterfactual dictionary with:
                - features: Counterfactual feature vector
                - distance: Distance from original patient
                - is_valid: Whether counterfactual satisfies constraints
                - violations: List of constraint violations
                - changes: Dictionary of feature changes
            or None if no valid counterfactual found
        """
        if modifiable_features is None:
            modifiable_features = list(range(len(patient)))

        logger.info(
            f"Generating counterfactual for target diagnosis: {target_diagnosis}")

        best_counterfactual = None
        best_distance = np.inf

        for iteration in range(self.max_iterations):
            # Generate candidate counterfactual
            candidate = self._perturb_patient(patient, modifiable_features)

            # Check constraint satisfaction
            is_valid, violations = self._check_constraints(
                candidate, target_diagnosis)

            if is_valid:
                # Compute distance
                distance = self._compute_distance(patient, candidate)

                # Update best counterfactual if closer
                if distance < best_distance:
                    best_distance = distance
                    best_counterfactual = {
                        'features': candidate.copy(),
                        'distance': distance,
                        'is_valid': True,
                        'violations': [],
                        'changes': self._identify_changes(
                            patient,
                            candidate,
                            feature_names),
                        'target_diagnosis': target_diagnosis}

                    logger.debug(
                        f"Iteration {iteration}: Found valid counterfactual (distance={
                            distance:.3f})")

                    # Early stopping if very close
                    if distance < 1e-3:
                        break

        if best_counterfactual is not None:
            logger.info(
                f"✓ Counterfactual found (distance={
                    best_distance:.3f})")
            return best_counterfactual
        else:
            logger.warning(
                f"✗ No valid counterfactual found after {
                    self.max_iterations} iterations")
            return None

    def generate_counterfactuals(self,
                                 patient: np.ndarray,
                                 target_diagnosis: str,
                                 n_counterfactuals: int = 5,
                                 feature_names: Optional[List[str]] = None) -> List[Dict]:
        """
        Generate multiple diverse counterfactuals.

        Args:
            patient: Original patient feature vector
            target_diagnosis: Desired diagnosis
            n_counterfactuals: Number of counterfactuals to generate
            feature_names: Optional feature names

        Returns:
            List of counterfactual dictionaries
        """
        logger.info(
            f"Generating {n_counterfactuals} counterfactuals for {target_diagnosis}...")

        counterfactuals = []

        for i in range(n_counterfactuals):
            # Randomize feature modification order for diversity
            np.random.seed(self.random_state + i)

            cf = self.generate_counterfactual(
                patient,
                target_diagnosis,
                feature_names=feature_names
            )

            if cf is not None:
                counterfactuals.append(cf)
            else:
                logger.warning(
                    f"Failed to generate counterfactual {
                        i + 1}/{n_counterfactuals}")

        logger.info(
            f"✓ Generated {
                len(counterfactuals)}/{n_counterfactuals} valid counterfactuals")

        return counterfactuals

    def _perturb_patient(
            self,
            patient: np.ndarray,
            modifiable_features: List[int]) -> np.ndarray:
        """
        Perturb patient features to create candidate counterfactual.

        Strategy:
        - Randomly select subset of modifiable features
        - Apply Gaussian noise with adaptive magnitude
        - Ensure features remain in valid range

        Args:
            patient: Original patient vector
            modifiable_features: Indices of features to modify

        Returns:
            Perturbed patient vector
        """
        candidate = patient.copy()

        # Select random subset of features to modify (1-5 features)
        n_features_to_modify = np.random.randint(
            1, min(6, len(modifiable_features) + 1))
        features_to_modify = np.random.choice(
            modifiable_features, size=n_features_to_modify, replace=False)

        # Apply perturbations
        for feat_idx in features_to_modify:
            # Adaptive noise magnitude (larger for categorical, smaller for
            # continuous)
            if np.abs(patient[feat_idx]) < 2.0:  # Likely binary/categorical
                noise_scale = 0.5
            else:
                noise_scale = 0.1 * np.abs(patient[feat_idx])

            # Add Gaussian noise
            noise = np.random.randn() * noise_scale
            candidate[feat_idx] = patient[feat_idx] + noise

            # Clip to reasonable range [-10, 10] for numerical stability
            candidate[feat_idx] = np.clip(candidate[feat_idx], -10, 10)

        return candidate

    def _check_constraints(self, patient: np.ndarray,
                           target_diagnosis: str) -> Tuple[bool, List[str]]:
        """
        Check if patient satisfies TiTrATE constraints.

        Args:
            patient: Patient feature vector
            target_diagnosis: Expected diagnosis

        Returns:
            (is_valid, violations): Constraint satisfaction status and violation list
        """
        # Placeholder constraint check (would use actual TiTrATEFormalizer)
        # For demo purposes, assume most counterfactuals are valid
        if self.constraint_checker is None:
            # Mock validation
            is_valid = np.random.rand() > 0.3  # 70% success rate
            violations = [] if is_valid else ['mock_constraint_violation']
            return is_valid, violations

        # Real validation using constraint checker
        try:
            is_valid, violations = self.constraint_checker.validate_all_constraints(
                patient)
            return is_valid, violations
        except Exception as e:
            logger.warning(f"Constraint check failed: {e}")
            return False, [str(e)]

    def _compute_distance(
            self,
            original: np.ndarray,
            counterfactual: np.ndarray) -> float:
        """
        Compute distance between original and counterfactual.

        Args:
            original: Original patient vector
            counterfactual: Counterfactual patient vector

        Returns:
            Distance value
        """
        if self.distance_metric == 'l1':
            return np.linalg.norm(original - counterfactual, ord=1)
        elif self.distance_metric == 'l2':
            return np.linalg.norm(original - counterfactual, ord=2)
        else:
            raise ValueError(
                f"Unknown distance metric: {
                    self.distance_metric}")

    def _identify_changes(self,
                          original: np.ndarray,
                          counterfactual: np.ndarray,
                          feature_names: Optional[List[str]] = None) -> Dict:
        """
        Identify which features changed between original and counterfactual.

        Args:
            original: Original patient vector
            counterfactual: Counterfactual vector
            feature_names: Optional feature names

        Returns:
            Dictionary mapping feature indices/names to (original, counterfactual, delta)
        """
        changes = {}

        for i in range(len(original)):
            if np.abs(original[i] - counterfactual[i]) > 1e-6:
                feat_name = feature_names[i] if feature_names else f"feature_{i}"
                changes[feat_name] = {
                    'original': float(original[i]),
                    'counterfactual': float(counterfactual[i]),
                    'delta': float(counterfactual[i] - original[i])
                }

        return changes

    def validate_plausibility(
            self,
            original: np.ndarray,
            counterfactual: np.ndarray) -> Dict:
        """
        Assess clinical plausibility of counterfactual.

        Args:
            original: Original patient
            counterfactual: Counterfactual patient

        Returns:
            Plausibility metrics
        """
        # Constraint satisfaction
        is_valid, violations = self._check_constraints(
            counterfactual, 'unknown')

        # Feature distance
        l1_distance = np.linalg.norm(original - counterfactual, ord=1)
        l2_distance = np.linalg.norm(original - counterfactual, ord=2)

        # Number of changed features
        changed_features = np.sum(np.abs(original - counterfactual) > 1e-6)

        # Plausibility heuristic: fewer changes = more plausible
        # Score in [0, 1], higher is better
        plausibility_score = 1.0 / (1.0 + changed_features / len(original))

        return {
            'is_valid': is_valid,
            'violations': violations,
            'l1_distance': l1_distance,
            'l2_distance': l2_distance,
            'changed_features': int(changed_features),
            'plausibility_score': plausibility_score
        }

    def compute_actionable_insights(self,
                                    patient: np.ndarray,
                                    possible_diagnoses: List[str],
                                    feature_names: Optional[List[str]] = None) -> List[Dict]:
        """
        Find minimal changes needed for each possible diagnosis flip.

        Args:
            patient: Patient feature vector
            possible_diagnoses: List of possible alternative diagnoses
            feature_names: Optional feature names

        Returns:
            List of actionable insights (one per diagnosis)
        """
        logger.info(
            f"Computing actionable insights for {
                len(possible_diagnoses)} diagnoses...")

        insights = []

        for target_diag in possible_diagnoses:
            # Generate counterfactual for this diagnosis
            cf = self.generate_counterfactual(
                patient, target_diag, feature_names)

            if cf is not None:
                insights.append({
                    'target_diagnosis': target_diag,
                    'distance': cf['distance'],
                    'changes': cf['changes'],
                    'is_valid': cf['is_valid']
                })
            else:
                insights.append({
                    'target_diagnosis': target_diag,
                    'distance': np.inf,
                    'changes': {},
                    'is_valid': False
                })

        # Sort by distance (most achievable first)
        insights.sort(key=lambda x: x['distance'])

        logger.info(
            f"✓ Computed insights for {len([i for i in insights if i['is_valid']])} valid counterfactuals")

        return insights

    def summary(self) -> Dict:
        """
        Get summary statistics of counterfactual validation.

        Returns:
            Dictionary with validation statistics
        """
        if not self.counterfactuals:
            return {'status': 'no_counterfactuals'}

        distances = [cf['distance'] for cf in self.counterfactuals]
        valid_count = sum(cf['is_valid'] for cf in self.counterfactuals)

        return {
            'status': 'validated',
            'total_counterfactuals': len(self.counterfactuals),
            'valid_counterfactuals': valid_count,
            'validity_rate': valid_count / len(self.counterfactuals),
            'avg_distance': float(np.mean(distances)),
            'min_distance': float(np.min(distances)),
            'max_distance': float(np.max(distances))
        }

    def __repr__(self) -> str:
        """String representation"""
        return (f"CounterfactualValidator(max_iter={self.max_iterations}, "
                f"metric='{self.distance_metric}')")


# Main demonstration
if __name__ == '__main__':
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    logger.info("CounterfactualValidator - Demo Mode")

    # Create mock patient data
    np.random.seed(42)

    # Mock patient: 20 features representing clinical presentation
    patient = np.array([
        1.0,   # Feature 0: Age (scaled)
        0.0,   # Feature 1: Gender (0=M, 1=F)
        1.5,   # Feature 2: Symptom duration
        3.0,
        # Feature 3: Trigger type (0=spontaneous, 1=positional,
        # 2=head_movement, 3=exertion)
        1.0,   # Feature 4: Nystagmus present
        2.0,   # Feature 5: Nystagmus direction
        0.5,   # Feature 6: Vertigo intensity
        1.2,   # Feature 7: Episode duration
        # ... 12 more features
        0.8, 0.3, 1.1, 0.6, 0.9, 1.4, 0.2, 0.7, 1.0, 0.4, 0.5, 1.3
    ])

    feature_names = [
        'age',
        'gender',
        'duration',
        'trigger',
        'nystagmus',
        'nyst_direction',
        'vertigo_intensity',
        'episode_duration',
        'hearing_loss',
        'tinnitus',
        'headache',
        'aura',
        'family_history',
        'previous_episodes',
        'medication_response',
        'dix_hallpike_positive',
        'head_impulse_test',
        'cerebellar_signs',
        'autonomic_symptoms',
        'photophobia']

    logger.info(f"Patient features: {patient.shape}")

    # Initialize validator (without constraint checker for demo)
    validator = CounterfactualValidator(
        constraint_checker=None,  # Mock validation
        max_iterations=50,
        distance_metric='l2'
    )

    # Generate counterfactuals for different diagnoses
    diagnoses = ['BPPV', 'VM', 'VN', 'Stroke']

    logger.info(
        f"\nGenerating counterfactuals for {
            len(diagnoses)} diagnoses...")

    for diagnosis in diagnoses:
        logger.info(f"\n{'=' * 80}")
        logger.info(f"Target Diagnosis: {diagnosis}")
        logger.info(f"{'=' * 80}")

        counterfactuals = validator.generate_counterfactuals(
            patient,
            target_diagnosis=diagnosis,
            n_counterfactuals=3,
            feature_names=feature_names
        )

        for i, cf in enumerate(counterfactuals, 1):
            logger.info(f"\nCounterfactual {i}:")
            logger.info(f"  Distance: {cf['distance']:.3f}")
            logger.info(f"  Valid: {cf['is_valid']}")
            logger.info(f"  Changes ({len(cf['changes'])}):")
            for feat, change in list(
                    cf['changes'].items())[:5]:  # Show top 5 changes
                logger.info(
                    f"    - {feat}: {change['original']:.2f} → {change['counterfactual']:.2f} "
                    f"(Δ={change['delta']:+.2f})"
                )

    # Actionable insights
    logger.info(f"\n{'=' * 80}")
    logger.info("Computing Actionable Insights")
    logger.info(f"{'=' * 80}")

    insights = validator.compute_actionable_insights(
        patient, diagnoses, feature_names)

    logger.info("\nMinimal changes needed for diagnosis flip:")
    for insight in insights:
        logger.info(
            f"  {insight['target_diagnosis']:10s}: "
            f"distance={insight['distance']:.3f}, "
            f"changes={len(insight['changes'])}, "
            f"valid={insight['is_valid']}"
        )

    logger.info("\n✓ Demo complete!")
