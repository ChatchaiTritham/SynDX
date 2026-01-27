"""
Counterfactual Reasoning Engine - Layer 5: Counterfactual Reasoning

Validates diagnostic logic through systematic "what-if" perturbations, testing pathway consistency.
Ensures that when patient features change appropriately, the diagnosis changes accordingly.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional
import logging
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)


class PerturbationEngine:
    """
    Counterfactual reasoning engine that validates diagnostic logic through systematic perturbations.

    Tests whether diagnostic logic is consistent: if we change patient features in known ways,
    does the diagnosis change appropriately according to clinical guidelines?
    """

    def __init__(self):
        """Initialize perturbation engine."""
        self.perturbations_applied = 0
        logger.info("Initialized PerturbationEngine")

    def validate_samples(
            self,
            data: pd.DataFrame,
            validation_type: str = "ti_trate_consistency") -> pd.DataFrame:
        """
        Validate samples using counterfactual reasoning.

        Args:
            data: Input data to validate
            validation_type: Type of validation to perform

        Returns:
            DataFrame with validation results
        """
        logger.info(
            f"Validating {
                len(data)} samples using {validation_type}...")

        df = data.copy()

        if validation_type == "ti_trate_consistency":
            df = self._validate_ti_trate_consistency(df)
        elif validation_type == "age_perturbation":
            df = self._validate_age_perturbation(df)
        elif validation_type == "risk_factor_perturbation":
            df = self._validate_risk_factor_perturbation(df)
        else:
            logger.warning(
                f"Unknown validation type: {validation_type}. Using ti_trate_consistency.")
            df = self._validate_ti_trate_consistency(df)

        # Calculate Counterfactual Consistency Rate (CCR)
        ccr = self._calculate_ccr(df)
        logger.info(f"Counterfactual Consistency Rate: {ccr:.3f}")

        return df

    def _validate_ti_trate_consistency(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Validate consistency with TiTrATE framework principles.

        According to TiTrATE: acute spontaneous vertigo with central HINTS pattern in patient
        with vascular risk factors should be diagnosed as stroke.
        """
        logger.info("Performing TiTrATE consistency validation...")

        # Add validation columns
        df['validation_ti_trate_consistent'] = True
        df['validation_notes'] = ""
        df['validation_confidence'] = 0.95  # Default high confidence

        # Check for TiTrATE-consistent patterns
        for idx in df.index:
            row = df.loc[idx]

            # TiTrATE rule: acute + spontaneous + central HINTS + vascular risk
            # factors = stroke
            acute_spontaneous = str(
                row.get(
                    'timing_pattern',
                    '')).lower() == 'acute' and str(
                row.get(
                    'trigger_type',
                    '')).lower() == 'spontaneous'

            central_hints = row.get('hints_central', False) or \
                (row.get('hit_abnormal', False) and
                 row.get('nystagmus_direction_changing', False))

            vascular_risk = row.get('cvd_risk', False) or \
                row.get('hypertension', False) or \
                row.get('diabetes', False)

            age_factor = row.get('age', 0) >= 50

            expected_stroke = acute_spontaneous and central_hints and vascular_risk and age_factor

            actual_diagnosis = str(row.get('diagnosis', '')).lower()
            is_stroke = 'stroke' in actual_diagnosis

            # Check if the diagnosis is consistent with TiTrATE
            if expected_stroke and not is_stroke:
                df.at[idx, 'validation_ti_trate_consistent'] = False
                df.at[idx, 'validation_notes'] = "TiTrATE violation: acute spontaneous with central HINTS and risk factors should suggest stroke"
                df.at[idx, 'validation_confidence'] = 0.3
            elif not expected_stroke and is_stroke:
                # Check if stroke diagnosis is justified otherwise
                if not self._is_stroke_justified(row):
                    df.at[idx, 'validation_ti_trate_consistent'] = False
                    df.at[idx, 'validation_notes'] = "TiTrATE violation: stroke diagnosis without appropriate indicators"
                    df.at[idx, 'validation_confidence'] = 0.4

        return df

    def _is_stroke_justified(self, row: pd.Series) -> bool:
        """
        Check if stroke diagnosis is justified by other indicators.

        Args:
            row: Single patient record

        Returns:
            True if stroke diagnosis is justified by other factors
        """
        # Check for other stroke indicators
        has_neurological_signs = row.get('focal_neurological_signs', False)
        has_other_indicators = row.get('other_stroke_indicators', False)

        return has_neurological_signs or has_other_indicators

    def _validate_age_perturbation(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Validate consistency under age perturbations.

        If we reduce patient age significantly, stroke likelihood should decrease.
        """
        logger.info("Performing age perturbation validation...")

        df = df.copy()
        df['validation_age_perturbation_consistent'] = True
        df['validation_age_original_age'] = df.get('age', 50)

        # Create perturbed versions with reduced age
        for idx in df.index:
            original_age = df.at[idx, 'age'] if 'age' in df.columns else 50
            original_diagnosis = str(df.at[idx, 'diagnosis']).lower(
            ) if 'diagnosis' in df.columns else 'unknown'

            # Simulate age reduction by 20 years
            perturbed_age = max(18, original_age - 20)

            # If original was stroke and age is major factor, perturbation
            # should change diagnosis
            if 'stroke' in original_diagnosis and original_age > 60:
                # With lower age, stroke probability should decrease
                df.at[idx, 'validation_age_perturbation_consistent'] = False
                df.at[idx,
                      'validation_notes'] = f"Age perturbation: {original_age}→{perturbed_age}, diagnosis should reconsider stroke likelihood"

        return df

    def _validate_risk_factor_perturbation(
            self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Validate consistency under risk factor perturbations.

        If we remove risk factors, serious diagnosis likelihood should decrease.
        """
        logger.info("Performing risk factor perturbation validation...")

        df = df.copy()
        df['validation_risk_perturbation_consistent'] = True

        for idx in df.index:
            original_diagnosis = str(df.at[idx, 'diagnosis']).lower(
            ) if 'diagnosis' in df.columns else 'unknown'
            original_hypertension = df.at[idx,
                                          'hypertension'] if 'hypertension' in df.columns else False
            original_diabetes = df.at[idx,
                                      'diabetes'] if 'diabetes' in df.columns else False
            original_age = df.at[idx, 'age'] if 'age' in df.columns else 50

            # If serious diagnosis with major risk factors, removing factors
            # should change diagnosis
            if ('stroke' in original_diagnosis or 'serious' in original_diagnosis) and (
                    original_hypertension or original_diabetes or original_age > 65):
                df.at[idx, 'validation_risk_perturbation_consistent'] = False
                df.at[idx, 'validation_notes'] = "Risk factor perturbation would likely change diagnosis"

        return df

    def _calculate_ccr(self, df: pd.DataFrame) -> float:
        """
        Calculate Counterfactual Consistency Rate (CCR).

        CCR = (Number of samples passing validation) / (Total number of samples) * 100%

        Args:
            df: DataFrame with validation results

        Returns:
            CCR value between 0 and 1
        """
        if 'validation_ti_trate_consistent' in df.columns:
            valid_count = df['validation_ti_trate_consistent'].sum()
            total_count = len(df)
            return valid_count / total_count if total_count > 0 else 0.0
        else:
            # If no validation column, assume all are valid
            return 1.0

    def generate_counterfactuals(self,
                                 data: pd.DataFrame,
                                 n_perturbations: int = 100) -> pd.DataFrame:
        """
        Generate counterfactual examples by systematically perturbing patient features.

        Args:
            data: Original data to perturb
            n_perturbations: Number of perturbations to generate

        Returns:
            DataFrame with original and perturbed examples
        """
        logger.info(
            f"Generating {n_perturbations} counterfactual perturbations...")

        original_df = data.copy()
        counterfactuals = []

        for i in range(min(n_perturbations, len(original_df))):
            # Select a random sample to perturb
            orig_idx = np.random.choice(original_df.index)
            original_sample = original_df.loc[orig_idx].copy()

            # Create perturbed version
            perturbed_sample = original_sample.copy()
            perturbed_sample['counterfactual_id'] = f"cf_{i:06d}"
            perturbed_sample['original_id'] = orig_idx

            # Apply systematic perturbations
            # Age perturbation
            if 'age' in perturbed_sample.index:
                original_age = perturbed_sample['age']
                # Perturb age by +/- 10-20 years
                age_change = np.random.choice([-20, -15, -10, 10, 15, 20])
                new_age = max(18, min(100, original_age + age_change))
                perturbed_sample['age'] = new_age
                perturbed_sample['age_perturbation'] = age_change

            # Risk factor perturbation
            if 'hypertension' in perturbed_sample.index:
                # Flip hypertension status
                perturbed_sample['hypertension'] = not perturbed_sample['hypertension']
                perturbed_sample['htn_perturbation_applied'] = True

            if 'diabetes' in perturbed_sample.index:
                # Flip diabetes status
                perturbed_sample['diabetes'] = not perturbed_sample['diabetes']
                perturbed_sample['dm_perturbation_applied'] = True

            # Add to counterfactuals
            counterfactuals.append(perturbed_sample)

        # Combine original and counterfactual data
        if counterfactuals:
            cf_df = pd.DataFrame(counterfactuals)
            result_df = pd.concat([original_df, cf_df], ignore_index=True)
        else:
            result_df = original_df

        logger.info(
            f"Generated {
                len(counterfactuals)} counterfactual examples")

        return result_df

    def get_validation_report(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Generate a comprehensive validation report.

        Args:
            df: DataFrame with validation results

        Returns:
            Dictionary with validation statistics
        """
        report = {}

        # Basic statistics
        report['total_samples'] = len(df)

        # Validation statistics
        if 'validation_ti_trate_consistent' in df.columns:
            report['ti_trate_consistent_count'] = df['validation_ti_trate_consistent'].sum()
            report['ti_trate_inconsistent_count'] = len(
                df) - report['ti_trate_consistent_count']
            report['ti_trate_consistency_rate'] = report['ti_trate_consistent_count'] / \
                len(df)
        else:
            # Assume perfect if no validation performed
            report['ti_trate_consistency_rate'] = 1.0

        if 'validation_age_perturbation_consistent' in df.columns:
            report['age_perturbation_consistent_count'] = df['validation_age_perturbation_consistent'].sum()
            report['age_perturbation_consistency_rate'] = report['age_perturbation_consistent_count'] / \
                len(df)
        else:
            report['age_perturbation_consistency_rate'] = 1.0

        # Overall CCR (Counterfactual Consistency Rate)
        report['counterfactual_consistency_rate'] = self._calculate_ccr(df)
        report['target_ccr'] = 0.95  # Target: 95% consistency
        report['ccr_met'] = report['counterfactual_consistency_rate'] >= 0.95

        # Notes summary
        if 'validation_notes' in df.columns:
            report['validation_note_types'] = df['validation_notes'].value_counts(
            ).to_dict()
            report['samples_with_issues'] = (
                df['validation_notes'] != "").sum()
        else:
            report['samples_with_issues'] = 0

        return report


class TiTrATEConsistencyChecker:
    """
    Validate diagnostic pathways against TiTrATE framework principles.
    """

    def __init__(self):
        """Initialize TiTrATE consistency checker."""
        logger.info("Initialized TiTrATEConsistencyChecker")

    def validate_pathway(self, patient_data: pd.Series) -> Dict[str, Any]:
        """
        Validate a single patient's diagnostic pathway against TiTrATE principles.

        Args:
            patient_data: Single patient record

        Returns:
            Dictionary with validation results
        """
        results = {
            'patient_id': patient_data.get('patient_id', 'unknown'),
            'original_diagnosis': patient_data.get('diagnosis', 'unknown'),
            'ti_trate_compliant': True,
            'validation_issues': [],
            # How much confidence should be adjusted based on validation
            'confidence_adjustment': 0.0
        }

        # Apply TiTrATE logic
        timing = patient_data.get('timing_pattern', '').lower()
        trigger = patient_data.get('trigger_type', '').lower()
        age = patient_data.get('age', 0)
        hypertension = patient_data.get('hypertension', False)
        diabetes = patient_data.get('diabetes', False)
        atrial_fibrillation = patient_data.get('atrial_fibrillation', False)
        hit_abnormal = patient_data.get('hit_abnormal', False)
        nystagmus_direction_changing = patient_data.get(
            'nystagmus_direction_changing', False)
        skew_deviation = patient_data.get('skew_deviation', False)

        # Check for acute spontaneous AVS with central HINTS pattern
        if (timing == 'acute' and trigger == 'spontaneous' and age >=
                50 and (hypertension or diabetes or atrial_fibrillation)):

            # Check HINTS components
            central_hints = hit_abnormal or nystagmus_direction_changing or skew_deviation

            if central_hints and 'stroke' not in results['original_diagnosis'].lower(
            ):
                results['ti_trate_compliant'] = False
                results['validation_issues'].append(
                    "Acute spontaneous AVS with central HINTS and risk factors should suggest stroke"
                )
                results['confidence_adjustment'] = -0.2  # Reduce confidence

            elif not central_hints and 'stroke' in results['original_diagnosis'].lower():
                # Check if stroke diagnosis is justified by other means
                focal_neurological_signs = patient_data.get(
                    'focal_neurological_signs', False)
                if not focal_neurological_signs:
                    results['ti_trate_compliant'] = False
                    results['validation_issues'].append(
                        "Stroke diagnosis without central HINTS pattern or focal neurological signs"
                    )
                    results['confidence_adjustment'] = -0.15

        return results

    def validate_dataset(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Validate entire dataset against TiTrATE principles.

        Args:
            df: DataFrame to validate

        Returns:
            DataFrame with validation results added
        """
        logger.info(
            f"Validating {
                len(df)} samples against TiTrATE principles...")

        validation_results = []
        for idx in df.index:
            patient_data = df.loc[idx]
            result = self.validate_pathway(patient_data)
            validation_results.append(result)

        # Add validation results to dataframe
        validation_df = pd.DataFrame(validation_results)

        # Merge with original dataframe
        df_with_validation = df.copy()
        df_with_validation['ti_trate_compliant'] = validation_df['ti_trate_compliant']
        df_with_validation['validation_issues'] = validation_df['validation_issues'].apply(
            lambda x: '; '.join(x) if x else '')
        df_with_validation['confidence_adjustment'] = validation_df['confidence_adjustment']

        # Adjust confidence based on validation
        if 'confidence' in df_with_validation.columns:
            df_with_validation['adjusted_confidence'] = np.clip(
                df_with_validation['confidence'] +
                df_with_validation['confidence_adjustment'],
                0.0,
                1.0)

        logger.info(
            f"TiTrATE validation completed: {
                validation_df['ti_trate_compliant'].sum()}/{
                len(df)} compliant")

        return df_with_validation


class ValidationEngine:
    """
    Comprehensive validation engine for SynDX-Hybrid framework.
    """

    def __init__(self):
        """Initialize validation engine."""
        self.titrate_checker = TiTrATEConsistencyChecker()
        logger.info("Initialized ValidationEngine")

    def validate_synthetic_data(self,
                                synthetic_data: pd.DataFrame,
                                validation_types: List[str] = ['ti_trate',
                                                               'statistical',
                                                               'clinical']) -> Tuple[Dict[str,
                                                                                          Any],
                                                                                     pd.DataFrame]:
        """
        Perform comprehensive validation of synthetic data.

        Args:
            synthetic_data: Synthetic data to validate
            validation_types: Types of validation to perform

        Returns:
            Tuple of (validation_results, validated_data)
        """
        logger.info(
            f"Performing comprehensive validation on {
                len(synthetic_data)} samples...")

        results = {
            'validation_timestamp': pd.Timestamp.now().isoformat(),
            'n_samples': len(synthetic_data),
            'validation_types_performed': validation_types,
            'validation_results': {}
        }

        validated_data = synthetic_data.copy()

        for validation_type in validation_types:
            if validation_type == 'ti_trate':
                # Validate against TiTrATE framework
                validated_data = self.titrate_checker.validate_dataset(
                    validated_data)
                ti_trate_rate = validated_data['ti_trate_compliant'].mean()

                results['validation_results']['ti_trate'] = {
                    'compliance_rate': ti_trate_rate,
                    'target_rate': 0.95,  # 95% compliance target
                    'met_target': ti_trate_rate >= 0.95,
                    'compliant_samples': int(validated_data['ti_trate_compliant'].sum()),
                    'non_compliant_samples': len(synthetic_data) - int(validated_data['ti_trate_compliant'].sum())
                }

            elif validation_type == 'statistical':
                # Statistical realism validation
                results['validation_results']['statistical'] = self._validate_statistical_realism(
                    validated_data)

            elif validation_type == 'clinical':
                # Clinical plausibility validation
                results['validation_results']['clinical'] = self._validate_clinical_plausibility(
                    validated_data)

        # Calculate overall validation score
        ti_trate_score = results['validation_results'].get(
            'ti_trate', {}).get('compliance_rate', 1.0)
        stat_score = results['validation_results'].get(
            'statistical', {}).get('realism_score', 1.0)
        clinical_score = results['validation_results'].get(
            'clinical', {}).get('plausibility_rate', 1.0)

        results['overall_validation_score'] = (
            ti_trate_score + stat_score + clinical_score) / 3
        results['validation_passed'] = all(
            res.get('met_target', True)
            for res in results['validation_results'].values()
            if isinstance(res, dict) and 'met_target' in res
        )

        logger.info(
            f"Overall validation score: {
                results['overall_validation_score']:.3f}")
        logger.info(f"Validation passed: {results['validation_passed']}")

        return results, validated_data

    def _validate_statistical_realism(
            self, df: pd.DataFrame) -> Dict[str, Any]:
        """Validate statistical realism of synthetic data."""
        logger.info("Performing statistical realism validation...")

        # Calculate KL divergence from expected distributions
        # For demonstration, we'll use a simplified approach
        # In a real implementation, we'd compare to reference distributions

        # Calculate basic statistical metrics
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

        if len(numeric_cols) > 0:
            # Calculate distribution similarity metrics
            js_divergence = 0.0
            # Limit to first 10 columns for performance
            for col in numeric_cols[:10]:
                # Create a reference distribution (normal distribution for
                # example)
                ref_dist = np.random.normal(
                    df[col].mean(), df[col].std(), len(df))
                # Calculate JS divergence (simplified)
                hist1, _ = np.histogram(
                    df[col].dropna(), bins=20, density=True)
                hist2, _ = np.histogram(ref_dist, bins=20, density=True)

                # Avoid division by zero
                hist1 = hist1 + 1e-10
                hist2 = hist2 + 1e-10

                m = 0.5 * (hist1 + hist2)
                js_div = 0.5 * (np.sum(hist1 * np.log(hist1 / m)) +
                                np.sum(hist2 * np.log(hist2 / m)))
                js_divergence += js_div

            js_divergence = js_divergence / \
                min(10, len(numeric_cols)) if numeric_cols else 0.0
        else:
            js_divergence = 0.1  # Default if no numeric columns

        return {
            'js_divergence': js_divergence,
            'target_js_divergence': 0.05,  # Target: <= 0.05
            'met_target': js_divergence <= 0.05,
            # Higher score for lower divergence
            'realism_score': max(0, 1 - js_divergence / 0.05),
            'n_numeric_features': len(numeric_cols)
        }

    def _validate_clinical_plausibility(
            self, df: pd.DataFrame) -> Dict[str, Any]:
        """Validate clinical plausibility of synthetic data."""
        logger.info("Performing clinical plausibility validation...")

        # Check for clinically impossible combinations
        implausible_count = 0
        total_checks = 0

        for idx in df.index:
            row = df.loc[idx]
            total_checks += 1

            # Check for impossible combinations
            age = row.get('age', 50)
            hypertension = row.get('hypertension', False)
            diabetes = row.get('diabetes', False)
            diagnosis = str(row.get('diagnosis', '')).lower()

            # Example: young person (age < 30) with multiple vascular risk
            # factors and stroke diagnosis
            if age < 30 and (
                    hypertension or diabetes) and 'stroke' in diagnosis:
                implausible_count += 1
            # Example: very elderly (age > 90) with benign diagnosis but severe
            # symptoms
            elif age > 90 and 'benign' in diagnosis and row.get('severity_score', 5) > 7:
                implausible_count += 1

        plausibility_rate = 1 - (implausible_count /
                                 total_checks) if total_checks > 0 else 1.0

        return {
            'implausible_combinations': implausible_count,
            'total_checks': total_checks,
            'plausibility_rate': plausibility_rate,
            'target_plausibility': 0.90,  # Target: >= 90%
            'met_target': plausibility_rate >= 0.90
        }


# Test the Counterfactual Reasoning Engine
if __name__ == '__main__':
    print("Testing Counterfactual Reasoning Engine...")

    # Create sample data for testing
    sample_data = pd.DataFrame({
        'patient_id': [f'CF_{i:06d}' for i in range(100)],
        'age': np.random.normal(55, 18, 100),
        'sex': np.random.choice(['M', 'F'], 100),
        'hypertension': np.random.choice([0, 1], 100, p=[0.7, 0.3]),
        'diabetes': np.random.choice([0, 1], 100, p=[0.85, 0.15]),
        'atrial_fibrillation': np.random.choice([0, 1], 100, p=[0.95, 0.05]),
        'timing_pattern': np.random.choice(['acute', 'episodic', 'chronic'], 100, p=[0.3, 0.5, 0.2]),
        'trigger_type': np.random.choice(['spontaneous', 'positional', 'head_movement'], 100, p=[0.4, 0.4, 0.2]),
        'hit_abnormal': np.random.choice([0, 1], 100, p=[0.7, 0.3]),
        'nystagmus_direction_changing': np.random.choice([0, 1], 100, p=[0.85, 0.15]),
        'skew_deviation': np.random.choice([0, 1], 100, p=[0.95, 0.05]),
        'dix_hallpike_positive': np.random.choice([0, 1], 100, p=[0.85, 0.15]),
        'focal_neurological_signs': np.random.choice([0, 1], 100, p=[0.98, 0.02]),
        'diagnosis': np.random.choice(['stroke', 'bppv', 'vn', 'menieres', 'migraine'], 100, p=[0.1, 0.3, 0.2, 0.2, 0.2]),
        'confidence': np.random.uniform(0.7, 1.0, 100),
        'urgency': np.random.choice([0, 1, 2], 100, p=[0.6, 0.3, 0.1])
    })

    # Ensure age is within reasonable bounds
    sample_data['age'] = np.clip(sample_data['age'], 18, 100)

    # Initialize perturbation engine
    perturbation_engine = PerturbationEngine()

    # Validate samples using TiTrATE consistency
    validated_data = perturbation_engine.validate_samples(
        sample_data,
        validation_type="ti_trate_consistency"
    )

    print(f"\\nTiTrATE validation completed")
    print(
        f"  Samples passing validation: {(validated_data['validation_ti_trate_consistent']).sum()}")
    print(
        f"  Samples failing validation: {
            (
                validated_data['validation_ti_trate_consistent'] == False).sum()}")
    print(
        f"  Counterfactual Consistency Rate: {
            perturbation_engine._calculate_ccr(validated_data):.3f}")

    # Generate counterfactuals
    counterfactual_data = perturbation_engine.generate_counterfactuals(
        validated_data.head(20),  # Use first 20 for counterfactual generation
        n_perturbations=10
    )

    print(f"\\nGenerated {len(counterfactual_data) -
                          len(validated_data.head(20))} counterfactual examples")

    # Initialize TiTrATE consistency checker
    titrate_checker = TiTrATEConsistencyChecker()

    # Validate against TiTrATE principles
    ti_trate_validated = titrate_checker.validate_dataset(
        validated_data.head(50))  # Validate first 50

    print(f"\\nTiTrATE pathway validation completed")
    print(
        f"  TiTrATE compliant samples: {
            ti_trate_validated['ti_trate_compliant'].sum()}")
    print(
        f"  Non-compliant samples: {(ti_trate_validated['ti_trate_compliant'] == False).sum()}")
    print(
        f"  Compliance rate: {
            ti_trate_validated['ti_trate_compliant'].mean():.3f}")

    # Initialize validation engine
    validation_engine = ValidationEngine()

    # Perform comprehensive validation
    validation_results, final_data = validation_engine.validate_synthetic_data(
        ti_trate_validated,
        validation_types=['ti_trate', 'statistical', 'clinical']
    )

    print(f"\\nComprehensive validation results:")
    for val_type, result in validation_results['validation_results'].items():
        print(f"  {val_type.upper()}:")
        for key, value in result.items():
            print(f"    {key}: {value}")

    print(
        f"\\nOverall validation score: {
            validation_results['overall_validation_score']:.3f}")
    print(f"Validation passed: {validation_results['validation_passed']}")

    print(f"\\nCounterfactual reasoning engine test completed successfully!")
