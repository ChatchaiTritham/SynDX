"""
Triate Clinical Pathway Classifier

Implements the three-stage clinical decision model:
1. Triage - how urgent is this patient?
2. Diagnose - what's the likely diagnosis?
3. Disposition - where should this patient go? (ER, specialist, home)

This is based on Snae Namahoot et al.'s Triate framework for emergency medicine.
We use it to validate that our synthetic patients follow realistic clinical
pathways - not just statistically similar but clinically coherent.

Sub-Phase 3.3 in the paper.
"""

import pandas as pd
import numpy as np
import logging
from typing import Dict, Optional, Union

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TriateClassifier:
    """
    Triate Clinical Pathway Classifier for SynDX Framework.

    Implements:
    - Sub-Phase 3.3: Triate Clinical Pathway Classification

    The Triate Model integrates three decision stages:
    1. Triage: Urgency-based initial classification
    2. Diagnose: AI-driven diagnostic prediction
    3. Disposition: Care pathway determination

    Clinical Pathways:
    - ER (Emergency Room): High-risk conditions requiring immediate care
    - Specialist_OPD (Outpatient Department): Moderate-risk requiring specialist review
    - Home_Observation: Low-risk suitable for home monitoring
    """

    def __init__(self, diagnostic_model=None, clinical_triage_rules=None):
        """
 Initializes the TriateClassifier with an optional diagnostic model and triage rules.
 Args:
 diagnostic_model: The trained SynDX diagnostic model (e.g., RandomForestClassifier).
 This model's output (diagnosis) is used as an indicator.
 clinical_triage_rules (dict): Expert-defined rules for urgency classification
 (e.g., loaded from triage_rules.json).
 """
        self.diagnostic_model = diagnostic_model
        self.clinical_triage_rules = clinical_triage_rules

    def extract_relevant_indicators(self, patient_features, diagnosis_output):
        """
 Extracts key indicators for triage from patient features and the diagnostic model's output.
 These indicators are then used to apply clinical triage rules.

 Args:
 patient_features (dict or pd.Series): Subset of patient's initial features (e.g., vital signs, key symptoms, risk factors).
 These should be consistent with features used in triage rules.
 diagnosis_output (str): The primary diagnosis output from the SynDX diagnostic model.
 e.g., 'Stroke', 'BPPV', 'Vestibular Migraine'.
 Returns:
 dict: A dictionary of extracted indicators relevant for triage.
 """
        indicators = {
            'diagnosis_output': diagnosis_output,
            'age': patient_features.get('age', 0),
            'symptom_duration_days': patient_features.get('symptom_duration_days', 0),
            'nystagmus_type': patient_features.get('nystagmus_type', 'none'),
            'headache': patient_features.get('headache', 0),
            'tinnitus': patient_features.get('tinnitus', 0),
            'stroke_risk_factor': patient_features.get('stroke_risk_factor', 0),
            # Add more features as needed from your simulated_scenarios.csv
        }

        # Derive additional boolean indicators based on raw features for rule
        # checking
        indicators['has_severe_nystagmus'] = 1 if indicators['nystagmus_type'] in [
            'vertical', 'torsional'] else 0
        # Dummy rule
        indicators['inability_to_ambulate'] = 1 if indicators['age'] > 70 else 0
        # Dummy rule for vital signs
        indicators['abnormal_vital_signs'] = 1 if indicators['age'] > 65 else 0

        return indicators

    def check_rule_set(self, indicators, criteria_set):
        """
 Checks if the extracted indicators satisfy a given set of triage criteria.
 This function applies the logical rules defined in triage_rules.json.

 Args:
 indicators (dict): Dictionary of extracted indicators for a patient.
 criteria_set (dict): A specific set of rules (e.g., ER_Criteria, Specialist_OPD_Criteria).
 Returns:
 bool: True if criteria are met, False otherwise.
 """
        # Iterate through criteria and check if all conditions are met
        # This is a simplified rule checker. Real rules would be more complex and
        # handle AND/OR logic.

        # Check diagnosis suspicion
        if 'diagnosis_suspicion' in criteria_set:
            if indicators['diagnosis_output'] not in criteria_set['diagnosis_suspicion']:
                return False  # Diagnosis does not match suspicion list

        # Check other boolean/threshold criteria
        if criteria_set.get('vital_signs_abnormal'):
            if indicators.get('abnormal_vital_signs', 0) == 0:
                return False
        if criteria_set.get('symptoms_severe'):
            if indicators.get(
                    'has_severe_nystagmus',
                    0) == 0 and indicators.get(
                    'inability_to_ambulate',
                    0) == 0:
                return False
        if criteria_set.get('no_immediate_danger'):
            if indicators.get('diagnosis_output') in [
                    'Stroke', 'TIA'] or indicators.get(
                    'abnormal_vital_signs', 0) == 1:
                return False
        if criteria_set.get('symptoms_persistent'):
            if indicators.get('symptom_duration_days', 0) < 7:
                return False  # Dummy rule for persistent

        # If all checks pass for the given criteria_set, return True
        return True

        # =========================================================================
        # SUB-PHASE 3.3: Triate Clinical Pathway Classification
        # =========================================================================

    def triate_patient_pathway_classification(self,
                                              diagnosis_output: str,
                                              patient_features: Union[Dict, pd.Series],
                                              clinical_triage_rules: Dict) -> str:
        """
 Sub-Phase 3.3: Triate Clinical Pathway Classification

 Classifies patient urgency and determines appropriate care pathway using Triate Model.

 Triate Decision Algorithm:
 1. Extract clinical indicators from patient features + diagnosis
 2. Apply expert-defined triage rules hierarchically:
 - Check ER criteria first (highest urgency)
 - Check Specialist OPD criteria second (moderate urgency)
 - Default to Home Observation (lowest urgency)

 ER Criteria (High Priority):
 - Stroke/TIA diagnosis suspicion
 - Abnormal vital signs
 - Severe symptoms (inability to ambulate, severe nystagmus)

 Specialist OPD Criteria (Moderate Priority):
 - Persistent symptoms (>7 days)
 - Complex presentations requiring specialist review
 - No immediate danger but needs follow-up

 Home Observation Criteria (Low Priority):
 - Self-limiting conditions
 - Mild symptoms
 - Stable vital signs

 Args:
 diagnosis_output: Primary diagnosis from SynDX diagnostic model
 (e.g., 'Stroke', 'BPPV', 'Vestibular Migraine')
 patient_features: Initial patient features for triage
 (vital signs, key symptoms, risk factors)
 clinical_triage_rules: Expert-defined rules for urgency
 (loaded from triage_rules.json)

 Returns:
 Patient pathway: 'ER', 'Specialist_OPD', or 'Home_Observation'

 Example:
 >>> classifier = TriateClassifier()
 >>> features = {'age': 72, 'stroke_risk_factor': 1, 'nystagmus_type': 'vertical'}
 >>> pathway = classifier.triate_patient_pathway_classification(
 ... diagnosis_output='Stroke',
 ... patient_features=features,
 ... clinical_triage_rules=triage_rules
 ... )
 >>> print(pathway)
 'ER'
 """
        urgency_indicators = self.extract_relevant_indicators(
            patient_features, diagnosis_output)

        if self.check_rule_set(
                urgency_indicators,
                clinical_triage_rules["ER_Criteria"]):
            return "ER"
        elif self.check_rule_set(urgency_indicators, clinical_triage_rules["Specialist_OPD_Criteria"]):
            return "Specialist_OPD"
        else:
            return "Home_Observation"

    def classify(self, patient: Union[Dict, pd.Series],
                 diagnosis: Optional[str] = None) -> str:
        """
 Simplified interface for triage classification.

 Automatically determines triage pathway based on patient features and diagnosis.

 Args:
 patient: Patient features as dict or Series
 diagnosis: Optional diagnosis. If None, will be inferred from patient data

 Returns:
 Triage pathway: 'ER', 'Specialist_OPD', or 'Home_Observation'

 Example:
 >>> classifier = TriateClassifier()
 >>> patient = {'age': 72, 'diagnosis': 'Stroke', 'nystagmus_type': 'central'}
 >>> pathway = classifier.classify(patient)
 >>> print(pathway) # 'ER'
 """
        # Extract diagnosis
        if diagnosis is None:
            if isinstance(patient, dict):
                diagnosis = patient.get('diagnosis', 'Unknown')
            else:
                diagnosis = getattr(patient, 'diagnosis', 'Unknown')

        # Convert to string if needed
        diagnosis = str(diagnosis).upper()

        # Default triage rules
        default_rules = {
            "ER_Criteria": {
                "diagnosis_suspicion": [
                    "STROKE",
                    "TIA",
                    "ACUTE_STROKE"],
                "vital_signs_abnormal": False,
                "symptoms_severe": False},
            "Specialist_OPD_Criteria": {
                "diagnosis_suspicion": [
                    "VESTIBULAR_MIGRAINE",
                    "MENIERES",
                    "VESTIBULAR_NEURITIS",
                    "LABYRINTHITIS",
                    "MS"],
                "symptoms_persistent": True,
                "no_immediate_danger": True}}

        # Use provided rules or default
        rules = self.clinical_triage_rules if self.clinical_triage_rules else default_rules

        # Call main classification method
        return self.triate_patient_pathway_classification(
            diagnosis_output=diagnosis,
            patient_features=patient,
            clinical_triage_rules=rules
        )

    def simulate_baseline_predictions_proba(self, X_simulated_eval):
        """
 Simulates baseline predictions (probabilities) for comparison, representing a rule-based
 standard practice that does not use advanced AI/synthetic data.

 Args:
 X_simulated_eval (pd.DataFrame): Simulated patient features for evaluation.
 Returns:
 np.array: Predicted probabilities for the positive class (e.g., Stroke).
 """
        # Dummy simulation of a simple rule-based baseline
        # E.g., higher probability for 'Stroke' if age > 60 and stroke_risk_factor
        # is present.

        if 'age' in X_simulated_eval.columns and 'stroke_risk_factor' in X_simulated_eval.columns:
            # Simple rule for demonstration: If old and has risk factor, higher
            # chance of stroke
            probas = X_simulated_eval.apply(lambda row:
                                            0.8 if row['age'] > 60 and row['stroke_risk_factor'] == 1 else
                                            0.3 if row['age'] > 50 else
                                            0.1, axis=1)
        else:
            # Default to random low prob if columns missing
            probas = np.random.rand(len(X_simulated_eval)) * 0.5

        return probas.values
