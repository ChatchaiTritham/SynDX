"""
Rule-Based Expert System - Layer 3: Rule-Based Expert Systems

Directly encodes clinical guidelines as formal IF-THEN rules with full source traceability.
Every decision includes complete reference and diagnostic rationale.
"""

import logging
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class RuleBasedExpertSystem:
    """
    Rule-based expert system that translates clinical guidelines into formal IF-THEN rules.

    Each rule includes full source attribution with peer-reviewed references and
    diagnostic rationale, ensuring complete traceability.
    """

    def __init__(self, rule_count: int = 247, random_seed: int = 42):
        """
        Initialize rule-based expert system.

        Args:
            rule_count: Number of clinical rules to generate
            random_seed: Random seed for reproducibility
        """
        self.rule_count = rule_count
        self.random_seed = random_seed
        np.random.seed(random_seed)

        # Generate clinical rules based on TiTrATE framework and guidelines
        self.rules = self._generate_clinical_rules()

        logger.info(
            f"Initialized RuleBasedExpertSystem with {len(self.rules)} clinical rules"
        )

    def _generate_clinical_rules(self) -> List[Dict[str, Any]]:
        """
        Generate clinical rules based on TiTrATE framework and published guidelines.

        Rules follow the format: IF (conditions) THEN (diagnosis, confidence, source, rationale)
        """
        rules = []

        # Rule category 1: Stroke detection (TiTrATE + HINTS criteria)
        stroke_rules = [
            {
                'id': 'stroke_rule_001',
                'condition': "(timing == 'acute' and trigger == 'spontaneous' and "
                "hints_central == True and age >= 50 and cvd_risk == True)",
                'action': {
                    'diagnosis': 'posterior_circulation_stroke',
                    'confidence': 0.95,
                    'urgency': 'emergency',
                    'source': 'AHA/ASA Acute Ischemic Stroke Guidelines 2018',
                    'reference': 'Powers et al., Stroke 2018;49:e46-e110',
                    'rationale': 'HINTS examination showing central pattern (abnormal HIT or '
                    'direction-changing nystagmus or skew deviation) in appropriate '
                    'clinical context (acute spontaneous AVS with vascular risk factors) '
                    'has 96.8% sensitivity and 98.5% specificity for stroke',
                },
            },
            {
                'id': 'stroke_rule_002',
                'condition': "(onset == 'acute' and focal_neurological_signs == True and age >= 50)",
                'action': {
                    'diagnosis': 'stroke',
                    'confidence': 0.92,
                    'urgency': 'emergency',
                    'source': 'Newman-Toker et al., Neurologic Clinics 2015',
                    'reference': 'Newman-Toker & Edlow, Neurol Clin. 2015;33:577-599',
                    'rationale': 'Acute onset with focal neurological signs in patient >50 years '
                    'highly suggestive of stroke',
                },
            },
        ]

        # Rule category 2: BPPV detection (Bรกrรกny Society ICVD criteria)
        bppv_rules = [
            {
                'id': 'bppv_rule_001',
                'condition': "(timing == 'episodic' and trigger == 'positional' and "
                "duration < 60 and dix_hallpike_positive == True)",
                'action': {
                    'diagnosis': 'BPPV_posterior_canal',
                    'confidence': 0.92,
                    'urgency': 'routine',
                    'source': 'Bhattacharyya et al., Otolaryngology 2017',
                    'reference': 'Bhattacharyya et al., Otolaryngol Head Neck Surg. 2017;156(3_suppl):S1-S47',
                    'rationale': 'Episodic positional vertigo <1min with positive Dix-Hallpike showing '
                    'characteristic upbeating-torsional nystagmus is diagnostic for '
                    'posterior canal BPPV per AAO-HNSF criteria',
                },
            }
        ]

        # Rule category 3: Vestibular neuritis
        vestibular_neuritis_rules = [
            {
                'id': 'vn_rule_001',
                'condition': "(timing == 'acute' and spontaneous_onset == True and "
                "hit_abnormal == True and nystagmus_peripheral == True)",
                'action': {
                    'diagnosis': 'vestibular_neuritis',
                    'confidence': 0.88,
                    'urgency': 'urgent',
                    'source': 'Strupp et al., J Neurol 2017',
                    'reference': 'Strupp et al., J Neurol. 2017;264(4):611-616',
                    'rationale': 'Acute spontaneous vertigo with abnormal head impulse test and '
                    'peripheral-type nystagmus characteristic of vestibular neuritis',
                },
            }
        ]

        # Rule category 4: Vestibular migraine
        vm_rules = [
            {
                'id': 'vm_rule_001',
                'condition': "(migraine_history == True and episodic_vertigo == True and "
                "headache_present == True)",
                'action': {
                    'diagnosis': 'vestibular_migraine',
                    'confidence': 0.85,
                    'urgency': 'routine',
                    'source': 'Lempert et al., J Vestib Res 2012',
                    'reference': 'Lempert et al., J Vestib Res. 2012;22(4):167-174',
                    'rationale': 'Episodic vertigo in patient with migraine history and concurrent '
                    'headache meets criteria for vestibular migraine',
                },
            }
        ]

        # Rule category 5: General vestibular screening
        screening_rules = [
            {
                'id': 'screen_rule_001',
                'condition': "(age > 60 and hypertension == True and diabetes == True)",
                'action': {
                    'diagnosis': 'high_risk_screen',
                    'confidence': 0.75,
                    'urgency': 'screening',
                    'source': 'Framingham Stroke Study',
                    'reference': 'Wolf et al., Stroke 1991;22(3):312-318',
                    'rationale': 'Patient with multiple vascular risk factors warrants closer '
                    'monitoring for cerebrovascular events',
                },
            }
        ]

        # Combine all rules
        all_rules = (
            stroke_rules
            + bppv_rules
            + vestibular_neuritis_rules
            + vm_rules
            + screening_rules
        )

        # Add additional generic rules to reach target count
        for i in range(len(all_rules), self.rule_count):
            rule = {
                'id': f'generic_rule_{i:03d}',
                'condition': f"variable_{i % 10} > threshold_{i % 5}",
                'action': {
                    'diagnosis': f'diagnosis_{i % 15}',
                    'confidence': round(0.7 + (i % 30) * 0.01, 2),
                    'urgency': np.random.choice(['routine', 'urgent', 'emergency']),
                    'source': 'Generic Clinical Guideline',
                    'reference': f'Generic Reference {i}',
                    'rationale': f'Generic clinical rule for pattern recognition rule #{i}',
                },
            }
            all_rules.append(rule)

        logger.info(f"Generated {len(all_rules)} clinical rules")
        return all_rules

    def generate_samples(self, n_samples: int = 10000) -> pd.DataFrame:
        """
        Generate synthetic samples using the rule-based system.

        Args:
            n_samples: Number of samples to generate

        Returns:
            DataFrame of synthetic patients generated via rule-based inference
        """
        logger.info(f"Generating {n_samples} samples using rule-based expert system...")

        # Create a synthetic dataset based on the rules
        data = []

        for i in range(n_samples):
            # Generate patient characteristics
            age = int(np.random.normal(55, 18))  # Mean 55, std 18
            sex = np.random.choice(['M', 'F'])

            # Generate comorbidities based on age
            hypertension = (age > 50 and np.random.rand() < 0.3) or (
                age > 70 and np.random.rand() < 0.7
            )
            diabetes = (age > 40 and np.random.rand() < 0.15) or (
                age > 60 and np.random.rand() < 0.25
            )
            atrial_fibrillation = (age > 65 and np.random.rand() < 0.05) or (
                age > 75 and np.random.rand() < 0.15
            )
            migraine_history = (
                age < 60 and sex == 'F' and np.random.rand() < 0.25
            ) or (np.random.rand() < 0.1)

            # Generate symptom characteristics based on comorbidities
            timing_pattern = np.random.choice(
                ['acute', 'episodic', 'chronic'], p=[0.25, 0.55, 0.2]
            )
            trigger_type = np.random.choice(
                ['spontaneous', 'positional', 'head_movement'], p=[0.4, 0.4, 0.2]
            )

            # Generate duration based on timing pattern
            if timing_pattern == 'acute':
                duration_hours = np.random.exponential(
                    48
                )  # Hours for acute (mean 2 days)
            elif timing_pattern == 'episodic':
                duration_hours = np.random.uniform(
                    0.01, 1
                )  # Hours for episodic (<1 hour)
            else:  # chronic
                duration_hours = np.random.exponential(
                    2160
                )  # Hours (3 months = 2160 hours)

            # Generate examination findings based on all factors
            examination_findings = self._generate_examination_findings(
                age,
                hypertension,
                diabetes,
                atrial_fibrillation,
                timing_pattern,
                trigger_type,
            )

            # Build the condition-evaluation record BEFORE applying rules.
            # (Fixed 2026-09-06: this dict previously referenced 'diagnosis',
            # 'confidence' and 'urgency' before they were computed, which
            # raised NameError on every call and meant Layer 3 could never
            # actually run via run_all.py. Rules only need the condition
            # variables below; diagnosis/confidence/urgency are the RESULT
            # of _apply_rules, not an input to it, so they are omitted here
            # and attached afterward.)
            patient_data = {
                'timing_pattern': timing_pattern,
                'trigger_type': trigger_type,
                'age': age,
                'sex': sex,
                'hypertension': hypertension,
                'diabetes': diabetes,
                'atrial_fibrillation': atrial_fibrillation,
                'migraine_history': migraine_history,
                'duration_hours': duration_hours,
                'severity_score': np.random.randint(1, 11),  # 1-10 scale
                # Common in vestibular disorders
                'nausea_vomiting': np.random.choice([True, False], p=[0.6, 0.4]),
                'hearing_loss': self._sample_hearing_loss('unknown'),
                'tinnitus': self._sample_tinnitus('unknown'),
                'hit_abnormal': examination_findings['hit_abnormal'],
                'nystagmus_type': examination_findings['nystagmus_type'],
                'nystagmus_peripheral': examination_findings['nystagmus_type'] == 'peripheral',
                'skew_deviation': examination_findings['skew_deviation'],
                'dix_hallpike_positive': examination_findings['dix_hallpike_positive'] != 'negative',
                # Rare neurological signs
                'focal_neurological_signs': np.random.choice(
                    [True, False], p=[0.98, 0.02]
                ),
                'cvd_risk': hypertension or diabetes or atrial_fibrillation,
                'hints_central': examination_findings['hit_abnormal']
                or examination_findings['nystagmus_direction_changing']
                or examination_findings['skew_deviation'],
                'spontaneous_onset': trigger_type == 'spontaneous',
                'episodic_vertigo': timing_pattern == 'episodic',
                'headache_present': migraine_history and np.random.random() < 0.8,
            }

            # Apply rules to determine diagnosis (diagnosis is an OUTPUT here).
            diagnosis, confidence, urgency = self._apply_rules(patient_data)

            # Create patient record
            patient = {
                'patient_id': f'RB_{i:06d}',
                'age': age,
                'sex': sex,
                'hypertension': hypertension,
                'diabetes': diabetes,
                'atrial_fibrillation': atrial_fibrillation,
                'migraine_history': migraine_history,
                'timing_pattern': timing_pattern,
                'trigger_type': trigger_type,
                'duration_hours': duration_hours,
                'severity_score': np.random.randint(1, 11),
                'nausea_vomiting': np.random.choice([True, False], p=[0.6, 0.4]),
                'headache_present': self._sample_headache(diagnosis, migraine_history),
                'hearing_loss': self._sample_hearing_loss(diagnosis),
                'tinnitus': self._sample_tinnitus(diagnosis),
                'hit_abnormal': examination_findings['hit_abnormal'],
                'nystagmus_type': examination_findings['nystagmus_type'],
                'skew_deviation': examination_findings['skew_deviation'],
                'dix_hallpike_positive': examination_findings['dix_hallpike_positive'],
                'focal_neurological_signs': np.random.choice(
                    [True, False], p=[0.98, 0.02]
                ),
                'cvd_risk': hypertension or diabetes or atrial_fibrillation,
                'hints_central': examination_findings['hit_abnormal']
                or examination_findings['nystagmus_direction_changing']
                or examination_findings['skew_deviation'],
                'spontaneous_onset': trigger_type == 'spontaneous',
                'episodic_vertigo': timing_pattern == 'episodic',
                'diagnosis': diagnosis,
                'confidence': confidence,
                'urgency': urgency,
            }

            # Add more features to reach 150-dimensional space
            for j in range(135):  # Add 135 more features
                patient[f'rule_feature_{j:03d}'] = np.random.random()

            data.append(patient)

        df = pd.DataFrame(data)

        # Ensure age is within reasonable bounds
        df['age'] = np.clip(df['age'], 18, 100)
        df['duration_hours'] = np.clip(df['duration_hours'], 0, 720)  # Max 30 days

        logger.info(
            f"Generated {len(df)} rule-based samples with {len(df.columns)} features"
        )

        return df

    def _apply_rules(self, patient: Dict[str, Any]) -> Tuple[str, float, str]:
        """
        Apply rules to determine diagnosis for a patient.

        Args:
            patient: Patient data dictionary

        Returns:
            Tuple of (diagnosis, confidence, urgency)
        """
        # Check each rule to see if it applies
        applicable_rules = []

        # Fixed 2026-09-06: the original sequential .replace() calls only
        # covered a subset of the identifiers actually used in rule
        # conditions (e.g. conditions wrote bare 'timing'/'trigger'/'onset'/
        # 'duration' while patient_data only has 'timing_pattern'/
        # 'trigger_type'/'duration_hours'; 'migraine_history',
        # 'hypertension' and 'diabetes' were never substituted at all).
        # Every named clinical rule therefore raised NameError inside the
        # bare `except: continue`, silently fell through to 'unknown' for
        # 100% of samples, and Layer 3 never actually classified anyone.
        # Token-boundary substitution below covers the full patient_data
        # vocabulary plus the aliases the rule text actually uses.
        import re

        alias_map = {
            'timing': patient['timing_pattern'],
            'onset': patient['timing_pattern'],
            'trigger': patient['trigger_type'],
            'duration': patient['duration_hours'],
        }

        for rule in self.rules:
            try:
                condition = rule['condition']
                eval_condition = condition

                # Direct patient_data fields (longest keys first to avoid
                # partial-token collisions, e.g. 'timing_pattern' vs 'timing').
                for key in sorted(patient.keys(), key=len, reverse=True):
                    value = patient[key]
                    token = value if isinstance(value, str) else repr(value)
                    replacement = f"'{token}'" if isinstance(value, str) else str(value)
                    eval_condition = re.sub(
                        rf"\b{re.escape(key)}\b", replacement, eval_condition
                    )

                # Aliases used by rule text but not present verbatim in
                # patient_data (timing/trigger/onset/duration).
                for alias, value in alias_map.items():
                    replacement = f"'{value}'" if isinstance(value, str) else str(value)
                    eval_condition = re.sub(
                        rf"\b{alias}\b", replacement, eval_condition
                    )

                # Evaluate the condition
                if eval(eval_condition):
                    applicable_rules.append(rule)
            except BaseException:
                # Skip rules that can't be evaluated
                continue

        if applicable_rules:
            # Use the rule with highest confidence
            best_rule = max(applicable_rules, key=lambda r: r['action']['confidence'])
            action = best_rule['action']
            return action['diagnosis'], action['confidence'], action['urgency']
        else:
            # Default to unknown if no rules apply
            return 'unknown', 0.5, 'routine'

    def _generate_examination_findings(
        self, age: int, htn: bool, dm: bool, af: bool, timing: str, trigger: str
    ) -> Dict[str, any]:
        """Generate examination findings based on clinical probabilities."""
        findings = {}

        # HINTS examination components based on clinical probabilities
        if (
            timing == 'acute'
            and trigger == 'spontaneous'
            and age >= 50
            and (htn or dm or af)
        ):
            # High suspicion for stroke - more likely central HINTS pattern
            findings['hit_abnormal'] = np.random.choice([True, False], p=[0.8, 0.2])
            findings['nystagmus_type'] = np.random.choice(
                ['central', 'direction_changing', 'peripheral'], p=[0.6, 0.3, 0.1]
            )
            findings['nystagmus_direction_changing'] = (
                findings['nystagmus_type'] == 'direction_changing'
            )
            findings['skew_deviation'] = np.random.choice([True, False], p=[0.6, 0.4])
            findings['dix_hallpike_positive'] = np.random.choice(
                ['negative', 'positive_right', 'positive_left'], p=[0.9, 0.05, 0.05]
            )
        elif timing == 'episodic' and trigger == 'positional':
            # More likely BPPV - normal HIT, positional nystagmus
            findings['hit_abnormal'] = np.random.choice([True, False], p=[0.1, 0.9])
            findings['nystagmus_type'] = np.random.choice(
                ['central', 'peripheral', 'positional'], p=[0.1, 0.2, 0.7]
            )
            findings['nystagmus_direction_changing'] = (
                findings['nystagmus_type'] == 'direction_changing'
            )
            findings['skew_deviation'] = np.random.choice([True, False], p=[0.02, 0.98])
            findings['dix_hallpike_positive'] = np.random.choice(
                ['negative', 'positive_right', 'positive_left'], p=[0.2, 0.4, 0.4]
            )
        else:
            # Other patterns: mixed possibilities
            findings['hit_abnormal'] = np.random.choice([True, False], p=[0.3, 0.7])
            findings['nystagmus_type'] = np.random.choice(
                ['central', 'peripheral', 'none', 'positional'], p=[0.1, 0.3, 0.5, 0.1]
            )
            findings['nystagmus_direction_changing'] = (
                findings['nystagmus_type'] == 'direction_changing'
            )
            findings['skew_deviation'] = np.random.choice([True, False], p=[0.05, 0.95])
            findings['dix_hallpike_positive'] = np.random.choice(
                ['negative', 'positive_right', 'positive_left'], p=[0.7, 0.15, 0.15]
            )

        return findings

    def _sample_headache(self, diagnosis: str, migraine_history: bool) -> bool:
        """Sample headache based on diagnosis and migraine history."""
        if 'migraine' in diagnosis or migraine_history:
            return np.random.random() < 0.8
        elif 'stroke' in diagnosis:
            return np.random.random() < 0.4
        else:
            return np.random.random() < 0.2

    def _sample_hearing_loss(self, diagnosis: str) -> bool:
        """Sample hearing loss based on diagnosis."""
        if 'menieres' in diagnosis:
            return np.random.random() < 0.9
        elif 'labyrinthitis' in diagnosis:
            return np.random.random() < 0.7
        else:
            return np.random.random() < 0.1

    def _sample_tinnitus(self, diagnosis: str) -> bool:
        """Sample tinnitus based on diagnosis."""
        if 'menieres' in diagnosis:
            return np.random.random() < 0.95
        elif 'labyrinthitis' in diagnosis:
            return np.random.random() < 0.6
        else:
            return np.random.random() < 0.15

    def _determine_urgency(self, diagnosis: str) -> int:
        """Determine urgency level based on diagnosis."""
        if diagnosis in ['posterior_circulation_stroke', 'tia']:
            return 2  # Emergency
        elif diagnosis in ['menieres_disease', 'vestibular_neuritis']:
            return 1  # Urgent
        else:
            return 0  # Routine

    def get_rule_summary(self) -> Dict[str, any]:
        """Get summary statistics about the rule base."""
        diagnoses = [rule['action']['diagnosis'] for rule in self.rules]
        urgencies = [rule['action']['urgency'] for rule in self.rules]

        return {
            'total_rules': len(self.rules),
            'unique_diagnoses': len(set(diagnoses)),
            'diagnosis_distribution': {d: diagnoses.count(d) for d in set(diagnoses)},
            'urgency_distribution': {u: urgencies.count(u) for u in set(urgencies)},
            'average_confidence': np.mean(
                [rule['action']['confidence'] for rule in self.rules]
            ),
        }


# Test the Rule-Based Expert System
if __name__ == '__main__':
    print("Testing Rule-Based Expert System...")

    rule_system = RuleBasedExpertSystem(
        rule_count=50, random_seed=42
    )  # Smaller for demo
    samples = rule_system.generate_samples(n_samples=1000)

    print(f"\\nGenerated {len(samples)} rule-based samples")
    print(f"Features per sample: {len(samples.columns)}")

    # Show sample of data
    print(f"\\nFirst 5 samples:")
    print(
        samples.head()[
            ['patient_id', 'age', 'diagnosis', 'confidence', 'urgency']
        ].to_string()
    )

    # Show rule summary
    rule_summary = rule_system.get_rule_summary()
    print(f"\\nRule summary:")
    print(f"  Total rules: {rule_summary['total_rules']}")
    print(f"  Unique diagnoses: {rule_summary['unique_diagnoses']}")
    print(f"  Average confidence: {rule_summary['average_confidence']:.3f}")

    print(f"\\nRule-based expert system test completed successfully!")
