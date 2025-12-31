"""
Archetype Generator

Generates 8,400 clinically plausible computational archetypes from
TiTrATE guidelines with constraint validation.
"""

import numpy as np
from typing import List, Dict, Tuple
import pandas as pd
from tqdm import tqdm
import logging
from itertools import product

from .titrate_formalizer import (
    TiTrATEFormalizer,
    ClinicalArchetype,
    ExaminationFindings,
    TimingPattern,
    TriggerType,
    DiagnosisCategory
)

logger = logging.getLogger(__name__)


class ArchetypeGenerator:
    """
    Generate computational archetypes from TiTrATE diagnostic framework.

    Produces ~8,400 constraint-validated archetypes representing
    clinically plausible patient scenarios.
    """

    def __init__(self, random_seed: int = 42):
        """
        Initialize archetype generator.

        Args:
            random_seed: Random seed for reproducibility
        """
        self.random_seed = random_seed
        np.random.seed(random_seed)
        self.archetypes: List[ClinicalArchetype] = []

    def generate_archetypes(self, n_target: int = 8400) -> List[ClinicalArchetype]:
        """
        Generate constraint-validated archetypes.

        Args:
            n_target: Target number of archetypes (default: 8400)

        Returns:
            List of valid ClinicalArchetype objects
        """
        logger.info(f"Generating {n_target} computational archetypes...")

        archetypes = []
        attempts = 0
        max_attempts = n_target * 10  # Safety limit

        pbar = tqdm(total=n_target, desc="Generating archetypes")

        while len(archetypes) < n_target and attempts < max_attempts:
            archetype = self._generate_single_archetype()
            attempts += 1

            # Validate with TiTrATE constraints
            if archetype.is_valid():
                archetypes.append(archetype)
                pbar.update(1)

        pbar.close()

        acceptance_rate = len(archetypes) / attempts * 100
        logger.info(f"Generated {len(archetypes)} valid archetypes")
        logger.info(f"Acceptance rate: {acceptance_rate:.1f}%")
        logger.info(f"Total attempts: {attempts}")

        self.archetypes = archetypes
        return archetypes

    def _generate_single_archetype(self) -> ClinicalArchetype:
        """Generate a single archetype with random but plausible values"""

        # Sample basic dimensions
        timing = np.random.choice(list(TimingPattern))
        trigger = np.random.choice(list(TriggerType))
        diagnosis = np.random.choice(list(DiagnosisCategory))

        # Generate demographics
        age = self._sample_age(diagnosis)
        gender = np.random.choice(["M", "F"])

        # Generate comorbidities (age-dependent)
        has_hypertension = self._sample_hypertension(age)
        has_diabetes = self._sample_diabetes(age)
        has_cvd = self._sample_cardiovascular_disease(age, has_hypertension, has_diabetes)
        has_migraine = self._sample_migraine(age, gender)

        # Generate symptoms based on timing and diagnosis
        symptom_duration = self._sample_symptom_duration(timing, diagnosis)
        symptom_severity = np.random.randint(1, 11)
        nausea_vomiting = np.random.rand() < 0.6  # Common in vestibular disorders
        headache = self._sample_headache(diagnosis)
        hearing_loss = self._sample_hearing_loss(diagnosis)
        tinnitus = self._sample_tinnitus(diagnosis)

        # Generate examination findings
        examination = self._generate_examination(diagnosis, age, has_cvd)

        # Generate urgency based on diagnosis
        urgency = self._determine_urgency(diagnosis, examination)

        # Generate 150-dimensional feature vector
        features = self._generate_feature_vector(
            timing, trigger, diagnosis, age, gender,
            has_hypertension, has_diabetes, has_cvd, has_migraine,
            symptom_duration, symptom_severity,
            nausea_vomiting, headache, hearing_loss, tinnitus,
            examination
        )

        archetype = ClinicalArchetype(
            timing=timing,
            trigger=trigger,
            examination=examination,
            diagnosis=diagnosis,
            features=features,
            urgency=urgency,
            age=age,
            gender=gender,
            has_hypertension=has_hypertension,
            has_diabetes=has_diabetes,
            has_cardiovascular_disease=has_cvd,
            has_migraine_history=has_migraine,
            symptom_duration_hours=symptom_duration,
            symptom_severity=symptom_severity,
            nausea_vomiting=nausea_vomiting,
            headache=headache,
            hearing_loss=hearing_loss,
            tinnitus=tinnitus
        )

        return archetype

    def _sample_age(self, diagnosis: DiagnosisCategory) -> int:
        """Sample age based on diagnosis epidemiology"""
        if diagnosis == DiagnosisCategory.BPPV:
            # BPPV more common in older adults
            return int(np.random.normal(60, 15))
        elif diagnosis == DiagnosisCategory.STROKE:
            # Stroke typically older with vascular risk
            return int(np.random.normal(70, 10))
        elif diagnosis == DiagnosisCategory.VESTIBULAR_MIGRAINE:
            # Vestibular migraine typically younger
            return int(np.random.normal(40, 12))
        else:
            # General population
            return int(np.random.normal(55, 18))

    def _sample_hypertension(self, age: int) -> bool:
        """Sample hypertension based on age"""
        prob = 0.1 + (age - 18) * 0.008  # Increases with age
        return np.random.rand() < np.clip(prob, 0, 0.7)

    def _sample_diabetes(self, age: int) -> bool:
        """Sample diabetes based on age"""
        prob = 0.05 + (age - 18) * 0.005
        return np.random.rand() < np.clip(prob, 0, 0.4)

    def _sample_cardiovascular_disease(self, age: int, has_htn: bool, has_dm: bool) -> bool:
        """Sample cardiovascular disease based on age and risk factors"""
        base_prob = 0.05 + (age - 18) * 0.006
        if has_htn:
            base_prob *= 1.5
        if has_dm:
            base_prob *= 1.3
        return np.random.rand() < np.clip(base_prob, 0, 0.5)

    def _sample_migraine(self, age: int, gender: str) -> bool:
        """Sample migraine history (more common in women, younger)"""
        base_prob = 0.15 if gender == "F" else 0.08
        age_factor = max(0, 1 - (age - 30) * 0.01)
        prob = base_prob * age_factor
        return np.random.rand() < np.clip(prob, 0, 0.3)

    def _sample_symptom_duration(self, timing: TimingPattern, diagnosis: DiagnosisCategory) -> float:
        """Sample symptom duration based on timing pattern"""
        if timing == TimingPattern.ACUTE:
            # Continuous > 24 hours
            return 24 + np.random.exponential(24)
        elif timing == TimingPattern.EPISODIC:
            if diagnosis == DiagnosisCategory.BPPV:
                # BPPV: seconds to minutes
                return np.random.uniform(0.01, 0.5)  # 0.6-30 minutes
            elif diagnosis == DiagnosisCategory.MENIERES:
                # Meniere's: hours
                return np.random.uniform(0.5, 12)
            else:
                # General episodic: minutes to hours
                return np.random.uniform(0.1, 6)
        else:  # CHRONIC
            # Persistent > 3 months = 2160 hours
            return 2160 + np.random.exponential(1000)

    def _sample_headache(self, diagnosis: DiagnosisCategory) -> bool:
        """Sample headache presence based on diagnosis"""
        if diagnosis in [DiagnosisCategory.MIGRAINE, DiagnosisCategory.VESTIBULAR_MIGRAINE]:
            return np.random.rand() < 0.9
        elif diagnosis == DiagnosisCategory.STROKE:
            return np.random.rand() < 0.4
        else:
            return np.random.rand() < 0.3

    def _sample_hearing_loss(self, diagnosis: DiagnosisCategory) -> bool:
        """Sample hearing loss based on diagnosis"""
        if diagnosis == DiagnosisCategory.MENIERES:
            return np.random.rand() < 0.8
        elif diagnosis == DiagnosisCategory.LABYRINTHITIS:
            return np.random.rand() < 0.5
        else:
            return np.random.rand() < 0.1

    def _sample_tinnitus(self, diagnosis: DiagnosisCategory) -> bool:
        """Sample tinnitus based on diagnosis"""
        if diagnosis == DiagnosisCategory.MENIERES:
            return np.random.rand() < 0.9
        elif diagnosis in [DiagnosisCategory.LABYRINTHITIS, DiagnosisCategory.VESTIBULAR_NEURITIS]:
            return np.random.rand() < 0.3
        else:
            return np.random.rand() < 0.15

    def _generate_examination(self, diagnosis: DiagnosisCategory,
                            age: int, has_cvd: bool) -> ExaminationFindings:
        """Generate examination findings based on diagnosis"""

        # HINTS exam
        if diagnosis == DiagnosisCategory.STROKE:
            hit = "normal"  # Central = normal HIT
            nystagmus = np.random.choice(["central", "none"])
            skew = True
        elif diagnosis == DiagnosisCategory.VESTIBULAR_NEURITIS:
            hit = "abnormal_peripheral"
            nystagmus = "peripheral"
            skew = False
        elif diagnosis == DiagnosisCategory.BPPV:
            hit = np.random.choice(["normal", "abnormal_peripheral"])
            nystagmus = "positional"
            skew = False
        else:
            hit = np.random.choice(["normal", "abnormal_peripheral"])
            nystagmus = np.random.choice(["none", "peripheral", "central"])
            skew = np.random.rand() < 0.05

        # Dix-Hallpike
        if diagnosis == DiagnosisCategory.BPPV:
            dix_hallpike = np.random.choice(["positive_right", "positive_left"])
        else:
            dix_hallpike = np.random.choice(["negative"] * 8 + ["positive_right", "positive_left"])

        # Romberg and gait
        if diagnosis in [DiagnosisCategory.STROKE, DiagnosisCategory.MS]:
            romberg = "abnormal"
            gait = np.random.choice(["ataxic", "unstable"])
        else:
            romberg = np.random.choice(["normal"] * 7 + ["abnormal"])
            gait = np.random.choice(["normal"] * 8 + ["unstable", "ataxic"])

        # Neurological signs
        neuro_signs = diagnosis in [DiagnosisCategory.STROKE, DiagnosisCategory.TIA, DiagnosisCategory.MS]

        # Vital signs
        bp_systolic = self._sample_blood_pressure(age, has_cvd, diagnosis)
        bp_diastolic = int(bp_systolic * np.random.uniform(0.55, 0.70))
        heart_rate = int(np.random.normal(75, 12))

        return ExaminationFindings(
            head_impulse_test=hit,
            nystagmus_type=nystagmus,
            test_of_skew=skew,
            dix_hallpike_result=dix_hallpike,
            romberg_test=romberg,
            gait_assessment=gait,
            neurological_signs=neuro_signs,
            blood_pressure_systolic=bp_systolic,
            blood_pressure_diastolic=bp_diastolic,
            heart_rate=np.clip(heart_rate, 45, 180)
        )

    def _sample_blood_pressure(self, age: int, has_cvd: bool,
                              diagnosis: DiagnosisCategory) -> int:
        """Sample blood pressure based on risk factors"""
        if diagnosis == DiagnosisCategory.ORTHOSTATIC:
            # Low BP for orthostatic hypotension
            return int(np.random.normal(95, 10))
        elif has_cvd or age > 65:
            # Higher BP with cardiovascular disease
            return int(np.random.normal(145, 20))
        else:
            # Normal range
            return int(np.random.normal(120, 15))

    def _determine_urgency(self, diagnosis: DiagnosisCategory,
                          exam: ExaminationFindings) -> int:
        """Determine urgency level based on diagnosis and exam"""
        if diagnosis in [DiagnosisCategory.STROKE, DiagnosisCategory.TIA]:
            return 2  # Emergency
        elif diagnosis in [DiagnosisCategory.MS] or exam.neurological_signs:
            return 1  # Urgent
        else:
            return 0  # Routine

    def _generate_feature_vector(self, timing, trigger, diagnosis, age, gender,
                                has_htn, has_dm, has_cvd, has_migraine,
                                symptom_duration, symptom_severity,
                                nausea, headache, hearing_loss, tinnitus,
                                exam: ExaminationFindings) -> np.ndarray:
        """
        Generate 150-dimensional feature vector.

        Features are organized as:
        - Demographics (10 features)
        - Comorbidities (20 features)
        - Symptoms (30 features)
        - Examination findings (40 features)
        - TiTrATE dimensions (20 features)
        - Temporal patterns (10 features)
        - Vital signs (10 features)
        - Derived features (10 features)
        """

        features = np.zeros(150)
        idx = 0

        # Demographics (10)
        features[idx] = age / 100.0  # Normalized age
        features[idx + 1] = 1.0 if gender == "M" else 0.0
        features[idx + 2] = 1.0 if gender == "F" else 0.0
        idx += 10

        # Comorbidities (20)
        features[idx] = float(has_htn)
        features[idx + 1] = float(has_dm)
        features[idx + 2] = float(has_cvd)
        features[idx + 3] = float(has_migraine)
        idx += 20

        # Symptoms (30)
        features[idx] = np.log1p(symptom_duration) / 10.0  # Log-normalized duration
        features[idx + 1] = symptom_severity / 10.0
        features[idx + 2] = float(nausea)
        features[idx + 3] = float(headache)
        features[idx + 4] = float(hearing_loss)
        features[idx + 5] = float(tinnitus)
        idx += 30

        # Examination findings (40)
        # One-hot encode HINTS components
        features[idx] = 1.0 if exam.head_impulse_test == "normal" else 0.0
        features[idx + 1] = 1.0 if exam.head_impulse_test == "abnormal_peripheral" else 0.0
        features[idx + 2] = 1.0 if exam.head_impulse_test == "abnormal_central" else 0.0
        features[idx + 3] = 1.0 if exam.nystagmus_type == "none" else 0.0
        features[idx + 4] = 1.0 if exam.nystagmus_type == "peripheral" else 0.0
        features[idx + 5] = 1.0 if exam.nystagmus_type == "central" else 0.0
        features[idx + 6] = 1.0 if exam.nystagmus_type == "positional" else 0.0
        features[idx + 7] = float(exam.test_of_skew)
        features[idx + 8] = 1.0 if exam.dix_hallpike_result == "negative" else 0.0
        features[idx + 9] = 1.0 if exam.dix_hallpike_result == "positive_right" else 0.0
        features[idx + 10] = 1.0 if exam.dix_hallpike_result == "positive_left" else 0.0
        features[idx + 11] = float(exam.neurological_signs)
        idx += 40

        # TiTrATE dimensions (20)
        # One-hot encode timing
        for i, t in enumerate(TimingPattern):
            features[idx + i] = 1.0 if timing == t else 0.0
        idx += len(TimingPattern)

        # One-hot encode trigger
        for i, t in enumerate(TriggerType):
            features[idx + i] = 1.0 if trigger == t else 0.0
        idx += len(TriggerType)

        # One-hot encode diagnosis
        for i, d in enumerate(DiagnosisCategory):
            features[idx + i] = 1.0 if diagnosis == d else 0.0
        idx += len(DiagnosisCategory) - 10  # Adjust to fit in 20 features

        # Vital signs (10)
        features[idx] = exam.blood_pressure_systolic / 200.0  # Normalized
        features[idx + 1] = exam.blood_pressure_diastolic / 140.0
        features[idx + 2] = exam.heart_rate / 180.0
        idx += 10

        # Derived features (10) - interaction terms
        features[idx] = (age / 100.0) * float(has_cvd)  # Age × CVD
        features[idx + 1] = symptom_severity / 10.0 * float(nausea)  # Severity × nausea
        idx += 10

        return features[:150]  # Ensure exactly 150 dimensions

    def to_dataframe(self) -> pd.DataFrame:
        """Convert archetypes to pandas DataFrame"""
        data = []
        for arch in self.archetypes:
            row = arch.to_dict()
            data.append(row)

        return pd.DataFrame(data)

    def save_archetypes(self, filepath: str):
        """Save archetypes to file"""
        df = self.to_dataframe()
        if filepath.endswith('.csv'):
            df.to_csv(filepath, index=False)
        elif filepath.endswith('.json'):
            df.to_json(filepath, orient='records', indent=2)
        elif filepath.endswith('.parquet'):
            df.to_parquet(filepath, index=False)
        else:
            raise ValueError(f"Unsupported file format: {filepath}")

        logger.info(f"Saved {len(self.archetypes)} archetypes to {filepath}")

    def get_statistics(self) -> Dict:
        """Get summary statistics of generated archetypes"""
        if not self.archetypes:
            return {}

        df = self.to_dataframe()

        stats = {
            "total_archetypes": len(self.archetypes),
            "diagnosis_distribution": df['diagnosis'].value_counts().to_dict(),
            "timing_distribution": df['timing'].value_counts().to_dict(),
            "trigger_distribution": df['trigger'].value_counts().to_dict(),
            "age_stats": {
                "mean": df['age'].mean(),
                "std": df['age'].std(),
                "min": df['age'].min(),
                "max": df['age'].max()
            },
            "urgency_distribution": df['urgency'].value_counts().to_dict(),
        }

        return stats


if __name__ == "__main__":
    # Test archetype generator
    logging.basicConfig(level=logging.INFO)

    generator = ArchetypeGenerator(random_seed=42)
    archetypes = generator.generate_archetypes(n_target=100)  # Test with 100

    print("\n" + "="*60)
    print("ARCHETYPE GENERATION TEST")
    print("="*60)

    stats = generator.get_statistics()
    print(f"\nGenerated {stats['total_archetypes']} archetypes")

    print("\nDiagnosis distribution:")
    for dx, count in sorted(stats['diagnosis_distribution'].items(), key=lambda x: -x[1])[:5]:
        print(f"  {dx}: {count}")

    print("\nAge statistics:")
    for key, val in stats['age_stats'].items():
        print(f"  {key}: {val:.1f}")

    # Save test archetypes
    import os
    os.makedirs("../../data/archetypes", exist_ok=True)
    generator.save_archetypes("../../data/archetypes/test_archetypes.csv")

    print("\nTest completed successfully!")
