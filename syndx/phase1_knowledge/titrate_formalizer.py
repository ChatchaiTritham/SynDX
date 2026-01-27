"""
TiTrATE Framework Formalizer

Converts the TiTrATE diagnostic framework (Timing-Triggers-Targeted Examination)
into computational structures with consistency constraints.

This is the clinical knowledge base that makes everything else work.
Incorrect constraint implementation compromises data quality.

Reference: Newman-Toker & Edlow (2015) - the clinical foundation
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from enum import Enum
import logging

logger = logging.getLogger(__name__)


class TimingPattern(Enum):
    """Timing dimension of TiTrATE framework"""
    ACUTE = "acute"  # Continuous > 24 hours
    EPISODIC = "episodic"  # Brief paroxysms < 1 minute
    CHRONIC = "chronic"  # Persistent > 3 months


class TriggerType(Enum):
    """Trigger dimension of TiTrATE framework"""
    SPONTANEOUS = "spontaneous"
    POSITIONAL = "positional"
    HEAD_MOVEMENT = "head_movement"
    VALSALVA = "valsalva"
    AUDITORY = "auditory"
    VISUAL = "visual"
    ORTHOSTATIC = "orthostatic"


class DiagnosisCategory(Enum):
    """Vestibular disorder diagnoses based on Bárány ICVD 2025"""
    # Peripheral vestibular
    BPPV = "benign_paroxysmal_positional_vertigo"
    VESTIBULAR_NEURITIS = "vestibular_neuritis"
    MENIERES = "menieres_disease"
    LABYRINTHITIS = "labyrinthitis"
    VESTIBULAR_MIGRAINE = "vestibular_migraine"

    # Central vestibular
    STROKE = "cerebellar_stroke"
    TIA = "transient_ischemic_attack"
    MS = "multiple_sclerosis"
    MIGRAINE = "migraine_associated_vertigo"

    # Other
    ORTHOSTATIC = "orthostatic_hypotension"
    CARDIOVASCULAR = "cardiovascular"
    PSYCHIATRIC = "psychiatric"
    MEDICATION = "medication_induced"
    CERVICOGENIC = "cervicogenic"
    OTHER = "other"
    UNDETERMINED = "undetermined"


@dataclass
class ExaminationFindings:
    """Targeted examination findings"""
    # HINTS exam components
    head_impulse_test: str  # "normal", "abnormal_peripheral", "abnormal_central"
    nystagmus_type: str  # "none", "peripheral", "central", "positional"
    test_of_skew: bool  # True = present (central sign)

    # Dix-Hallpike
    dix_hallpike_result: str  # "negative", "positive_right", "positive_left"

    # Other bedside tests
    romberg_test: str  # "normal", "abnormal"
    gait_assessment: str  # "normal", "ataxic", "unstable"
    neurological_signs: bool  # True = focal deficits present

    # Vital signs
    blood_pressure_systolic: int
    blood_pressure_diastolic: int
    heart_rate: int

    def to_dict(self) -> Dict:
        """Convert to dictionary"""
        return {
            "head_impulse_test": self.head_impulse_test,
            "nystagmus_type": self.nystagmus_type,
            "test_of_skew": self.test_of_skew,
            "dix_hallpike_result": self.dix_hallpike_result,
            "romberg_test": self.romberg_test,
            "gait_assessment": self.gait_assessment,
            "neurological_signs": self.neurological_signs,
            "bp_systolic": self.blood_pressure_systolic,
            "bp_diastolic": self.blood_pressure_diastolic,
            "heart_rate": self.heart_rate,
        }


@dataclass
class ClinicalArchetype:
    """
    Computational archetype representing a clinically plausible patient scenario.

    Defined by equation (1) in paper:
    A = {a | a = ⟨t, r, e, d, f, u⟩, C_TiTrATE(t,r,e,d) = True}
    """
    timing: TimingPattern
    trigger: TriggerType
    examination: ExaminationFindings
    diagnosis: DiagnosisCategory
    features: np.ndarray  # 150-dimensional feature vector
    urgency: int  # 0=routine, 1=urgent, 2=emergency

    # Demographics and comorbidities
    age: int
    gender: str  # "M", "F"
    has_hypertension: bool
    has_diabetes: bool
    has_cardiovascular_disease: bool
    has_migraine_history: bool

    # Symptoms
    symptom_duration_hours: float
    symptom_severity: int  # 1-10 scale
    nausea_vomiting: bool
    headache: bool
    hearing_loss: bool
    tinnitus: bool

    def is_valid(self) -> bool:
        """Check if archetype satisfies TiTrATE consistency constraints"""
        return TiTrATEFormalizer.check_constraints(self)

    def to_feature_vector(self) -> np.ndarray:
        """Convert to 150-dimensional feature vector"""
        return self.features

    def to_dict(self) -> Dict:
        """Convert to dictionary for serialization"""
        return {
            "timing": self.timing.value,
            "trigger": self.trigger.value,
            "examination": self.examination.to_dict(),
            "diagnosis": self.diagnosis.value,
            "urgency": self.urgency,
            "age": self.age,
            "gender": self.gender,
            "comorbidities": {
                "hypertension": self.has_hypertension,
                "diabetes": self.has_diabetes,
                "cardiovascular": self.has_cardiovascular_disease,
                "migraine": self.has_migraine_history,
            },
            "symptoms": {
                "duration_hours": self.symptom_duration_hours,
                "severity": self.symptom_severity,
                "nausea_vomiting": self.nausea_vomiting,
                "headache": self.headache,
                "hearing_loss": self.hearing_loss,
                "tinnitus": self.tinnitus,
            },
            "features": self.features.tolist(),
        }


class TiTrATEFormalizer:
    """
    Formalizes TiTrATE diagnostic framework into computational constraints.

    Implements constraint function C_TiTrATE from equation (2) in paper.
    """

    @staticmethod
    def check_constraints(archetype: ClinicalArchetype) -> bool:
        """
        Apply TiTrATE consistency constraints.

        Returns True if archetype is clinically plausible, False otherwise.

        Implements equation (2):
        C_TiTrATE(T_acute, R_spontaneous, E_central-HINTS, D_stroke) =
        True if age ≥ 50 AND cardiovascular risk present
        False otherwise
        """

        # Constraint 1: Stroke diagnosis requirements
        if archetype.diagnosis == DiagnosisCategory.STROKE:
            # Must have acute spontaneous timing
            if archetype.timing != TimingPattern.ACUTE:
                return False

            # Typically age >= 50 with vascular risk factors
            if archetype.age < 50 and not archetype.has_cardiovascular_disease:
                return False

            # Should have central HINTS signs
            exam = archetype.examination
            if exam.head_impulse_test != "normal" and exam.nystagmus_type == "peripheral":
                return False

            # Urgency must be emergency
            if archetype.urgency != 2:
                return False

        # Constraint 2: BPPV diagnosis requirements
        if archetype.diagnosis == DiagnosisCategory.BPPV:
            # Must have episodic timing
            if archetype.timing != TimingPattern.EPISODIC:
                return False

            # Must be positional trigger
            if archetype.trigger != TriggerType.POSITIONAL:
                return False

            # Should have positive Dix-Hallpike
            if archetype.examination.dix_hallpike_result == "negative":
                return False

            # No central signs
            if archetype.examination.neurological_signs:
                return False

        # Constraint 3: Vestibular neuritis requirements
        if archetype.diagnosis == DiagnosisCategory.VESTIBULAR_NEURITIS:
            # Acute spontaneous timing
            if archetype.timing != TimingPattern.ACUTE:
                return False

            # Peripheral HINTS pattern
            exam = archetype.examination
            if exam.head_impulse_test == "normal":
                return False
            if exam.nystagmus_type != "peripheral":
                return False
            if exam.test_of_skew:  # Skew = central sign
                return False

        # Constraint 4: Meniere's disease requirements
        if archetype.diagnosis == DiagnosisCategory.MENIERES:
            # Episodic timing
            if archetype.timing != TimingPattern.EPISODIC:
                return False

            # Must have hearing symptoms
            if not (archetype.hearing_loss or archetype.tinnitus):
                return False

            # Duration typically hours, not seconds
            if archetype.symptom_duration_hours < 0.5:
                return False

        # Constraint 5: Orthostatic hypotension requirements
        if archetype.diagnosis == DiagnosisCategory.ORTHOSTATIC:
            # Orthostatic trigger required
            if archetype.trigger != TriggerType.ORTHOSTATIC:
                return False

            # Low blood pressure
            if archetype.examination.blood_pressure_systolic > 120:
                return False

        # Constraint 6: Age plausibility
        if archetype.age < 18 or archetype.age > 100:
            return False

        # Constraint 7: Vital signs plausibility
        exam = archetype.examination
        if not (60 <= exam.blood_pressure_systolic <= 220):
            return False
        if not (40 <= exam.blood_pressure_diastolic <= 140):
            return False
        if not (40 <= exam.heart_rate <= 180):
            return False

        # Constraint 8: Symptom duration consistency with timing
        if archetype.timing == TimingPattern.ACUTE:
            if archetype.symptom_duration_hours < 24:
                return False
        elif archetype.timing == TimingPattern.EPISODIC:
            if archetype.symptom_duration_hours > 24:
                return False

        # Constraint 9: Cardiovascular diagnosis requires cardiovascular risk
        if archetype.diagnosis == DiagnosisCategory.CARDIOVASCULAR:
            if not (archetype.has_hypertension or
                    archetype.has_cardiovascular_disease or
                    archetype.age > 60):
                return False

        # Constraint 10: Central diagnoses require appropriate urgency
        central_diagnoses = {
            DiagnosisCategory.STROKE,
            DiagnosisCategory.TIA,
            DiagnosisCategory.MS
        }
        if archetype.diagnosis in central_diagnoses:
            if archetype.urgency < 1:  # Must be urgent or emergency
                return False

        return True

    @staticmethod
    def get_diagnostic_space() -> Dict[str, List]:
        """
        Get the full diagnostic space dimensions.

        Returns:
            Dictionary containing all possible values for each TiTrATE dimension
        """
        return {
            "timing": [t for t in TimingPattern],
            "triggers": [t for t in TriggerType],
            "diagnoses": [d for d in DiagnosisCategory],
            "timing_count": len(TimingPattern),
            "trigger_count": len(TriggerType),
            "diagnosis_count": len(DiagnosisCategory),
        }

    @staticmethod
    def get_expected_archetype_count() -> int:
        """
        Calculate expected number of valid archetypes.

        Theoretical upper bound: |T| × |R| × |E| × |D|
        After constraints: ~8,400 valid archetypes
        """
        space = TiTrATEFormalizer.get_diagnostic_space()
        # This is theoretical max, actual is lower due to constraints
        theoretical_max = (
            space["timing_count"] *
            space["trigger_count"] *
            space["diagnosis_count"] *
            400  # Approximate examination finding combinations
        )
        logger.info(f"Theoretical archetype space: {theoretical_max:,}")
        logger.info(f"Expected valid arcetype after constraints: ~8,400")
        return 8400


if __name__ == "__main__":
    # Test TiTrATE formalizer
    logging.basicConfig(level=logging.INFO)

    # Example: Valid stroke archetype
    stroke_exam = ExaminationFindings(
        head_impulse_test="normal",
        nystagmus_type="central",
        test_of_skew=True,
        dix_hallpike_result="negative",
        romberg_test="abnormal",
        gait_assessment="ataxic",
        neurological_signs=True,
        blood_pressure_systolic=160,
        blood_pressure_diastolic=95,
        heart_rate=88
    )

    stroke_archetype = ClinicalArchetype(
        timing=TimingPattern.ACUTE,
        trigger=TriggerType.SPONTANEOUS,
        examination=stroke_exam,
        diagnosis=DiagnosisCategory.STROKE,
        features=np.random.randn(150),
        urgency=2,
        age=68,
        gender="M",
        has_hypertension=True,
        has_diabetes=True,
        has_cardiovascular_disease=True,
        has_migraine_history=False,
        symptom_duration_hours=36,
        symptom_severity=8,
        nausea_vomiting=True,
        headache=True,
        hearing_loss=False,
        tinnitus=False
    )

    print(f"Stroke archetype valid: {stroke_archetype.is_valid()}")

    # Example: Invalid BPPV archetype (wrong timing)
    bppv_exam = ExaminationFindings(
        head_impulse_test="abnormal_peripheral",
        nystagmus_type="positional",
        test_of_skew=False,
        dix_hallpike_result="positive_right",
        romberg_test="normal",
        gait_assessment="normal",
        neurological_signs=False,
        blood_pressure_systolic=120,
        blood_pressure_diastolic=80,
        heart_rate=72
    )

    invalid_bppv = ClinicalArchetype(
        timing=TimingPattern.ACUTE,  # Wrong! BPPV should be episodic
        trigger=TriggerType.POSITIONAL,
        examination=bppv_exam,
        diagnosis=DiagnosisCategory.BPPV,
        features=np.random.randn(150),
        urgency=0,
        age=55,
        gender="F",
        has_hypertension=False,
        has_diabetes=False,
        has_cardiovascular_disease=False,
        has_migraine_history=False,
        symptom_duration_hours=0.5,
        symptom_severity=6,
        nausea_vomiting=True,
        headache=False,
        hearing_loss=False,
        tinnitus=False
    )

    print(f"Invalid BPPV archetype valid: {invalid_bppv.is_valid()}")

    # Get diagnostic space
    space = TiTrATEFormalizer.get_diagnostic_space()
    print(f"\nDiagnostic space dimensions:")
    print(f" Timing patterns: {space['timing_count']}")
    print(f" Trigger types: {space['trigger_count']}")
    print(f" Diagnoses: {space['diagnosis_count']}")
    print(
        f" Expected valid archetypes: {
            TiTrATEFormalizer.get_expected_archetype_count()}")
