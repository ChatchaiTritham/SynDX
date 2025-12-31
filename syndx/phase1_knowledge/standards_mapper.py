"""
Standards Mapper

Maps clinical archetypes to healthcare IT standards:
- HL7 FHIR R4
- SNOMED CT
- LOINC
- OMOP CDM
"""

from typing import Dict, List, Optional
import logging
from datetime import datetime
from .titrate_formalizer import ClinicalArchetype, DiagnosisCategory

logger = logging.getLogger(__name__)


class StandardsMapper:
    """
    Map clinical archetypes to healthcare IT standards.

    Provides code mappings for:
    - SNOMED CT (diagnosis codes)
    - LOINC (examination codes)
    - FHIR resource structures
    - OMOP CDM concepts
    """

    # SNOMED CT codes for vestibular disorders
    SNOMED_DIAGNOSIS_CODES = {
        DiagnosisCategory.BPPV: "44446003",
        DiagnosisCategory.VESTIBULAR_NEURITIS: "128722006",
        DiagnosisCategory.MENIERES: "13445001",
        DiagnosisCategory.LABYRINTHITIS: "75112008",
        DiagnosisCategory.VESTIBULAR_MIGRAINE: "445073001",
        DiagnosisCategory.STROKE: "230690007",  # Cerebellar stroke
        DiagnosisCategory.TIA: "266257000",
        DiagnosisCategory.MS: "24700007",
        DiagnosisCategory.MIGRAINE: "37796009",
        DiagnosisCategory.ORTHOSTATIC: "28651003",
        DiagnosisCategory.CARDIOVASCULAR: "49601007",  # Cardiovascular disorder
        DiagnosisCategory.PSYCHIATRIC: "74732009",  # Psychiatric disorder
        DiagnosisCategory.MEDICATION: "473010000",  # Medication side effect
        DiagnosisCategory.CERVICOGENIC: "202708005",  # Cervicogenic dizziness
        DiagnosisCategory.OTHER: "404684003",  # Clinical finding
        DiagnosisCategory.UNDETERMINED: "261665006",  # Unknown
    }

    # SNOMED CT codes for symptoms
    SNOMED_SYMPTOM_CODES = {
        "vertigo": "399153001",
        "dizziness": "404640003",
        "nausea": "422587007",
        "vomiting": "422400008",
        "headache": "25064002",
        "hearing_loss": "343087000",
        "tinnitus": "60862001",
        "nystagmus": "563001",
    }

    # LOINC codes for examinations
    LOINC_EXAM_CODES = {
        "hints_exam": "72107-6",
        "head_impulse_test": "72108-4",
        "dix_hallpike": "72109-2",
        "romberg_test": "72110-0",
        "gait_assessment": "72111-8",
        "blood_pressure_systolic": "8480-6",
        "blood_pressure_diastolic": "8462-4",
        "heart_rate": "8867-4",
    }

    # OMOP CDM concept IDs (standard concepts)
    OMOP_CONCEPT_IDS = {
        "bppv": 4226408,
        "vestibular_neuritis": 4230254,
        "menieres": 4027384,
        "stroke": 4043731,
        "vertigo": 4236484,
        "dizziness": 4223659,
        "blood_pressure": 3004249,
    }

    def __init__(self):
        """Initialize standards mapper"""
        pass

    def map_to_snomed(self, archetype: ClinicalArchetype) -> Dict[str, str]:
        """
        Map archetype to SNOMED CT codes.

        Args:
            archetype: ClinicalArchetype to map

        Returns:
            Dictionary of SNOMED CT codes
        """
        codes = {}

        # Primary diagnosis code
        codes["primary_diagnosis"] = self.SNOMED_DIAGNOSIS_CODES.get(
            archetype.diagnosis,
            "404684003"  # Default: Clinical finding
        )

        # Symptom codes
        symptom_codes = []
        if archetype.nausea_vomiting:
            symptom_codes.append(self.SNOMED_SYMPTOM_CODES["nausea"])
            symptom_codes.append(self.SNOMED_SYMPTOM_CODES["vomiting"])
        if archetype.headache:
            symptom_codes.append(self.SNOMED_SYMPTOM_CODES["headache"])
        if archetype.hearing_loss:
            symptom_codes.append(self.SNOMED_SYMPTOM_CODES["hearing_loss"])
        if archetype.tinnitus:
            symptom_codes.append(self.SNOMED_SYMPTOM_CODES["tinnitus"])

        # Always include primary symptom
        symptom_codes.append(self.SNOMED_SYMPTOM_CODES["vertigo"])

        codes["symptoms"] = symptom_codes

        # Examination finding codes
        if archetype.examination.nystagmus_type != "none":
            codes["nystagmus"] = self.SNOMED_SYMPTOM_CODES["nystagmus"]

        return codes

    def map_to_loinc(self, archetype: ClinicalArchetype) -> Dict[str, str]:
        """
        Map examination findings to LOINC codes.

        Args:
            archetype: ClinicalArchetype to map

        Returns:
            Dictionary of LOINC codes
        """
        codes = {}

        exam = archetype.examination

        # HINTS exam
        codes["hints_exam"] = self.LOINC_EXAM_CODES["hints_exam"]
        codes["head_impulse_test"] = self.LOINC_EXAM_CODES["head_impulse_test"]

        # Dix-Hallpike
        codes["dix_hallpike"] = self.LOINC_EXAM_CODES["dix_hallpike"]

        # Romberg
        codes["romberg_test"] = self.LOINC_EXAM_CODES["romberg_test"]

        # Gait
        codes["gait_assessment"] = self.LOINC_EXAM_CODES["gait_assessment"]

        # Vital signs
        codes["bp_systolic"] = self.LOINC_EXAM_CODES["blood_pressure_systolic"]
        codes["bp_diastolic"] = self.LOINC_EXAM_CODES["blood_pressure_diastolic"]
        codes["heart_rate"] = self.LOINC_EXAM_CODES["heart_rate"]

        return codes

    def map_to_fhir_condition(self, archetype: ClinicalArchetype,
                             patient_id: str = "patient-001") -> Dict:
        """
        Map to FHIR Condition resource (R4).

        Args:
            archetype: ClinicalArchetype to map
            patient_id: FHIR patient ID

        Returns:
            FHIR Condition resource as dictionary
        """
        snomed_codes = self.map_to_snomed(archetype)

        condition = {
            "resourceType": "Condition",
            "id": f"condition-{patient_id}",
            "clinicalStatus": {
                "coding": [{
                    "system": "http://terminology.hl7.org/CodeSystem/condition-clinical",
                    "code": "active",
                    "display": "Active"
                }]
            },
            "verificationStatus": {
                "coding": [{
                    "system": "http://terminology.hl7.org/CodeSystem/condition-ver-status",
                    "code": "confirmed",
                    "display": "Confirmed"
                }]
            },
            "category": [{
                "coding": [{
                    "system": "http://terminology.hl7.org/CodeSystem/condition-category",
                    "code": "encounter-diagnosis",
                    "display": "Encounter Diagnosis"
                }]
            }],
            "severity": {
                "coding": [{
                    "system": "http://snomed.info/sct",
                    "code": self._severity_code(archetype.symptom_severity),
                    "display": self._severity_display(archetype.symptom_severity)
                }]
            },
            "code": {
                "coding": [{
                    "system": "http://snomed.info/sct",
                    "code": snomed_codes["primary_diagnosis"],
                    "display": archetype.diagnosis.value.replace("_", " ").title()
                }],
                "text": archetype.diagnosis.value.replace("_", " ").title()
            },
            "subject": {
                "reference": f"Patient/{patient_id}"
            },
            "onsetDateTime": datetime.now().isoformat(),
            "recordedDate": datetime.now().isoformat(),
        }

        return condition

    def map_to_fhir_observation(self, archetype: ClinicalArchetype,
                               patient_id: str = "patient-001") -> List[Dict]:
        """
        Map examination findings to FHIR Observation resources.

        Args:
            archetype: ClinicalArchetype to map
            patient_id: FHIR patient ID

        Returns:
            List of FHIR Observation resources
        """
        loinc_codes = self.map_to_loinc(archetype)
        observations = []
        exam = archetype.examination

        # Blood pressure observation
        bp_obs = {
            "resourceType": "Observation",
            "id": f"bp-{patient_id}",
            "status": "final",
            "category": [{
                "coding": [{
                    "system": "http://terminology.hl7.org/CodeSystem/observation-category",
                    "code": "vital-signs",
                    "display": "Vital Signs"
                }]
            }],
            "code": {
                "coding": [{
                    "system": "http://loinc.org",
                    "code": "85354-9",
                    "display": "Blood pressure panel"
                }]
            },
            "subject": {"reference": f"Patient/{patient_id}"},
            "effectiveDateTime": datetime.now().isoformat(),
            "component": [
                {
                    "code": {
                        "coding": [{
                            "system": "http://loinc.org",
                            "code": loinc_codes["bp_systolic"],
                            "display": "Systolic blood pressure"
                        }]
                    },
                    "valueQuantity": {
                        "value": exam.blood_pressure_systolic,
                        "unit": "mmHg",
                        "system": "http://unitsofmeasure.org",
                        "code": "mm[Hg]"
                    }
                },
                {
                    "code": {
                        "coding": [{
                            "system": "http://loinc.org",
                            "code": loinc_codes["bp_diastolic"],
                            "display": "Diastolic blood pressure"
                        }]
                    },
                    "valueQuantity": {
                        "value": exam.blood_pressure_diastolic,
                        "unit": "mmHg",
                        "system": "http://unitsofmeasure.org",
                        "code": "mm[Hg]"
                    }
                }
            ]
        }
        observations.append(bp_obs)

        # Heart rate observation
        hr_obs = {
            "resourceType": "Observation",
            "id": f"hr-{patient_id}",
            "status": "final",
            "category": [{
                "coding": [{
                    "system": "http://terminology.hl7.org/CodeSystem/observation-category",
                    "code": "vital-signs",
                    "display": "Vital Signs"
                }]
            }],
            "code": {
                "coding": [{
                    "system": "http://loinc.org",
                    "code": loinc_codes["heart_rate"],
                    "display": "Heart rate"
                }]
            },
            "subject": {"reference": f"Patient/{patient_id}"},
            "effectiveDateTime": datetime.now().isoformat(),
            "valueQuantity": {
                "value": exam.heart_rate,
                "unit": "beats/minute",
                "system": "http://unitsofmeasure.org",
                "code": "/min"
            }
        }
        observations.append(hr_obs)

        # HINTS exam observation
        hints_obs = {
            "resourceType": "Observation",
            "id": f"hints-{patient_id}",
            "status": "final",
            "category": [{
                "coding": [{
                    "system": "http://terminology.hl7.org/CodeSystem/observation-category",
                    "code": "exam",
                    "display": "Exam"
                }]
            }],
            "code": {
                "coding": [{
                    "system": "http://loinc.org",
                    "code": loinc_codes["hints_exam"],
                    "display": "HINTS examination"
                }]
            },
            "subject": {"reference": f"Patient/{patient_id}"},
            "effectiveDateTime": datetime.now().isoformat(),
            "component": [
                {
                    "code": {
                        "coding": [{
                            "system": "http://loinc.org",
                            "code": loinc_codes["head_impulse_test"],
                            "display": "Head impulse test"
                        }]
                    },
                    "valueString": exam.head_impulse_test
                },
                {
                    "code": {
                        "text": "Nystagmus type"
                    },
                    "valueString": exam.nystagmus_type
                },
                {
                    "code": {
                        "text": "Test of skew"
                    },
                    "valueBoolean": exam.test_of_skew
                }
            ]
        }
        observations.append(hints_obs)

        return observations

    def map_to_omop_cdm(self, archetype: ClinicalArchetype,
                       person_id: int = 1) -> Dict[str, List[Dict]]:
        """
        Map to OMOP Common Data Model tables.

        Args:
            archetype: ClinicalArchetype to map
            person_id: OMOP person_id

        Returns:
            Dictionary with OMOP CDM table records
        """
        omop_data = {}

        # PERSON table
        omop_data["person"] = [{
            "person_id": person_id,
            "gender_concept_id": 8507 if archetype.gender == "M" else 8532,
            "year_of_birth": datetime.now().year - archetype.age,
            "race_concept_id": 0,  # Unknown
            "ethnicity_concept_id": 0,  # Unknown
        }]

        # CONDITION_OCCURRENCE table
        dx_concept_id = self.OMOP_CONCEPT_IDS.get(
            archetype.diagnosis.value,
            0
        )
        omop_data["condition_occurrence"] = [{
            "condition_occurrence_id": 1,
            "person_id": person_id,
            "condition_concept_id": dx_concept_id,
            "condition_start_date": datetime.now().date().isoformat(),
            "condition_type_concept_id": 32020,  # EHR encounter diagnosis
        }]

        # MEASUREMENT table (vital signs)
        omop_data["measurement"] = [
            {
                "measurement_id": 1,
                "person_id": person_id,
                "measurement_concept_id": 3004249,  # Blood pressure
                "measurement_date": datetime.now().date().isoformat(),
                "value_as_number": archetype.examination.blood_pressure_systolic,
                "unit_concept_id": 8876,  # mmHg
            },
            {
                "measurement_id": 2,
                "person_id": person_id,
                "measurement_concept_id": 3027018,  # Heart rate
                "measurement_date": datetime.now().date().isoformat(),
                "value_as_number": archetype.examination.heart_rate,
                "unit_concept_id": 8541,  # /min
            }
        ]

        return omop_data

    def _severity_code(self, severity: int) -> str:
        """Map severity score to SNOMED CT severity code"""
        if severity <= 3:
            return "255604002"  # Mild
        elif severity <= 7:
            return "6736007"   # Moderate
        else:
            return "24484000"  # Severe

    def _severity_display(self, severity: int) -> str:
        """Map severity score to display text"""
        if severity <= 3:
            return "Mild"
        elif severity <= 7:
            return "Moderate"
        else:
            return "Severe"


if __name__ == "__main__":
    # Test standards mapper
    from .titrate_formalizer import (
        TimingPattern, TriggerType, DiagnosisCategory,
        ExaminationFindings, ClinicalArchetype
    )
    import numpy as np
    import json

    # Create test archetype
    exam = ExaminationFindings(
        head_impulse_test="abnormal_peripheral",
        nystagmus_type="peripheral",
        test_of_skew=False,
        dix_hallpike_result="negative",
        romberg_test="abnormal",
        gait_assessment="unstable",
        neurological_signs=False,
        blood_pressure_systolic=145,
        blood_pressure_diastolic=88,
        heart_rate=76
    )

    archetype = ClinicalArchetype(
        timing=TimingPattern.ACUTE,
        trigger=TriggerType.SPONTANEOUS,
        examination=exam,
        diagnosis=DiagnosisCategory.VESTIBULAR_NEURITIS,
        features=np.random.randn(150),
        urgency=1,
        age=58,
        gender="F",
        has_hypertension=True,
        has_diabetes=False,
        has_cardiovascular_disease=False,
        has_migraine_history=False,
        symptom_duration_hours=36,
        symptom_severity=7,
        nausea_vomiting=True,
        headache=False,
        hearing_loss=False,
        tinnitus=True
    )

    mapper = StandardsMapper()

    print("="*60)
    print("STANDARDS MAPPING TEST")
    print("="*60)

    # SNOMED CT mapping
    print("\nSNOMED CT Codes:")
    snomed = mapper.map_to_snomed(archetype)
    print(json.dumps(snomed, indent=2))

    # LOINC mapping
    print("\nLOINC Codes:")
    loinc = mapper.map_to_loinc(archetype)
    print(json.dumps(loinc, indent=2))

    # FHIR Condition
    print("\nFHIR Condition Resource:")
    fhir_condition = mapper.map_to_fhir_condition(archetype, "patient-test")
    print(json.dumps(fhir_condition, indent=2))

    # FHIR Observations
    print("\nFHIR Observation Resources:")
    fhir_obs = mapper.map_to_fhir_observation(archetype, "patient-test")
    print(f"Generated {len(fhir_obs)} observations")

    # OMOP CDM
    print("\nOMOP CDM Tables:")
    omop = mapper.map_to_omop_cdm(archetype, person_id=123)
    for table, records in omop.items():
        print(f"  {table}: {len(records)} records")

    print("\nTest completed successfully!")
