# Clinical Guidelines and Gold Standards

This document provides references to the clinical guidelines and gold standards used in developing the SynDX framework, specifically for the TiTrATE (Triage for Acute Tabular Events) system.

---

## Overview

The SynDX framework's clinical knowledge extraction (Phase 1) is based on evidence-based clinical guidelines for acute dizziness and vertigo management. This document provides citations and links to access the official guidelines.

**Important**: These guidelines are copyrighted materials owned by their respective organizations. We provide references only. Please access the original sources for full content.

---

## Primary Clinical Guidelines

### 1. ACEP Clinical Policy: Critical Issues in the Evaluation and Management of Adult Patients Presenting to the Emergency Department with Acute Headache

**Citation**:
```
Godwin SA, Cherkas DS, Panagos PD, et al. Clinical policy: critical issues in the
evaluation and management of adult patients presenting to the emergency department
with acute headache. Ann Emerg Med. 2019;74(4):e41-e74.
```

**DOI**: [10.1016/j.annemergmed.2019.07.009](https://doi.org/10.1016/j.annemergmed.2019.07.009)

**Official Source**: [American College of Emergency Physicians (ACEP)](https://www.acep.org/patient-care/clinical-policies/)

**Relevance**: Provides evidence-based recommendations for evaluating acute presentations including dizziness associated with headache.

---

### 2. AHA/ASA Guidelines for the Early Management of Acute Ischemic Stroke

**Citation**:
```
Powers WJ, Rabinstein AA, Ackerson T, et al. Guidelines for the early management
of patients with acute ischemic stroke: 2019 update to the 2018 guidelines for
the early management of acute ischemic stroke. Stroke. 2019;50(12):e344-e418.
```

**DOI**: [10.1161/STR.0000000000000211](https://doi.org/10.1161/STR.0000000000000211)

**Official Source**: [American Heart Association/American Stroke Association](https://www.ahajournals.org/stroke)

**Relevance**: Critical for understanding stroke risk stratification in acute dizziness presentations, particularly for the central vs. peripheral distinction.

---

### 3. Benign Paroxysmal Positional Vertigo (BPPV) Clinical Practice Guidelines

**Citation**:
```
Bhattacharyya N, Gubbels SP, Schwartz SR, et al. Clinical practice guideline:
benign paroxysmal positional vertigo (update). Otolaryngol Head Neck Surg.
2017;156(3_suppl):S1-S47.
```

**DOI**: [10.1177/0194599816689667](https://doi.org/10.1177/0194599816689667)

**Official Source**: [American Academy of Otolaryngology-Head and Neck Surgery](https://www.entnet.org/)

**Relevance**: Gold standard for BPPV diagnosis and management, essential for vertigo presentations in TiTrATE.

---

### 4. Vestibular Neuritis and Labyrinthitis Guidelines

**Citation**:
```
Strupp M, Brandt T. Vestibular neuritis. Semin Neurol. 2020;40(1):39-46.
```

**DOI**: [10.1055/s-0039-3402735](https://doi.org/10.1055/s-0039-3402735)

**Relevance**: Defines diagnostic criteria for acute vestibular syndrome (AVS) and helps distinguish central from peripheral causes.

---

### 5. Meniere's Disease Diagnostic Criteria

**Citation**:
```
Lopez-Escamez JA, Carey J, Chung WH, et al. Diagnostic criteria for Menière's
disease. J Vestib Res. 2015;25(1):1-7.
```

**DOI**: [10.3233/VES-150549](https://doi.org/10.3233/VES-150549)

**Official Source**: [Bárány Society](https://www.barany-society.nl/)

**Relevance**: Provides definitive diagnostic criteria for Meniere's disease used in archetype generation.

---

## Clinical Decision Tools

### ABCD² Score (Stroke Risk Stratification)

**Citation**:
```
Johnston SC, Rothwell PM, Nguyen-Huynh MN, et al. Validation and refinement of
scores to predict very early stroke risk after transient ischaemic attack.
Lancet. 2007;369(9558):283-292.
```

**DOI**: [10.1016/S0140-6736(07)60150-0](https://doi.org/10.1016/S0140-6736(07)60150-0)

**Usage**: Implemented in TiTrATE for stroke risk assessment in acute dizziness.

**Calculator**: [MDCalc ABCD² Score](https://www.mdcalc.com/calc/715/abcd2-score-tia)

---

### HINTS Exam (Head Impulse, Nystagmus, Test of Skew)

**Citation**:
```
Kattah JC, Talkad AV, Wang DZ, Hsieh YH, Toker DE. HINTS to diagnose stroke in
the acute vestibular syndrome: three-step bedside oculomotor examination more
sensitive than early MRI diffusion-weighted imaging. Stroke. 2009;40(11):3504-3510.
```

**DOI**: [10.1161/STROKEAHA.109.551234](https://doi.org/10.1161/STROKEAHA.109.551234)

**Usage**: Gold standard for differentiating central vs. peripheral causes of acute vestibular syndrome.

**Tutorial**: [Stanford Medicine 25 - HINTS Exam](https://stanfordmedicine25.stanford.edu/the25/hints.html)

---

## Evidence Synthesis Sources

### Cochrane Reviews

**Acute Vestibular Syndrome Management**:
- [Cochrane Library - Vestibular Disorders](https://www.cochranelibrary.com/cdsr/topics/ENT-disorders/vestibular)

### UpToDate Clinical Resources

**Approach to the patient with dizziness**:
- Available at: [UpToDate.com](https://www.uptodate.com/) (subscription required)
- Topic ID: 434

---

## Implementation in SynDX

### How Guidelines Inform Archetype Generation

The clinical guidelines above inform the SynDX framework in the following ways:

1. **Diagnostic Criteria Extraction**: Formalized into computational rules
2. **Symptom-Disease Associations**: Used to create realistic symptom patterns
3. **Risk Stratification**: ABCD², HINTS inform triage decision logic
4. **Treatment Pathways**: Guide disposition decisions (ER, Specialist, Home Observation)

### Mapping to TiTrATE Dimensions

| Clinical Guideline | TiTrATE Dimension(s) |
|-------------------|---------------------|
| BPPV Guidelines | Onset Pattern, Triggers, Duration |
| Stroke Guidelines | Associated Symptoms, Risk Factors |
| HINTS Exam | Physical Exam Findings |
| Vestibular Neuritis | Onset Pattern, Duration, Severity |
| ABCD² Score | Risk Factors, Associated Symptoms |

---

## Data Sources for Validation

### Publicly Available Datasets

While full clinical guidelines are copyrighted, the following open datasets can be used for validation:

1. **MIMIC-III Clinical Database**
   - Citation: Johnson AEW, et al. MIMIC-III, a freely accessible critical care database. Sci Data. 2016;3:160035.
   - Access: [PhysioNet](https://physionet.org/content/mimiciii/)
   - License: PhysioNet Credentialed Health Data License 1.5.0

2. **eICU Collaborative Research Database**
   - Citation: Pollard TJ, et al. The eICU Collaborative Research Database. Sci Data. 2018;5:180178.
   - Access: [PhysioNet](https://physionet.org/content/eicu-crd/)

---

## Compliance and Attribution

### Copyright Notice

All clinical guidelines referenced in this document are copyrighted materials owned by their respective professional organizations:

- American College of Emergency Physicians (ACEP)
- American Heart Association (AHA) / American Stroke Association (ASA)
- American Academy of Otolaryngology-Head and Neck Surgery (AAO-HNS)
- Bárány Society

**Usage**: References provided under fair use for academic research purposes. Full guidelines must be accessed through official sources.

### How to Cite This Work

If you use the SynDX framework based on these guidelines, please cite:

```bibtex
@software{syndx2024,
  title = {SynDX: Explainable Synthetic Data Generation for Clinical Triage},
  author = {[Your Name]},
  year = {2024},
  note = {Clinical guidelines used with attribution. See data/guidelines/CLINICAL_GUIDELINES.md}
}
```

And cite the individual guidelines as appropriate for your specific use case.

---

## Updates and Maintenance

**Last Updated**: 2026-01-01

Clinical guidelines are regularly updated by their respective organizations. Users should:

1. Check for updated versions of referenced guidelines annually
2. Update archetype generation rules if recommendations change
3. Re-validate synthetic data if significant guideline changes occur

---

## Contact for Clinical Questions

For questions about clinical content or guideline interpretation:
- Review the original guideline documents
- Consult with domain experts (emergency physicians, neurologists, otolaryngologists)
- This is a research framework and should not be used for clinical decision-making

---

## Disclaimer

⚠️ **IMPORTANT**:

- This framework is for RESEARCH PURPOSES ONLY
- NOT validated for clinical use
- NOT a substitute for clinical judgment
- All clinical decisions must be made by qualified healthcare professionals
- Synthetic data does NOT represent real patients

---

## Additional Resources

### Professional Organizations

- **ACEP**: [www.acep.org](https://www.acep.org)
- **AHA/ASA**: [www.heart.org](https://www.heart.org) | [www.stroke.org](https://www.stroke.org)
- **AAO-HNS**: [www.entnet.org](https://www.entnet.org)
- **Bárány Society**: [www.barany-society.nl](https://www.barany-society.nl)

### Educational Resources

- **Stanford Medicine 25**: [stanfordmedicine25.stanford.edu](https://stanfordmedicine25.stanford.edu)
- **MDCalc**: [www.mdcalc.com](https://www.mdcalc.com)
- **NEJM Knowledge+**: [knowledgeplus.nejm.org](https://knowledgeplus.nejm.org)

---

**Document Version**: 1.0
**Repository**: SynDX Framework
**License**: See [LICENSE](../../LICENSE) for framework licensing
**Note**: Clinical guidelines retain their original copyright and licensing
