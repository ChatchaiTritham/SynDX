# SynDX: Explainable AI-Driven Synthetic Data Generation for Privacy-Preserving Differential Diagnosis of Vestibular Disorders

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.XXXXXXX.svg)](https://doi.org/10.5281/zenodo.XXXXXXX)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![arXiv](https://img.shields.io/badge/arXiv-2025.XXXXX-b31b1b.svg)](https://arxiv.org/abs/2025.XXXXX)

## ⚠️ Important Notice

**This is preliminary work without clinical validation.** All validation in the accompanying paper uses synthetic data only. This framework has NOT been validated against real patient outcomes and should NOT be used for clinical decision-making without prospective clinical trials.

## Overview

SynDX is a framework that generates synthetic medical data by integrating clinical guidelines with explainable AI techniques, eliminating the need for real patient records entirely. The system generates privacy-preserving synthetic patient data for vestibular disorder diagnosis research.

### Key Features

- **Guideline-Driven Generation**: Based on TiTrATE diagnostic framework and Bárány Society ICVD 2025 classification
- **XAI Integration**: SHAP-guided feature importance, NMF archetype extraction, counterfactual validation
- **Privacy-Preserving**: Differential privacy (ε=1.0) without requiring real patient data
- **Standards Compliant**: HL7 FHIR, SNOMED CT, LOINC, DICOM, OMOP CDM
- **Reproducible**: Complete implementation with Docker support

## Architecture

SynDX operates in three phases:

1. **Phase 1: Clinical Knowledge Extraction**
   - Extract 8,400 computational archetypes from TiTrATE/Bárány guidelines
   - Apply consistency constraints and standards mapping

2. **Phase 2: XAI-Driven Synthesis**
   - NMF latent archetype extraction (r=20)
   - VAE latent space modeling (d=50)
   - SHAP-guided feature reweighting
   - TiTrATE-constrained counterfactual validation
   - Differential privacy noise injection

3. **Phase 3: Multi-Level Validation**
   - Statistical realism (KL divergence, Jensen-Shannon, Wasserstein)
   - Diagnostic performance (ROC-AUC, sensitivity, specificity)
   - XAI fidelity (SHAP alignment, TiTrATE coverage)
   - Scalability analysis

## Installation

### Prerequisites

- Python 3.9 or higher
- CUDA 11.8+ (for GPU acceleration, optional but recommended)
- 16GB+ RAM (128GB recommended for large-scale generation)

### Using pip

```bash
git clone https://github.com/chatchai.tritham/SynDX.git
cd SynDX
pip install -r requirements.txt
```

### Using Docker

```bash
docker pull chatchaitritham/syndx:latest
docker run -it --gpus all -v $(pwd)/outputs:/app/outputs chatchaitritham/syndx:latest
```

### Using Conda

```bash
conda env create -f environment.yml
conda activate syndx
```

## Quick Start

### Generate 10,000 Synthetic Patients

```python
from syndx import SynDXPipeline

# Initialize pipeline
pipeline = SynDXPipeline(
    n_archetypes=8400,
    nmf_components=20,
    vae_latent_dim=50,
    epsilon=1.0  # Differential privacy budget
)

# Phase 1: Extract clinical knowledge
archetypes = pipeline.extract_archetypes(
    guidelines=['titrate', 'barany_icvd_2025']
)

# Phase 2: Generate synthetic patients
synthetic_patients = pipeline.generate(
    n_patients=10000,
    convergence_threshold=0.05
)

# Phase 3: Validate
validation_results = pipeline.validate(
    synthetic_patients,
    metrics=['statistical', 'diagnostic', 'xai']
)

# Export to FHIR
pipeline.export_fhir(synthetic_patients, 'outputs/synthetic_patients.json')
```

### Using Jupyter Notebooks

See the `notebooks/` directory for detailed tutorials:

- `01_data_generation_tutorial.ipynb` - Step-by-step generation walkthrough
- `02_statistical_validation.ipynb` - Statistical realism analysis
- `03_diagnostic_performance.ipynb` - Diagnostic classifier training
- `04_xai_analysis.ipynb` - SHAP and counterfactual explanation
- `05_fhir_export.ipynb` - Standards-compliant data export

## Project Structure

```
SynDX/
├── syndx/                      # Main package
│   ├── __init__.py
│   ├── phase1_knowledge/       # Clinical knowledge extraction
│   │   ├── titrate_formalizer.py
│   │   ├── archetype_generator.py
│   │   └── standards_mapper.py
│   ├── phase2_synthesis/       # XAI-driven synthesis
│   │   ├── nmf_extractor.py
│   │   ├── vae_model.py
│   │   ├── shap_reweighter.py
│   │   ├── counterfactual_validator.py
│   │   └── differential_privacy.py
│   ├── phase3_validation/      # Multi-level validation
│   │   ├── statistical_metrics.py
│   │   ├── diagnostic_evaluator.py
│   │   └── xai_fidelity.py
│   └── utils/                  # Utilities
│       ├── fhir_exporter.py
│       ├── snomed_mapper.py
│       └── data_loader.py
├── notebooks/                  # Jupyter tutorials
├── data/                       # Input guidelines and reference data
├── models/                     # Pre-trained models
├── outputs/                    # Generated synthetic datasets
├── tests/                      # Unit tests
├── docs/                       # Documentation
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
├── environment.yml
├── setup.py
└── README.md
```

## Results (Synthetic-to-Synthetic Validation Only)

⚠️ **Note**: These metrics evaluate internal consistency on synthetic test data, NOT real patient generalization.

### Statistical Realism
- KL Divergence: 0.042 ± 0.004
- Jensen-Shannon Divergence: 0.031 ± 0.003
- Wasserstein Distance: 0.053 ± 0.005

### Internal Diagnostic Performance
- ROC-AUC: 0.89 ± 0.02
- Sensitivity: 0.91 ± 0.03
- Specificity: 0.86 ± 0.03

### XAI Fidelity
- SHAP Fidelity: 91.6% (95% CI: 88.9-94.3%)
- TiTrATE Coverage: 94.2% (95% CI: 91.8-96.6%)

## Citation

If you use SynDX in your research, please cite:

```bibtex
@article{tritham2025syndx,
  title={SynDX: Explainable AI-Driven Synthetic Data Generation for Privacy-Preserving Differential Diagnosis of Vestibular Disorders},
  author={Tritham, Chatchai and Namahoot, Chakkrit Snae},
  journal={IEEE Access},
  year={2025},
  note={Preliminary work without clinical validation},
  doi={10.1109/ACCESS.2025.XXXXXXX}
}
```

## Limitations

### Critical Limitation: No Real Patient Validation

**All validation uses synthetic data.** We have NOT validated against real patient outcomes. Performance metrics measure internal consistency within the synthetic framework—they do NOT prove clinical utility.

Before clinical deployment, SynDX requires:
- Prospective validation on real patient cohorts (500-1,000 ED patients planned)
- Multi-center clinical trials
- Regulatory approval (FDA 510(k) pathway)

### Other Limitations
- Domain-specific to vestibular disorders (adaptation required for other specialties)
- Computational requirements: GPU recommended (NVIDIA A100 used in development)
- Guideline currency: Based on TiTrATE 2015 and Bárány ICVD 2025

## Use Cases

**Appropriate Uses** (without real patient validation):
- Algorithm prototyping and development
- Benchmark creation for comparing diagnostic algorithms
- Privacy-first model architecture exploration
- Educational demonstrations
- Reproducibility in published research

**Inappropriate Uses** (requires clinical validation):
- Clinical decision support
- Actual patient diagnosis
- Regulatory submissions
- Clinical trial enrollment decisions

## Contributing

We welcome contributions! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

### Development Setup

```bash
git clone https://github.com/chatchai.tritham/SynDX.git
cd SynDX
pip install -e ".[dev]"
pytest tests/
```

## License

This project is licensed under the MIT License - see [LICENSE](LICENSE) file for details.

## Acknowledgments

- Based on TiTrATE framework (Newman-Toker & Edlow, 2015)
- Bárány Society ICVD 2025 classification
- Supported by [Funding Agency - To Be Updated]

## Contact

- **Chatchai Tritham** - chatchai.tritham@nu.ac.th
- **Chakkrit Snae Namahoot** - chakkrits@nu.ac.th

Department of Computer Science and Information Technology
Faculty of Science, Naresuan University
Phitsanulok 65000, Thailand

## Future Work

- **Phase 1 Mini-Validation Pilot**: 100 real ED patients (3 months)
- **Phase 2 Multi-Center Trial**: 500-1,000 patients (12 months)
- **Phase 3 Longitudinal Outcomes**: 24-month follow-up
- Domain extension: Chest pain, neurological deficits, chronic diseases
- Federated learning version
- FDA 510(k) regulatory pathway

## Changelog

See [CHANGELOG.md](CHANGELOG.md) for version history.

---

**Disclaimer**: This software is provided for research purposes only. It is NOT approved for clinical use. Always consult qualified healthcare professionals for medical decisions.
