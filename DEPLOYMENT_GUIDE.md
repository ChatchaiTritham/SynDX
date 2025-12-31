# SynDX Deployment Guide

## ⚠️ Important Notice

**This is preliminary work without clinical validation.**
All validation uses synthetic data only. Do **NOT** use for clinical decision-making.

---

## Quick Start

### 1. Clone Repository

```bash
git clone https://github.com/chatchai.tritham/SynDX.git
cd SynDX
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
pip install -e .
```

### 3. Generate Example Dataset

```bash
python scripts/generate_example_dataset.py
```

This will create:
- `data/archetypes/example_archetypes.csv` (500 archetypes)
- `outputs/synthetic_patients/example_synthetic_patients.csv` (1,000 patients)
- `outputs/synthetic_patients/example_synthetic_patients.json`
- `outputs/synthetic_patients/example_dataset_metadata.json`

### 4. Run Jupyter Notebooks

```bash
jupyter notebook notebooks/01_Quick_Start_Tutorial.ipynb
```

---

## Docker Deployment

### Build Docker Image

```bash
docker build -t syndx:latest .
```

### Run with Docker Compose

```bash
# Start Jupyter Lab
docker-compose up

# Access at: http://localhost:8888
```

### Run CLI Mode

```bash
docker-compose --profile cli up syndx-cli
```

---

## Project Structure

```
SynDX/
├── syndx/                      # Main Python package
│   ├── phase1_knowledge/       # Clinical knowledge extraction
│   │   ├── titrate_formalizer.py
│   │   ├── archetype_generator.py
│   │   └── standards_mapper.py
│   ├── phase2_synthesis/       # XAI-driven synthesis
│   │   ├── nmf_extractor.py
│   │   ├── vae_model.py (placeholder)
│   │   ├── shap_reweighter.py (placeholder)
│   │   ├── counterfactual_validator.py (placeholder)
│   │   └── differential_privacy.py (placeholder)
│   ├── phase3_validation/      # Multi-level validation
│   │   ├── statistical_metrics.py
│   │   └── ... (placeholders)
│   ├── utils/                  # Utilities
│   │   ├── data_loader.py
│   │   └── ... (placeholders)
│   └── pipeline.py             # Main orchestrator
├── notebooks/                  # Jupyter tutorials
│   └── 01_Quick_Start_Tutorial.ipynb
├── scripts/                    # Helper scripts
│   └── generate_example_dataset.py
├── data/                       # Input data
│   └── archetypes/
├── outputs/                    # Generated outputs
│   ├── synthetic_patients/
│   ├── figures/
│   └── metrics/
├── tests/                      # Unit tests
├── Dockerfile                  # Docker configuration
├── docker-compose.yml          # Docker Compose
├── requirements.txt            # Python dependencies
├── setup.py                    # Package setup
├── README.md                   # Main documentation
├── CITATION.cff                # Citation metadata
└── LICENSE                     # MIT License
```

---

## Generated Files

After running `generate_example_dataset.py`:

### 1. Archetypes CSV (`data/archetypes/example_archetypes.csv`)

Contains 500 computationally-generated clinical archetypes with:
- TiTrATE dimensions (timing, triggers, examination)
- Demographics (age, gender)
- Comorbidities (hypertension, diabetes, CVD)
- Symptoms (nausea, headache, hearing loss, tinnitus)
- 150-dimensional feature vectors

### 2. Synthetic Patients CSV (`outputs/synthetic_patients/example_synthetic_patients.csv`)

Contains 1,000 synthetic patient records:
- Patient IDs (syn-000000 to syn-000999)
- 150-dimensional feature vectors
- Generated via NMF-based latent archetype sampling

### 3. Metadata JSON (`outputs/synthetic_patients/example_dataset_metadata.json`)

Includes:
- Generation parameters
- Validation metrics
- Citation information
- Timestamp and version

---

## Next Steps for Full Implementation

The current release includes:

✅ **Implemented:**
- Phase 1: TiTrATE formalization and archetype generation
- NMF latent archetype extraction
- Basic synthetic data generation
- Statistical validation framework
- Standards mapping (FHIR, SNOMED CT)
- Docker deployment

⏳ **Pending (Placeholders):**
- VAE training loop (requires PyTorch implementation)
- SHAP feature importance calculation
- Counterfactual validation
- Differential privacy noise injection
- Diagnostic classifier training
- XAI fidelity metrics
- FHIR export functionality

To complete the implementation:

1. **Implement VAE Model** (`syndx/phase2_synthesis/vae_model.py`)
   - Encoder/Decoder networks
   - ELBO training loop
   - Latent space sampling

2. **Implement SHAP Reweighting** (`syndx/phase2_synthesis/shap_reweighter.py`)
   - Train XGBoost classifier on archetypes
   - Compute SHAP values
   - Reweight sampling probabilities

3. **Implement Counterfactual Validation** (`syndx/phase2_synthesis/counterfactual_validator.py`)
   - Gradient-based counterfactual search
   - TiTrATE constraint checking
   - Iterative refinement

4. **Implement Differential Privacy** (`syndx/phase2_synthesis/differential_privacy.py`)
   - Laplace mechanism
   - Sensitivity calculation
   - Privacy budget tracking

5. **Implement Diagnostic Evaluation** (`syndx/phase3_validation/diagnostic_evaluator.py`)
   - Train/test split
   - Classifier training
   - ROC-AUC, sensitivity, specificity

6. **Implement XAI Fidelity** (`syndx/phase3_validation/xai_fidelity.py`)
   - SHAP fidelity measurement
   - TiTrATE coverage calculation

7. **Implement FHIR Export** (`syndx/utils/fhir_exporter.py`)
   - Map to FHIR resources
   - Generate Condition, Observation resources
   - Export Bundle

---

## Testing

```bash
# Run unit tests (when implemented)
pytest tests/

# Run with coverage
pytest --cov=syndx tests/

# Run specific test
pytest tests/test_archetype_generator.py
```

---

## Publishing to GitHub

### 1. Initialize Git Repository

```bash
cd SynDX
git init
git add .
git commit -m "Initial commit: SynDX v0.1.0 (preliminary work without clinical validation)"
```

### 2. Create GitHub Repository

Go to https://github.com/new and create repository:
- Name: `SynDX`
- Description: "Explainable AI-Driven Synthetic Data Generation (Preliminary)"
- Public repository
- Don't initialize with README (we have one)

### 3. Push to GitHub

```bash
git remote add origin https://github.com/chatchai.tritham/SynDX.git
git branch -M main
git push -u origin main
```

### 4. Create GitHub Release

```bash
# Tag release
git tag -a v0.1.0 -m "v0.1.0: Initial release (preliminary)"
git push origin v0.1.0
```

Then go to GitHub → Releases → Draft a new release:
- Tag: `v0.1.0`
- Title: `v0.1.0 - Initial Release (Preliminary Work)`
- Description: Copy from CHANGELOG.md
- Attach: Example datasets (zip files)

### 5. Get DOI from Zenodo

1. Go to https://zenodo.org
2. Link your GitHub repository
3. Create new version upload
4. Fill in metadata:
   - Title: "SynDX: Explainable AI-Driven Synthetic Data Generation"
   - Authors: Chatchai Tritham, Chakkrit Snae Namahoot
   - Keywords: synthetic data, XAI, vestibular disorders
   - Notes: "Preliminary work without clinical validation"
5. Publish to get DOI
6. Update README.md and CITATION.cff with DOI

---

## Citation

Once published, cite as:

```bibtex
@software{tritham2025syndx_software,
  author = {Tritham, Chatchai and Namahoot, Chakkrit Snae},
  title = {SynDX: Explainable AI-Driven Synthetic Data Generation},
  year = {2025},
  publisher = {GitHub},
  url = {https://github.com/chatchai.tritham/SynDX},
  doi = {10.5281/zenodo.XXXXXXX}
}

@article{tritham2025syndx_paper,
  author = {Tritham, Chatchai and Namahoot, Chakkrit Snae},
  title = {SynDX: Explainable AI-Driven Synthetic Data Generation for
           Privacy-Preserving Differential Diagnosis of Vestibular Disorders},
  journal = {IEEE Access},
  year = {2025},
  doi = {10.1109/ACCESS.2025.XXXXXXX},
  note = {Preliminary work without clinical validation}
}
```

---

## Support

- **Issues**: https://github.com/chatchai.tritham/SynDX/issues
- **Email**: chatchai.tritham@nu.ac.th
- **Institution**: Naresuan University, Thailand

---

## License

MIT License - See [LICENSE](LICENSE) file for details.

---

**Disclaimer**: This software is for research purposes only. It has NOT been clinically validated. Do NOT use for patient care without prospective clinical trials.
