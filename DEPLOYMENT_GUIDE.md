# Getting SynDX Running on Your Machine

## 🚨 Quick Reminder

This is PhD research code that's only been tested on synthetic data. **Don't use it for actual patient care.** We need proper clinical trials first.

---

## The Fast Track (If You Just Want to See It Work)

### Step 1: Grab the Code

```bash
git clone https://github.com/ChatchaiTritham/SynDX.git
cd SynDX
```

### Step 2: Install Everything

```bash
pip install -r requirements.txt
pip install -e .
```

That `-e` flag installs in "editable" mode so you can tweak things without reinstalling.

### Step 3: Make Some Fake Patients

```bash
python scripts/generate_example_dataset.py
```

This creates:
- 500 clinical archetypes in `data/archetypes/example_archetypes.csv`
- 1,000 synthetic patients in `outputs/synthetic_patients/`
- Metadata JSON with all the generation details

### Step 4: Play Around in Jupyter

```bash
jupyter notebook notebooks/01_Quick_Start_Tutorial.ipynb
```

---

## Docker (If Dependencies Are Being Annoying)

Sometimes pip just doesn't cooperate. Docker to the rescue:

### Build It

```bash
docker build -t syndx:latest .
```

### Run with Jupyter

```bash
docker-compose up

# Then open your browser to http://localhost:8888
```

### Command Line Mode

```bash
docker-compose --profile cli up syndx-cli
```

---

## What You'll Get After Generation

### The Archetypes CSV

`data/archetypes/example_archetypes.csv` has 500 rows with:
- **TiTrATE dimensions**: Timing patterns, triggers, physical exam findings
- **Demographics**: Age, gender (following real ED population distributions)
- **Comorbidities**: Hypertension, diabetes, cardiovascular disease
- **Symptoms**: Nausea, headache, hearing loss, tinnitus, etc.
- **Feature vector**: 150 dimensions capturing everything above

### The Synthetic Patients CSV

`outputs/synthetic_patients/example_synthetic_patients.csv` contains 1,000 fake patients:
- Patient IDs like `syn-000000` through `syn-000999`
- Same 150-dimensional feature space
- Generated using NMF to mix and match archetypes realistically

### Metadata JSON

All the nerdy details:
- What parameters we used
- Validation metrics (KL divergence, etc.)
- Timestamps and version info
- How to cite this if you use it

---

## Project Layout (Where Everything Lives)

```
SynDX/
├── syndx/                      # The actual framework
│   ├── phase1_knowledge/       # Turning guidelines into archetypes
│   │   ├── titrate_formalizer.py
│   │   ├── archetype_generator.py
│   │   └── standards_mapper.py
│   ├── phase2_synthesis/       # Making synthetic patients
│   │   ├── nmf_extractor.py
│   │   ├── vae_model.py
│   │   ├── shap_reweighter.py
│   │   ├── counterfactual_validator.py
│   │   └── differential_privacy.py
│   ├── phase3_validation/      # Checking if it worked
│   │   ├── statistical_metrics.py
│   │   ├── diagnostic_evaluator.py
│   │   └── xai_fidelity.py
│   └── utils/                  # Helper stuff
├── notebooks/                  # Jupyter tutorials
├── data/                       # Input guidelines
├── outputs/                    # Generated datasets
├── tests/                      # Unit tests
└── docs/                       # More documentation
```

---

## What's Implemented vs. What's Still TODO

### ✅ Working Right Now

- **Phase 1**: TiTrATE formalization and archetype generation
- **NMF extraction**: Finding latent patterns in archetypes
- **Basic synthesis**: Creating synthetic patients
- **Statistical validation**: Checking distributions make sense
- **Standards mapping**: FHIR, SNOMED CT readiness
- **Docker deployment**: Containerized everything

### ⏳ On the Roadmap (Currently Placeholders)

We wrote the scaffolding, but these need full implementations:

- **VAE training**: Need to finish the PyTorch training loop
- **SHAP reweighting**: Feature importance-guided sampling
- **Counterfactual validation**: Making sure small changes behave right
- **Differential privacy**: Adding the privacy noise properly
- **Diagnostic classifiers**: Training and evaluating ML models
- **XAI fidelity metrics**: Measuring explanation quality
- **FHIR export**: Actually outputting proper FHIR bundles

If you're looking to contribute, these are great places to start!

---

## Running Tests

```bash
# All tests
pytest tests/

# With coverage report
pytest --cov=syndx tests/

# Just one file
pytest tests/test_archetype_generator.py
```

---

## Pushing to GitHub (When You're Ready)

### Initialize Git

```bash
cd SynDX
git init
git add .
git commit -m "Initial commit: SynDX v0.1.0 - preliminary research code"
```

### Create the Repo on GitHub

Head to https://github.com/new:
- Name it `SynDX`
- Make it public (or private if you prefer)
- Skip the README init (we already have one)

### Push It Up

```bash
git remote add origin https://github.com/YourUsername/SynDX.git
git branch -M main
git push -u origin main
```

### Tag a Release

```bash
git tag -a v0.1.0 -m "v0.1.0: First release (preliminary, no clinical validation)"
git push origin v0.1.0
```

Then create a release on GitHub and attach the example datasets as zip files.

### Getting a DOI from Zenodo

1. Log into https://zenodo.org
2. Connect your GitHub repo
3. Upload your release
4. Fill in the metadata:
   - **Title**: SynDX: Explainable AI-Driven Synthetic Data Generation
   - **Authors**: You and your advisor
   - **Keywords**: synthetic data, XAI, vestibular disorders, differential diagnosis
   - **Notes**: "Preliminary work without clinical validation"
5. Publish and grab your DOI
6. Update the README badges with the real DOI

---

## How to Cite This

Once it's published:

```bibtex
@software{tritham2025syndx_software,
  author = {Tritham, Chatchai and Namahoot, Chakkrit Snae},
  title = {SynDX: Explainable AI-Driven Synthetic Data Generation},
  year = {2025},
  publisher = {GitHub},
  url = {https://github.com/ChatchaiTritham/SynDX},
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

## Need Help?

- **Bug reports**: Open an issue on GitHub
- **Email**: chatchai.tritham@nu.ac.th
- **Institution**: Naresuan University, Thailand

---

## License

MIT License - see the [LICENSE](LICENSE) file for full text.

---

**Legal Stuff**: This is research software only. Not validated for clinical use. Don't diagnose patients with this without proper clinical trials and regulatory approval.
