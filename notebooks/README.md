# Jupyter Notebooks - Learn by Doing

These notebooks walk you through the whole SynDX pipeline, from archetypes to validation.

---

## Which Notebook Should I Start With?

**Just want to see it work?** → `01_Quick_Start_Tutorial.ipynb` (5 min)

**Want to understand each phase?** → Work through 02, 03, 04 in order (1-2 hours total)

**Ready to reproduce the paper?** → Jump to `05_Complete_Pipeline_End_to_End.ipynb` (45 min)

---

## What's in Each Notebook

### 01: Quick Start (5-10 minutes)
**Good for**: First-time users

The basics:
- Set up the pipeline
- Generate a few archetypes
- Make some synthetic patients
- See what the output looks like

No deep dives, just enough to understand what SynDX does.

---

### 02: Phase 1 - Clinical Knowledge Extraction (15-20 minutes)
**Good for**: Understanding where archetypes come from

We cover:
- How TiTrATE rules get formalized
- Archetype generation (with all 150 features)
- Demographics and diagnosis distributions
- Healthcare standards mapping (FHIR, SNOMED CT, LOINC)

**Charts you'll see** (~12 visualizations):
- Diagnosis breakdowns
- Age/gender distributions
- Feature correlation heatmaps
- Data quality dashboards

---

### 03: Phase 2 - XAI-Driven Synthesis (30-40 minutes)
**Good for**: The ML/AI folks

This is where it gets interesting:
- **NMF**: Extract r=20 latent factors from archetypes
- **VAE**: Train a variational autoencoder (architecture: 512→256→128→20→128→256→512)
- **SHAP**: Calculate feature importance for explainability
- **Sampling**: Generate synthetic patients from latent space

**Charts you'll see** (~15 visualizations):
- NMF component heatmaps
- VAE training curves (loss, reconstruction, KL divergence)
- Real vs synthetic comparisons
- Latent space projections
- SHAP importance plots

---

### 04: Phase 3 - Validation (25-35 minutes)
**Good for**: Checking if it actually worked

Multi-level validation:
- **Statistical realism**: KL divergence, Jensen-Shannon, Wasserstein (target: < 0.05)
- **Diagnostic performance**: Train classifiers, check ROC-AUC (target: > 0.80)
- **Triage classification**: ER vs Specialist vs Home
- **Clinical coherence**: Do the synthetic patients make medical sense?

**Charts you'll see** (~20 visualizations):
- ROC curves
- Confusion matrices
- Distribution comparisons
- Comprehensive metrics dashboard

---

### 05: Complete Pipeline (45-60 minutes) ⭐
**Good for**: Reproducing the IEEE Access paper

This runs everything end-to-end with paper-matching config:

```python
n_archetypes = 8400
nmf_components = 20
vae_latent_dim = 20
n_synthetic = 10000
```

**What you get:**
- 8,400 archetypes saved to `data/archetypes/full_archetypes_8400.csv`
- 10,000 synthetic patients in `outputs/synthetic_patients/`
- Trained VAE model in `models/pretrained/`
- Metadata JSON with all the generation details

**Expected metrics:**
- KL Divergence: < 0.05 ✅
- Mean Coherence: > 0.80 ✅
- Total runtime: ~5-10 minutes on CPU, ~2-3 minutes with GPU

---

## Setup (Do This First)

```bash
# Make a virtual environment
python -m venv venv
source venv/bin/activate # Windows: venv\Scripts\activate

# Install stuff
pip install -r ../requirements.txt

# Fire up Jupyter
jupyter lab
```

---

## What You'll Learn

### Technical Stuff
- Formalizing clinical guidelines into code
- NMF for latent factor extraction
- VAE architecture and training
- SHAP for explainability
- Multi-level validation strategies

### Medical AI Concepts
- Synthetic data for healthcare (the privacy-safe way)
- Clinical coherence validation
- Triage systems
- Healthcare standards (FHIR, SNOMED, LOINC)

### Reproducibility
- How to match paper results exactly
- Quality benchmarks and thresholds
- Proper visualization and reporting

---

## Tweaking the Config

In Notebook 05, you can mess with these:

```python
# How much data to generate
n_archetypes = 8400 # More = better coverage, slower
n_synthetic = 10000 # How many fake patients you want

# Model architecture
nmf_components = 20 # Latent factors (10-50 works)
vae_latent_dim = 20 # VAE bottleneck (10-100)
vae_hidden_dims = [512, 256, 128] # Network layers

# Training
vae_epochs = 100 # More epochs = better fit, but slower
vae_batch_size = 64 # Bigger batches = faster, needs more RAM
vae_lr = 1e-3 # Learning rate

# Quality thresholds
kl_threshold = 0.05 # Maximum divergence we accept
roc_auc_threshold = 0.80 # Minimum classifier performance
coherence_threshold = 0.80 # Clinical plausibility cutoff
```

---

## Common Problems

**Out of memory?**
```python
vae_batch_size = 32 # Reduce batch size
n_synthetic = 5000 # Generate fewer patients
```

**Training too slow?**
```python
device = 'cuda' if torch.cuda.is_available() else 'cpu' # Use GPU
vae_epochs = 50 # Reduce epochs for testing
```

**Import errors?**
```bash
# Make sure you're in the virtual environment
source venv/bin/activate
pip install -r ../requirements.txt
```

**Jupyter kernel keeps crashing?**
```bash
# Increase memory limit
jupyter notebook --NotebookApp.max_buffer_size=10000000000
```

---

## Citation

Using these notebooks in your research? Cite the paper:

```bibtex
@article{tritham2025syndx,
 title={SynDX: Explainable AI-Driven Synthetic Data Generation for
 Privacy-Preserving Differential Diagnosis of Vestibular Disorders},
 author={Tritham, Chatchai and Namahoot, Chakkrit Snae},
 journal={IEEE Access},
 year={2025},
 doi={10.1109/ACCESS.2025.XXXXXXX}
}
```

---

## Need Help?

- **Bug reports**: Open an issue on GitHub
- **Questions**: Email chatchai.tritham@nu.ac.th
- **Docs**: Check the main README.md

---

**Checklist before you start:**
- [ ] Python 3.9+ installed
- [ ] Virtual environment set up
- [ ] Dependencies installed
- [ ] Jupyter running
- [ ] At least 8GB RAM (16GB better)
- [ ] GPU with CUDA (optional, but way faster)

---

*Last updated: 2026-01-01*
*Maintainer: Chatchai Tritham, Naresuan University*
