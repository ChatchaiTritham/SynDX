# SynDX

## Overview

Synthetic validation and explainability evidence package for dizziness-focused clinical decision support experiments.

This repository is part of an eight-repository clinical decision-support research portfolio. Current status: manuscript or component package in preparation. The repository role is **manuscript**.

## Standard Repository Layout

| Path | Purpose |
|---|---|
| `src/` | Package source code: `syndx` |
| `tests/` | Unit, smoke, and behavior checks |
| `scripts/` | Reproducibility and export scripts |
| `examples/` | Runnable examples and demonstrations |
| `figures/`, `visualizations/`, `outputs/`, `results/` | Generated visual and result artifacts |
| `data/`, `models/`, `evaluation/` | Dataset, model, and evaluation assets when used by this repo |
| `FIGURE_MANIFEST.csv` | Curated figure inventory for manuscript or component evidence |
| `pyproject.toml`, `setup.py`, `requirements.txt`, `pytest.ini` | Python package and test configuration |

## Architecture Flow

```mermaid
flowchart LR
    A[Input data or scenario] --> B[Core package logic]
    B --> C[Safety and quality checks]
    C --> D[Metrics and audit outputs]
    D --> E[Curated figures and result artifacts]
```

## Core Logic

- Build archetype and synthetic cohorts.
- Evaluate diagnostic and privacy behavior.
- Compute SHAP, NMF, and counterfactual evidence.
- Export focused validation figures.

## Key Formulas And Rules

- Utility gap: Delta = metric_synthetic - metric_archetype
- XAI fidelity: F = mean(corr(SHAP_real, SHAP_synth), rank_agreement, top_k_overlap)
- Counterfactual quality: Q = g(sparsity, plausibility, diversity, proximity)

## Data, Results, Charts, And Graphs

The curated visual set is controlled by FIGURE_MANIFEST.csv and currently lists **3** figure entries. The manifest links figure IDs, roles, source scripts, source data, captions, sections, timestamps, and export DPI.

| ID | Role | PNG | PDF |
|---|---|---|---|
| SynDX-F1 | manuscript | `figures\manuscript\fig1_shap_importance_clinical.png` | `figures\manuscript\fig1_shap_importance_clinical.pdf` |
| SynDX-F2 | manuscript | `figures\manuscript\fig2_focused_validation_metrics.png` | `figures\manuscript\fig2_focused_validation_metrics.pdf` |
| SynDX-F3 | manuscript | `figures\manuscript\fig3_counterfactual_quality_profile.png` | `figures\manuscript\fig3_counterfactual_quality_profile.pdf` |

## Reproduce

```powershell
cd D:\PhD-NU\Manuscript\GitHub\SynDX
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
pip install -e .

# Regenerate every headline metric deterministically (seed = 42):
python scripts\run_all.py
```

`scripts/run_all.py` builds a seeded synthetic vestibular cohort with the real
package modules, computes the metrics below, and writes them to
`results/metrics.json` and `results/metrics.csv`. Running it twice yields
byte-identical output. The figure scripts in `examples/` read SynDX's own values
from `results/metrics.json` rather than hard-coded constants.

If figure-generation scripts are present, run the matching script listed in `FIGURE_MANIFEST.csv` from the repository root.

## Reproduced Metrics (seed = 42)

All values below are the actual output of `python scripts/run_all.py` on a
10,000-record seeded synthetic cohort across 16 vestibular diagnostic
categories. They are computed, not asserted; the manuscript numbers are
reconciled to them.

| Metric | Value |
|---|---|
| Cohort size | 10,000 archetypes |
| Diagnostic categories | 16 |
| Statistical realism: mean per-feature KL divergence (synthetic vs archetype) | 0.071 |
| Statistical realism: mean Jensen-Shannon divergence | 0.009 |
| Statistical realism: mean Wasserstein-1 distance | 0.078 |
| Downstream classifier (XGBoost, 70/30, label excluded): macro ROC-AUC | 0.85 |
| Downstream classifier: macro specificity | 0.96 |
| Downstream classifier: macro sensitivity | 0.43 |
| Downstream classifier: macro F1 | 0.35 |
| TiTrATE constraint satisfaction (retained cohort) | 100% |
| TiTrATE candidate acceptance rate | 71.6% |
| Guideline traceability of populated features | 100% |
| Counterfactual reaction rate (timing-flip perturbation) | 72.4% |

**Honesty notes.** The diagnosis one-hot block is excluded from the classifier
input to prevent label leakage (leaving it in inflates ROC-AUC toward 1.0). The
lower macro sensitivity/F1 are driven by under-determined catch-all categories
(e.g. "other", "undetermined") with no distinctive clinical signature; these are
genuine outputs, not tuned figures. No expert-plausibility, inter-rater (Fleiss
kappa), or real-patient metric is produced here, because none can be reproduced
by code; those are described in the manuscript as planned validation.

## Verification Criteria

- Root metadata and package files are present.
- Source paths follow `src/<package>/...` where the package shape allows it.
- Tests pass with `python -m pytest -q`.
- Curated figures are listed in `FIGURE_MANIFEST.csv` rather than inferred from every raw image file.
- Manuscript status wording stays conservative: in preparation, implementation, supplementary, or reproducibility/component evidence as appropriate.
- No local manuscript path, external assistant wording, or software metadata block is kept in the repository text.

## Portfolio Relationship

| Repository | Role |
|---|---|
| BASICS-CDSS | Beyond-accuracy evaluation methodology |
| TRI-X | Framework-level package |
| ORASR | Routing and safety-action component |
| DRAS-5 | Dynamic risk-state component |
| SAFE-Gate | Safety-gated ensemble framework |
| SynDX | Synthetic validation and explainability evidence |
| SURgul | SRGL/governance reproducibility component |
| TRI-X-CDSS | Integration and implementation package |
| Selective-CDSS | Risk-controlled selective-prediction (abstention) component |
| Causal-CDSS | Causal-inference evaluation component |
| Beyond-Accuracy | Simulation-based safety/calibration evaluation framework |

## Contact

**Chatchai Tritham**  
Department of Computer Science and Information Technology, Faculty of Science, Naresuan University, Phitsanulok 65000, Thailand  
Email: chatchait66@nu.ac.th  
ORCID: 0000-0001-7899-228X

**Chakkrit Snae Namahoot**  
Department of Computer Science and Information Technology, Faculty of Science, Naresuan University, Phitsanulok 65000, Thailand  
Email: chakkrits@nu.ac.th  
ORCID: 0000-0003-4660-4590