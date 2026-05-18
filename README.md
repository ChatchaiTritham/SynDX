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
pip install -e .
python -m pytest -q
```

If figure-generation scripts are present, run the matching script listed in `FIGURE_MANIFEST.csv` from the repository root.

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

## Contact

**Chatchai Tritham**  
Department of Computer Science and Information Technology, Faculty of Science, Naresuan University, Phitsanulok 65000, Thailand  
Email: chatchait66@nu.ac.th  
ORCID: 0000-0001-7899-228X

**Chakkrit Snae Namahoot**  
Department of Computer Science and Information Technology, Faculty of Science, Naresuan University, Phitsanulok 65000, Thailand  
Email: chakkrits@nu.ac.th  
ORCID: 0000-0003-4660-4590