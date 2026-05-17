# SynDX

## Overview

SynDX provides the synthetic dizziness-case generation and validation pipeline
used to support methodological evaluation and reproducibility.

## Installation

```bash
pip install -e .
```

## Quickstart

```python
from syndx import SynDXPipeline

pipeline = SynDXPipeline(n_archetypes=100, nmf_components=5, vae_latent_dim=8)
print(pipeline.n_archetypes)
```

## Repository Structure

- `src/syndx/`: importable package
- `tests/`: automated tests
- `scripts/`: figure generation and validation utilities
- `examples/`: example usage
- `notebooks/`: research notebooks

## Tutorials And Demos

- Scripts:
  - `scripts/generate_all_figures.py`: consolidated figure generation
  - `scripts/generate_manuscript_figures.py`: curated manuscript figure generation
  - `scripts/generate_2d_3d_visualizations.py`: visualization workflow
  - `scripts/validate_xai_framework.py`: XAI validation workflow
  - `scripts/compute_performance_metrics.py`: performance metric generation
- Example scripts:
  - `examples/vestibular_demo.py`
  - `examples/vestibular_demo_with_viz.py`
  - `examples/complete_visualization_suite.py`
  - `examples/academic_visualizations.py`
  - `examples/advanced_academic_charts.py`
- Notebooks:
  - `notebooks/01_Quick_Start_Tutorial.ipynb`
  - `notebooks/02_Phase1_Knowledge_Extraction.ipynb`
  - `notebooks/03_Phase2_VAE_and_XAI_Synthesis.ipynb`
  - `notebooks/04_Phase3_Validation_and_Metrics.ipynb`
  - `notebooks/05_Complete_Pipeline_End_to_End.ipynb`

## Curated Manuscript Figures

Curated manuscript figures are maintained for a manuscript that is still in
preparation. This status does not imply publication, acceptance, or final
journal readiness for every raw demo, legacy, or exploratory image in
`outputs/`.

Regenerate the curated manuscript figure set:

```bash
python scripts/generate_manuscript_figures.py
```

Outputs:

- `figures/manuscript/`: PDF and PNG manuscript figures
- `FIGURE_MANIFEST.csv`: figure role, source script, source data, caption, and
  intended article section

The broader `outputs/` tree remains a reproducibility archive and should not be
treated as the final manuscript figure set unless a figure is promoted into the
manifest.

## Cross-Repository Tutorial Charts

- `../tutorial_surface_comparison.png`: scripts vs examples vs notebooks across all repositories
- `../tutorial_asset_density.png`: interactive/tutorial asset density normalized by repository size
- `../tutorial_maturity_report.md`: combined maturity summary

## Package Scope

The package contains the main pipeline plus phase-specific modules for
knowledge extraction, synthesis, validation, and supporting utilities.

## Source Layout

This repository uses the recommended `src/<package_name>` layout.
Importable code lives in `src/syndx/`.

## Testing

```bash
pytest tests -v
```

## Manuscript Alignment

The SynDX manuscript is still in preparation and owns the synthetic validation
and XAI evidence contribution in the research program. This repository supports
the manuscript's technical claims through:

- formulas: TiTrATE constraint filtering, ensemble weighting, counterfactual
  consistency rate, and computational-complexity expressions
- pseudocode/logic: five-layer pipeline covering combinatorial enumeration,
  Bayesian dependency modeling, rule-based guideline encoding, provenance
  tracking, and counterfactual validation
- data/results: synthetic archetype coverage, synthetic patient generation,
  statistical realism, diagnostic coherence, pathway coverage, and provenance
  traceability
- figure artifacts: clinical SHAP importance, focused validation metrics, and
  counterfactual quality profile under `figures/manuscript/`

The current manuscript package may still use manuscript-local or TikZ/text
figures. `FIGURE_MANIFEST.csv` is the repository-side source of truth for
promoted SynDX figure artifacts.

## License

- MIT; see `LICENSE`

## Contact

### Contact Author

**Chatchai Tritham** (Author)

- Email: [chatchait66@nu.ac.th](mailto:chatchait66@nu.ac.th)
- ORCID: [0000-0001-7899-228X](https://orcid.org/0000-0001-7899-228X)
- Department of Computer Science and Information Technology
- Faculty of Science, Naresuan University
- Phitsanulok 65000, Thailand

### Supervisor

**Chakkrit Snae Namahoot**

- E-mail: [chakkrits@nu.ac.th](mailto:chakkrits@nu.ac.th)
- ORCID: [0000-0003-4660-4590](https://orcid.org/0000-0003-4660-4590)
- Department of Computer Science and Information Technology
- Faculty of Science, Naresuan University
- Phitsanulok 65000, Thailand
