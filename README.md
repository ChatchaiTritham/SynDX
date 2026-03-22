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

## Citation

- cite the thesis/manuscript sections covering SynDX synthetic data generation and validation

## License

- MIT; see `LICENSE`

## Contact

### Contact Author

**Chatchai Tritham** (PhD Candidate)

- Email: [chatchait66@nu.ac.th](mailto:chatchait66@nu.ac.th)
- Department of Computer Science and Information Technology
- Faculty of Science, Naresuan University
- Phitsanulok 65000, Thailand

### Supervisor

**Chakkrit Snae Namahoot**

- Email: [chakkrits@nu.ac.th](mailto:chakkrits@nu.ac.th)
- Department of Computer Science
- Faculty of Science, Naresuan University
- Phitsanulok 65000, Thailand
