# Reproducibility

This document records, honestly, which manuscript numbers are regenerated from
the committed source code, and which cannot be. Every "code value" below is
produced by the deterministic driver `scripts/run_all.py` (fixed `seed = 42`);
nothing in that driver is hand-typed to match the manuscript. The manuscript has
been reconciled to these computed values.

## How to reproduce

```bash
python -m venv .venv
.\.venv\Scripts\Activate.ps1   # PowerShell  (use source .venv/bin/activate on POSIX)
pip install -r requirements.txt
pip install -e .
python scripts/run_all.py      # writes results/metrics.json + results/metrics.csv
```

`scripts/run_all.py` drives the real package modules end to end:

1. `ArchetypeGenerator` (Phase 1) builds TiTrATE-constraint-validated archetypes
   from documented, diagnosis-specific clinical distributions (age, comorbidity,
   symptom, HINTS/Dix-Hallpike examination, timing/trigger). 10,000 valid
   archetypes are retained across 16 vestibular diagnostic categories.
2. A synthetic cohort is created exactly as `SynDXPipeline.generate` does it
   (archetype feature vectors + small Gaussian noise, `sigma = 0.1`).
3. `StatisticalMetrics.compute_all_metrics` computes per-feature KL / JS /
   Wasserstein between the archetype and synthetic feature matrices.
4. `DiagnosticEvaluator` trains an XGBoost classifier on a 70/30 split. **The
   one-hot diagnosis block is excluded from the inputs to prevent label
   leakage**; leaving it in inflates ROC-AUC toward 1.0.
5. The TiTrATE constraint checker measures the candidate acceptance rate and a
   counterfactual reaction rate (timing-flip perturbation); feature traceability
   is the share of populated features covered by the documented schema.

The run is deterministic: repeated executions produce byte-identical
`results/metrics.json`.

## Computed values (seed = 42)

| Metric | Code value | In manuscript |
|---|---|---|
| Mean per-feature KL divergence (synthetic vs archetype) | 0.071 | abstract, Table (statistical realism) |
| Mean Jensen-Shannon divergence | 0.009 | abstract, Table |
| Mean Wasserstein-1 distance | 0.078 | abstract, Table |
| Downstream classifier macro ROC-AUC (label excluded) | 0.85 | abstract, Table (diagnostic) |
| Downstream classifier macro specificity | 0.96 | Table |
| Downstream classifier macro sensitivity | 0.43 | Table |
| Downstream classifier macro F1 | 0.35 | Table |
| TiTrATE constraint satisfaction (retained cohort) | 100% | Table (clinical reasoning) |
| TiTrATE candidate acceptance rate | 71.6% | Table |
| Feature traceability (populated features) | 100% | Table |
| Counterfactual reaction rate (timing flip) | 72.4% | Table |
| Cohort size / categories | 10,000 / 16 | throughout |

## What CANNOT be reproduced by code (and how the manuscript handles it)

- **Expert plausibility study (3 neurologists, plausibility 4.4/5, % plausible,
  diagnostic correctness, Fleiss' kappa = 0.87).** These are human-study results.
  No expert ratings, rating data, or kappa computation exist in the repository,
  and none can be produced by code. The manuscript **removes** all such reported
  numbers and reframes the expert study explicitly as *planned* future
  validation.
- **Head-to-head baselines (Synthea / MedGAN / CTGAN on our cohort).** These
  external generators were not re-implemented or re-run here. The manuscript
  **drops** all "in our evaluation" baseline numbers and makes no controlled
  numerical comparison against them; any baseline figure that remains is cited
  from the source publication and labelled as such.
- **Quantitative layer ablation with significance tests.** Not part of the
  committed pipeline. The manuscript **removes** the fabricated per-layer table
  and p-values and presents only the design-level role of each layer, flagging a
  controlled ablation as future work.
- **Sensitivity analysis, bias audit, structure validation by clinicians.**
  Reframed in the manuscript as planned/needed analyses rather than completed
  results.

## Honesty notes

- The mean KL (0.071) sits just above the 0.05 target; the median per-feature KL
  is 0, and JS/Wasserstein are very low, so the bulk of features are reproduced
  almost exactly while a few sharp distributions raise the mean. This is reported
  transparently rather than tuned.
- Macro sensitivity/F1 are low because several catch-all categories ("other",
  "undetermined", etc.) carry no distinctive clinical signature; these are
  genuine outputs of a hard 16-class task, not adjusted figures.

## Changes made for reproducibility

- Added `scripts/run_all.py` — deterministic driver writing `results/metrics.json`
  and `results/metrics.csv` from the real modules, with label-leakage removed.
- De-hardcoded figure scripts: SynDX's own values in
  `examples/comparative_academic_charts.py` and
  `scripts/demo_advanced_visualization.py` are read from `results/metrics.json`;
  the fabricated expert-plausibility target was removed from the latter.
- Rewired `scripts/verify_implementation.py` to print real computed metrics from
  `results/metrics.json` instead of hard-coded manuscript "targets".
- Fixed `scripts/demo.py`, which imported a non-existent module
  (`syn_dx_hybrid.pipeline`); it now drives the real `syndx.pipeline.SynDXPipeline`.
- Fixed an indentation bug in `ArchetypeGenerator.to_dataframe` that appended only
  the last row to the DataFrame.
- Added `xgboost` to `requirements.txt` (required by the diagnostic evaluator).
- Committed a small seeded sample of the cohort at `data/archetype_sample.csv`.
