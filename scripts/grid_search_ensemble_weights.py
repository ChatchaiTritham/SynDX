"""
Grid search over the three-layer ensemble weights [w_comb, w_bayes, w_rules].

The manuscript reported w = [0.35, 0.30, 0.35] "from a grid search," but no
grid-search script existed in the repository and the code's actual default
was a 5-element vector [0.25, 0.20, 0.25, 0.15, 0.15] (see
src/syndx/syndx_temp/ensemble_integration/weighted_merger.py and
main_pipeline.py). This script performs a REAL grid search and its output
(results/ensemble_grid_search.csv) is what run_full_pipeline.py reads to
pick the reported weights - no weight in the manuscript is hand-typed.

Search space: w_comb, w_bayes, w_rules on a 0.1 grid over {0.1, ..., 0.8},
constrained to sum to 1.0 (all (w1, w2) pairs with w3 = 1 - w1 - w2 >= 0.1).
Selection criterion: macro-F1 of an XGBoost classifier (seed 42) trained on
the merged ensemble cohort and evaluated on a held-out 30% split, using the
shared 7-class diagnosis taxonomy defined in run_full_pipeline.py. Each grid
point is evaluated on an independent validation pool (separate from the
final N=10,000 pipeline run in run_full_pipeline.py) to avoid selecting
weights on the same data used for the reported headline numbers.

Run:
    python scripts/grid_search_ensemble_weights.py
"""

import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC = REPO_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

sys.path.insert(0, str(REPO_ROOT / "scripts"))
from run_full_pipeline import (  # noqa: E402
    build_layer1,
    build_layer2,
    build_layer3,
    macro_f1_for_weights,
)

SEED = 42
N_VALIDATION_POOL = 6000  # smaller, independent pool used only for weight selection
GRID_STEP = 0.1
MIN_WEIGHT = 0.1

RESULTS_DIR = REPO_ROOT / "results"


def grid_points(step=GRID_STEP, min_w=MIN_WEIGHT):
    vals = np.round(np.arange(min_w, 0.81, step), 2)
    pts = []
    for w1 in vals:
        for w2 in vals:
            w3 = round(1.0 - w1 - w2, 2)
            if w3 >= min_w - 1e-9:
                pts.append((round(w1, 2), round(w2, 2), w3))
    return pts


def main():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    print("[grid_search] building validation pools for Layers 1-3 "
          f"(n={N_VALIDATION_POOL} each, seed={SEED})...")
    _df1, X1, y1, _ = build_layer1(N_VALIDATION_POOL)
    _df2, X2, y2, _ = build_layer2(N_VALIDATION_POOL)
    _df3, X3, y3, _cols3, _gen = build_layer3(N_VALIDATION_POOL)

    pts = grid_points()
    print(f"[grid_search] evaluating {len(pts)} weight combinations...")

    rows = []
    for (w1, w2, w3) in pts:
        f1 = macro_f1_for_weights(
            [w1, w2, w3], [X1, X2, X3], [y1, y2, y3],
            total_n=4000, seed=SEED,
        )
        rows.append({"w_comb": w1, "w_bayes": w2, "w_rules": w3, "macro_f1": f1})

    grid_df = pd.DataFrame(rows).sort_values("macro_f1", ascending=False).reset_index(drop=True)
    grid_df.to_csv(RESULTS_DIR / "ensemble_grid_search.csv", index=False)

    best = grid_df.iloc[0]
    print("\n[grid_search] top 5 weight combinations by macro-F1:")
    print(grid_df.head(5).to_string(index=False))
    print(f"\n[grid_search] SELECTED weights: "
          f"w_comb={best['w_comb']}, w_bayes={best['w_bayes']}, w_rules={best['w_rules']} "
          f"(macro-F1={best['macro_f1']:.4f})")
    print(f"[grid_search] full grid written to: {RESULTS_DIR / 'ensemble_grid_search.csv'}")


if __name__ == "__main__":
    main()
