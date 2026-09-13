"""
Scalability benchmark for the full 3-layer + ensemble pipeline.

The manuscript's Table 7 (runtime 4.2-467.5 min, peak RAM, R^2=0.998 log-log
fit up to N=100,000) had no backing script anywhere in the repository (see
DMKD_DEEP_AUDIT.md, item 6). This script actually measures wall-clock
runtime and peak RSS memory for the real Layer1+Layer2+Layer3+merge pipeline
(scripts/run_full_pipeline.py's core, minus the diagnostic-classifier
training step, which is a fixed downstream cost independent of N) at a set
of cohort sizes. It reports ONLY sizes that were actually run in this
environment; it does not extrapolate to sizes larger than measured.

IMPORTANT (methodology note): runtime and peak memory are measured in TWO
SEPARATE passes per size. tracemalloc's per-allocation Python-level tracing
adds substantial overhead to allocation-heavy code -- confirmed empirically
here (build_layer3(10000) took ~4.3 min with tracemalloc off vs. ~35 min
with it on for the whole timed region, roughly 8x inflation). Running both
measurements in one tracemalloc-active pass therefore produces a
contaminated, unusable wall-clock number. Pass 1 times the pipeline with
plain time.perf_counter() and tracemalloc fully OFF. Pass 2 re-runs the same
pipeline (same n, same seed -> same deterministic workload) with tracemalloc
on, discarding its elapsed time and keeping only the peak-memory reading.

Run:
    python scripts/scalability_benchmark.py
"""

import sys
import time
import tracemalloc
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

from run_full_pipeline import build_layer1, build_layer2, build_layer3, weighted_pool_merge  # noqa: E402

SEED = 42
SIZES = [500, 1000, 2000, 5000, 10000]
WEIGHTS = None  # filled from grid search output if available

RESULTS_DIR = REPO_ROOT / "results"


def run_pipeline(n, weights):
    """Runs the full pipeline once and returns the ensemble output length."""
    _df1, X1, y1, _ = build_layer1(n)
    _df2, X2, y2, _ = build_layer2(n)
    _df3, X3, y3, _c, _g = build_layer3(n)
    X_ens, y_ens, _src, _q = weighted_pool_merge([X1, X2, X3], [y1, y2, y3], weights, n, seed=SEED)
    return len(X_ens)


def run_one(n, weights):
    # Pass 1: true wall-clock timing, tracemalloc OFF (profiler-free).
    t0 = time.perf_counter()
    n_out = run_pipeline(n, weights)
    elapsed = time.perf_counter() - t0

    # Pass 2: separate re-run, tracemalloc ON, for peak memory only.
    # Same n/seed => same deterministic workload; this pass's elapsed time
    # is discarded because tracemalloc's tracing overhead inflates it.
    tracemalloc.start()
    _n_out2 = run_pipeline(n, weights)
    _cur, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    return elapsed, peak / (1024 * 1024), n_out


def main():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    gs_path = RESULTS_DIR / "ensemble_grid_search.csv"
    if gs_path.exists():
        gdf = pd.read_csv(gs_path)
        best = gdf.loc[gdf["macro_f1"].idxmax()]
        weights = [best["w_comb"], best["w_bayes"], best["w_rules"]]
    else:
        weights = [1 / 3, 1 / 3, 1 / 3]
        print("[scalability] no grid-search CSV found, using equal weights for the benchmark only")

    rows = []
    for n in SIZES:
        print(f"[scalability] running n={n}...")
        elapsed, peak_mb, n_out = run_one(n, weights)
        rows.append({
            "n_patients": n,
            "runtime_seconds": elapsed,
            "runtime_minutes": elapsed / 60.0,
            "peak_memory_mb": peak_mb,
            "n_ensemble_rows_produced": n_out,
        })
        print(f"    runtime={elapsed:.2f}s  peak_mem={peak_mb:.1f}MB")

    df = pd.DataFrame(rows)
    # Log-log linear fit over the sizes actually measured, for reference only.
    logn = np.log10(df["n_patients"].values)
    logt = np.log10(df["runtime_seconds"].values)
    slope, intercept = np.polyfit(logn, logt, 1)
    pred = slope * logn + intercept
    ss_res = np.sum((logt - pred) ** 2)
    ss_tot = np.sum((logt - np.mean(logt)) ** 2)
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else float("nan")

    df.to_csv(RESULTS_DIR / "scalability_benchmark.csv", index=False)
    with open(RESULTS_DIR / "scalability_fit.txt", "w") as f:
        f.write(f"log-log slope (complexity exponent): {slope:.4f}\n")
        f.write(f"R^2 of log-log fit: {r2:.4f}\n")
        f.write(f"Measured sizes: {SIZES}\n")
        f.write("No sizes beyond the measured range are reported or extrapolated.\n")

    print(f"\n[scalability] log-log slope = {slope:.3f} (1.0 = linear O(N)), R^2 = {r2:.4f}")
    print(f"[scalability] results written to {RESULTS_DIR / 'scalability_benchmark.csv'}")


if __name__ == "__main__":
    main()
