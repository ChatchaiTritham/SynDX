"""
Focused deterministic unit tests for SynDX pure functions.

Targets:
- StatisticalMetrics (KL / JS / Wasserstein) - phase3_validation.statistical_metrics
- DifferentialPrivacy (gradient clipping, privacy accounting) - phase2_synthesis.differential_privacy
- SNOMEDMapper (label -> code lookup) - utils.snomed_mapper
- constants (package metadata sanity)

These tests use tiny hand-made inputs and seed-42 reproducibility. No network,
no datasets, no training runs.
"""

import sys
from pathlib import Path

import numpy as np
import pytest

# Ensure the repo's src/ is importable even if pytest rootdir differs.
SRC_PATH = Path(__file__).resolve().parents[1] / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

from syndx import constants  # noqa: E402
from syndx.phase2_synthesis.differential_privacy import DifferentialPrivacy  # noqa: E402
from syndx.phase3_validation.statistical_metrics import StatisticalMetrics  # noqa: E402
from syndx.utils.snomed_mapper import SNOMEDMapper  # noqa: E402


# ---------------------------------------------------------------------------
# StatisticalMetrics
# ---------------------------------------------------------------------------

def test_kl_divergence_identical_is_zero():
    """KL(P || P) must be exactly 0 (no information loss for identical dist)."""
    p = np.array([0.2, 0.3, 0.5])
    kl = StatisticalMetrics.kl_divergence(p, p)
    assert kl == pytest.approx(0.0, abs=1e-9)


def test_kl_divergence_nonnegative_and_handles_unnormalized():
    """KL is >= 0 and the function normalizes raw (unnormalized) counts."""
    p = np.array([10.0, 20.0, 30.0])  # not a probability vector
    q = np.array([30.0, 20.0, 10.0])
    kl = StatisticalMetrics.kl_divergence(p, q)
    assert kl > 0.0
    assert np.isfinite(kl)


def test_js_divergence_bounds_and_symmetry():
    """JS divergence is in [0, 1], zero for identical, and symmetric."""
    p = np.array([0.1, 0.4, 0.5])
    q = np.array([0.5, 0.4, 0.1])
    js_pq = StatisticalMetrics.jensen_shannon_divergence(p, q)
    js_qp = StatisticalMetrics.jensen_shannon_divergence(q, p)
    js_self = StatisticalMetrics.jensen_shannon_divergence(p, p)
    assert 0.0 <= js_pq <= 1.0
    assert js_pq == pytest.approx(js_qp, abs=1e-9)  # symmetric
    assert js_self == pytest.approx(0.0, abs=1e-9)


def test_wasserstein_known_shift():
    """1D Wasserstein between two delta-like sample sets equals the offset."""
    a = np.zeros(100)
    b = np.full(100, 3.0)
    w = StatisticalMetrics.wasserstein_distance_1d(a, b)
    assert w == pytest.approx(3.0, abs=1e-9)


def test_compute_all_metrics_structure_and_seed42_reproducible():
    """compute_all_metrics returns expected keys with per-feature lists,
    and is reproducible under a fixed seed."""
    def make_data():
        rng = np.random.RandomState(42)
        arch = rng.randn(60, 4)
        synth = rng.randn(60, 4)
        return arch, synth

    arch1, synth1 = make_data()
    res1 = StatisticalMetrics.compute_all_metrics(arch1, synth1)

    for key in ("kl_divergence", "js_divergence", "wasserstein", "summary"):
        assert key in res1
    assert len(res1["kl_divergence"]) == 4  # one entry per feature
    assert set(res1["summary"]) >= {"mean_kl", "mean_js", "mean_wasserstein"}

    arch2, synth2 = make_data()
    res2 = StatisticalMetrics.compute_all_metrics(arch2, synth2)
    assert res2["summary"]["mean_kl"] == pytest.approx(res1["summary"]["mean_kl"])
    assert res2["summary"]["mean_wasserstein"] == pytest.approx(
        res1["summary"]["mean_wasserstein"]
    )


# ---------------------------------------------------------------------------
# DifferentialPrivacy
# ---------------------------------------------------------------------------

def test_dp_noise_multiplier_calibration_formula():
    """Auto-calibrated noise multiplier matches the analytic Gaussian formula
    sigma = C * sqrt(2 ln(1.25/delta)) / eps, with multiplier = sigma / C."""
    dp = DifferentialPrivacy(epsilon=1.0, delta=1e-5, clip_norm=1.0)
    expected = np.sqrt(2 * np.log(1.25 / 1e-5)) / 1.0
    assert dp.noise_multiplier == pytest.approx(expected, rel=1e-9)
    assert dp.noise_multiplier > 0


def test_dp_clip_gradients_bounds_norm_and_stats():
    """Gradients above clip_norm are scaled down to exactly clip_norm; small
    gradients are untouched; reported stats are correct."""
    dp = DifferentialPrivacy(epsilon=1.0, delta=1e-5, clip_norm=1.0)
    big = np.array([3.0, 4.0])      # L2 norm = 5.0 -> must be clipped to 1.0
    small = np.array([0.1, 0.0])    # L2 norm = 0.1 -> untouched
    clipped, stats = dp.clip_gradients([big, small])

    assert np.linalg.norm(clipped[0]) == pytest.approx(1.0, abs=1e-9)
    assert np.allclose(clipped[1], small)
    assert stats["max_norm"] == pytest.approx(5.0, abs=1e-9)
    assert stats["clip_fraction"] == pytest.approx(0.5, abs=1e-9)  # 1 of 2 clipped


def test_dp_clip_gradients_empty_raises():
    """Empty gradient list must raise ValueError (guard clause)."""
    dp = DifferentialPrivacy(epsilon=1.0, delta=1e-5, clip_norm=1.0)
    with pytest.raises(ValueError):
        dp.clip_gradients([])


def test_dp_privacy_budget_monotonic_and_check():
    """Privacy spent increases with steps, and check/get are consistent."""
    dp = DifferentialPrivacy(epsilon=1.0, delta=1e-5, clip_norm=1.0)
    eps_small, _ = dp.get_privacy_spent(steps=10, batch_size=64, n_samples=8400)
    eps_large, _ = dp.get_privacy_spent(steps=1000, batch_size=64, n_samples=8400)
    assert eps_large > eps_small
    assert eps_small >= 0.0
    # check_privacy_budget agrees with the raw epsilon comparison
    ok = dp.check_privacy_budget(steps=10, batch_size=64, n_samples=8400)
    assert ok == (eps_small <= dp.epsilon)


# ---------------------------------------------------------------------------
# SNOMEDMapper
# ---------------------------------------------------------------------------

def test_snomed_mapper_known_and_unknown():
    """Known terms map (case/space-insensitive); unknown -> UNKNOWN."""
    m = SNOMEDMapper()
    assert m.map_term("  Vertigo ")["code"] == "C0042571"
    assert m.map_term("STROKE")["code"] == "C0038454"
    out = m.map_term("nonexistent-term")
    assert out["code"] == "UNKNOWN"
    assert out["term"] == "nonexistent-term"  # original term preserved


# ---------------------------------------------------------------------------
# constants
# ---------------------------------------------------------------------------

def test_constants_sane_defaults():
    """Package metadata constants have the documented values/ranges."""
    assert constants.PACKAGE_NAME == "syndx"
    assert constants.DEFAULT_RANDOM_SEED == 42
    assert 0.0 < constants.DEFAULT_CONFIDENCE_LEVEL < 1.0
    assert len(constants.DEFAULT_FIGURE_SIZE) == 2
