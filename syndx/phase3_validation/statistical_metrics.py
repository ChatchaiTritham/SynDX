"""
Statistical Realism Metrics

Measures how close the synthetic data distribution is to the archetype distribution.

We use three complementary metrics:
- KL Divergence (target: < 0.05) - measures information loss
- Jensen-Shannon Divergence - symmetric version of KL
- Wasserstein Distance - measures "earth mover's distance"

If these numbers are bad, your synthetic data doesn't represent the archetypes.
Equations (13-15) in the paper.
"""

import numpy as np
from scipy import stats
from scipy.spatial.distance import jensenshannon
from scipy.stats import wasserstein_distance
from typing import Dict, Tuple
import logging

logger = logging.getLogger(__name__)


class StatisticalMetrics:
    """
    Compute statistical realism metrics between synthetic and archetype data.

    Implements equations (13-15) from paper.
    """

    @staticmethod
    def kl_divergence(
            p: np.ndarray,
            q: np.ndarray,
            epsilon: float = 1e-10) -> float:
        """
 Compute Kullback-Leibler divergence D_KL(P || Q).

 D_KL(P || Q) = Σ P(i) log(P(i) / Q(i))

 Args:
 p: True distribution (archetypes)
 q: Approximate distribution (synthetic)
 epsilon: Small constant to avoid log(0)

 Returns:
 KL divergence value
 """
        p = np.asarray(p, dtype=float)
        q = np.asarray(q, dtype=float)

        # Normalize to probability distributions
        p = p / (p.sum() + epsilon)
        q = q / (q.sum() + epsilon)

        # Add epsilon to avoid division by zero
        p = p + epsilon
        q = q + epsilon

        # Re-normalize
        p = p / p.sum()
        q = q / q.sum()

        kl = np.sum(p * np.log(p / q))
        return float(kl)

    @staticmethod
    def jensen_shannon_divergence(p: np.ndarray, q: np.ndarray) -> float:
        """
 Compute Jensen-Shannon divergence D_JS(P || Q).

 D_JS(P || Q) = 0.5 * D_KL(P || M) + 0.5 * D_KL(Q || M)
 where M = 0.5 * (P + Q)

 Args:
 p: Distribution 1 (archetypes)
 q: Distribution 2 (synthetic)

 Returns:
 Jensen-Shannon divergence value (0 to 1)
 """
        p = np.asarray(p, dtype=float)
        q = np.asarray(q, dtype=float)

        # Normalize
        p = p / p.sum()
        q = q / q.sum()

        # Use scipy's implementation (returns sqrt of JS divergence)
        js_dist = jensenshannon(p, q)

        # Square to get actual divergence
        js_div = js_dist ** 2

        return float(js_div)

    @staticmethod
    def wasserstein_distance_1d(p_samples: np.ndarray,
                                q_samples: np.ndarray) -> float:
        """
 Compute 1D Wasserstein distance (Earth Mover's Distance).

 W_1(P, Q) = inf E_{(x,y)~γ} [||x - y||]

 Args:
 p_samples: Samples from distribution P
 q_samples: Samples from distribution Q

 Returns:
 Wasserstein-1 distance
 """
        return wasserstein_distance(p_samples, q_samples)

    @staticmethod
    def compute_all_metrics(archetype_data: np.ndarray,
                            synthetic_data: np.ndarray,
                            feature_names: list = None) -> Dict:
        """
 Compute all statistical metrics for each feature.

 Args:
 archetype_data: Archetype feature matrix (n_arch × n_feat)
 synthetic_data: Synthetic feature matrix (n_syn × n_feat)
 feature_names: Optional list of feature names

 Returns:
 Dictionary of metric results
 """
        logger.info("Computing statistical realism metrics...")

        n_features = archetype_data.shape[1]
        if feature_names is None:
            feature_names = [f"feature_{i}" for i in range(n_features)]

        results = {
            "kl_divergence": [],
            "js_divergence": [],
            "wasserstein": [],
            "feature_names": feature_names
        }

        for i in range(n_features):
            arch_feat = archetype_data[:, i]
            syn_feat = synthetic_data[:, i]

            # Create histograms for discrete metrics
            bins = min(50, len(np.unique(arch_feat)))
            p_hist, _ = np.histogram(arch_feat, bins=bins, density=True)
            q_hist, _ = np.histogram(syn_feat, bins=bins, density=True)

            # KL divergence
            kl = StatisticalMetrics.kl_divergence(p_hist, q_hist)
            results["kl_divergence"].append(kl)

            # JS divergence
            js = StatisticalMetrics.jensen_shannon_divergence(p_hist, q_hist)
            results["js_divergence"].append(js)

            # Wasserstein distance
            wass = StatisticalMetrics.wasserstein_distance_1d(
                arch_feat, syn_feat)
            results["wasserstein"].append(wass)

        # Compute summary statistics
        results["summary"] = {
            "mean_kl": np.mean(results["kl_divergence"]),
            "mean_js": np.mean(results["js_divergence"]),
            "mean_wasserstein": np.mean(results["wasserstein"]),
            "median_kl": np.median(results["kl_divergence"]),
            "median_js": np.median(results["js_divergence"]),
            "median_wasserstein": np.median(results["wasserstein"]),
        }

        logger.info(f"Mean KL divergence: {results['summary']['mean_kl']:.4f}")
        logger.info(f"Mean JS divergence: {results['summary']['mean_js']:.4f}")
        logger.info(
            f"Mean Wasserstein: {
                results['summary']['mean_wasserstein']:.4f}")

        return results


if __name__ == "__main__":
    # Test statistical metrics
    logging.basicConfig(level=logging.INFO)

    # Generate test distributions
    np.random.seed(42)
    p = np.random.dirichlet([1] * 10, size=100)
    q = np.random.dirichlet([1.2] * 10, size=100)

    metrics = StatisticalMetrics()

    print("Testing statistical metrics...")
    kl = metrics.kl_divergence(p[0], q[0])
    print(f"KL divergence: {kl:.4f}")

    js = metrics.jensen_shannon_divergence(p[0], q[0])
    print(f"JS divergence: {js:.4f}")

    samples_p = np.random.normal(0, 1, 1000)
    samples_q = np.random.normal(0.1, 1.05, 1000)
    wass = metrics.wasserstein_distance_1d(samples_p, samples_q)
    print(f"Wasserstein distance: {wass:.4f}")

    print("\nStatistical metrics test completed!")
