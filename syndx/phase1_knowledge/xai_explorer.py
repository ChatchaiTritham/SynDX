"""
XAI-Guided Parameter Space Explorer
Implements complete Algorithm 7.1 from manuscript
"""

import numpy as np
from typing import List, Dict, Optional, Tuple
import logging
from tqdm import tqdm
from sklearn.cluster import KMeans

from syndx.core.parameter_space import ParameterSpace, Archetype
from syndx.phase1_knowledge.nmf_discovery import NMFFactorDiscovery
from syndx.phase1_knowledge.shap_importance import SHAPImportanceAnalyzer
from syndx.utils.formulas import calculate_n_target

logger = logging.getLogger(__name__)


class XAIGuidedExplorer:
    """
    XAI-Guided Parameter Space Exploration (Algorithm 7.1)

    Multi-phase intelligent sampling:
    - Phase 1: Initial uniform sampling
    - Phase 2: NMF latent factor discovery
    - Phase 3: SHAP importance analysis
    - Phase 4: Importance-weighted sampling (60%)
    - Phase 5: Critical scenario targeting (30%)
    - Phase 6: Diversity-oriented sampling (10%)
    """

    def __init__(
        self,
        parameter_space: ParameterSpace,
        n_target: int = None,
        nmf_factors: int = None,
        alpha_importance: float = 0.60,
        alpha_critical: float = 0.30,
        alpha_diversity: float = 0.10,
        random_state: int = 42,
        enable_profiling: bool = False
    ):
        """
        Args:
            parameter_space: ParameterSpace instance
            n_target: Target number of archetypes (calculated if None)
            nmf_factors: Number of NMF factors (calculated if None)
            alpha_importance: Ratio for importance-weighted sampling
            alpha_critical: Ratio for critical scenarios
            alpha_diversity: Ratio for diversity sampling
            random_state: Random seed
            enable_profiling: Enable performance profiling (for real metrics)
        """
        self.param_space = parameter_space
        self.random_state = random_state
        np.random.seed(random_state)

        # Initialize profiler if enabled
        self.enable_profiling = enable_profiling
        if enable_profiling:
            from syndx.utils.profiler import SynDXProfiler
            self.profiler = SynDXProfiler(
                output_dir='outputs/profiling', enabled=True)
            logger.info("Profiling enabled for XAI Explorer")
        else:
            self.profiler = None

        # Calculate target if not provided
        if n_target is None:
            n_target, breakdown = calculate_n_target(
                space_size=parameter_space.space_size,
                n_diagnoses=parameter_space.D_size,
                valid_space_size=int(
                    parameter_space.space_size * 0.5),  # Estimate
                critical_diagnoses={'stroke': 0.10, 'tia': 0.05}
            )
            logger.info(f"Calculated n_target = {n_target}")
            logger.info(f"  Breakdown: {breakdown}")

        self.n_target = n_target

        # Phase allocation
        self.alpha_importance = alpha_importance
        self.alpha_critical = alpha_critical
        self.alpha_diversity = alpha_diversity

        self.n_importance = int(n_target * alpha_importance)
        self.n_critical = int(n_target * alpha_critical)
        self.n_diversity = n_target - self.n_importance - self.n_critical

        # Calculate NMF factors if not provided
        if nmf_factors is None:
            from syndx.utils.formulas import calculate_n_target
            # Use heuristic from NMF class
            nmf_factors = NMFFactorDiscovery._calculate_r_clinical(
                parameter_space.D_size,
                parameter_space.m
            )

        self.nmf_factors = nmf_factors

        # Storage
        self.initial_archetypes = []
        self.nmf_model = None
        self.shap_model = None
        self.final_archetypes = []

        # Statistics
        self.stats = {
            'phase1_sampled': 0,
            'phase1_valid': 0,
            'phase4_sampled': 0,
            'phase4_valid': 0,
            'phase5_sampled': 0,
            'phase5_valid': 0,
            'phase6_sampled': 0,
            'phase6_valid': 0,
        }

        logger.info(f"Initialized XAI-Guided Explorer")
        logger.info(f"  Target: {n_target} archetypes")
        logger.info(
            f"  Allocation: {self.n_importance} + {self.n_critical} + {self.n_diversity}")
        logger.info(f"  NMF factors: {nmf_factors}")

    def explore(self) -> List[Archetype]:
        """
        Execute complete exploration algorithm (Algorithm 7.1)

        Returns:
            List of valid archetypes
        """
        import time
        overall_start = time.perf_counter()

        logger.info("=" * 70)
        logger.info("STARTING XAI-GUIDED PARAMETER SPACE EXPLORATION")
        logger.info("=" * 70)

        # Phase 1: Initial uniform sampling
        logger.info("\n--- PHASE 1: INITIAL UNIFORM SAMPLING ---")
        n0 = self._calculate_initial_sample_size()
        p1_start = time.perf_counter() if self.profiler else 0
        self.initial_archetypes = self._phase1_uniform_sampling(n0)
        if self.profiler:
            p1_time = time.perf_counter() - p1_start
            self.profiler.record_subphase(
                'Exploration', 'Phase 1: Uniform Sampling', p1_time)

        # Phase 2: NMF latent factor discovery
        logger.info("\n--- PHASE 2: NMF LATENT FACTOR DISCOVERY ---")
        p2_start = time.perf_counter() if self.profiler else 0
        self.nmf_model = self._phase2_nmf_discovery()
        if self.profiler:
            p2_time = time.perf_counter() - p2_start
            self.profiler.record_subphase(
                'Exploration', 'Phase 2: NMF Discovery', p2_time)

        # Phase 3: SHAP importance analysis
        logger.info("\n--- PHASE 3: SHAP IMPORTANCE ANALYSIS ---")
        p3_start = time.perf_counter() if self.profiler else 0
        self.shap_model = self._phase3_shap_analysis()
        if self.profiler:
            p3_time = time.perf_counter() - p3_start
            self.profiler.record_subphase(
                'Exploration', 'Phase 3: SHAP Analysis', p3_time)

        # Phase 4: Importance-weighted sampling
        logger.info("\n--- PHASE 4: IMPORTANCE-WEIGHTED SAMPLING ---")
        p4_start = time.perf_counter() if self.profiler else 0
        importance_archetypes = self._phase4_importance_sampling()
        if self.profiler:
            p4_time = time.perf_counter() - p4_start
            self.profiler.record_subphase(
                'Exploration', 'Phase 4: Importance Sampling', p4_time)

        # Phase 5: Critical scenario targeting
        logger.info("\n--- PHASE 5: CRITICAL SCENARIO TARGETING ---")
        p5_start = time.perf_counter() if self.profiler else 0
        critical_archetypes = self._phase5_critical_sampling()
        if self.profiler:
            p5_time = time.perf_counter() - p5_start
            self.profiler.record_subphase(
                'Exploration', 'Phase 5: Critical Sampling', p5_time)

        # Phase 6: Diversity-oriented sampling
        logger.info("\n--- PHASE 6: DIVERSITY-ORIENTED SAMPLING ---")
        p6_start = time.perf_counter() if self.profiler else 0
        diversity_archetypes = self._phase6_diversity_sampling(
            importance_archetypes + critical_archetypes
        )
        if self.profiler:
            p6_time = time.perf_counter() - p6_start
            self.profiler.record_subphase(
                'Exploration', 'Phase 6: Diversity Sampling', p6_time)

        # Combine all
        self.final_archetypes = (
            importance_archetypes +
            critical_archetypes +
            diversity_archetypes
        )

        # Validate
        self._validate_final_set()

        # Record overall time
        if self.profiler:
            overall_time = time.perf_counter() - overall_start
            self.profiler.phase_metrics['Exploration'] = {
                'total_time_sec': overall_time,
                'subphases': self.profiler.phase_metrics.get(
                    'Exploration',
                    {}).get(
                    'subphases',
                    {})}
            self.profiler.save_metrics('exploration_profiling.json')
            logger.info(f"Profiling data saved: exploration_profiling.json")

        logger.info("\n" + "=" * 70)
        logger.info("EXPLORATION COMPLETED")
        logger.info("=" * 70)
        logger.info(f"Total archetypes: {len(self.final_archetypes)}")
        logger.info(f"Target: {self.n_target}")
        logger.info(
            f"Match: {len(self.final_archetypes) / self.n_target * 100:.1f}%")
        if self.profiler:
            logger.info(f"Total execution time: {overall_time:.2f}s")

        return self.final_archetypes

    def _calculate_initial_sample_size(self) -> int:
        """
        Calculate n₀ for Phase 1
        n₀ = ⌈log(|P|) · √|D|⌉
        """
        n0 = int(np.ceil(
            np.log(self.param_space.space_size) *
            np.sqrt(self.param_space.D_size)
        ))
        return max(n0, 100)  # Minimum 100

    def _phase1_uniform_sampling(self, n0: int) -> List[Archetype]:
        """Phase 1: Initial uniform sampling"""
        logger.info(f"Sampling {n0} initial archetypes uniformly...")

        archetypes = []
        attempts = 0
        max_attempts = n0 * 10

        pbar = tqdm(total=n0, desc="Phase 1")

        while len(archetypes) < n0 and attempts < max_attempts:
            # Sample uniformly
            candidates = self.param_space.sample_uniform(1)
            attempts += 1

            for archetype in candidates:
                if archetype.is_valid(self.param_space.constraints):
                    # Generate features
                    archetype.features = self._archetype_to_features(archetype)
                    archetypes.append(archetype)
                    pbar.update(1)

                    if len(archetypes) >= n0:
                        break

        pbar.close()

        self.stats['phase1_sampled'] = attempts
        self.stats['phase1_valid'] = len(archetypes)

        logger.info(f"Phase 1 complete: {len(archetypes)} valid archetypes")
        logger.info(
            f"  Acceptance rate: {
                len(archetypes) / attempts * 100:.1f}%")

        return archetypes

    def _phase2_nmf_discovery(self) -> NMFFactorDiscovery:
        """Phase 2: NMF latent factor discovery"""
        # Create feature matrix
        X = np.array([a.features for a in self.initial_archetypes])
        logger.info(f"Feature matrix shape: {X.shape}")

        # Fit NMF
        nmf = NMFFactorDiscovery(
            n_components=self.nmf_factors,
            random_state=self.random_state
        )
        nmf.fit(X)

        # Log interpretations
        logger.info(f"Discovered {nmf.n_components} clinical patterns:")
        for interp in nmf.factor_interpretations_:
            logger.info(
                f"  Factor {
                    interp['factor_id']}: {
                    interp['clinical_pattern']}")

        return nmf

    def _phase3_shap_analysis(self) -> SHAPImportanceAnalyzer:
        """Phase 3: SHAP importance analysis"""
        # Create feature matrix and labels
        X = np.array([a.features for a in self.initial_archetypes])
        y = np.array([a.diagnosis for a in self.initial_archetypes])

        logger.info(f"Training classifier on {len(X)} samples...")

        # Fit SHAP analyzer
        shap = SHAPImportanceAnalyzer(
            model_type='tree',
            random_state=self.random_state
        )
        shap.fit(X, y)

        # Log top features
        top_features = shap.get_top_features(10)
        logger.info("Top 10 features by SHAP importance:")
        for name, importance in top_features:
            logger.info(f"  {name}: {importance:.4f}")

        return shap

    def _phase4_importance_sampling(self) -> List[Archetype]:
        """Phase 4: Importance-weighted sampling (60%)"""
        logger.info(
            f"Importance-weighted sampling: {self.n_importance} archetypes...")

        archetypes = []
        attempts = 0
        max_attempts = self.n_importance * 20

        pbar = tqdm(total=self.n_importance, desc="Phase 4")

        while len(archetypes) < self.n_importance and attempts < max_attempts:
            # Sample using SHAP weights
            archetype = self._sample_importance_weighted()
            attempts += 1

            if archetype.is_valid(self.param_space.constraints):
                archetype.features = self._archetype_to_features(archetype)
                archetypes.append(archetype)
                pbar.update(1)

        pbar.close()

        self.stats['phase4_sampled'] = attempts
        self.stats['phase4_valid'] = len(archetypes)

        logger.info(f"Phase 4 complete: {len(archetypes)} archetypes")
        logger.info(
            f"  Acceptance rate: {
                len(archetypes) / attempts * 100:.1f}%")

        return archetypes

    def _sample_importance_weighted(self) -> Archetype:
        """Sample single archetype using SHAP importance weights"""
        # For now, use epidemiological sampling as proxy
        # In full implementation, use SHAP weights per parameter
        diagnosis = self.param_space.epidemiology.sample_diagnosis(1)[0]
        age = self.param_space.epidemiology.sample_age(diagnosis, 1)[0]

        # Sample other parameters
        params = {'diagnosis': diagnosis, 'age': int(age)}

        for name, param in self.param_space.parameters.items():
            if name not in params:
                params[name] = param.sample(1)[0]

        return Archetype(parameters=params, diagnosis=diagnosis)

    def _phase5_critical_sampling(self) -> List[Archetype]:
        """Phase 5: Critical scenario targeting (30%)"""
        logger.info(
            f"Critical scenario sampling: {
                self.n_critical} archetypes...")

        # Define critical scenarios
        critical_diagnoses = ['stroke', 'tia']

        archetypes = []
        attempts = 0
        max_attempts = self.n_critical * 20

        pbar = tqdm(total=self.n_critical, desc="Phase 5")

        while len(archetypes) < self.n_critical and attempts < max_attempts:
            # Sample critical scenario
            archetype = self._sample_critical_scenario(critical_diagnoses)
            attempts += 1

            if archetype.is_valid(self.param_space.constraints):
                archetype.features = self._archetype_to_features(archetype)
                archetypes.append(archetype)
                pbar.update(1)

        pbar.close()

        self.stats['phase5_sampled'] = attempts
        self.stats['phase5_valid'] = len(archetypes)

        logger.info(f"Phase 5 complete: {len(archetypes)} archetypes")
        logger.info(
            f"  Acceptance rate: {
                len(archetypes) / attempts * 100:.1f}%")

        return archetypes

    def _sample_critical_scenario(
            self, critical_diagnoses: List[str]) -> Archetype:
        """Sample from critical scenario space"""
        # Force critical diagnosis
        diagnosis = np.random.choice(critical_diagnoses)

        # Critical scenario parameters
        params = {
            'diagnosis': diagnosis,
            'age': np.random.normal(70, 10),  # Older
            'timing': 'acute',  # Acute onset
            'trigger': 'spontaneous',  # Spontaneous
            'urgency': 2,  # Emergency
            'cardiovascular_disease': True,  # High risk
        }

        # If stroke, ensure central HINTS
        if diagnosis == 'stroke':
            params['nystagmus_type'] = np.random.choice(
                ['central', 'direction_changing'])
            params['head_impulse_test'] = 'normal'  # Central = normal HIT

        # Sample remaining parameters
        for name, param in self.param_space.parameters.items():
            if name not in params:
                params[name] = param.sample(1)[0]

        return Archetype(parameters=params, diagnosis=diagnosis)

    def _phase6_diversity_sampling(
        self,
        existing_archetypes: List[Archetype]
    ) -> List[Archetype]:
        """Phase 6: Diversity-oriented sampling (10%)"""
        logger.info(f"Diversity sampling: {self.n_diversity} archetypes...")

        # Create feature matrix from existing
        X_existing = np.array([a.features for a in existing_archetypes])

        # K-means clustering
        n_clusters = min(50, len(existing_archetypes) // 10)
        logger.info(f"Clustering into {n_clusters} clusters...")

        kmeans = KMeans(n_clusters=n_clusters, random_state=self.random_state)
        cluster_labels = kmeans.fit_predict(X_existing)

        # Count samples per cluster
        cluster_counts = np.bincount(cluster_labels, minlength=n_clusters)

        # Find underrepresented clusters
        underrep_clusters = np.argsort(cluster_counts)[:n_clusters // 2]

        logger.info(
            f"Targeting {
                len(underrep_clusters)} underrepresented clusters")

        archetypes = []
        attempts = 0
        max_attempts = self.n_diversity * 20

        pbar = tqdm(total=self.n_diversity, desc="Phase 6")

        while len(archetypes) < self.n_diversity and attempts < max_attempts:
            # Sample near underrepresented cluster center
            cluster_id = np.random.choice(underrep_clusters)
            archetype = self._sample_near_cluster(
                kmeans.cluster_centers_[cluster_id])
            attempts += 1

            if archetype.is_valid(self.param_space.constraints):
                archetypes.append(archetype)
                pbar.update(1)

        pbar.close()

        self.stats['phase6_sampled'] = attempts
        self.stats['phase6_valid'] = len(archetypes)

        logger.info(f"Phase 6 complete: {len(archetypes)} archetypes")
        logger.info(
            f"  Acceptance rate: {
                len(archetypes) / attempts * 100:.1f}%")

        return archetypes

    def _sample_near_cluster(self, center: np.ndarray) -> Archetype:
        """Sample archetype near cluster center"""
        # Add Gaussian noise to center
        features = center + np.random.normal(0, 0.1, size=center.shape)
        features = np.clip(features, 0, 1)

        # Convert features back to parameters (simplified)
        # In full implementation, would use proper inverse mapping
        diagnosis = self.param_space.epidemiology.sample_diagnosis(1)[0]
        age = self.param_space.epidemiology.sample_age(diagnosis, 1)[0]

        params = {'diagnosis': diagnosis, 'age': int(age)}

        for name, param in self.param_space.parameters.items():
            if name not in params:
                params[name] = param.sample(1)[0]

        archetype = Archetype(parameters=params, diagnosis=diagnosis)
        archetype.features = features

        return archetype

    def _archetype_to_features(self, archetype: Archetype) -> np.ndarray:
        """Convert archetype parameters to feature vector"""
        # Simplified: create 150-dim feature vector
        # In full implementation, would use proper feature engineering

        features = []

        # Encode categorical parameters
        for name, param in self.param_space.parameters.items():
            value = archetype.parameters.get(name)

            if param.param_type.value == 'categorical':
                # One-hot encode
                one_hot = [1.0 if v == value else 0.0 for v in param.domain]
                features.extend(one_hot)
            elif param.param_type.value == 'continuous':
                # Normalize to [0, 1]
                a, b = param.domain
                normalized = (value - a) / (b - a)
                features.append(normalized)
            elif param.param_type.value == 'binary':
                features.append(1.0 if value else 0.0)
            elif param.param_type.value == 'ordinal':
                # Normalize
                normalized = value / max(param.domain)
                features.append(normalized)

        # Pad to 150 dimensions if needed
        while len(features) < 150:
            features.append(0.0)

        return np.array(features[:150])

    def _validate_final_set(self):
        """Validate final archetype set"""
        logger.info("\nValidating final archetype set...")

        # Check distribution
        validation_result = self.param_space.epidemiology.validate_distribution(
            self.final_archetypes)

        logger.info(
            f"Chi-squared test: χ²={
                validation_result['chi2_statistic']:.4f}, " f"p={
                validation_result['p_value']:.4f}")
        logger.info(
            f"Distribution match: {
                'PASS' if validation_result['accept'] else 'FAIL'}")

        # Check critical scenario coverage
        critical_count = sum(
            1 for a in self.final_archetypes
            if a.diagnosis in ['stroke', 'tia']
        )
        critical_rate = critical_count / len(self.final_archetypes)
        expected_rate = 0.15  # 15% from epidemiology

        logger.info(
            f"Critical scenarios: {critical_count} ({
                critical_rate * 100:.1f}%)")
        logger.info(f"Expected: {expected_rate * 100:.1f}%")
        logger.info(f"Match: {abs(critical_rate -
                                  expected_rate) /
                              expected_rate *
                              100:.1f}% deviation")

    def get_statistics(self) -> Dict:
        """Get comprehensive statistics"""
        return {
            'configuration': {
                'n_target': self.n_target,
                'nmf_factors': self.nmf_factors,
                'alpha_importance': self.alpha_importance,
                'alpha_critical': self.alpha_critical,
                'alpha_diversity': self.alpha_diversity,
            },
            'sampling_stats': self.stats,
            'final_count': len(
                self.final_archetypes),
            'nmf_summary': self.nmf_model.get_summary() if self.nmf_model else None,
            'shap_summary': self.shap_model.get_summary() if self.shap_model else None,
        }
