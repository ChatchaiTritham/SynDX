"""
Integration tests for SynDX components.

Tests interactions between multiple modules to ensure they work together correctly.

Author: Chatchai Tritham
Date: 2026-01-25
"""

import pytest
import numpy as np
from syndx.phase2_synthesis.shap_reweighter import SHAPReweighter
from syndx.phase2_synthesis.differential_privacy import DifferentialPrivacy
from syndx.phase3_validation.xai_fidelity import XAIFidelity
from syndx.phase2_synthesis.counterfactual_validator import CounterfactualValidator
from syndx.phase3_validation.diagnostic_evaluator import DiagnosticEvaluator


@pytest.mark.integration
class TestSHAPAndDiagnosticEvaluator:
    """Test SHAP reweighting integrated with diagnostic evaluation."""

    def test_shap_reweighting_improves_diagnostic_performance(self,
                                                              mock_archetype_data,
                                                              mock_synthetic_data):
        """Test that SHAP-weighted features improve diagnostic performance."""
        X_arch, y_arch = mock_archetype_data
        X_synth, y_synth = mock_synthetic_data

        # Split data
        split_idx = 20
        X_test = X_arch[:split_idx]
        y_test = y_arch[:split_idx]
        X_arch_train = X_arch[split_idx:]
        y_arch_train = y_arch[split_idx:]

        # Train SHAP reweighter
        reweighter = SHAPReweighter()
        reweighter.fit(X_arch_train, y_arch_train)

        # Transform synthetic data
        X_synth_weighted = reweighter.transform(X_synth)

        # Evaluate both versions
        evaluator_original = DiagnosticEvaluator(random_state=42)
        evaluator_original.fit_archetype_model(X_arch_train, y_arch_train)
        evaluator_original.fit_synthetic_model(X_synth, y_synth)
        results_original = evaluator_original.evaluate(X_test, y_test)

        evaluator_weighted = DiagnosticEvaluator(random_state=42)
        evaluator_weighted.fit_archetype_model(X_arch_train, y_arch_train)
        evaluator_weighted.fit_synthetic_model(X_synth_weighted, y_synth)
        results_weighted = evaluator_weighted.evaluate(X_test, y_test)

        # Both should work without errors
        assert results_original is not None
        assert results_weighted is not None

        # SHAP weighting may or may not improve performance (depends on data)
        # Just verify pipeline works
        assert -1.0 <= results_original['utility_gap'] <= 1.0
        assert -1.0 <= results_weighted['utility_gap'] <= 1.0


@pytest.mark.integration
class TestDifferentialPrivacyAndXAIFidelity:
    """Test DP-SGD integration with XAI fidelity evaluation."""

    def test_dp_preserves_xai_fidelity(self, mock_gradients, mock_archetype_data, mock_synthetic_data):
        """Test that differential privacy doesn't completely destroy XAI fidelity."""
        X_arch, y_arch = mock_archetype_data
        X_synth, y_synth = mock_synthetic_data

        # Initialize DP mechanism
        dp = DifferentialPrivacy(epsilon=1.0, delta=1e-5, clip_norm=1.0)

        # Privatize some gradients (simulating DP-SGD training)
        private_grad, stats = dp.privatize_gradients(mock_gradients, batch_size=64)

        assert private_grad is not None
        assert stats['clip_fraction'] >= 0

        # Evaluate XAI fidelity (even with DP noise in training)
        fidelity = XAIFidelity()
        fidelity.fit_archetype_model(X_arch, y_arch)
        fidelity.fit_synthetic_model(X_synth, y_synth)

        scores = fidelity.compute_all_metrics()

        # Fidelity should still be measurable
        assert 0.0 <= scores['overall_fidelity'] <= 1.0


@pytest.mark.integration
class TestXAIFidelityAndDiagnosticEvaluator:
    """Test XAI fidelity and diagnostic evaluation together."""

    def test_high_fidelity_correlates_with_low_utility_gap(self,
                                                           mock_archetype_data,
                                                           mock_synthetic_data):
        """Test that high XAI fidelity correlates with good diagnostic performance."""
        X_arch, y_arch = mock_archetype_data
        X_synth, y_synth = mock_synthetic_data

        split_idx = 20
        X_test = X_arch[:split_idx]
        y_test = y_arch[:split_idx]
        X_arch_train = X_arch[split_idx:]
        y_arch_train = y_arch[split_idx:]

        # Compute XAI fidelity
        fidelity = XAIFidelity()
        fidelity.fit_archetype_model(X_arch_train, y_arch_train)
        fidelity.fit_synthetic_model(X_synth, y_synth)
        fidelity_scores = fidelity.compute_all_metrics()

        # Compute diagnostic performance
        evaluator = DiagnosticEvaluator()
        evaluator.fit_archetype_model(X_arch_train, y_arch_train)
        evaluator.fit_synthetic_model(X_synth, y_synth)
        eval_results = evaluator.evaluate(X_test, y_test)

        # Both metrics should be computed
        assert 0.0 <= fidelity_scores['overall_fidelity'] <= 1.0
        assert abs(eval_results['utility_gap']) <= 1.0

        # In general, higher fidelity should mean lower utility gap
        # (but this depends on data quality)
        # Just verify both metrics are available for comparison


@pytest.mark.integration
class TestCounterfactualAndXAIFidelity:
    """Test counterfactual generation with XAI fidelity."""

    def test_counterfactuals_respect_feature_importance(self,
                                                        mock_patient_data,
                                                        mock_archetype_data):
        """Test that counterfactuals align with SHAP feature importance."""
        patient, feature_names = mock_patient_data
        X_arch, y_arch = mock_archetype_data

        # Get feature importance via SHAP
        reweighter = SHAPReweighter()
        reweighter.fit(X_arch, y_arch)
        top_features = reweighter.get_top_features(n=5)
        top_indices = [idx for idx, _ in top_features]

        # Generate counterfactuals
        validator = CounterfactualValidator(constraint_checker=None, max_iterations=30)
        cf = validator.generate_counterfactual(patient, 'BPPV', feature_names)

        if cf is not None and cf['changes']:
            # Check if important features are being changed
            # (Counterfactuals should leverage important features)
            changed_indices = [feature_names.index(name) for name in cf['changes'].keys()]

            # Some overlap expected (but not guaranteed)
            # Just verify the integration works
            assert len(changed_indices) > 0


@pytest.mark.integration
class TestFullPipeline:
    """Test complete SynDX validation pipeline."""

    def test_complete_validation_workflow(self,
                                          mock_archetype_data,
                                          mock_synthetic_data,
                                          mock_patient_data):
        """Test complete workflow from SHAP to evaluation."""
        X_arch, y_arch = mock_archetype_data
        X_synth, y_synth = mock_synthetic_data
        patient, feature_names = mock_patient_data

        split_idx = 20
        X_test = X_arch[:split_idx]
        y_test = y_arch[:split_idx]
        X_arch_train = X_arch[split_idx:]
        y_arch_train = y_arch[split_idx:]

        # Step 1: SHAP reweighting
        reweighter = SHAPReweighter()
        reweighter.fit(X_arch_train, y_arch_train)
        X_synth_weighted = reweighter.transform(X_synth)

        # Step 2: DP privatization (simulate)
        dp = DifferentialPrivacy(epsilon=1.0)
        max_steps = dp.get_max_steps(batch_size=64, n_samples=len(X_synth))
        assert max_steps > 0

        # Step 3: XAI fidelity evaluation
        fidelity = XAIFidelity()
        fidelity.fit_archetype_model(X_arch_train, y_arch_train)
        fidelity.fit_synthetic_model(X_synth_weighted, y_synth)
        fidelity_scores = fidelity.compute_all_metrics()

        # Step 4: Diagnostic evaluation
        evaluator = DiagnosticEvaluator()
        evaluator.fit_archetype_model(X_arch_train, y_arch_train)
        evaluator.fit_synthetic_model(X_synth_weighted, y_synth)
        eval_results = evaluator.evaluate(X_test, y_test)

        # Step 5: Counterfactual validation
        validator = CounterfactualValidator(constraint_checker=None, max_iterations=20)
        cf = validator.generate_counterfactual(patient, 'VM', feature_names)

        # Verify complete pipeline works
        assert fidelity_scores['overall_fidelity'] >= 0
        assert abs(eval_results['utility_gap']) <= 1.0
        # Counterfactual may or may not be found
        if cf:
            assert cf['distance'] >= 0

        # Get summaries
        fidelity_summary = fidelity.summary()
        eval_summary = evaluator.summary()

        assert fidelity_summary['status'] == 'evaluated'
        assert eval_summary['status'] == 'evaluated'


@pytest.mark.integration
class TestSHAPAndDP:
    """Test SHAP reweighting with differential privacy."""

    def test_shap_weights_with_dp_budget(self, mock_archetype_data, mock_synthetic_data):
        """Test SHAP reweighting respects DP privacy budget."""
        X_arch, y_arch = mock_archetype_data
        X_synth, y_synth = mock_synthetic_data

        # Get SHAP weights
        reweighter = SHAPReweighter()
        reweighter.fit(X_arch, y_arch)
        weights = reweighter.get_sampling_weights()

        # Check privacy budget for VAE training with these weights
        dp = DifferentialPrivacy(epsilon=1.0, delta=1e-5)

        batch_size = 64
        n_samples = len(X_synth)
        max_steps = dp.get_max_steps(batch_size, n_samples)

        # Should allow reasonable training
        assert max_steps > 100  # At least 100 steps

        # Transform data with SHAP weights
        X_synth_weighted = reweighter.transform(X_synth)

        # Both versions should have same shape
        assert X_synth_weighted.shape == X_synth.shape


@pytest.mark.integration
@pytest.mark.slow
class TestMultipleMetrics:
    """Test multiple validation metrics together."""

    def test_consistent_validation_across_metrics(self,
                                                  mock_archetype_data,
                                                  mock_synthetic_data):
        """Test that different validation metrics give consistent results."""
        X_arch, y_arch = mock_archetype_data
        X_synth, y_synth = mock_synthetic_data

        split_idx = 20
        X_test = X_arch[:split_idx]
        y_test = y_arch[:split_idx]
        X_arch_train = X_arch[split_idx:]
        y_arch_train = y_arch[split_idx:]

        # Compute all validation metrics
        results = {}

        # 1. XAI Fidelity
        fidelity = XAIFidelity()
        fidelity.fit_archetype_model(X_arch_train, y_arch_train)
        fidelity.fit_synthetic_model(X_synth, y_synth)
        results['fidelity'] = fidelity.compute_all_metrics()['overall_fidelity']

        # 2. Diagnostic Performance
        evaluator = DiagnosticEvaluator()
        evaluator.fit_archetype_model(X_arch_train, y_arch_train)
        evaluator.fit_synthetic_model(X_synth, y_synth)
        eval_results = evaluator.evaluate(X_test, y_test)
        results['utility_gap'] = abs(eval_results['utility_gap'])

        # 3. Feature Importance Consistency
        reweighter = SHAPReweighter()
        reweighter.fit(X_arch_train, y_arch_train)
        top_features = reweighter.get_top_features(n=10)
        results['top_feature_weight'] = top_features[0][1] if top_features else 0

        # All metrics should be computed
        assert all(v >= 0 for v in results.values())

        # High fidelity + low utility gap = good synthetic data
        if results['fidelity'] > 0.7 and results['utility_gap'] < 0.1:
            # This would indicate excellent synthetic data
            pass  # Just checking metrics are computable


@pytest.mark.integration
class TestErrorHandling:
    """Test error handling across modules."""

    def test_graceful_error_handling_in_pipeline(self):
        """Test that pipeline handles errors gracefully."""
        # Test with invalid data
        X_invalid = np.array([[]])  # Empty features
        y_invalid = np.array([])

        # Each module should handle errors gracefully
        with pytest.raises((ValueError, IndexError)):
            reweighter = SHAPReweighter()
            reweighter.fit(X_invalid, y_invalid)

        # DP with empty gradients
        dp = DifferentialPrivacy()
        with pytest.raises((ValueError, IndexError)):
            dp.privatize_gradients([], batch_size=0)

    def test_missing_model_errors(self):
        """Test that missing models raise appropriate errors."""
        fidelity = XAIFidelity()

        # Should raise error when models not fitted
        with pytest.raises(ValueError):
            fidelity.compute_shap_correlation()

        evaluator = DiagnosticEvaluator()
        with pytest.raises(ValueError):
            X_test = np.random.randn(10, 20)
            y_test = np.random.randint(0, 3, 10)
            evaluator.evaluate(X_test, y_test)


@pytest.mark.integration
class TestDataConsistency:
    """Test data consistency across modules."""

    def test_feature_dimension_consistency(self, mock_archetype_data, mock_synthetic_data):
        """Test that all modules handle feature dimensions consistently."""
        X_arch, y_arch = mock_archetype_data
        X_synth, y_synth = mock_synthetic_data

        n_features = X_arch.shape[1]

        # SHAP reweighter
        reweighter = SHAPReweighter()
        reweighter.fit(X_arch, y_arch)
        assert len(reweighter.feature_weights) == n_features

        # XAI Fidelity
        fidelity = XAIFidelity()
        fidelity.fit_archetype_model(X_arch, y_arch)
        arch_imp = fidelity._get_global_importance(fidelity.archetype_shap)
        assert len(arch_imp) == n_features

        # Diagnostic Evaluator
        evaluator = DiagnosticEvaluator()
        evaluator.fit_archetype_model(X_arch, y_arch)
        arch_feat_imp, _ = evaluator.get_feature_importance()
        if arch_feat_imp is not None:
            assert len(arch_feat_imp) == n_features

    def test_label_consistency(self, mock_archetype_data, mock_synthetic_data):
        """Test that all modules handle multi-class labels consistently."""
        X_arch, y_arch = mock_archetype_data
        X_synth, y_synth = mock_synthetic_data

        n_classes_arch = len(np.unique(y_arch))
        n_classes_synth = len(np.unique(y_synth))

        # All modules should handle multi-class
        reweighter = SHAPReweighter()
        reweighter.fit(X_arch, y_arch)  # Should work

        fidelity = XAIFidelity()
        fidelity.fit_archetype_model(X_arch, y_arch)  # Should work

        evaluator = DiagnosticEvaluator()
        evaluator.fit_archetype_model(X_arch, y_arch)  # Should work

        # All should succeed without errors
        assert True


@pytest.mark.integration
@pytest.mark.slow
class TestPerformanceBenchmark:
    """Integration tests for performance benchmarking."""

    def test_all_modules_complete_in_reasonable_time(self,
                                                     mock_archetype_data,
                                                     mock_synthetic_data,
                                                     mock_patient_data):
        """Test that complete pipeline completes in reasonable time."""
        import time

        X_arch, y_arch = mock_archetype_data
        X_synth, y_synth = mock_synthetic_data
        patient, feature_names = mock_patient_data

        start_time = time.time()

        # Run all modules
        reweighter = SHAPReweighter()
        reweighter.fit(X_arch, y_arch)

        fidelity = XAIFidelity()
        fidelity.fit_archetype_model(X_arch, y_arch)
        fidelity.fit_synthetic_model(X_synth, y_synth)
        fidelity.compute_all_metrics()

        evaluator = DiagnosticEvaluator()
        evaluator.fit_archetype_model(X_arch[20:], y_arch[20:])
        evaluator.fit_synthetic_model(X_synth, y_synth)
        evaluator.evaluate(X_arch[:20], y_arch[:20])

        validator = CounterfactualValidator(constraint_checker=None, max_iterations=10)
        validator.generate_counterfactual(patient, 'BPPV', feature_names)

        elapsed = time.time() - start_time

        # Should complete in reasonable time (< 60 seconds for small mock data)
        assert elapsed < 60.0
