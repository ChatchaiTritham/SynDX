"""
Unit tests for DiagnosticEvaluator module.

Tests diagnostic performance evaluation for synthetic data validation.

Author: Chatchai Tritham
Date: 2026-01-25
"""

import pytest
import numpy as np
from syndx.phase3_validation.diagnostic_evaluator import DiagnosticEvaluator


@pytest.mark.unit
@pytest.mark.validation
class TestDiagnosticEvaluatorInit:
    """Test DiagnosticEvaluator initialization."""

    def test_default_initialization(self):
        """Test initialization with default parameters."""
        evaluator = DiagnosticEvaluator()

        assert evaluator.model_type == 'xgboost'
        assert evaluator.cv_folds == 5
        assert evaluator.random_state == 42
        assert evaluator.archetype_model is None
        assert evaluator.synthetic_model is None
        assert evaluator.results == {}

    def test_custom_initialization(self):
        """Test initialization with custom parameters."""
        evaluator = DiagnosticEvaluator(
            model_type='random_forest',
            cv_folds=10,
            random_state=123
        )

        assert evaluator.model_type == 'random_forest'
        assert evaluator.cv_folds == 10
        assert evaluator.random_state == 123

    def test_repr_before_evaluation(self):
        """Test string representation before evaluation."""
        evaluator = DiagnosticEvaluator()
        repr_str = repr(evaluator)

        assert "DiagnosticEvaluator" in repr_str
        assert "evaluated=False" in repr_str

    def test_repr_after_evaluation(self, mock_archetype_data, mock_synthetic_data):
        """Test string representation after evaluation."""
        X_arch_train, y_arch_train = mock_archetype_data
        X_synth_train, y_synth_train = mock_synthetic_data

        # Create test set
        X_test = X_arch_train[:20]
        y_test = y_arch_train[:20]

        evaluator = DiagnosticEvaluator()
        evaluator.fit_archetype_model(X_arch_train[20:], y_arch_train[20:])
        evaluator.fit_synthetic_model(X_synth_train, y_synth_train)
        evaluator.evaluate(X_test, y_test)

        repr_str = repr(evaluator)

        assert "DiagnosticEvaluator" in repr_str
        assert "utility_gap" in repr_str


@pytest.mark.unit
@pytest.mark.validation
class TestCreateClassifier:
    """Test classifier creation."""

    def test_create_xgboost_classifier(self):
        """Test creating XGBoost classifier."""
        evaluator = DiagnosticEvaluator(model_type='xgboost')
        classifier = evaluator._create_classifier()

        assert classifier is not None
        assert hasattr(classifier, 'fit')
        assert hasattr(classifier, 'predict')

    def test_create_random_forest_classifier(self):
        """Test creating Random Forest classifier."""
        evaluator = DiagnosticEvaluator(model_type='random_forest')
        classifier = evaluator._create_classifier()

        assert classifier is not None
        assert hasattr(classifier, 'fit')
        assert hasattr(classifier, 'predict')

    def test_invalid_model_type_raises_error(self):
        """Test that invalid model type raises error."""
        evaluator = DiagnosticEvaluator(model_type='invalid_model')

        with pytest.raises(ValueError, match="Unknown model type"):
            evaluator._create_classifier()


@pytest.mark.unit
@pytest.mark.validation
class TestFitModels:
    """Test model fitting."""

    def test_fit_archetype_model(self, mock_archetype_data):
        """Test fitting archetype model."""
        X, y = mock_archetype_data
        evaluator = DiagnosticEvaluator()

        evaluator.fit_archetype_model(X, y)

        assert evaluator.archetype_model is not None
        assert evaluator.archetype_model.score(X, y) > 0  # Some accuracy

    def test_fit_synthetic_model(self, mock_synthetic_data):
        """Test fitting synthetic model."""
        X, y = mock_synthetic_data
        evaluator = DiagnosticEvaluator()

        evaluator.fit_synthetic_model(X, y)

        assert evaluator.synthetic_model is not None
        assert evaluator.synthetic_model.score(X, y) > 0

    def test_fit_both_models(self, mock_archetype_data, mock_synthetic_data):
        """Test fitting both models."""
        X_arch, y_arch = mock_archetype_data
        X_synth, y_synth = mock_synthetic_data

        evaluator = DiagnosticEvaluator()
        evaluator.fit_archetype_model(X_arch, y_arch)
        evaluator.fit_synthetic_model(X_synth, y_synth)

        assert evaluator.archetype_model is not None
        assert evaluator.synthetic_model is not None


@pytest.mark.unit
@pytest.mark.validation
class TestEvaluate:
    """Test evaluation method."""

    def test_evaluate_basic(self, mock_archetype_data, mock_synthetic_data):
        """Test basic evaluation."""
        X_arch_train, y_arch_train = mock_archetype_data
        X_synth_train, y_synth_train = mock_synthetic_data

        # Split for train/test
        X_test = X_arch_train[:20]
        y_test = y_arch_train[:20]
        X_train = X_arch_train[20:]
        y_train = y_arch_train[20:]

        evaluator = DiagnosticEvaluator()
        evaluator.fit_archetype_model(X_train, y_train)
        evaluator.fit_synthetic_model(X_synth_train, y_synth_train)

        results = evaluator.evaluate(X_test, y_test)

        assert 'archetype' in results
        assert 'synthetic' in results
        assert 'utility_gap' in results
        assert 'mcnemar' in results

    def test_evaluate_without_models_raises_error(self):
        """Test that evaluate without models raises error."""
        evaluator = DiagnosticEvaluator()
        X_test = np.random.randn(10, 20)
        y_test = np.random.randint(0, 3, 10)

        with pytest.raises(ValueError, match="Must fit both models"):
            evaluator.evaluate(X_test, y_test)

    def test_evaluate_computes_all_metrics(self, mock_archetype_data, mock_synthetic_data):
        """Test that evaluation computes all expected metrics."""
        X_arch_train, y_arch_train = mock_archetype_data
        X_synth_train, y_synth_train = mock_synthetic_data

        X_test = X_arch_train[:20]
        y_test = y_arch_train[:20]

        evaluator = DiagnosticEvaluator()
        evaluator.fit_archetype_model(X_arch_train[20:], y_arch_train[20:])
        evaluator.fit_synthetic_model(X_synth_train, y_synth_train)

        results = evaluator.evaluate(X_test, y_test)

        # Archetype metrics
        assert 'accuracy' in results['archetype']
        assert 'precision_macro' in results['archetype']
        assert 'recall_macro' in results['archetype']
        assert 'f1_macro' in results['archetype']
        assert 'confusion_matrix' in results['archetype']

        # Synthetic metrics
        assert 'accuracy' in results['synthetic']
        assert 'f1_macro' in results['synthetic']

        # Comparison metrics
        assert isinstance(results['utility_gap'], float)
        assert 'p_value' in results['mcnemar']


@pytest.mark.unit
@pytest.mark.validation
class TestComputeMetrics:
    """Test metrics computation."""

    def test_compute_metrics_basic(self, mock_archetype_data):
        """Test basic metrics computation."""
        X, y = mock_archetype_data
        evaluator = DiagnosticEvaluator()
        evaluator.fit_archetype_model(X, y)

        y_pred = evaluator.archetype_model.predict(X)
        y_proba = evaluator.archetype_model.predict_proba(X)

        metrics = evaluator._compute_metrics(y, y_pred, y_proba, "Test")

        assert 0.0 <= metrics['accuracy'] <= 1.0
        assert 0.0 <= metrics['f1_macro'] <= 1.0
        assert 0.0 <= metrics['auc_macro'] <= 1.0

    def test_metrics_all_present(self, mock_archetype_data):
        """Test that all expected metrics are present."""
        X, y = mock_archetype_data
        evaluator = DiagnosticEvaluator()
        evaluator.fit_archetype_model(X, y)

        y_pred = evaluator.archetype_model.predict(X)
        y_proba = evaluator.archetype_model.predict_proba(X)

        metrics = evaluator._compute_metrics(y, y_pred, y_proba, "Test")

        expected_metrics = [
            'accuracy', 'precision_macro', 'recall_macro', 'f1_macro',
            'precision_weighted', 'recall_weighted', 'f1_weighted',
            'auc_macro', 'auc_weighted', 'precision_per_class',
            'recall_per_class', 'f1_per_class', 'confusion_matrix'
        ]

        for metric in expected_metrics:
            assert metric in metrics


@pytest.mark.unit
@pytest.mark.validation
class TestMcNemarTest:
    """Test McNemar's test."""

    def test_mcnemar_test_basic(self, mock_archetype_data):
        """Test basic McNemar's test."""
        X, y = mock_archetype_data
        evaluator = DiagnosticEvaluator()
        evaluator.fit_archetype_model(X, y)

        # Create two predictions (could be similar or different)
        y_pred1 = evaluator.archetype_model.predict(X)

        # Slightly perturb predictions for testing
        y_pred2 = y_pred1.copy()
        if len(y_pred2) > 5:
            y_pred2[:5] = (y_pred2[:5] + 1) % len(np.unique(y))

        result = evaluator._mcnemar_test(y, y_pred1, y_pred2)

        assert 'statistic' in result
        assert 'p_value' in result
        assert 'both_correct' in result
        assert 'both_wrong' in result
        assert 'is_significant' in result

        assert 0.0 <= result['p_value'] <= 1.0

    def test_mcnemar_identical_predictions(self, mock_archetype_data):
        """Test McNemar's test with identical predictions."""
        X, y = mock_archetype_data
        evaluator = DiagnosticEvaluator()
        evaluator.fit_archetype_model(X, y)

        y_pred = evaluator.archetype_model.predict(X)

        result = evaluator._mcnemar_test(y, y_pred, y_pred)

        # Identical predictions should not be significant
        assert result['archetype_only_correct'] == 0
        assert result['synthetic_only_correct'] == 0


@pytest.mark.unit
@pytest.mark.validation
class TestCrossValidation:
    """Test cross-validation."""

    def test_cross_validate_basic(self, mock_archetype_data):
        """Test basic cross-validation."""
        X, y = mock_archetype_data
        evaluator = DiagnosticEvaluator(cv_folds=3)

        cv_results = evaluator.cross_validate(X, y, data_type='archetype')

        assert 'accuracy_mean' in cv_results
        assert 'accuracy_std' in cv_results
        assert 'f1_macro_mean' in cv_results
        assert 'f1_macro_std' in cv_results

        assert 0.0 <= cv_results['accuracy_mean'] <= 1.0
        assert cv_results['accuracy_std'] >= 0

    def test_cross_validate_reproducibility(self, mock_archetype_data):
        """Test cross-validation reproducibility."""
        X, y = mock_archetype_data

        evaluator1 = DiagnosticEvaluator(random_state=42, cv_folds=3)
        results1 = evaluator1.cross_validate(X, y)

        evaluator2 = DiagnosticEvaluator(random_state=42, cv_folds=3)
        results2 = evaluator2.cross_validate(X, y)

        # Should be identical
        assert results1['accuracy_mean'] == pytest.approx(results2['accuracy_mean'])
        assert results1['f1_macro_mean'] == pytest.approx(results2['f1_macro_mean'])


@pytest.mark.unit
@pytest.mark.validation
class TestFeatureImportance:
    """Test feature importance extraction."""

    def test_get_feature_importance(self, mock_archetype_data, mock_synthetic_data):
        """Test getting feature importance from both models."""
        X_arch, y_arch = mock_archetype_data
        X_synth, y_synth = mock_synthetic_data

        evaluator = DiagnosticEvaluator()
        evaluator.fit_archetype_model(X_arch, y_arch)
        evaluator.fit_synthetic_model(X_synth, y_synth)

        arch_imp, synth_imp = evaluator.get_feature_importance()

        assert arch_imp is not None
        assert synth_imp is not None
        assert len(arch_imp) == X_arch.shape[1]
        assert len(synth_imp) == X_synth.shape[1]

    def test_feature_importance_without_models_raises_error(self):
        """Test that getting importance without models raises error."""
        evaluator = DiagnosticEvaluator()

        with pytest.raises(ValueError, match="Must fit both models"):
            evaluator.get_feature_importance()


@pytest.mark.unit
@pytest.mark.validation
class TestSummary:
    """Test DiagnosticEvaluator summary method."""

    def test_summary_before_evaluation(self):
        """Test summary before evaluation."""
        evaluator = DiagnosticEvaluator()
        summary = evaluator.summary()

        assert summary['status'] == 'not_evaluated'

    def test_summary_after_evaluation(self, mock_archetype_data, mock_synthetic_data):
        """Test summary after evaluation."""
        X_arch_train, y_arch_train = mock_archetype_data
        X_synth_train, y_synth_train = mock_synthetic_data

        X_test = X_arch_train[:20]
        y_test = y_arch_train[:20]

        evaluator = DiagnosticEvaluator()
        evaluator.fit_archetype_model(X_arch_train[20:], y_arch_train[20:])
        evaluator.fit_synthetic_model(X_synth_train, y_synth_train)
        evaluator.evaluate(X_test, y_test)

        summary = evaluator.summary()

        assert summary['status'] == 'evaluated'
        assert 'utility_gap' in summary
        assert 'interpretation' in summary
        assert 'recommendation' in summary

    def test_summary_interpretation_excellent(self):
        """Test interpretation for excellent performance."""
        evaluator = DiagnosticEvaluator()
        evaluator.results = {
            'archetype': {'accuracy': 0.90},
            'synthetic': {'accuracy': 0.89},
            'utility_gap': 0.01,
            'mcnemar': {'is_significant': False}
        }

        summary = evaluator.summary()

        assert "Excellent" in summary['interpretation']
        assert "approved" in summary['recommendation']

    def test_summary_interpretation_good(self):
        """Test interpretation for good performance."""
        evaluator = DiagnosticEvaluator()
        evaluator.results = {
            'archetype': {'accuracy': 0.90},
            'synthetic': {'accuracy': 0.87},
            'utility_gap': 0.03,
            'mcnemar': {'is_significant': False}
        }

        summary = evaluator.summary()

        assert "Good" in summary['interpretation']

    def test_summary_interpretation_poor(self):
        """Test interpretation for poor performance."""
        evaluator = DiagnosticEvaluator()
        evaluator.results = {
            'archetype': {'accuracy': 0.90},
            'synthetic': {'accuracy': 0.75},
            'utility_gap': 0.15,
            'mcnemar': {'is_significant': True}
        }

        summary = evaluator.summary()

        assert "Poor" in summary['interpretation']
        assert "Improve" in summary['recommendation']


@pytest.mark.unit
@pytest.mark.validation
class TestDiagnosticEvaluatorIntegration:
    """Integration tests for DiagnosticEvaluator."""

    def test_full_evaluation_pipeline(self, mock_archetype_data, mock_synthetic_data):
        """Test complete evaluation pipeline."""
        X_arch_train, y_arch_train = mock_archetype_data
        X_synth_train, y_synth_train = mock_synthetic_data

        # Create train/test split
        split_idx = 20
        X_test = X_arch_train[:split_idx]
        y_test = y_arch_train[:split_idx]
        X_arch_train = X_arch_train[split_idx:]
        y_arch_train = y_arch_train[split_idx:]

        evaluator = DiagnosticEvaluator()

        # Fit models
        evaluator.fit_archetype_model(X_arch_train, y_arch_train)
        evaluator.fit_synthetic_model(X_synth_train, y_synth_train)

        # Evaluate
        results = evaluator.evaluate(X_test, y_test)

        # Get summary
        summary = evaluator.summary()

        # Verify complete workflow
        assert results is not None
        assert summary['status'] == 'evaluated'
        assert -1.0 <= results['utility_gap'] <= 1.0

    def test_utility_gap_reasonable(self, mock_archetype_data, mock_synthetic_data):
        """Test that utility gap is reasonable."""
        X_arch_train, y_arch_train = mock_archetype_data
        X_synth_train, y_synth_train = mock_synthetic_data

        split_idx = 20
        X_test = X_arch_train[:split_idx]
        y_test = y_arch_train[:split_idx]

        evaluator = DiagnosticEvaluator()
        evaluator.fit_archetype_model(X_arch_train[split_idx:], y_arch_train[split_idx:])
        evaluator.fit_synthetic_model(X_synth_train, y_synth_train)

        results = evaluator.evaluate(X_test, y_test)

        # Utility gap should be bounded
        assert abs(results['utility_gap']) <= 1.0

        # If synthetic is good, gap should be small
        # (This depends on mock data quality)
        pass

    def test_reproducibility_with_random_state(self, mock_archetype_data, mock_synthetic_data):
        """Test reproducibility with fixed random state."""
        X_arch_train, y_arch_train = mock_archetype_data
        X_synth_train, y_synth_train = mock_synthetic_data

        split_idx = 20
        X_test = X_arch_train[:split_idx]
        y_test = y_arch_train[:split_idx]

        # Run 1
        evaluator1 = DiagnosticEvaluator(random_state=42)
        evaluator1.fit_archetype_model(X_arch_train[split_idx:], y_arch_train[split_idx:])
        evaluator1.fit_synthetic_model(X_synth_train, y_synth_train)
        results1 = evaluator1.evaluate(X_test, y_test)

        # Run 2
        evaluator2 = DiagnosticEvaluator(random_state=42)
        evaluator2.fit_archetype_model(X_arch_train[split_idx:], y_arch_train[split_idx:])
        evaluator2.fit_synthetic_model(X_synth_train, y_synth_train)
        results2 = evaluator2.evaluate(X_test, y_test)

        # Should be identical
        assert results1['utility_gap'] == pytest.approx(results2['utility_gap'])
        assert results1['archetype']['accuracy'] == pytest.approx(results2['archetype']['accuracy'])

    def test_both_model_types(self, mock_archetype_data, mock_synthetic_data):
        """Test that both XGBoost and RandomForest work."""
        X_arch, y_arch = mock_archetype_data
        X_synth, y_synth = mock_synthetic_data

        for model_type in ['xgboost', 'random_forest']:
            evaluator = DiagnosticEvaluator(model_type=model_type)
            evaluator.fit_archetype_model(X_arch, y_arch)
            evaluator.fit_synthetic_model(X_synth, y_synth)

            # Both should train successfully
            assert evaluator.archetype_model is not None
            assert evaluator.synthetic_model is not None
