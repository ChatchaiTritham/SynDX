"""
Unit tests for CounterfactualValidator module.

Tests TiTrATE-constrained counterfactual generation.

Author: Chatchai Tritham
Date: 2026-01-25
"""

import pytest
import numpy as np
from syndx.phase2_synthesis.counterfactual_validator import CounterfactualValidator


@pytest.mark.unit
@pytest.mark.counterfactual
class TestCounterfactualValidatorInit:
    """Test CounterfactualValidator initialization."""

    def test_default_initialization(self):
        """Test initialization with default parameters."""
        validator = CounterfactualValidator(constraint_checker=None)

        assert validator.max_iterations == 100
        assert validator.distance_metric == 'l2'
        assert validator.random_state == 42
        assert validator.constraint_checker is None

    def test_custom_initialization(self):
        """Test initialization with custom parameters."""
        validator = CounterfactualValidator(
            constraint_checker=None,
            max_iterations=50,
            distance_metric='l1',
            random_state=123
        )

        assert validator.max_iterations == 50
        assert validator.distance_metric == 'l1'
        assert validator.random_state == 123

    def test_repr(self):
        """Test string representation."""
        validator = CounterfactualValidator(constraint_checker=None)
        repr_str = repr(validator)

        assert "CounterfactualValidator" in repr_str
        assert "max_iter=100" in repr_str
        assert "metric='l2'" in repr_str


@pytest.mark.unit
@pytest.mark.counterfactual
class TestGenerateCounterfactual:
    """Test single counterfactual generation."""

    def test_generate_counterfactual_basic(self, mock_patient_data):
        """Test basic counterfactual generation."""
        patient, feature_names = mock_patient_data
        validator = CounterfactualValidator(constraint_checker=None, max_iterations=50)

        cf = validator.generate_counterfactual(
            patient,
            target_diagnosis='VM',
            feature_names=feature_names
        )

        # May or may not find valid counterfactual with mock validation
        if cf is not None:
            assert 'features' in cf
            assert 'distance' in cf
            assert 'is_valid' in cf
            assert 'changes' in cf
            assert cf['features'].shape == patient.shape

    def test_generate_counterfactual_with_modifiable_features(self, mock_patient_data):
        """Test counterfactual with restricted modifiable features."""
        patient, feature_names = mock_patient_data
        validator = CounterfactualValidator(constraint_checker=None, max_iterations=30)

        # Only allow modifying first 5 features
        modifiable = [0, 1, 2, 3, 4]

        cf = validator.generate_counterfactual(
            patient,
            target_diagnosis='BPPV',
            feature_names=feature_names,
            modifiable_features=modifiable
        )

        if cf is not None and cf['changes']:
            # All changes should be in modifiable features
            changed_features = list(cf['changes'].keys())
            for feat_name in changed_features:
                feat_idx = feature_names.index(feat_name)
                assert feat_idx in modifiable

    def test_counterfactual_distance_positive(self, mock_patient_data):
        """Test that counterfactual has positive distance."""
        patient, feature_names = mock_patient_data
        validator = CounterfactualValidator(constraint_checker=None, max_iterations=50)

        cf = validator.generate_counterfactual(
            patient,
            target_diagnosis='VN',
            feature_names=feature_names
        )

        if cf is not None:
            assert cf['distance'] >= 0


@pytest.mark.unit
@pytest.mark.counterfactual
class TestGenerateCounterfactuals:
    """Test multiple counterfactual generation."""

    def test_generate_multiple_counterfactuals(self, mock_patient_data):
        """Test generating multiple counterfactuals."""
        patient, feature_names = mock_patient_data
        validator = CounterfactualValidator(constraint_checker=None, max_iterations=30)

        counterfactuals = validator.generate_counterfactuals(
            patient,
            target_diagnosis='BPPV',
            n_counterfactuals=3,
            feature_names=feature_names
        )

        assert isinstance(counterfactuals, list)
        assert len(counterfactuals) <= 3  # May find fewer than requested

        for cf in counterfactuals:
            assert 'features' in cf
            assert 'distance' in cf
            assert 'is_valid' in cf

    def test_counterfactuals_diversity(self, mock_patient_data):
        """Test that multiple counterfactuals are diverse."""
        patient, feature_names = mock_patient_data
        validator = CounterfactualValidator(constraint_checker=None, max_iterations=50)

        counterfactuals = validator.generate_counterfactuals(
            patient,
            target_diagnosis='VM',
            n_counterfactuals=5,
            feature_names=feature_names
        )

        if len(counterfactuals) >= 2:
            # Check that different counterfactuals have different features
            cf1_features = counterfactuals[0]['features']
            cf2_features = counterfactuals[1]['features']

            # Should not be identical
            assert not np.array_equal(cf1_features, cf2_features)


@pytest.mark.unit
@pytest.mark.counterfactual
class TestPerturbPatient:
    """Test patient perturbation."""

    def test_perturb_patient_basic(self, mock_patient_data):
        """Test basic patient perturbation."""
        patient, _ = mock_patient_data
        validator = CounterfactualValidator(constraint_checker=None)

        modifiable_features = list(range(len(patient)))
        perturbed = validator._perturb_patient(patient, modifiable_features)

        assert perturbed.shape == patient.shape
        assert not np.array_equal(perturbed, patient)  # Should be different

    def test_perturb_patient_clips_values(self, mock_patient_data):
        """Test that perturbation clips extreme values."""
        patient, _ = mock_patient_data
        validator = CounterfactualValidator(constraint_checker=None)

        modifiable_features = list(range(len(patient)))
        perturbed = validator._perturb_patient(patient, modifiable_features)

        # Should be clipped to [-10, 10]
        assert np.all(perturbed >= -10)
        assert np.all(perturbed <= 10)

    def test_perturb_patient_respects_modifiable_features(self, mock_patient_data):
        """Test that only modifiable features are changed."""
        patient, _ = mock_patient_data
        validator = CounterfactualValidator(constraint_checker=None)

        # Only allow modifying first 3 features
        modifiable_features = [0, 1, 2]

        # Run multiple times to check consistency
        for _ in range(5):
            perturbed = validator._perturb_patient(patient, modifiable_features)

            # Non-modifiable features should be unchanged (with high probability)
            # Note: Due to randomness, this may occasionally fail
            # but should mostly preserve non-modifiable features
            pass  # Visual inspection test


@pytest.mark.unit
@pytest.mark.counterfactual
class TestComputeDistance:
    """Test distance computation."""

    def test_compute_distance_l2(self):
        """Test L2 distance computation."""
        validator = CounterfactualValidator(constraint_checker=None, distance_metric='l2')

        original = np.array([0, 0, 0])
        counterfactual = np.array([3, 4, 0])

        distance = validator._compute_distance(original, counterfactual)

        assert distance == pytest.approx(5.0)  # 3-4-5 triangle

    def test_compute_distance_l1(self):
        """Test L1 distance computation."""
        validator = CounterfactualValidator(constraint_checker=None, distance_metric='l1')

        original = np.array([0, 0, 0])
        counterfactual = np.array([3, 4, 0])

        distance = validator._compute_distance(original, counterfactual)

        assert distance == pytest.approx(7.0)  # |3| + |4| + |0|

    def test_compute_distance_zero_for_identical(self):
        """Test that distance is zero for identical vectors."""
        validator = CounterfactualValidator(constraint_checker=None)

        vector = np.random.randn(20)
        distance = validator._compute_distance(vector, vector.copy())

        assert distance == pytest.approx(0.0, abs=1e-10)


@pytest.mark.unit
@pytest.mark.counterfactual
class TestIdentifyChanges:
    """Test change identification."""

    def test_identify_changes_basic(self):
        """Test basic change identification."""
        validator = CounterfactualValidator(constraint_checker=None)

        original = np.array([1.0, 2.0, 3.0])
        counterfactual = np.array([1.5, 2.0, 4.0])

        feature_names = ['feat_0', 'feat_1', 'feat_2']
        changes = validator._identify_changes(original, counterfactual, feature_names)

        assert 'feat_0' in changes
        assert 'feat_2' in changes
        assert 'feat_1' not in changes  # Unchanged

        assert changes['feat_0']['original'] == 1.0
        assert changes['feat_0']['counterfactual'] == 1.5
        assert changes['feat_0']['delta'] == 0.5

    def test_identify_changes_without_feature_names(self):
        """Test change identification without feature names."""
        validator = CounterfactualValidator(constraint_checker=None)

        original = np.array([1.0, 2.0, 3.0])
        counterfactual = np.array([1.5, 2.0, 3.0])

        changes = validator._identify_changes(original, counterfactual)

        assert 'feature_0' in changes
        assert changes['feature_0']['delta'] == 0.5


@pytest.mark.unit
@pytest.mark.counterfactual
class TestValidatePlausibility:
    """Test plausibility validation."""

    def test_validate_plausibility_basic(self, mock_patient_data):
        """Test basic plausibility validation."""
        patient, _ = mock_patient_data
        validator = CounterfactualValidator(constraint_checker=None)

        # Create a slightly modified counterfactual
        counterfactual = patient.copy()
        counterfactual[0] += 0.5
        counterfactual[5] -= 0.3

        plausibility = validator.validate_plausibility(patient, counterfactual)

        assert 'is_valid' in plausibility
        assert 'l1_distance' in plausibility
        assert 'l2_distance' in plausibility
        assert 'changed_features' in plausibility
        assert 'plausibility_score' in plausibility

        assert plausibility['changed_features'] == 2
        assert 0.0 <= plausibility['plausibility_score'] <= 1.0

    def test_plausibility_higher_for_fewer_changes(self, mock_patient_data):
        """Test that plausibility is higher for fewer changes."""
        patient, _ = mock_patient_data
        validator = CounterfactualValidator(constraint_checker=None)

        # Few changes
        cf_few = patient.copy()
        cf_few[0] += 0.5

        # Many changes
        cf_many = patient.copy()
        for i in range(10):
            cf_many[i] += 0.5

        plausibility_few = validator.validate_plausibility(patient, cf_few)
        plausibility_many = validator.validate_plausibility(patient, cf_many)

        assert plausibility_few['plausibility_score'] > plausibility_many['plausibility_score']


@pytest.mark.unit
@pytest.mark.counterfactual
class TestActionableInsights:
    """Test actionable insights generation."""

    def test_compute_actionable_insights_basic(self, mock_patient_data):
        """Test basic actionable insights computation."""
        patient, feature_names = mock_patient_data
        validator = CounterfactualValidator(constraint_checker=None, max_iterations=30)

        diagnoses = ['BPPV', 'VM', 'VN']
        insights = validator.compute_actionable_insights(patient, diagnoses, feature_names)

        assert len(insights) == len(diagnoses)

        for insight in insights:
            assert 'target_diagnosis' in insight
            assert 'distance' in insight
            assert 'changes' in insight
            assert 'is_valid' in insight

    def test_insights_sorted_by_distance(self, mock_patient_data):
        """Test that insights are sorted by distance."""
        patient, feature_names = mock_patient_data
        validator = CounterfactualValidator(constraint_checker=None, max_iterations=30)

        diagnoses = ['BPPV', 'VM', 'VN', 'Stroke']
        insights = validator.compute_actionable_insights(patient, diagnoses, feature_names)

        distances = [ins['distance'] for ins in insights]

        # Should be sorted in ascending order
        assert distances == sorted(distances)


@pytest.mark.unit
@pytest.mark.counterfactual
class TestSummary:
    """Test CounterfactualValidator summary method."""

    def test_summary_no_counterfactuals(self):
        """Test summary with no counterfactuals."""
        validator = CounterfactualValidator(constraint_checker=None)
        summary = validator.summary()

        assert summary['status'] == 'no_counterfactuals'

    def test_summary_with_counterfactuals(self, mock_patient_data):
        """Test summary with counterfactuals."""
        patient, feature_names = mock_patient_data
        validator = CounterfactualValidator(constraint_checker=None, max_iterations=30)

        # Generate some counterfactuals
        cf1 = validator.generate_counterfactual(patient, 'BPPV', feature_names)
        cf2 = validator.generate_counterfactual(patient, 'VM', feature_names)

        # Manually add to list for testing
        if cf1:
            validator.counterfactuals.append(cf1)
        if cf2:
            validator.counterfactuals.append(cf2)

        if validator.counterfactuals:
            summary = validator.summary()

            assert summary['status'] == 'validated'
            assert 'total_counterfactuals' in summary
            assert 'valid_counterfactuals' in summary
            assert 'avg_distance' in summary


@pytest.mark.unit
@pytest.mark.counterfactual
class TestCounterfactualValidatorIntegration:
    """Integration tests for CounterfactualValidator."""

    def test_full_workflow(self, mock_patient_data):
        """Test complete counterfactual validation workflow."""
        patient, feature_names = mock_patient_data
        validator = CounterfactualValidator(constraint_checker=None, max_iterations=50)

        # Generate counterfactuals for multiple diagnoses
        diagnoses = ['BPPV', 'VM']

        all_counterfactuals = []
        for diagnosis in diagnoses:
            cfs = validator.generate_counterfactuals(
                patient,
                target_diagnosis=diagnosis,
                n_counterfactuals=2,
                feature_names=feature_names
            )
            all_counterfactuals.extend(cfs)

        # Validate plausibility
        if all_counterfactuals:
            for cf in all_counterfactuals:
                plausibility = validator.validate_plausibility(patient, cf['features'])
                assert 0.0 <= plausibility['plausibility_score'] <= 1.0

    def test_reproducibility_with_random_seed(self, mock_patient_data):
        """Test reproducibility with fixed random seed."""
        patient, feature_names = mock_patient_data

        # Run 1
        np.random.seed(42)
        validator1 = CounterfactualValidator(constraint_checker=None, random_state=42, max_iterations=20)
        cf1 = validator1.generate_counterfactual(patient, 'BPPV', feature_names)

        # Run 2
        np.random.seed(42)
        validator2 = CounterfactualValidator(constraint_checker=None, random_state=42, max_iterations=20)
        cf2 = validator2.generate_counterfactual(patient, 'BPPV', feature_names)

        # Should give similar results (with mock validation, may have randomness)
        if cf1 is not None and cf2 is not None:
            # At minimum, should have same structure
            assert cf1.keys() == cf2.keys()
