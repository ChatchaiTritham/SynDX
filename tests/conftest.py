"""
Pytest configuration and shared fixtures for SynDX tests.

This module provides common fixtures used across multiple test files,
including mock data generation, test configurations, and cleanup utilities.

Author: Chatchai Tritham
Date: 2026-01-25
"""

import pytest
import numpy as np
import pandas as pd
from pathlib import Path
import tempfile
import shutil


# ============================================================================
# Test Configuration
# ============================================================================

@pytest.fixture(scope="session")
def test_config():
    """Global test configuration."""
    return {
        'random_state': 42,
        'n_samples': 100,
        'n_features': 50,
        'n_classes': 4,
        'epsilon': 1.0,
        'delta': 1e-5,
        'clip_norm': 1.0
    }


# ============================================================================
# Data Fixtures
# ============================================================================

@pytest.fixture(scope="function")
def mock_archetype_data(test_config):
    """Generate mock archetype data for testing."""
    np.random.seed(test_config['random_state'])

    n_samples = test_config['n_samples']
    n_features = test_config['n_features']

    # Generate features with some structure
    X = np.random.randn(n_samples, n_features)

    # Amplify important features
    important_features = [0, 5, 10, 15, 20]
    for feat_idx in important_features:
        X[:, feat_idx] *= 3.0

    # Generate labels based on important features
    y = np.zeros(n_samples, dtype=int)
    for i in range(n_samples):
        score = np.sum(X[i, important_features])
        if score > 4:
            y[i] = 3  # Stroke
        elif score > 2:
            y[i] = 2  # VN
        elif score > 0:
            y[i] = 1  # VM
        else:
            y[i] = 0  # BPPV

    return X, y


@pytest.fixture(scope="function")
def mock_synthetic_data(test_config):
    """Generate mock synthetic data for testing."""
    np.random.seed(test_config['random_state'] + 1)

    n_samples = test_config['n_samples']
    n_features = test_config['n_features']

    # Generate features with slightly different structure
    X = np.random.randn(n_samples, n_features)

    important_features = [0, 5, 10, 15, 20]
    for feat_idx in important_features:
        X[:, feat_idx] *= 2.8  # Slightly weaker signal

    # Generate labels
    y = np.zeros(n_samples, dtype=int)
    for i in range(n_samples):
        score = np.sum(X[i, important_features])
        if score > 3.5:
            y[i] = 3
        elif score > 1.5:
            y[i] = 2
        elif score > -0.5:
            y[i] = 1
        else:
            y[i] = 0

    return X, y


@pytest.fixture(scope="function")
def mock_patient_data(test_config):
    """Generate a single mock patient for counterfactual testing."""
    np.random.seed(test_config['random_state'])

    n_features = 20
    patient = np.random.randn(n_features)

    # Set specific feature values for testing
    patient[0] = 1.0   # Age
    patient[1] = 0.0   # Gender
    patient[2] = 1.5   # Duration
    patient[3] = 3.0   # Trigger type

    feature_names = [
        'age', 'gender', 'duration', 'trigger', 'nystagmus', 'nyst_direction',
        'vertigo_intensity', 'episode_duration', 'hearing_loss', 'tinnitus',
        'headache', 'aura', 'family_history', 'previous_episodes',
        'medication_response', 'dix_hallpike_positive', 'head_impulse_test',
        'cerebellar_signs', 'autonomic_symptoms', 'photophobia'
    ]

    return patient, feature_names


@pytest.fixture(scope="function")
def mock_gradients(test_config):
    """Generate mock per-sample gradients for DP testing."""
    np.random.seed(test_config['random_state'])

    batch_size = 64
    grad_dim = 100

    gradients = []
    for i in range(batch_size):
        grad = np.random.randn(grad_dim)
        # Make some gradients large to trigger clipping
        if i % 10 == 0:
            grad *= 5.0
        gradients.append(grad)

    return gradients


@pytest.fixture(scope="function")
def mock_nmf_matrices(test_config):
    """Generate mock NMF factor matrices for interaction fidelity testing."""
    np.random.seed(test_config['random_state'])

    n_factors = 20
    n_features = test_config['n_features']

    # Archetype NMF matrix
    W_arch = np.random.rand(n_features, n_factors)
    H_arch = np.random.rand(n_factors, test_config['n_samples'])

    # Synthetic NMF matrix (similar but with noise)
    W_synth = W_arch + np.random.randn(n_features, n_factors) * 0.1
    H_synth = H_arch + np.random.randn(n_factors, test_config['n_samples']) * 0.1

    return (W_arch, H_arch), (W_synth, H_synth)


# ============================================================================
# File System Fixtures
# ============================================================================

@pytest.fixture(scope="function")
def temp_output_dir():
    """Create a temporary output directory for test files."""
    temp_dir = tempfile.mkdtemp(prefix="syndx_test_")
    yield Path(temp_dir)
    # Cleanup after test
    if Path(temp_dir).exists():
        shutil.rmtree(temp_dir)


@pytest.fixture(scope="function")
def temp_figure_dir(temp_output_dir):
    """Create a temporary directory for figure outputs."""
    fig_dir = temp_output_dir / "figures"
    fig_dir.mkdir(exist_ok=True)
    return fig_dir


# ============================================================================
# Cleanup Fixtures
# ============================================================================

@pytest.fixture(scope="function", autouse=True)
def reset_random_state(test_config):
    """Reset random state before each test."""
    np.random.seed(test_config['random_state'])
    yield
    # No cleanup needed


@pytest.fixture(scope="session", autouse=True)
def setup_test_environment():
    """Setup test environment before all tests."""
    # Suppress warnings during tests
    import warnings
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    warnings.filterwarnings("ignore", category=UserWarning)

    yield

    # Cleanup after all tests
    pass


# ============================================================================
# Pytest Hooks
# ============================================================================

def pytest_configure(config):
    """Configure pytest with custom markers."""
    config.addinivalue_line("markers", "unit: Unit tests")
    config.addinivalue_line("markers", "integration: Integration tests")
    config.addinivalue_line("markers", "slow: Slow-running tests")
    config.addinivalue_line("markers", "validation: Validation tests")
    config.addinivalue_line("markers", "visualization: Visualization tests")
    config.addinivalue_line("markers", "privacy: Privacy mechanism tests")
    config.addinivalue_line("markers", "fidelity: XAI fidelity tests")
    config.addinivalue_line("markers", "counterfactual: Counterfactual generation tests")
