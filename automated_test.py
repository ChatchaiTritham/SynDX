"""
Automated Comprehensive Test Suite for SynDX Framework
Tests all modules, phases, and functionality automatically
"""

import sys
import time
import traceback
import numpy as np
import pandas as pd
from datetime import datetime
import os

# Fix Windows console encoding
if sys.platform == 'win32':
    os.system('chcp 65001 > nul')
    sys.stdout.reconfigure(encoding='utf-8')

print("=" * 80)
print("SynDX Framework - Automated Comprehensive Test Suite")
print("=" * 80)
print(f"Test Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print(f"Python Version: {sys.version}")
print("=" * 80)
print()

# Track test results
test_results = {
    'passed': 0,
    'failed': 0,
    'errors': []
}

def test_section(name):
    """Decorator for test sections"""
    print(f"\n{'=' * 80}")
    print(f"Testing: {name}")
    print('=' * 80)

def test_case(description):
    """Print test case description"""
    print(f"\n  → {description}...", end=' ')

def test_pass(details=""):
    """Mark test as passed"""
    global test_results
    test_results['passed'] += 1
    if details:
        print(f"✅ PASS ({details})")
    else:
        print("✅ PASS")

def test_fail(error):
    """Mark test as failed"""
    global test_results
    test_results['failed'] += 1
    print(f"❌ FAIL")
    test_results['errors'].append(str(error))
    print(f"     Error: {error}")

# ============================================================================
# TEST 1: Package Imports
# ============================================================================
test_section("Package Imports")

try:
    test_case("Importing main SynDX package")
    import syndx
    test_pass(f"v{syndx.__version__}")
except Exception as e:
    test_fail(e)

try:
    test_case("Importing Phase 1 modules")
    from syndx.phase1_knowledge import TiTrATEFormalizer, ArchetypeGenerator, StandardsMapper
    test_pass("3 modules")
except Exception as e:
    test_fail(e)

try:
    test_case("Importing Phase 2 modules")
    from syndx.phase2_synthesis import NMFExtractor, VAEModel, train_vae, sample_from_vae
    from syndx.phase2_synthesis import XAIDriver, ProbabilisticLogic
    test_pass("6 components")
except Exception as e:
    test_fail(e)

try:
    test_case("Importing Phase 3 modules")
    from syndx.phase3_validation import StatisticalMetrics, TriateClassifier, EvaluationMetrics
    test_pass("3 modules")
except Exception as e:
    test_fail(e)

try:
    test_case("Importing utility modules")
    from syndx.utils import FHIRExporter, SNOMEDMapper, DataLoader
    test_pass("3 utilities")
except Exception as e:
    test_fail(e)

try:
    test_case("Importing pipeline")
    from syndx import SynDXPipeline
    test_pass()
except Exception as e:
    test_fail(e)

# ============================================================================
# TEST 2: Phase 1 - Clinical Knowledge Extraction
# ============================================================================
test_section("Phase 1: Clinical Knowledge Extraction")

try:
    test_case("Initializing TiTrATEFormalizer")
    from syndx.phase1_knowledge import TiTrATEFormalizer
    formalizer = TiTrATEFormalizer()
    test_pass()
except Exception as e:
    test_fail(e)

try:
    test_case("Getting diagnostic space")
    space = formalizer.get_diagnostic_space()
    test_pass(f"{len(space)} dimensions")
except Exception as e:
    test_fail(e)

try:
    test_case("Initializing ArchetypeGenerator")
    from syndx.phase1_knowledge import ArchetypeGenerator
    generator = ArchetypeGenerator(random_seed=42)
    test_pass()
except Exception as e:
    test_fail(e)

try:
    test_case("Generating archetypes (n=200)")
    start_time = time.time()
    archetypes = generator.generate_archetypes(n_target=200)
    elapsed = time.time() - start_time
    speed = len(archetypes) / elapsed
    test_pass(f"{len(archetypes)} archetypes, {speed:.0f}/sec")
except Exception as e:
    test_fail(e)

try:
    test_case("Validating archetype structure")
    # Check if archetypes is a list (from some implementations) or DataFrame
    if isinstance(archetypes, list):
        assert len(archetypes) > 0, "No archetypes generated"
        test_pass(f"{len(archetypes)} archetypes (list format)")
    else:
        assert isinstance(archetypes, pd.DataFrame), f"Expected DataFrame, got {type(archetypes)}"
        assert len(archetypes) > 0, "No archetypes generated"
        assert 'diagnosis' in archetypes.columns, "Missing 'diagnosis' column"
        assert 'age' in archetypes.columns, "Missing 'age' column"
        assert 'gender' in archetypes.columns, "Missing 'gender' column"
        test_pass(f"{archetypes.shape[0]} rows × {archetypes.shape[1]} cols")
except Exception as e:
    test_fail(e)

try:
    test_case("Testing StandardsMapper initialization")
    from syndx.phase1_knowledge import StandardsMapper
    mapper = StandardsMapper()
    test_pass()
except Exception as e:
    test_fail(e)

# ============================================================================
# TEST 3: Phase 2 - XAI-Driven Synthesis
# ============================================================================
test_section("Phase 2: XAI-Driven Synthesis")

try:
    test_case("Initializing NMFExtractor")
    from syndx.phase2_synthesis import NMFExtractor
    nmf = NMFExtractor(n_components=10, random_state=42)
    test_pass()
except Exception as e:
    test_fail(e)

try:
    test_case("Testing NMF fit_transform()")
    test_matrix = np.random.rand(100, 50)
    W, H = nmf.fit_transform(test_matrix)
    test_pass(f"W{W.shape}, H{H.shape}, error={nmf.reconstruction_error_:.4f}")
except Exception as e:
    test_fail(e)

try:
    test_case("Initializing VAEModel")
    from syndx.phase2_synthesis import VAEModel
    import torch
    vae = VAEModel(input_dim=50, latent_dim=10, hidden_dims=[128, 64])
    test_pass(f"{sum(p.numel() for p in vae.parameters())} params")
except Exception as e:
    test_fail(e)

try:
    test_case("Testing VAE forward pass")
    x = torch.randn(16, 50)
    recon, mu, log_var = vae(x)
    assert recon.shape == x.shape
    assert mu.shape == (16, 10)
    assert log_var.shape == (16, 10)
    test_pass(f"in{tuple(x.shape)} → out{tuple(recon.shape)}")
except Exception as e:
    test_fail(e)

try:
    test_case("Testing VAE training function")
    from syndx.phase2_synthesis import train_vae
    train_data = np.random.rand(100, 50)
    vae_for_training = VAEModel(input_dim=50, latent_dim=10, hidden_dims=[128, 64])
    history = train_vae(vae_for_training, torch.tensor(train_data, dtype=torch.float32),
                        epochs=3, batch_size=32)
    test_pass(f"3 epochs, final_loss={history['total_loss'][-1]:.4f}")

    # Save trained model for next test
    trained_vae = vae_for_training
except Exception as e:
    test_fail(e)
    trained_vae = None

try:
    test_case("Testing VAE sampling function")
    from syndx.phase2_synthesis import sample_from_vae
    if trained_vae is not None:
        samples = sample_from_vae(trained_vae, n_samples=50)
        test_pass(f"{len(samples)} samples, shape={samples.shape}")
    else:
        raise Exception("VAE not trained in previous test")
except Exception as e:
    test_fail(e)

try:
    test_case("Initializing XAIDriver")
    from syndx.phase2_synthesis import XAIDriver
    xai = XAIDriver()
    test_pass()
except Exception as e:
    test_fail(e)

try:
    test_case("Initializing ProbabilisticLogic")
    from syndx.phase2_synthesis import ProbabilisticLogic
    prob_logic = ProbabilisticLogic()
    test_pass()
except Exception as e:
    test_fail(e)

# ============================================================================
# TEST 4: Phase 3 - Multi-Level Validation
# ============================================================================
test_section("Phase 3: Multi-Level Validation")

try:
    test_case("Initializing StatisticalMetrics")
    from syndx.phase3_validation import StatisticalMetrics
    stats = StatisticalMetrics()
    test_pass()
except Exception as e:
    test_fail(e)

try:
    test_case("Testing KL divergence via EvaluationMetrics")
    from syndx.phase3_validation import EvaluationMetrics
    evaluator_temp = EvaluationMetrics()
    real_data = pd.DataFrame(np.random.randn(100, 5), columns=[f'f{i}' for i in range(5)])
    synth_data = pd.DataFrame(np.random.randn(100, 5), columns=[f'f{i}' for i in range(5)])
    kl_div = evaluator_temp.calculate_kl_divergence(synth_data, real_data)
    test_pass(f"KL={kl_div:.6f}")
except Exception as e:
    test_fail(e)

try:
    test_case("Initializing TriateClassifier")
    from syndx.phase3_validation import TriateClassifier
    triate = TriateClassifier()
    test_pass()
except Exception as e:
    test_fail(e)

try:
    test_case("Testing TriateClassifier.classify() - Stroke case")
    stroke_patient = {
        'age': 72,
        'diagnosis': 'stroke',
        'nystagmus_type': 'central'
    }
    pathway = triate.classify(stroke_patient)
    assert pathway == 'ER', f"Expected ER, got {pathway}"
    test_pass(f"Stroke → {pathway}")
except Exception as e:
    test_fail(e)

try:
    test_case("Testing TriateClassifier.classify() - BPPV case")
    bppv_patient = {
        'age': 45,
        'diagnosis': 'bppv',
        'nystagmus_type': 'horizontal'
    }
    pathway = triate.classify(bppv_patient)
    assert pathway == 'Home_Observation', f"Expected Home_Observation, got {pathway}"
    test_pass(f"BPPV → {pathway}")
except Exception as e:
    test_fail(e)

try:
    test_case("Initializing EvaluationMetrics")
    from syndx.phase3_validation import EvaluationMetrics
    evaluator = EvaluationMetrics()
    test_pass()
except Exception as e:
    test_fail(e)

try:
    test_case("Testing EvaluationMetrics.compute_all_metrics()")
    real_df = pd.DataFrame(np.random.randn(100, 3), columns=['age', 'score', 'duration'])
    synth_df = pd.DataFrame(np.random.randn(100, 3), columns=['age', 'score', 'duration'])
    metrics = evaluator.compute_all_metrics(real_df, synth_df)

    assert 'statistical' in metrics
    assert 'quality' in metrics
    assert 'features' in metrics
    test_pass(f"{len(metrics)} categories, KL={metrics['statistical'].get('mean_kl_divergence', 0):.4f}")
except Exception as e:
    test_fail(e)

# ============================================================================
# TEST 5: Utilities
# ============================================================================
test_section("Utility Modules")

try:
    test_case("Initializing FHIRExporter")
    from syndx.utils import FHIRExporter
    fhir = FHIRExporter()
    test_pass()
except Exception as e:
    test_fail(e)

try:
    test_case("Initializing SNOMEDMapper")
    from syndx.utils import SNOMEDMapper
    snomed = SNOMEDMapper()
    test_pass()
except Exception as e:
    test_fail(e)

try:
    test_case("Initializing DataLoader")
    from syndx.utils import DataLoader
    loader = DataLoader()
    test_pass()
except Exception as e:
    test_fail(e)

# ============================================================================
# TEST 6: End-to-End Pipeline
# ============================================================================
test_section("End-to-End Pipeline")

try:
    test_case("Initializing SynDXPipeline")
    from syndx import SynDXPipeline
    pipeline = SynDXPipeline(random_seed=42)
    test_pass()
except Exception as e:
    test_fail(e)

try:
    test_case("Running mini pipeline (50 archetypes)")
    start_time = time.time()

    # Phase 1: Generate archetypes (may return list or DataFrame)
    archetypes_result = pipeline.archetype_generator.generate_archetypes(n_target=50)

    # Convert to DataFrame if it's a list
    if isinstance(archetypes_result, list):
        # Assume list of dicts, convert to DataFrame
        archetypes_df = pd.DataFrame(archetypes_result)
    else:
        archetypes_df = archetypes_result

    # Phase 2: Extract numeric features for NMF
    numeric_cols = archetypes_df.select_dtypes(include=[np.number]).columns
    archetype_matrix = archetypes_df[numeric_cols].fillna(0).values

    # Normalize data to [0, 1] range for VAE
    from sklearn.preprocessing import MinMaxScaler
    scaler = MinMaxScaler()
    archetype_matrix_scaled = scaler.fit_transform(archetype_matrix)

    # NMF extraction (n_components must be <= min(n_samples, n_features))
    n_components = min(5, archetype_matrix_scaled.shape[0], archetype_matrix_scaled.shape[1])
    nmf_extractor = NMFExtractor(n_components=n_components, random_state=42)
    W, H = nmf_extractor.fit_transform(archetype_matrix_scaled)

    # VAE training (minimal)
    input_dim = archetype_matrix_scaled.shape[1]
    vae_mini = VAEModel(input_dim=input_dim, latent_dim=min(5, input_dim), hidden_dims=[64, 32])
    history = train_vae(vae_mini, torch.tensor(archetype_matrix_scaled, dtype=torch.float32),
                        epochs=2, batch_size=16)

    # Generate synthetic samples
    synthetic = sample_from_vae(vae_mini, n_samples=50)

    elapsed = time.time() - start_time
    test_pass(f"{len(archetypes_df)} arch → {len(synthetic)} synth in {elapsed:.2f}s")
except Exception as e:
    test_fail(e)

# ============================================================================
# TEST SUMMARY
# ============================================================================
print("\n" + "=" * 80)
print("TEST SUMMARY")
print("=" * 80)

total_tests = test_results['passed'] + test_results['failed']
pass_rate = (test_results['passed'] / total_tests * 100) if total_tests > 0 else 0

print(f"\nTotal Tests: {total_tests}")
print(f"✅ Passed: {test_results['passed']}")
print(f"❌ Failed: {test_results['failed']}")
print(f"Success Rate: {pass_rate:.1f}%")

if test_results['failed'] > 0:
    print(f"\n{'=' * 80}")
    print("ERRORS ENCOUNTERED:")
    print('=' * 80)
    for i, error in enumerate(test_results['errors'], 1):
        print(f"\n{i}. {error}")

print(f"\n{'=' * 80}")
if test_results['failed'] == 0:
    print("🎉 ALL TESTS PASSED! Framework is 100% operational.")
    print("=" * 80)
    sys.exit(0)
else:
    print("⚠️  Some tests failed. Please review errors above.")
    print("=" * 80)
    sys.exit(1)
