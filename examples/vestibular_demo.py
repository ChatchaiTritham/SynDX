"""
Complete Vestibular Domain Demo
Demonstrates full XAI-guided parameter space exploration
"""

from syndx.utils.formulas import (
    calculate_n_target,
    calculate_r_clinical,
    calculate_complexity_factor
)
from syndx.phase1_knowledge.xai_explorer import XAIGuidedExplorer
from syndx.phase1_knowledge.domain_config import create_vestibular_domain
from pathlib import Path
import logging
import numpy as np
import sys
import os
sys.path.insert(
    0,
    os.path.abspath(
        os.path.join(
            os.path.dirname(__file__),
            '..')))


# Import SynDX modules

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def main():
    """
    Complete demonstration of XAI-guided parameter space exploration
    for vestibular domain (dizziness and vertigo)
    """

    print("=" * 80)
    print(" " * 20 + "SYNDX: VESTIBULAR DOMAIN DEMO")
    print("=" * 80)
    print()

    # ========================================================================
    # Step 1: Create Parameter Space
    # ========================================================================
    print("Step 1: Creating parameter space...")
    print("-" * 80)

    param_space = create_vestibular_domain()

    print(f"✓ Parameter space created")
    print(f"  - Parameters (m): {param_space.m}")
    print(f"  - Diagnoses (|D|): {param_space.D_size}")
    print(f"  - Total combinations (|P|): {param_space.space_size:,}")
    print(f"  - Acceptance rate (ρ): {param_space.acceptance_rate:.3f}")
    print()

    # ========================================================================
    # Step 2: Calculate Target Archetypes
    # ========================================================================
    print("Step 2: Calculating target archetypes...")
    print("-" * 80)

    psi = calculate_complexity_factor('vestibular')

    n_target, breakdown = calculate_n_target(
        space_size=param_space.space_size,
        n_diagnoses=param_space.D_size,
        valid_space_size=int(param_space.space_size * param_space.acceptance_rate),
        psi=psi,
        critical_diagnoses={'stroke': 0.10, 'tia': 0.05}
    )

    print(f"✓ Target calculated: n_target = {n_target}")
    print(f"  - Statistical requirement: {breakdown['n_statistical']}")
    print(f"  - Coverage requirement: {breakdown['n_coverage']}")
    print(f"  - Clinical requirement: {breakdown['n_clinical']}")
    print(f"  - Optimal formula: {breakdown['n_optimal']}")
    print(f"  - Complexity factor (ψ): {psi}")
    print()

    # ========================================================================
    # Step 3: Calculate NMF Factors
    # ========================================================================
    print("Step 3: Calculating NMF factors...")
    print("-" * 80)

    r = calculate_r_clinical(param_space.D_size, param_space.m)

    print(f"✓ NMF factors: r = {r}")
    print(f"  - Formula: r = ⌈log₂(|D|) + √(m/10)⌉")
    print(
        f"  - log₂({param_space.D_size}) = {np.log2(param_space.D_size):.2f}")
    print(f"  - √({param_space.m}/10) = {np.sqrt(param_space.m / 10):.2f}")
    print()

    # ========================================================================
    # Step 4: Initialize Explorer
    # ========================================================================
    print("Step 4: Initializing XAI-Guided Explorer...")
    print("-" * 80)

    explorer = XAIGuidedExplorer(
        parameter_space=param_space,
        n_target=n_target,
        nmf_factors=r,
        alpha_importance=0.60,
        alpha_critical=0.30,
        alpha_diversity=0.10,
        random_state=42
    )

    print(f"✓ Explorer initialized")
    print(f"  - Importance-weighted: {explorer.n_importance} (60%)")
    print(f"  - Critical scenarios: {explorer.n_critical} (30%)")
    print(f"  - Diversity-oriented: {explorer.n_diversity} (10%)")
    print()

    # ========================================================================
    # Step 5: Execute Exploration
    # ========================================================================
    print("Step 5: Executing XAI-guided exploration...")
    print("-" * 80)
    print()

    archetypes = explorer.explore()

    print()
    print("=" * 80)
    print(" " * 25 + "EXPLORATION COMPLETED!")
    print("=" * 80)
    print()

    # ========================================================================
    # Step 6: Display Results
    # ========================================================================
    print("Step 6: Results Summary")
    print("-" * 80)

    stats = explorer.get_statistics()

    print(f"Final archetypes generated: {stats['final_count']}")
    print(f"Target: {stats['configuration']['n_target']}")
    print(
        f"Achievement: {
            stats['final_count'] / stats['configuration']['n_target'] * 100:.1f}%")
    print()

    print("Sampling Statistics:")
    print(f"  Phase 1 (Uniform):")
    print(f"    - Sampled: {stats['sampling_stats']['phase1_sampled']}")
    print(f"    - Valid: {stats['sampling_stats']['phase1_valid']}")
    print(
        f"    - Rate: {stats['sampling_stats']['phase1_valid'] / stats['sampling_stats']['phase1_sampled'] * 100:.1f}%")
    print()
    print(f"  Phase 4 (Importance):")
    print(f"    - Sampled: {stats['sampling_stats']['phase4_sampled']}")
    print(f"    - Valid: {stats['sampling_stats']['phase4_valid']}")
    print(
        f"    - Rate: {stats['sampling_stats']['phase4_valid'] / stats['sampling_stats']['phase4_sampled'] * 100:.1f}%")
    print()
    print(f"  Phase 5 (Critical):")
    print(f"    - Sampled: {stats['sampling_stats']['phase5_sampled']}")
    print(f"    - Valid: {stats['sampling_stats']['phase5_valid']}")
    print(
        f"    - Rate: {stats['sampling_stats']['phase5_valid'] / stats['sampling_stats']['phase5_sampled'] * 100:.1f}%")
    print()
    print(f"  Phase 6 (Diversity):")
    print(f"    - Sampled: {stats['sampling_stats']['phase6_sampled']}")
    print(f"    - Valid: {stats['sampling_stats']['phase6_valid']}")
    print(
        f"    - Rate: {stats['sampling_stats']['phase6_valid'] / stats['sampling_stats']['phase6_sampled'] * 100:.1f}%")
    print()

    # NMF Results
    if stats['nmf_summary']:
        print("NMF Factor Discovery:")
        print(f"  - Factors: {stats['nmf_summary']['n_components']}")
        print(
            f"  - Reconstruction error: {stats['nmf_summary']['reconstruction_error']:.4f}")
        print(f"  - Discovered patterns:")
        for interp in stats['nmf_summary']['factor_interpretations']:
            print(
                f"    Factor {
                    interp['factor_id']}: {
                    interp['clinical_pattern']}")
        print()

    # SHAP Results
    if stats['shap_summary']:
        print("SHAP Feature Importance (Top 10):")
        for name, importance in stats['shap_summary']['top_10_features']:
            print(f"  {name:<40} {importance:.4f}")
        print()

    # ========================================================================
    # Step 7: Sample Output
    # ========================================================================
    print("Step 7: Sample Archetypes")
    print("-" * 80)

    print("\nFirst 5 archetypes:")
    for i, archetype in enumerate(archetypes[:5]):
        print(f"\nArchetype {i + 1}:")
        print(f"  Diagnosis: {archetype.diagnosis}")
        print(f"  Age: {archetype.parameters.get('age', 'N/A')}")
        print(f"  Timing: {archetype.parameters.get('timing', 'N/A')}")
        print(f"  Trigger: {archetype.parameters.get('trigger', 'N/A')}")
        print(
            f"  HINTS - Nystagmus: {archetype.parameters.get('nystagmus_type', 'N/A')}")
        print(f"  Urgency: {archetype.parameters.get('urgency', 'N/A')}")

    print()
    print("=" * 80)
    print(" " * 25 + "DEMO COMPLETED!")
    print("=" * 80)
    print()

    # ========================================================================
    # Step 8: Visualizations
    # ========================================================================
    print("Step 8: Generating visualizations...")
    print("-" * 80)

    try:
        from examples.visualization import create_all_visualizations

        output_dir = Path("outputs/vestibular_demo")
        output_dir.mkdir(parents=True, exist_ok=True)

        create_all_visualizations(
            explorer=explorer,
            archetypes=archetypes,
            param_space=param_space,
            output_dir=output_dir
        )

        print(f"✓ Visualizations saved to: {output_dir}")

    except Exception as e:
        print(f"⚠ Visualization skipped: {e}")

    print()
    print("Demo complete! Check outputs/ directory for results.")

    return explorer, archetypes


if __name__ == "__main__":
    explorer, archetypes = main()
