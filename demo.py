"""
SynDX-Hybrid Demo Script

Demonstrates basic usage of the SynDX-Hybrid framework for synthetic medical data generation.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__)))

from syn_dx_hybrid.pipeline import SynDXHybridPipeline

def main():
    print("="*60)
    print("SYNDX-HYBRID FRAMEWORK DEMO")
    print("="*60)

    print("\nInitializing SynDX-Hybrid Pipeline with small parameters for demo...")

    # Initialize pipeline with smaller parameters for demo
    pipeline = SynDXHybridPipeline(
        n_archetypes=50,        # Smaller number for demo
        bayesian_nodes=10,      # Fewer nodes for demo
        rule_base_size=20,      # Smaller rule base for demo
        random_seed=42
    )

    print("OK Pipeline initialized successfully")
    print(f"  - Archetypes: {pipeline.n_archetypes}")
    print(f"  - Bayesian nodes: {pipeline.bayesian_nodes}")
    print(f"  - Rule base size: {pipeline.rule_base_size}")

    print("\nRunning full five-layer pipeline...")

    # Generate synthetic data
    synthetic_data = pipeline.run_full_pipeline(n_patients=100)

    print(f"\nOK Pipeline completed successfully!")
    print(f"  - Generated {len(synthetic_data)} synthetic patients")

    # Show some statistics
    stats = pipeline.get_statistics()
    print(f"\nConfiguration:")
    for key, value in stats['configuration'].items():
        print(f"  {key}: {value}")

    print(f"\nLayer Statistics:")
    for key, value in stats['layer_statistics'].items():
        print(f"  {key}: {value}")

    print("\nOK Demo completed successfully!")
    print("="*60)

if __name__ == '__main__':
    main()