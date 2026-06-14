"""
SynDX Demo Script

Demonstrates basic usage of the SynDX pipeline for synthetic medical data
generation, using the actual installed package (syndx). Runs a small, seeded
end-to-end pass: archetype extraction, synthetic generation, and validation.
"""

import os
import sys

sys.path.insert(
    0, os.path.join(os.path.dirname(os.path.dirname(__file__)), "src")
)

from syndx.pipeline import SynDXPipeline


def main():
    print("=" * 60)
    print("SYNDX FRAMEWORK DEMO")
    print("=" * 60)

    print("\nInitializing SynDX pipeline with small parameters for demo...")

    # Initialize pipeline with smaller parameters for demo
    pipeline = SynDXPipeline(
        n_archetypes=100,   # Smaller number for demo
        nmf_components=20,
        epsilon=1.0,
        random_seed=42,
    )

    print("OK Pipeline initialized successfully")
    print(f"  - Archetypes: {pipeline.n_archetypes}")
    print(f"  - NMF components: {pipeline.nmf_components}")
    print(f"  - Privacy epsilon: {pipeline.epsilon}")

    print("\nRunning pipeline (extract -> generate -> validate)...")

    archetypes = pipeline.extract_archetypes()
    synthetic_data = pipeline.generate(n_patients=500)
    results = pipeline.validate(synthetic_data)

    print("\nOK Pipeline completed successfully!")
    print(f"  - Valid archetypes:       {len(archetypes)}")
    print(f"  - Synthetic patients:     {len(synthetic_data)}")
    print(f"  - Validation result keys: {list(results.keys())}")

    print("\nOK Demo completed successfully!")
    print("=" * 60)


if __name__ == '__main__':
    main()
