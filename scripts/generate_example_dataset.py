"""
Generate Example Synthetic Dataset

Creates a 1,000-patient synthetic dataset for demonstration purposes.

Usage:
    python scripts/generate_example_dataset.py
"""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd
from datetime import datetime
import logging

from syndx import SynDXPipeline
from syndx.phase1_knowledge import ArchetypeGenerator
from syndx.utils import DataLoader

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def main():
    """Generate example dataset"""
    logger.info("="*70)
    logger.info("SYNDX EXAMPLE DATASET GENERATOR")
    logger.info("="*70)
    logger.info("")
    logger.info("⚠️  WARNING: Preliminary work without clinical validation")
    logger.info("   This dataset is for demonstration purposes only")
    logger.info("")

    # Set random seed
    np.random.seed(42)

    # Initialize pipeline
    logger.info("Initializing SynDX pipeline...")
    pipeline = SynDXPipeline(
        n_archetypes=500,      # Moderate size for demo
        nmf_components=20,
        vae_latent_dim=50,
        epsilon=1.0,
        random_seed=42
    )

    # Phase 1: Extract archetypes
    logger.info("\nPhase 1: Extracting clinical archetypes...")
    archetypes = pipeline.extract_archetypes(
        guidelines=['titrate', 'barany_icvd_2025']
    )
    logger.info(f"Generated {len(archetypes)} valid archetypes")

    # Save archetypes
    archetype_dir = Path('data/archetypes')
    archetype_dir.mkdir(parents=True, exist_ok=True)

    archetype_df = pipeline.archetype_generator.to_dataframe()
    archetype_path = archetype_dir / 'example_archetypes.csv'
    archetype_df.to_csv(archetype_path, index=False)
    logger.info(f"Saved archetypes to: {archetype_path}")

    # Get archetype statistics
    stats = pipeline.archetype_generator.get_statistics()
    logger.info("\nArchetype Statistics:")
    logger.info(f"  Total: {stats['total_archetypes']}")
    logger.info(f"  Age (mean±std): {stats['age_stats']['mean']:.1f}±{stats['age_stats']['std']:.1f} years")
    logger.info(f"  Age range: {stats['age_stats']['min']}-{stats['age_stats']['max']} years")

    logger.info("\n  Top 5 diagnoses:")
    for dx, count in sorted(stats['diagnosis_distribution'].items(),
                           key=lambda x: -x[1])[:5]:
        pct = count / stats['total_archetypes'] * 100
        logger.info(f"    {dx}: {count} ({pct:.1f}%)")

    # Phase 2: Generate synthetic patients
    logger.info("\nPhase 2: Generating synthetic patients...")
    n_patients = 1000
    synthetic_patients = pipeline.generate(
        n_patients=n_patients,
        convergence_threshold=0.05
    )
    logger.info(f"Generated {len(synthetic_patients)} synthetic patients")

    # Save synthetic patients
    output_dir = Path('outputs/synthetic_patients')
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save as CSV
    csv_path = output_dir / 'example_synthetic_patients.csv'
    synthetic_patients.to_csv(csv_path, index=False)
    logger.info(f"Saved CSV to: {csv_path}")

    # Save as JSON
    json_path = output_dir / 'example_synthetic_patients.json'
    synthetic_patients.to_json(json_path, orient='records', indent=2)
    logger.info(f"Saved JSON to: {json_path}")

    # Phase 3: Validate
    logger.info("\nPhase 3: Validating synthetic data...")
    validation_results = pipeline.validate(
        synthetic_patients,
        metrics=['statistical']
    )

    logger.info("\nValidation Results:")
    for metric, value in validation_results.get('statistical', {}).items():
        if isinstance(value, float):
            logger.info(f"  {metric}: {value:.4f}")

    # Generate metadata
    metadata = {
        "dataset_name": "SynDX Example Synthetic Dataset",
        "version": "0.1.0",
        "generation_date": datetime.now().isoformat(),
        "n_archetypes": len(archetypes),
        "n_synthetic_patients": len(synthetic_patients),
        "parameters": {
            "nmf_components": pipeline.nmf_components,
            "vae_latent_dim": pipeline.vae_latent_dim,
            "epsilon": pipeline.epsilon,
            "random_seed": pipeline.random_seed,
        },
        "validation": validation_results,
        "warning": "Preliminary work without clinical validation. NOT for clinical use.",
        "citation": {
            "title": "SynDX: Explainable AI-Driven Synthetic Data Generation",
            "authors": ["Tritham, C.", "Namahoot, C.S."],
            "year": 2025,
            "journal": "IEEE Access",
            "note": "Preliminary work without clinical validation"
        }
    }

    # Save metadata
    import json
    metadata_path = output_dir / 'example_dataset_metadata.json'
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    logger.info(f"Saved metadata to: {metadata_path}")

    # Summary
    logger.info("\n" + "="*70)
    logger.info("DATASET GENERATION COMPLETE")
    logger.info("="*70)
    logger.info(f"\nGenerated files:")
    logger.info(f"  1. Archetypes (CSV):    {archetype_path}")
    logger.info(f"  2. Synthetic data (CSV): {csv_path}")
    logger.info(f"  3. Synthetic data (JSON): {json_path}")
    logger.info(f"  4. Metadata:            {metadata_path}")
    logger.info(f"\nTotal synthetic patients: {n_patients}")
    logger.info("\n⚠️  IMPORTANT REMINDER:")
    logger.info("   This dataset is for research and demonstration only.")
    logger.info("   It has NOT been validated on real patients.")
    logger.info("   Do NOT use for clinical decision-making.")


if __name__ == "__main__":
    main()
