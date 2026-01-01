"""
Data Loader Module

Provides I/O operations for SynDX datasets including computational archetypes,
synthetic patient records, and validation datasets.

Supported Formats:
    - CSV: Comma-separated values for interoperability
    - JSON: JavaScript Object Notation for structured data
    - Parquet: Apache Parquet for efficient columnar storage

This module implements standardized data loading patterns to ensure consistency
across the SynDX pipeline phases.
"""

import pandas as pd
import numpy as np
import json
from pathlib import Path
from typing import Union, Dict, List
import logging

logger = logging.getLogger(__name__)


class DataLoader:
    """Load and manage SynDX datasets"""

    @staticmethod
    def load_archetypes(filepath: Union[str, Path]) -> pd.DataFrame:
        """Load archetype dataset"""
        filepath = Path(filepath)

        if filepath.suffix == '.csv':
            df = pd.read_csv(filepath)
        elif filepath.suffix == '.json':
            df = pd.read_json(filepath, orient='records')
        elif filepath.suffix == '.parquet':
            df = pd.read_parquet(filepath)
        else:
            raise ValueError(f"Unsupported format: {filepath.suffix}")

        logger.info(f"Loaded {len(df)} archetypes from {filepath}")
        return df

    @staticmethod
    def load_synthetic_patients(filepath: Union[str, Path]) -> pd.DataFrame:
        """Load synthetic patient dataset"""
        return DataLoader.load_archetypes(filepath)

    @staticmethod
    def save_dataset(data: pd.DataFrame, filepath: Union[str, Path]):
        """Save dataset to file"""
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)

        if filepath.suffix == '.csv':
            data.to_csv(filepath, index=False)
        elif filepath.suffix == '.json':
            data.to_json(filepath, orient='records', indent=2)
        elif filepath.suffix == '.parquet':
            data.to_parquet(filepath, index=False)
        else:
            raise ValueError(f"Unsupported format: {filepath.suffix}")

        logger.info(f"Saved {len(data)} records to {filepath}")
