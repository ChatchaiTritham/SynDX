"""Data loading utilities for SynDX."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd


class DataLoader:
    """Load tabular data from CSV or parquet paths."""

    def load(self, path: str | Path, **kwargs: Any) -> pd.DataFrame:
        data_path = Path(path)
        if data_path.suffix.lower() == ".parquet":
            return pd.read_parquet(data_path, **kwargs)
        return pd.read_csv(data_path, **kwargs)
