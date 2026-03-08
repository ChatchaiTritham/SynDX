"""Console entry points for the SynDX package."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict


def generate() -> int:
    """Generate a tiny sample payload for CLI smoke usage."""
    sample = {"status": "generated", "framework": "SynDX"}
    print(json.dumps(sample))
    return 0


def validate() -> int:
    """Emit a small validation result for CLI smoke usage."""
    report: Dict[str, Any] = {"status": "validated", "framework": "SynDX"}
    print(json.dumps(report))
    return 0
