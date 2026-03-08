"""FHIR-like exporter utilities for SynDX."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict


class FHIRExporter:
    """Export generic dictionaries into a lightweight FHIR bundle."""

    def export(self, payload: Dict[str, Any], output_path: str | Path) -> Path:
        bundle = {
            "resourceType": "Bundle",
            "type": "collection",
            "entry": [{"resource": {"resourceType": "Observation", "valueString": json.dumps(payload)}}],
        }
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(bundle, indent=2), encoding="utf-8")
        return path
