"""SNOMED mapping placeholders for SynDX."""

from __future__ import annotations

from typing import Dict


class SNOMEDMapper:
    """Map plain-text labels to lightweight SNOMED-like codes."""

    DEFAULT_CODES = {
        "vertigo": "C0042571",
        "stroke": "C0038454",
        "bppv": "C0155502",
    }

    def map_term(self, term: str) -> Dict[str, str]:
        normalized = term.strip().lower()
        return {"term": term, "code": self.DEFAULT_CODES.get(normalized, "UNKNOWN")}
