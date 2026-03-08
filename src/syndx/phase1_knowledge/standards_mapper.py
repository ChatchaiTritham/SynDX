"""Lightweight standards mapping helpers for SynDX."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List


@dataclass
class StandardsMapping:
    """Container for a mapped clinical record."""

    source: str
    target_standard: str
    mapped_record: Dict[str, Any]
    notes: List[str] = field(default_factory=list)


class StandardsMapper:
    """Provide a small stable API for standards-oriented mapping."""

    def __init__(self, target_standard: str = "FHIR"):
        self.target_standard = target_standard

    def map_record(self, record: Dict[str, Any], source: str = "generic") -> StandardsMapping:
        """Wrap a record with minimal standards metadata."""
        return StandardsMapping(
            source=source,
            target_standard=self.target_standard,
            mapped_record=dict(record),
            notes=[f"Mapped to {self.target_standard}"],
        )

    def map_records(self, records: Iterable[Dict[str, Any]], source: str = "generic") -> List[StandardsMapping]:
        """Map multiple records with the same target standard."""
        return [self.map_record(record, source=source) for record in records]
