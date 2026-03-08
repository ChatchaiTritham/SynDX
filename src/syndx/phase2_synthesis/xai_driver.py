"""Minimal XAI driver for package compatibility."""

from __future__ import annotations

from typing import Any, Dict


class XAIDriver:
    """Return simple metadata about explainability execution."""

    def run(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        return {"status": "completed", "keys": sorted(payload.keys())}
