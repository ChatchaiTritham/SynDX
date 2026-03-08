"""Smoke tests for SynDX package imports and core entry points."""

import json

import numpy as np

from syndx import SynDXPipeline
from syndx.cli import generate, validate
from syndx.phase1_knowledge import StandardsMapper, TiTrATEFormalizer
from syndx.phase2_synthesis import NMFExtractor
from syndx.phase3_validation import StatisticalMetrics
from syndx.utils import DataLoader, FHIRExporter, SNOMEDMapper


def test_package_imports_expose_core_api() -> None:
    assert SynDXPipeline is not None
    assert StandardsMapper is not None
    assert TiTrATEFormalizer is not None
    assert NMFExtractor is not None
    assert StatisticalMetrics is not None
    assert DataLoader is not None
    assert FHIRExporter is not None
    assert SNOMEDMapper is not None


def test_cli_entry_points_return_success(capsys) -> None:
    assert generate() == 0
    assert validate() == 0
    output = capsys.readouterr().out.strip().splitlines()
    assert json.loads(output[0])["status"] == "generated"
    assert json.loads(output[1])["status"] == "validated"


def test_pipeline_instantiates_and_runs_minimal_validation() -> None:
    pipeline = SynDXPipeline(n_archetypes=10, nmf_components=2, vae_latent_dim=4)
    pipeline.archetypes = [object()] * 4
    pipeline.archetype_matrix = np.abs(np.random.default_rng(42).normal(size=(4, 3)))

    synthetic = pipeline.generate(n_patients=5)
    results = pipeline.validate(synthetic)

    assert len(synthetic) == 5
    assert "statistical" in results
