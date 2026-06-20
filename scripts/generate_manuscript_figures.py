"""Generate curated manuscript figures for SynDX.

This script creates the small article-ready figure set used for top-tier
readiness claims. It keeps broad demo outputs in ``outputs/`` and writes the
curated, reproducible submission set to ``figures/manuscript/`` with a figure
manifest.

Visualization style and the save/load helpers are imported, byte-identical,
from the vendored ``pubviz.py`` (mirrors _management/FIGURE_STYLE.md). Data is
always loaded from ``outputs/`` / ``results/`` at run time -- never hardcoded.

Encoding choices follow Cleveland--McGill: position/length on a common scale
(dot/bar small multiples, horizontal bars) is preferred over angle (radar) or
area (pie/donut), which are weak perceptual channels.
"""

from __future__ import annotations

import argparse
import csv
import json
from datetime import datetime
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt

from pubviz import PALETTE, apply_pub_style, load_results, results_dir, save_fig  # noqa: F401


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_DIR = ROOT / "figures" / "manuscript"
DEFAULT_MANIFEST = ROOT / "FIGURE_MANIFEST.csv"
DPI = 300

# Semantic aliases drawn from the shared Okabe-Ito palette (no color-only
# encoding; markers/hatches carry the distinction where multiple series appear).
COLORS = {
    "blue": PALETTE[0],
    "orange": PALETTE[1],
    "green": PALETTE[2],
    "pink": PALETTE[3],
    "amber": PALETTE[4],
    "skyblue": PALETTE[5],
    "black": PALETTE[6],
}

CLINICAL_FEATURE_LABELS = {
    "vascular_risk_score": "Vascular risk score",
    "vertigo_severity": "Vertigo severity",
    "stroke_history": "Prior stroke/TIA",
    "diabetes": "Diabetes",
    "symptom_duration_days": "Symptom duration",
    "neurological_signs": "Neurological signs",
    "tinnitus": "Tinnitus",
    "nystagmus_intensity": "Nystagmus intensity",
    "imbalance_score": "Imbalance score",
    "headache": "Headache",
    "age": "Age",
    "hearing_loss": "Hearing loss",
    "hypertension": "Hypertension",
    "cardiac_disease": "Cardiac disease",
    "positional_trigger": "Positional trigger",
}


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def save_figure(fig: plt.Figure, output_dir: Path, stem: str) -> tuple[Path, Path]:
    """Write matched vector PDF + 300-dpi PNG via the canonical helper."""
    save_fig(fig, stem, out_dir=str(output_dir))
    plt.close(fig)
    return output_dir / f"{stem}.png", output_dir / f"{stem}.pdf"


def format_feature_name(feature: str) -> str:
    return CLINICAL_FEATURE_LABELS.get(feature, feature.replace("_", " ").title())


def figure1_shap_importance(output_dir: Path) -> dict[str, str]:
    report_path = ROOT / "outputs" / "xai_explanations" / "reports" / "shap_report.json"
    report = load_json(report_path)
    features = report["feature_importance"]["BPPV"][:12]

    labels = [format_feature_name(row["feature"]) for row in features][::-1]
    values = [float(row["importance"]) for row in features][::-1]

    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    ax.barh(labels, values, color=COLORS["blue"], edgecolor=COLORS["black"], linewidth=0.5)
    ax.set_xlabel("Mean absolute SHAP value (dimensionless)")
    ax.set_ylabel("Clinical feature")
    ax.set_title("Clinically labelled SHAP feature importance")
    ax.grid(axis="x")
    ax.grid(axis="y", visible=False)
    ax.set_xlim(0, max(values) * 1.20)

    # Value labels on every bar (position on a common scale + explicit number).
    for y_pos, value in enumerate(values):
        ax.text(value + max(values) * 0.02, y_pos, f"{value:.3f}", va="center", fontsize=8)

    png_path, pdf_path = save_figure(fig, output_dir, "fig1_shap_importance_clinical")
    return {
        "figure_id": "SynDX-F1",
        "role": "manuscript",
        "png": str(png_path.relative_to(ROOT)),
        "pdf": str(pdf_path.relative_to(ROOT)),
        "source_script": "scripts/generate_manuscript_figures.py",
        "source_data": str(report_path.relative_to(ROOT)),
        "caption": (
            "Clinically labelled SHAP feature-importance panel for SynDX "
            "dizziness-case modeling; bars annotated with mean absolute SHAP "
            "values. Per-sample SHAP arrays are not exported, so no bootstrap "
            "confidence interval is shown (see human-review note)."
        ),
        "article_section": "Explainability validation",
    }


def figure2_validation_metrics(output_dir: Path) -> dict[str, str]:
    shap_path = ROOT / "outputs" / "validation_demo" / "shap" / "validation_metrics.json"
    counterfactual_path = (
        ROOT / "outputs" / "validation_demo" / "counterfactual" / "validation_metrics.json"
    )
    nmf_path = ROOT / "outputs" / "validation_demo" / "nmf" / "validation_metrics.json"

    shap = load_json(shap_path)
    counterfactual = load_json(counterfactual_path)
    nmf = load_json(nmf_path)

    metrics = [
        ("SHAP bootstrap\nrank correlation", shap["bootstrap_consistency"]["mean_rank_correlation"]),
        ("Counterfactual\nsuccess rate", counterfactual["success_rate"]),
        ("Clinically plausible\ncounterfactuals", counterfactual["clinical_plausibility"]["percent_plausible"] / 100.0),
        ("NMF variance\nretained", nmf["pca_comparison"]["nmf_variance"]),
    ]
    labels = [item[0] for item in metrics]
    values = [float(item[1]) for item in metrics]
    colors = PALETTE[: len(values)]
    # Distinct hatches so bars remain separable in grayscale / for color-blind readers.
    hatches = ["", "//", "..", "xx"]

    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    bars = ax.bar(labels, values, color=colors, edgecolor=COLORS["black"], linewidth=0.6)
    for bar, hatch in zip(bars, hatches):
        bar.set_hatch(hatch)
    ax.set_ylim(0, 1.0)
    ax.set_ylabel("Score (0-1, dimensionless)")
    ax.set_xlabel("Validation metric")
    ax.set_title("Focused validation metrics")
    ax.grid(axis="y")
    ax.grid(axis="x", visible=False)

    for bar, value in zip(bars, values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            value + 0.025,
            f"{value:.2f}",
            ha="center",
            va="bottom",
            fontsize=9,
        )

    png_path, pdf_path = save_figure(fig, output_dir, "fig2_focused_validation_metrics")
    return {
        "figure_id": "SynDX-F2",
        "role": "manuscript",
        "png": str(png_path.relative_to(ROOT)),
        "pdf": str(pdf_path.relative_to(ROOT)),
        "source_script": "scripts/generate_manuscript_figures.py",
        "source_data": "; ".join(
            [
                str(shap_path.relative_to(ROOT)),
                str(counterfactual_path.relative_to(ROOT)),
                str(nmf_path.relative_to(ROOT)),
            ]
        ),
        "caption": "Focused SynDX validation metrics split from dense dashboard-style outputs.",
        "article_section": "Validation results",
    }


def figure3_counterfactual_quality(output_dir: Path) -> dict[str, str]:
    """Counterfactual-quality profile as a dot/bar small-multiple panel.

    Replaces the former radar/spider chart. Each of the four quality criteria is
    a separate row plotted on a shared 0-1 position scale (Cleveland--McGill:
    position on a common scale reads far more accurately than angle on a radar).
    """
    metrics_path = ROOT / "outputs" / "validation_demo" / "counterfactual" / "validation_metrics.json"
    metrics = load_json(metrics_path)

    # Same normalizations as before, but rendered on a linear common scale.
    sparsity = 1.0 - min(metrics["sparsity"]["mean"] / 10.0, 1.0)
    plausibility = metrics["clinical_plausibility"]["mean"] / 5.0
    diversity = min(metrics["diversity"]["mean_pairwise_distance"] / 10.0, 1.0)
    proximity = 1.0 - min(metrics["proximity"]["mean_l2"] / 12.0, 1.0)

    rows = [
        ("Sparsity", sparsity),
        ("Clinical plausibility", plausibility),
        ("Diversity", diversity),
        ("Proximity (L2, inverted)", proximity),
    ]
    labels = [r[0] for r in rows][::-1]
    values = [r[1] for r in rows][::-1]
    y_pos = list(range(len(labels)))

    fig, ax = plt.subplots(figsize=(7.0, 3.6))
    # Thin reference bar + emphasised dot = Cleveland dot plot on a common scale.
    ax.hlines(y=y_pos, xmin=0, xmax=values, color="#cfd6de", linewidth=2.4, zorder=1)
    ax.scatter(values, y_pos, s=70, color=COLORS["blue"], edgecolor=COLORS["black"],
               linewidth=0.6, zorder=3)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels)
    ax.set_xlim(0, 1.0)
    ax.set_xticks([0.0, 0.25, 0.5, 0.75, 1.0])
    ax.set_xlabel("Normalised quality score (0-1, higher is better)")
    ax.set_title("Counterfactual quality profile")
    ax.grid(axis="x")
    ax.grid(axis="y", visible=False)

    for yp, value in zip(y_pos, values):
        ax.text(value + 0.02, yp, f"{value:.2f}", va="center", fontsize=9)

    png_path, pdf_path = save_figure(fig, output_dir, "fig3_counterfactual_quality_profile")
    return {
        "figure_id": "SynDX-F3",
        "role": "manuscript",
        "png": str(png_path.relative_to(ROOT)),
        "pdf": str(pdf_path.relative_to(ROOT)),
        "source_script": "scripts/generate_manuscript_figures.py",
        "source_data": str(metrics_path.relative_to(ROOT)),
        "caption": (
            "Counterfactual quality profile (dot plot on a common 0-1 scale) "
            "summarizing sparsity, clinical plausibility, diversity, and "
            "proximity; replaces a radar chart for accurate magnitude reading."
        ),
        "article_section": "Counterfactual validation",
    }


def write_manifest(rows: list[dict[str, str]], manifest_path: Path) -> None:
    fieldnames = [
        "figure_id",
        "role",
        "png",
        "pdf",
        "source_script",
        "source_data",
        "caption",
        "article_section",
        "generated_at",
        "dpi",
    ]
    generated_at = datetime.now().isoformat(timespec="seconds")
    with manifest_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({**row, "generated_at": generated_at, "dpi": str(DPI)})


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate curated SynDX manuscript figures")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    args = parser.parse_args()

    apply_pub_style()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    rows = [
        figure1_shap_importance(args.output_dir),
        figure2_validation_metrics(args.output_dir),
        figure3_counterfactual_quality(args.output_dir),
    ]
    write_manifest(rows, args.manifest)

    print(f"Generated {len(rows)} curated manuscript figures in {args.output_dir}")
    print(f"Wrote manifest: {args.manifest}")


if __name__ == "__main__":
    main()
