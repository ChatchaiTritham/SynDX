"""Generate curated manuscript figures for SynDX.

This script creates the small article-ready figure set used for top-tier
readiness claims. It keeps broad demo outputs in ``outputs/`` and writes the
curated, reproducible submission set to ``figures/manuscript/`` with a figure
manifest.
"""

from __future__ import annotations

import argparse
import csv
import json
from datetime import datetime
from pathlib import Path
from typing import Any

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_DIR = ROOT / "figures" / "manuscript"
DEFAULT_MANIFEST = ROOT / "FIGURE_MANIFEST.csv"
DPI = 300

# Canonical Top-Tier figure style (shared across all PhD repos).
# Color-blind-safe (Okabe-Ito) palette — use in this order.
PALETTE = ["#0072B2", "#D55E00", "#009E73", "#CC79A7", "#E69F00", "#56B4E9", "#000000"]

# Semantic aliases drawn from the shared palette (no color-only encoding;
# markers/linestyles carry the distinction where multiple series appear).
COLORS = {
    "blue": PALETTE[0],
    "orange": PALETTE[1],
    "green": PALETTE[2],
    "pink": PALETTE[3],
    "amber": PALETTE[4],
    "skyblue": PALETTE[5],
    "black": PALETTE[6],
    # backward-compatible semantic names retained, remapped to Okabe-Ito
    "safe": PALETTE[2],
    "teal": PALETTE[5],
    "gray": PALETTE[6],
}


def apply_pub_style() -> None:
    """Apply the canonical publication style (see _management/FIGURE_STYLE.md)."""
    mpl.rcParams.update(
        {
            "figure.dpi": 150,
            "savefig.dpi": DPI,
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.02,
            "font.family": "serif",
            "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
            "mathtext.fontset": "stix",
            "font.size": 10,
            "axes.titlesize": 11,
            "axes.labelsize": 10,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "legend.fontsize": 9,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.linewidth": 0.8,
            "axes.grid": True,
            "grid.alpha": 0.3,
            "grid.linewidth": 0.6,
            "lines.linewidth": 1.6,
            "lines.markersize": 5,
            "legend.frameon": False,
            "figure.constrained_layout.use": True,
            "axes.prop_cycle": mpl.cycler(color=PALETTE),
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )

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
    png_path = output_dir / f"{stem}.png"
    pdf_path = output_dir / f"{stem}.pdf"
    fig.savefig(png_path, dpi=DPI, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)
    return png_path, pdf_path


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
    ax.set_xlim(0, max(values) * 1.18)

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
        "caption": "Clinically labelled SHAP feature-importance panel for SynDX dizziness-case modeling.",
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
    ax.set_ylabel("Score (0–1, dimensionless)")
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
    metrics_path = ROOT / "outputs" / "validation_demo" / "counterfactual" / "validation_metrics.json"
    metrics = load_json(metrics_path)

    labels = ["Sparsity", "Clinical\nplausibility", "Diversity", "Proximity\n(L2, inverted)"]
    sparsity = 1.0 - min(metrics["sparsity"]["mean"] / 10.0, 1.0)
    plausibility = metrics["clinical_plausibility"]["mean"] / 5.0
    diversity = min(metrics["diversity"]["mean_pairwise_distance"] / 10.0, 1.0)
    proximity = 1.0 - min(metrics["proximity"]["mean_l2"] / 12.0, 1.0)
    values = [sparsity, plausibility, diversity, proximity]

    angles = np.linspace(0, 2 * np.pi, len(labels), endpoint=False).tolist()
    values_closed = values + values[:1]
    angles_closed = angles + angles[:1]

    fig, ax = plt.subplots(
        figsize=(5.4, 5.0), subplot_kw={"polar": True}, layout="constrained"
    )
    ax.plot(
        angles_closed,
        values_closed,
        color=COLORS["blue"],
        linewidth=1.8,
        linestyle="-",
        marker="o",
        markersize=5,
    )
    ax.fill(angles_closed, values_closed, color=COLORS["blue"], alpha=0.18)
    ax.set_xticks(angles)
    ax.set_xticklabels(labels)
    ax.set_ylim(0, 1)
    ax.set_yticks([0.25, 0.5, 0.75, 1.0])
    ax.set_yticklabels(["0.25", "0.50", "0.75", "1.00"], fontsize=8)
    ax.grid(alpha=0.3, linewidth=0.6)
    ax.set_title("Counterfactual quality profile (normalised 0–1)", pad=18)

    png_path, pdf_path = save_figure(fig, output_dir, "fig3_counterfactual_quality_profile")
    return {
        "figure_id": "SynDX-F3",
        "role": "manuscript",
        "png": str(png_path.relative_to(ROOT)),
        "pdf": str(pdf_path.relative_to(ROOT)),
        "source_script": "scripts/generate_manuscript_figures.py",
        "source_data": str(metrics_path.relative_to(ROOT)),
        "caption": "Counterfactual quality profile summarizing sparsity, plausibility, diversity, and proximity.",
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
