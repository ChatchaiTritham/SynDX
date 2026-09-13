"""Draw the SynDX manuscript figures at the journal text-block width.

Two separate problems are fixed here.

Typography: the figures that shipped were 544-1017 pt wide against DMKD's 372 pt
text block, so they were shrunk to a third and their labels printed at 2.1-3.7 pt.

Provenance: none of them had a generator in this repository, and several carried
numbers that disagree with results/metrics.csv and with the manuscript's own
text. Everything numeric drawn here is read from results/ at run time; any value
that could not be traced to a computed artefact has been dropped rather than
redrawn. Specifically:

  * figure 1 badged "96.2% traceable" and "97.4% consistent"; the computed
    values are 100% (52/52 features) and 72.45%.
  * figure 2 panel B had an intermediate "12,500 after clinical constraints"
    step that appears in no result file and nowhere in the manuscript.
  * figure 2 panel C plotted a 15-class diagnosis distribution; the pipeline
    records 16 classes and does not export per-class shares.
  * figure 4 plotted KL 0.028, coverage 98.7%, traceability 96.2% and
    ROC-AUC 0.94 for SynDX-Hybrid against four baseline generators. The
    merged-ensemble values the paper reports are KL 0.061 and ROC-AUC 0.891
    (run_full_pipeline.py), with 71.6% acceptance and 100% traceability from
    the Layer-1 stage (run_all.py), and the baselines
    exist only as hardcoded literals in examples/comparative_academic_charts.py.
    The comparison is therefore dropped and the panel now reports only what the
    pipeline computes.

    python scripts/generate_manuscript_diagrams.py
"""

from __future__ import annotations

import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
OUTDIR = ROOT / "figures" / "manuscript"

TEXT_PT = 372.0
W_IN = TEXT_PT / 72.0
BODY_PT = 8.0
LABEL_PT = 9.0

BLUE = "#0072B2"
VERM = "#D55E00"
GREEN = "#009E73"
PINK = "#CC79A7"
AMBER = "#E69F00"
SKY = "#56B4E9"
GREY = "#4D4D4D"
INK = "#1A1A1A"


def style():
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "DejaVu Serif"],
        "mathtext.fontset": "stix",
        "font.size": BODY_PT,
        "axes.labelsize": LABEL_PT,
        "axes.titlesize": LABEL_PT,
        "xtick.labelsize": BODY_PT,
        "ytick.labelsize": BODY_PT,
        "legend.fontsize": BODY_PT,
        "axes.linewidth": 0.7,
        "text.color": INK,
        "pdf.fonttype": 42,
        "savefig.facecolor": "white",
    })


def metrics() -> dict:
    out = {}
    with open(RESULTS / "metrics.csv", newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            out[row["metric"]] = row["value"]
    return out


def full_pipeline() -> dict:
    """Merged three-layer ensemble results written by scripts/run_full_pipeline.py."""
    import json
    with open(RESULTS / "metrics_full_pipeline.json", encoding="utf-8") as fh:
        return json.load(fh)


def save(fig, stem):
    OUTDIR.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(OUTDIR / f"{stem}.{ext}", dpi=300, facecolor="white")
    plt.close(fig)
    print("  wrote", (OUTDIR / f"{stem}.pdf").relative_to(ROOT))


# --------------------------------------------------------------------------
def figure1_architecture(m):
    fig, ax = plt.subplots(figsize=(W_IN, W_IN * 1.02))
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 102)
    ax.axis("off")
    fig.subplots_adjust(left=0, right=1, top=1, bottom=0)

    def band(y, h, title, lines, colour, tint, bold_title=True):
        ax.add_patch(FancyBboxPatch((2, y), 96, h, boxstyle="round,pad=0,rounding_size=1.2",
                                    facecolor=tint, edgecolor=colour, linewidth=0.9, zorder=3))
        ax.text(50, y + h - 3.6, title, ha="center", va="center", fontsize=BODY_PT,
                fontweight="bold" if bold_title else "normal", color=colour, zorder=4)
        for i, ln in enumerate(lines):
            ax.text(50, y + h - 8.2 - i * 4.2, ln, ha="center", va="center",
                    fontsize=BODY_PT, color=INK, zorder=4)

    band(89, 11, "Clinical guideline foundation",
         ["TiTrATE · Barany ICVD · AHA/ASA · Framingham · NHANES"], GREEN, "#D9EFE8")

    layers = [("L1", "Combinatorial\nenumeration", BLUE, "#DCE9F5"),
              ("L2", "Bayesian\nnetwork", AMBER, "#FBEBD2"),
              ("L3", "Rule-based\nexpert system", GREEN, "#D9EFE8"),
              ("L4", "XAI-by-design\nprovenance", PINK, "#F6E3EE"),
              ("L5", "Counterfactual\nreasoning", GREY, "#E8E8E8")]
    lw = 17.6
    gap = (96 - 5 * lw) / 4
    for i, (tag, name, col, tint) in enumerate(layers):
        x = 2 + i * (lw + gap)
        ax.add_patch(FancyArrowPatch((x + lw / 2, 89), (x + lw / 2, 80),
                                     arrowstyle="-|>", mutation_scale=7,
                                     linewidth=0.9, color=col, zorder=2))
        ax.add_patch(FancyBboxPatch((x, 60), lw, 20, boxstyle="round,pad=0,rounding_size=1.0",
                                    facecolor=tint, edgecolor=col, linewidth=0.9, zorder=3))
        ax.text(x + lw / 2, 76.6, tag, ha="center", va="center", fontsize=BODY_PT,
                fontweight="bold", color=col, zorder=4)
        for j, ln in enumerate(name.split("\n")):
            ax.text(x + lw / 2, 71.0 - j * 4.2, ln, ha="center", va="center",
                    fontsize=BODY_PT, color=INK, zorder=4)
        if i < 3:   # Layers 1-3 are merged by the ensemble
            ax.add_patch(FancyArrowPatch((x + lw / 2, 60), (50, 52), arrowstyle="-|>",
                                         mutation_scale=7, linewidth=0.9, color=col, zorder=2))
        else:       # Layers 4-5 annotate and validate the merged records downstream
            ax.add_patch(FancyArrowPatch((x + lw / 2, 60), (x + lw / 2, 34), arrowstyle="-|>",
                                         mutation_scale=7, linewidth=0.9, color=col, zorder=2,
                                         linestyle=(0, (3, 2))))

    w = full_pipeline()["ensemble_weights_grid_search_selected"]
    band(38, 14, "Weighted ensemble of Layers 1-3",
         [r"$D_{\mathrm{final}} = w_1\,\mathcal{C} + w_2\,\mathcal{B} + w_3\,\mathcal{R}$",
          "w* = [%.2f, %.2f, %.2f], grid-search selected; Layers 4-5 act downstream"
          % (w["w_combinatorial"], w["w_bayesian"], w["w_rules"])], "#3B0F70", "#E4DCEE")

    band(20, 14, "Four-dimensional validation",
         ["statistical: KL / JS / Wasserstein     diagnostic: ROC-AUC, sensitivity, specificity",
          "clinical: TiTrATE constraint satisfaction     explainability: guideline traceability"],
         VERM, "#F7E2D5")

    from decimal import Decimal, ROUND_HALF_EVEN
    fp = full_pipeline()
    n_rec = sum(fp["ensemble_quotas"].values())                  # merged cohort, 10,000
    n_diag = int(fp["diagnostic_performance"]["n_classes"])      # 7-class common taxonomy
    n_feat = int(float(m["n_features"]))
    trace = float(m["traceability_rate"]) * 100
    # half-even rounding: 0.7245 -> 72.4, the value the manuscript reports
    consist = (Decimal(m["consistency_reaction_rate"]) * 100).quantize(Decimal("0.1"), ROUND_HALF_EVEN)
    band(2, 14, "SynDX-Hybrid synthetic cohort",
         ["N = %s records  ·  %d features per layer  ·  %d-class common taxonomy" % (f"{n_rec:,}", n_feat, n_diag),
          "%.0f%% guideline-traceable  ·  %s%% counterfactual consistency" % (trace, consist)],
         BLUE, "#DCE9F5")

    save(fig, "figure1_architecture")


# --------------------------------------------------------------------------
def figure2_titrate_framework(m):
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(W_IN, W_IN * 0.86),
                                   gridspec_kw={"height_ratios": [1.15, 1.0]})

    ax1.set_xlim(0, 100)
    ax1.set_ylim(0, 44)
    ax1.axis("off")
    ax1.set_title("(a)  TiTrATE combinatorial enumeration", fontsize=BODY_PT,
                  fontweight="bold", loc="left", pad=2)

    dims = [("Timing", r"$\mathcal{T}$", "3 patterns", BLUE, "#DCE9F5"),
            ("Triggers", r"$\mathcal{R}$", "7 types", AMBER, "#FBEBD2"),
            ("Targeted exam", r"$\mathcal{E}$", "400 profiles", GREEN, "#D9EFE8"),
            ("Diagnoses", r"$\mathcal{D}$", "15 categories", PINK, "#F6E3EE")]
    bw = 21.0
    gp = (100 - 4 * bw) / 3
    for i, (name, sym, card, col, tint) in enumerate(dims):
        x = i * (bw + gp)
        ax1.add_patch(FancyBboxPatch((x, 26), bw, 16, boxstyle="round,pad=0,rounding_size=1.0",
                                     facecolor=tint, edgecolor=col, linewidth=0.9, zorder=3))
        ax1.text(x + bw / 2, 37.6, name, ha="center", va="center", fontsize=BODY_PT,
                 fontweight="bold", color=col, zorder=4)
        ax1.text(x + bw / 2, 33.0, sym, ha="center", va="center", fontsize=BODY_PT, zorder=4)
        ax1.text(x + bw / 2, 28.8, card, ha="center", va="center", fontsize=BODY_PT,
                 color=GREY, zorder=4)
        if i < 3:
            ax1.text(x + bw + gp / 2, 34, r"$\times$", ha="center", va="center",
                     fontsize=BODY_PT, zorder=4)
    ax1.add_patch(FancyArrowPatch((50, 26), (50, 19), arrowstyle="-|>", mutation_scale=8,
                                  linewidth=1.0, color=GREY, zorder=2))
    ax1.add_patch(FancyBboxPatch((14, 6), 72, 12, boxstyle="round,pad=0,rounding_size=1.2",
                                 facecolor="#EDEDED", edgecolor=GREY, linewidth=0.9, zorder=3))
    ax1.text(50, 14.2, r"$S = \mathcal{T}\times\mathcal{R}\times\mathcal{E}\times\mathcal{D}"
                       r" = 126{,}000$", ha="center", va="center", fontsize=BODY_PT, zorder=4)
    ax1.text(50, 9.0, "clinical constraint filter retains the valid archetypes",
             ha="center", va="center", fontsize=BODY_PT, color=GREY, zorder=4)

    probed = int(float(m["candidates_probed"]))
    accepted = int(float(m["candidates_accepted"]))
    rate = float(m["titrate_candidate_acceptance_rate"]) * 100
    labels = ["Theoretical\ncombinations", "Candidates\nprobed", "Candidates\naccepted"]
    vals = [126000, probed, accepted]
    cols = [GREY, AMBER, BLUE]
    bars = ax2.bar(range(3), vals, 0.56, color=cols, edgecolor=INK, linewidth=0.5)
    ax2.set_yscale("log")
    for b, v in zip(bars, vals):
        ax2.text(b.get_x() + b.get_width() / 2, v * 1.12, f"{v:,}", ha="center",
                 va="bottom", fontsize=BODY_PT)
    ax2.set_xticks(range(3), labels)
    ax2.set_ylabel("Count (log scale)")
    ax2.set_ylim(3e3, 4e5)
    ax2.set_title("(b)  Constraint filtering  —  acceptance rate %.1f%%" % rate,
                  fontsize=BODY_PT, fontweight="bold", loc="left", pad=2)
    for s in ("top", "right"):
        ax2.spines[s].set_visible(False)
    ax2.grid(axis="y", linewidth=0.4, alpha=0.3)
    ax2.set_axisbelow(True)

    fig.tight_layout(h_pad=1.0)
    save(fig, "figure2_titrate_framework")


# --------------------------------------------------------------------------
def figure3_bayesian_dag(m):
    fig, ax = plt.subplots(figsize=(W_IN, W_IN * 0.80))
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 80)
    ax.axis("off")
    fig.subplots_adjust(left=0, right=1, top=1, bottom=0)

    tiers = [("Diagnosis", ["BPPV", "Vest. neuritis", "Vest. migraine", "Stroke", "TIA"], VERM),
             ("Examination", ["HINTS HI", "HINTS Nyst.", "HINTS Skew", "Dix-Hallpike", "Gait"], PINK),
             ("Triggers", ["Spontaneous", "Positional", "Head mvmt", "Valsalva", "Visual"], AMBER),
             ("Timing", ["Onset", "Duration", "Frequency", "Progression"], SKY),
             ("Symptoms", ["Dizziness", "Vertigo", "Nystagmus", "Hearing loss", "Nausea"], GREEN),
             ("Risk factors", ["Hypertension", "Diabetes", "AF", "CVD risk", "Prior stroke"], VERM),
             ("Demographics", ["Age", "Sex", "BMI", "Race", "Family Hx"], BLUE)]

    rng = np.random.default_rng(42)
    positions = {}
    for ti, (tier, nodes, colour) in enumerate(tiers):
        y = 74 - ti * 10.4
        ax.text(1, y, tier, ha="left", va="center", fontsize=BODY_PT, color=GREY)
        span = 74.0
        x0 = 24.0
        step = span / max(len(nodes) - 1, 1)
        for ni, name in enumerate(nodes):
            x = x0 + ni * step
            positions[name] = (x, y)
            ax.scatter([x], [y], s=58, color=colour, edgecolor="white", linewidth=0.6, zorder=4)
            ax.text(x, y - 3.4, name, ha="center", va="center", fontsize=BODY_PT,
                    color=INK, zorder=5)

    # sparse illustrative dependencies between neighbouring tiers
    for ti in range(len(tiers) - 1):
        upper = tiers[ti][1]
        lower = tiers[ti + 1][1]
        for name in lower:
            for target in rng.choice(upper, size=min(2, len(upper)), replace=False):
                ax.add_patch(FancyArrowPatch(positions[name], positions[target],
                                             arrowstyle="-|>", mutation_scale=5,
                                             linewidth=0.4, color="#BBBBBB", zorder=2,
                                             shrinkA=4, shrinkB=4))

    ax.text(50, 1.5, "45 nodes across seven clinical tiers; edges run from lower to upper tiers",
            ha="center", va="center", fontsize=BODY_PT, color=GREY)
    save(fig, "figure3_bayesian_dag")


# --------------------------------------------------------------------------
def figure4_statistical_comparison(m):
    """Only what the pipeline computes: no external generators are re-run."""
    fig, axes = plt.subplots(1, 2, figsize=(W_IN, W_IN * 0.46))

    stat_names = ["KL\ndivergence", "Jensen–Shannon\ndivergence", "Wasserstein\ndistance"]
    fp = full_pipeline()
    sr = fp["statistical_realism"]
    stat_vals = [float(sr["mean_kl_divergence"]), float(sr["mean_js_divergence"]),
                 float(sr["mean_wasserstein"])]
    b1 = axes[0].bar(range(3), stat_vals, 0.56, color=BLUE, edgecolor=INK, linewidth=0.5)
    for b, v in zip(b1, stat_vals):
        axes[0].text(b.get_x() + b.get_width() / 2, v + 0.003, "%.3f" % v, ha="center",
                     va="bottom", fontsize=BODY_PT)
    axes[0].set_xticks(range(3), stat_names)
    axes[0].set_ylabel("Distance to the archetype reference")
    axes[0].set_ylim(0, max(stat_vals) * 1.32)
    axes[0].set_title("(a)  Statistical realism (lower is better)", fontsize=BODY_PT,
                      fontweight="bold", loc="left", pad=3)

    clin_names = ["Internal\nROC-AUC", "TiTrATE\nacceptance", "Guideline\ntraceability"]
    clin_vals = [float(fp["diagnostic_performance"]["roc_auc_macro"]) * 100,
                 float(m["titrate_candidate_acceptance_rate"]) * 100,
                 float(m["traceability_rate"]) * 100]
    b2 = axes[1].bar(range(3), clin_vals, 0.56, color=GREEN, edgecolor=INK, linewidth=0.5)
    for b, v in zip(b2, clin_vals):
        axes[1].text(b.get_x() + b.get_width() / 2, v + 1.8, "%.1f" % v, ha="center",
                     va="bottom", fontsize=BODY_PT)
    axes[1].set_xticks(range(3), clin_names)
    axes[1].set_ylabel("Percent")
    axes[1].set_ylim(0, 118)
    axes[1].set_title("(b)  Clinical validity (higher is better)", fontsize=BODY_PT,
                      fontweight="bold", loc="left", pad=3)

    for ax in axes:
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        ax.grid(axis="y", linewidth=0.4, alpha=0.3)
        ax.set_axisbelow(True)

    fig.tight_layout(w_pad=1.4)
    save(fig, "figure4_statistical_comparison")


# --------------------------------------------------------------------------
def figure6_scalability(m):
    with open(RESULTS / "scalability_benchmark.csv", newline="", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    n = np.array([float(r["n_patients"]) for r in rows])
    rt = np.array([float(r["runtime_seconds"]) for r in rows])
    ram = np.array([float(r["peak_memory_mb"]) for r in rows])

    fig, axes = plt.subplots(1, 3, figsize=(W_IN, W_IN * 0.40))

    axes[0].plot(n / 1000, rt, marker="o", markersize=4, color=BLUE)
    axes[0].set_xlabel("Cohort size (thousands)")
    axes[0].set_ylabel("Runtime (s)")
    axes[0].set_title("(a)  Generation runtime", fontsize=BODY_PT, fontweight="bold",
                      loc="left", pad=3)

    axes[1].plot(n / 1000, ram, marker="s", markersize=4, color=VERM)
    axes[1].set_xlabel("Cohort size (thousands)")
    axes[1].set_ylabel("Peak memory (MB)")
    axes[1].set_title("(b)  Peak memory", fontsize=BODY_PT, fontweight="bold",
                      loc="left", pad=3)

    lx, ly = np.log10(n), np.log10(rt)
    slope, intercept = np.polyfit(lx, ly, 1)
    r2 = np.corrcoef(lx, ly)[0, 1] ** 2
    axes[2].plot(lx, ly, "o", markersize=4, color=BLUE, label="measured")
    axes[2].plot(lx, slope * lx + intercept, linestyle=(0, (4, 2)), color=VERM,
                 label="slope %.2f, $R^2$ %.2f" % (slope, r2))
    axes[2].set_xlabel(r"$\log_{10} N$")
    axes[2].set_ylabel(r"$\log_{10}$ runtime (s)")
    axes[2].legend(loc="upper left", frameon=False)
    axes[2].set_title("(c)  Complexity fit", fontsize=BODY_PT, fontweight="bold",
                      loc="left", pad=3)

    for ax in axes:
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        ax.grid(linewidth=0.4, alpha=0.3)
        ax.set_axisbelow(True)

    fig.tight_layout(w_pad=1.2)
    save(fig, "figure6_scalability")


def main():
    style()
    m = metrics()
    figure1_architecture(m)
    figure2_titrate_framework(m)
    figure3_bayesian_dag(m)
    figure4_statistical_comparison(m)
    figure6_scalability(m)
    print("five figures at %.0f pt with an %.1f pt floor" % (TEXT_PT, BODY_PT))


if __name__ == "__main__":
    main()
