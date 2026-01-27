"""
Comprehensive Visualization Module
Creates publication-quality figures for manuscript
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import List, Dict
import pandas as pd
from collections import Counter

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['font.size'] = 10
plt.rcParams['axes.labelsize'] = 11
plt.rcParams['axes.titlesize'] = 12
plt.rcParams['legend.fontsize'] = 9


def create_all_visualizations(
    explorer,
    archetypes: List,
    param_space,
    output_dir: Path
):
    """
    Create all visualizations for the paper

    Figures created:
    1. Parameter space overview
    2. Diagnosis distribution (observed vs expected)
    3. NMF factor loadings heatmap
    4. SHAP feature importance
    5. Age distribution by diagnosis
    6. Multi-phase sampling statistics
    7. Critical scenario coverage
    8. Acceptance rate by phase
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Creating visualizations...")

    # Figure 1: Parameter Space Overview
    plot_parameter_space_overview(
        param_space,
        explorer,
        output_dir / "fig1_parameter_space.png"
    )

    # Figure 2: Diagnosis Distribution
    plot_diagnosis_distribution(
        archetypes,
        param_space.epidemiology,
        output_dir / "fig2_diagnosis_distribution.png"
    )

    # Figure 3: NMF Factor Heatmap
    if explorer.nmf_model:
        plot_nmf_factors(
            explorer.nmf_model,
            output_dir / "fig3_nmf_factors.png"
        )

    # Figure 4: SHAP Importance
    if explorer.shap_model:
        plot_shap_importance(
            explorer.shap_model,
            output_dir / "fig4_shap_importance.png"
        )

    # Figure 5: Age Distribution
    plot_age_distribution(
        archetypes,
        param_space.epidemiology,
        output_dir / "fig5_age_distribution.png"
    )

    # Figure 6: Sampling Statistics
    plot_sampling_statistics(
        explorer.stats,
        output_dir / "fig6_sampling_stats.png"
    )

    # Figure 7: Critical Scenario Coverage
    plot_critical_coverage(
        archetypes,
        output_dir / "fig7_critical_coverage.png"
    )

    # Figure 8: Acceptance Rates
    plot_acceptance_rates(
        explorer.stats,
        output_dir / "fig8_acceptance_rates.png"
    )

    print(f"✓ All visualizations saved to {output_dir}")


# ============================================================================
# Figure 1: Parameter Space Overview
# ============================================================================

def plot_parameter_space_overview(param_space, explorer, save_path: Path):
    """Comprehensive parameter space overview"""

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle(
        'Parameter Space Overview: Vestibular Domain',
        fontsize=14,
        fontweight='bold')

    # (A) Space size components
    ax = axes[0, 0]
    components = {
        'Timing': 3,
        'Trigger': 7,
        'HINTS Exam': 400,
        'Diagnoses': 15,
        'Risk Factors': 16,
        'Demographics': 102
    }
    colors = plt.cm.Set3(np.linspace(0, 1, len(components)))
    ax.bar(range(len(components)), components.values(), color=colors)
    ax.set_xticks(range(len(components)))
    ax.set_xticklabels(components.keys(), rotation=45, ha='right')
    ax.set_ylabel('Cardinality')
    ax.set_title('(A) Parameter Cardinalities')
    ax.set_yscale('log')
    ax.grid(axis='y', alpha=0.3)

    # Add values on bars
    for i, (k, v) in enumerate(components.items()):
        ax.text(i, v, f'{v}', ha='center', va='bottom', fontsize=8)

    # (B) Target calculation breakdown
    ax = axes[0, 1]
    stats = explorer.get_statistics()
    breakdown = {
        # Proportion
        'Statistical\nRequirement': stats['configuration']['n_target'] * 0.014,
        'Coverage\nRequirement': stats['configuration']['n_target'] * 0.134,
        'Clinical\nRequirement': stats['configuration']['n_target'] * 0.15,
        'Final\nTarget': stats['configuration']['n_target']
    }
    bars = ax.barh(
        range(
            len(breakdown)),
        breakdown.values(),
        color=[
            'lightblue',
            'lightgreen',
            'salmon',
            'gold'])
    ax.set_yticks(range(len(breakdown)))
    ax.set_yticklabels(breakdown.keys())
    ax.set_xlabel('Number of Archetypes')
    ax.set_title('(B) Target Archetype Calculation')
    ax.grid(axis='x', alpha=0.3)

    # Add values
    for i, (k, v) in enumerate(breakdown.items()):
        ax.text(v, i, f'{int(v):,}', ha='left', va='center', fontsize=8)

    # (C) Space size visualization
    ax = axes[1, 0]
    sizes = {
        'Total\nCombinations\n|P|': param_space.space_size,
        'Valid\nArchetypes\n|A|': int(
            param_space.space_size *
            param_space.acceptance_rate),
        'Target\nGenerated\nn_target': stats['configuration']['n_target']}
    colors_pie = ['#ff9999', '#66b3ff', '#99ff99']
    # Show as bar chart instead of pie for clarity
    ax.bar(range(len(sizes)), sizes.values(), color=colors_pie)
    ax.set_xticks(range(len(sizes)))
    ax.set_xticklabels(sizes.keys(), fontsize=9)
    ax.set_ylabel('Count')
    ax.set_title('(C) Parameter Space Sizes')
    ax.set_yscale('log')
    ax.grid(axis='y', alpha=0.3)

    # Add values
    for i, (k, v) in enumerate(sizes.items()):
        ax.text(i, v, f'{v:,}', ha='center',
                va='bottom', fontsize=7, rotation=0)

    # (D) Phase allocation
    ax = axes[1, 1]
    phases = [
        'Importance\nWeighted\n(60%)',
        'Critical\nScenarios\n(30%)',
        'Diversity\nOriented\n(10%)']
    counts = [explorer.n_importance, explorer.n_critical, explorer.n_diversity]
    colors_phases = ['steelblue', 'crimson', 'darkorange']
    wedges, texts, autotexts = ax.pie(
        counts,
        labels=phases,
        autopct='%1.1f%%',
        colors=colors_phases,
        startangle=90,
        textprops={'fontsize': 9}
    )
    ax.set_title('(D) Multi-Phase Sampling Allocation')

    plt.tight_layout()
    plt.savefig(save_path, bbox_inches='tight')
    plt.close()
    print(f"✓ Figure 1 saved: {save_path.name}")


# ============================================================================
# Figure 2: Diagnosis Distribution
# ============================================================================

def plot_diagnosis_distribution(archetypes, epidemiology, save_path: Path):
    """Compare observed vs expected diagnosis distribution"""

    fig, ax = plt.subplots(figsize=(12, 6))

    # Count observed
    observed = Counter([a.diagnosis for a in archetypes])
    diagnoses = sorted(epidemiology.diagnosis_dist.keys(),
                       key=lambda x: epidemiology.diagnosis_dist[x],
                       reverse=True)

    # Expected counts
    n_total = len(archetypes)
    expected = {d: epidemiology.expected_count(d, n_total) for d in diagnoses}

    # Create dataframe for plotting
    df = pd.DataFrame({
        'Diagnosis': diagnoses,
        'Expected': [expected[d] for d in diagnoses],
        'Observed': [observed.get(d, 0) for d in diagnoses]
    })

    x = np.arange(len(diagnoses))
    width = 0.35

    ax.bar(
        x - width / 2,
        df['Expected'],
        width,
        label='Expected (Epidemiology)',
        color='lightblue',
        alpha=0.8)
    ax.bar(
        x + width / 2,
        df['Observed'],
        width,
        label='Observed (Generated)',
        color='coral',
        alpha=0.8)

    ax.set_xlabel('Diagnosis')
    ax.set_ylabel('Count')
    ax.set_title(
        'Diagnosis Distribution: Expected vs Observed',
        fontsize=13,
        fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(diagnoses, rotation=45, ha='right')
    ax.legend()
    ax.grid(axis='y', alpha=0.3)

    # Add chi-squared test result
    validation = epidemiology.validate_distribution(archetypes)
    textstr = f"χ² = {
        validation['chi2_statistic']:.2f}, p = {
        validation['p_value']:.4f}\n"
    textstr += f"{'PASS' if validation['accept'] else 'FAIL'} (α = 0.05)"
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    ax.text(0.98, 0.97, textstr, transform=ax.transAxes, fontsize=9,
            verticalalignment='top', horizontalalignment='right', bbox=props)

    plt.tight_layout()
    plt.savefig(save_path, bbox_inches='tight')
    plt.close()
    print(f"✓ Figure 2 saved: {save_path.name}")


# ============================================================================
# Figure 3: NMF Factor Heatmap
# ============================================================================

def plot_nmf_factors(nmf_model, save_path: Path):
    """Heatmap of NMF factor loadings"""

    fig, ax = plt.subplots(figsize=(14, 8))

    # Get H matrix (factors × features)
    H = nmf_model.H_

    # Take top 50 features for visualization
    top_features_per_factor = 50
    mask = np.zeros_like(H, dtype=bool)

    for i in range(H.shape[0]):
        top_indices = np.argsort(H[i])[-top_features_per_factor:]
        mask[i, top_indices] = True

    H_filtered = np.where(mask, H, 0)

    # Plot heatmap
    sns.heatmap(
        H_filtered,
        cmap='YlOrRd',
        cbar_kws={'label': 'Factor Weight'},
        xticklabels=False,
        yticklabels=[f"Factor {i}" for i in range(H.shape[0])],
        ax=ax
    )

    ax.set_xlabel('Features')
    ax.set_ylabel('NMF Factors')
    ax.set_title(
        f'NMF Factor Loadings (r={
            nmf_model.n_components})',
        fontsize=13,
        fontweight='bold')

    # Add factor interpretations on the right
    for i, interp in enumerate(nmf_model.factor_interpretations_):
        ax.text(
            H.shape[1] + 5, i + 0.5,
            interp['clinical_pattern'],
            va='center', fontsize=8
        )

    plt.tight_layout()
    plt.savefig(save_path, bbox_inches='tight')
    plt.close()
    print(f"✓ Figure 3 saved: {save_path.name}")


# ============================================================================
# Figure 4: SHAP Importance
# ============================================================================

def plot_shap_importance(shap_model, save_path: Path):
    """SHAP feature importance bar chart"""

    fig, ax = plt.subplots(figsize=(10, 8))

    top_features = shap_model.get_top_features(20)
    names, importances = zip(*top_features)

    y_pos = np.arange(len(names))

    ax.barh(y_pos, importances, color='steelblue', alpha=0.8)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(names)
    ax.invert_yaxis()
    ax.set_xlabel('Mean |SHAP Value| (Feature Importance φⱼ)')
    ax.set_title(
        'Top 20 Features by SHAP Importance',
        fontsize=13,
        fontweight='bold')
    ax.grid(axis='x', alpha=0.3)

    # Add values
    for i, imp in enumerate(importances):
        ax.text(imp, i, f' {imp:.4f}', va='center', fontsize=8)

    plt.tight_layout()
    plt.savefig(save_path, bbox_inches='tight')
    plt.close()
    print(f"✓ Figure 4 saved: {save_path.name}")


# ============================================================================
# Figure 5: Age Distribution by Diagnosis
# ============================================================================

def plot_age_distribution(archetypes, epidemiology, save_path: Path):
    """Age distribution violin plots by diagnosis"""

    # Extract data
    data = []
    for a in archetypes:
        data.append({
            'diagnosis': a.diagnosis,
            'age': a.parameters.get('age', np.nan)
        })

    df = pd.DataFrame(data).dropna()

    # Select top 6 diagnoses by frequency
    top_diagnoses = df['diagnosis'].value_counts().head(6).index.tolist()
    df_filtered = df[df['diagnosis'].isin(top_diagnoses)]

    fig, ax = plt.subplots(figsize=(12, 6))

    # Violin plot
    parts = ax.violinplot(
        [df_filtered[df_filtered['diagnosis'] == d]['age'].values for d in top_diagnoses],
        positions=range(len(top_diagnoses)),
        showmeans=True,
        showmedians=True
    )

    # Add expected distributions (from epidemiology)
    for i, diagnosis in enumerate(top_diagnoses):
        if diagnosis in epidemiology.age_dist:
            mean, std = epidemiology.age_dist[diagnosis]
            ax.errorbar(
                i, mean, yerr=std,
                fmt='ro', markersize=8,
                capsize=5, capthick=2,
                label='Expected' if i == 0 else ''
            )

    ax.set_xticks(range(len(top_diagnoses)))
    ax.set_xticklabels(top_diagnoses, rotation=45, ha='right')
    ax.set_ylabel('Age (years)')
    ax.set_title(
        'Age Distribution by Diagnosis',
        fontsize=13,
        fontweight='bold')
    ax.legend()
    ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, bbox_inches='tight')
    plt.close()
    print(f"✓ Figure 5 saved: {save_path.name}")


# ============================================================================
# Figure 6: Sampling Statistics
# ============================================================================

def plot_sampling_statistics(stats, save_path: Path):
    """Multi-phase sampling statistics"""

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # (A) Samples per phase
    ax = axes[0]
    phases = [
        'Phase 1\n(Uniform)',
        'Phase 4\n(Importance)',
        'Phase 5\n(Critical)',
        'Phase 6\n(Diversity)']
    sampled = [
        stats['phase1_sampled'],
        stats['phase4_sampled'],
        stats['phase5_sampled'],
        stats['phase6_sampled']
    ]
    valid = [
        stats['phase1_valid'],
        stats['phase4_valid'],
        stats['phase5_valid'],
        stats['phase6_valid']
    ]

    x = np.arange(len(phases))
    width = 0.35

    ax.bar(
        x - width / 2,
        sampled,
        width,
        label='Total Sampled',
        color='lightcoral',
        alpha=0.8)
    ax.bar(
        x + width / 2,
        valid,
        width,
        label='Valid (Passed Constraints)',
        color='lightgreen',
        alpha=0.8)

    ax.set_ylabel('Count')
    ax.set_title('(A) Sampling Attempts per Phase', fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(phases)
    ax.legend()
    ax.grid(axis='y', alpha=0.3)

    # (B) Acceptance rates
    ax = axes[1]
    rates = [
        stats['phase1_valid'] / stats['phase1_sampled'] * 100,
        stats['phase4_valid'] / stats['phase4_sampled'] * 100,
        stats['phase5_valid'] / stats['phase5_sampled'] * 100,
        stats['phase6_valid'] / stats['phase6_sampled'] * 100
    ]

    colors_grad = plt.cm.RdYlGn(np.linspace(0.3, 0.9, len(phases)))
    bars = ax.bar(x, rates, color=colors_grad, alpha=0.8)

    ax.set_ylabel('Acceptance Rate (%)')
    ax.set_title('(B) Acceptance Rate per Phase', fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(phases)
    ax.axhline(
        50,
        color='red',
        linestyle='--',
        alpha=0.5,
        label='50% threshold')
    ax.legend()
    ax.grid(axis='y', alpha=0.3)

    # Add values on bars
    for bar, rate in zip(bars, rates):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2., height,
                f'{rate:.1f}%', ha='center', va='bottom', fontsize=9)

    plt.tight_layout()
    plt.savefig(save_path, bbox_inches='tight')
    plt.close()
    print(f"✓ Figure 6 saved: {save_path.name}")


# ============================================================================
# Figure 7: Critical Scenario Coverage
# ============================================================================

def plot_critical_coverage(archetypes, save_path: Path):
    """Critical scenario coverage analysis"""

    fig, ax = plt.subplots(figsize=(10, 6))

    # Count critical scenarios
    critical_diagnoses = ['stroke', 'tia']
    critical_count = sum(
        1 for a in archetypes if a.diagnosis in critical_diagnoses)
    non_critical_count = len(archetypes) - critical_count

    # Expected
    expected_critical = len(archetypes) * 0.15  # 15%

    # Create grouped bar chart
    categories = ['Critical Scenarios\n(Stroke/TIA)', 'Other Diagnoses']
    observed = [critical_count, non_critical_count]
    expected = [expected_critical, len(archetypes) - expected_critical]

    x = np.arange(len(categories))
    width = 0.35

    ax.bar(
        x - width / 2,
        expected,
        width,
        label='Expected (Epidemiology)',
        color='lightblue',
        alpha=0.8)
    ax.bar(
        x + width / 2,
        observed,
        width,
        label='Observed (Generated)',
        color='coral',
        alpha=0.8)

    ax.set_ylabel('Count')
    ax.set_title('Critical Scenario Coverage', fontsize=13, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(categories)
    ax.legend()
    ax.grid(axis='y', alpha=0.3)

    # Add percentages
    for i, (exp, obs) in enumerate(zip(expected, observed)):
        ax.text(i - width / 2,
                exp,
                f'{exp / len(archetypes) * 100:.1f}%',
                ha='center',
                va='bottom',
                fontsize=9)
        ax.text(i + width / 2,
                obs,
                f'{obs / len(archetypes) * 100:.1f}%',
                ha='center',
                va='bottom',
                fontsize=9)

    # Add match info
    match_rate = (1 - abs(critical_count - expected_critical) /
                  expected_critical) * 100
    textstr = f"Match: {match_rate:.1f}%\n"
    textstr += f"Deviation: {abs(critical_count - expected_critical) / expected_critical * 100:.1f}%"
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    ax.text(0.98, 0.97, textstr, transform=ax.transAxes, fontsize=10,
            verticalalignment='top', horizontalalignment='right', bbox=props)

    plt.tight_layout()
    plt.savefig(save_path, bbox_inches='tight')
    plt.close()
    print(f"✓ Figure 7 saved: {save_path.name}")


# ============================================================================
# Figure 8: Acceptance Rates Comparison
# ============================================================================

def plot_acceptance_rates(stats, save_path: Path):
    """Acceptance rate improvement across phases"""

    fig, ax = plt.subplots(figsize=(10, 6))

    phases = [
        'Phase 1\n(Uniform)',
        'Phase 4\n(Importance)',
        'Phase 5\n(Critical)',
        'Phase 6\n(Diversity)']
    rates = [
        stats['phase1_valid'] / stats['phase1_sampled'],
        stats['phase4_valid'] / stats['phase4_sampled'],
        stats['phase5_valid'] / stats['phase5_sampled'],
        stats['phase6_valid'] / stats['phase6_sampled']
    ]

    x = np.arange(len(phases))

    # Line plot with markers
    line = ax.plot(
        x,
        rates,
        marker='o',
        markersize=10,
        linewidth=2,
        color='steelblue')
    ax.scatter(
        x,
        rates,
        s=100,
        c=rates,
        cmap='RdYlGn',
        vmin=0,
        vmax=1,
        zorder=3)

    ax.set_xticks(x)
    ax.set_xticklabels(phases)
    ax.set_ylabel('Acceptance Rate (ρ)')
    ax.set_title(
        'Acceptance Rate Evolution Across Phases',
        fontsize=13,
        fontweight='bold')
    ax.set_ylim([0, 1])
    ax.grid(True, alpha=0.3)

    # Add value labels
    for i, rate in enumerate(rates):
        ax.text(i,
                rate + 0.05,
                f'{rate:.2%}',
                ha='center',
                fontsize=10,
                fontweight='bold')

    # Add efficiency gain annotation
    efficiency_gain = (rates[-1] - rates[0]) / rates[0] * 100
    textstr = f"Efficiency Gain: {efficiency_gain:+.1f}%"
    props = dict(boxstyle='round', facecolor='lightgreen', alpha=0.5)
    ax.text(
        0.98,
        0.05,
        textstr,
        transform=ax.transAxes,
        fontsize=11,
        verticalalignment='bottom',
        horizontalalignment='right',
        bbox=props)

    plt.tight_layout()
    plt.savefig(save_path, bbox_inches='tight')
    plt.close()
    print(f"✓ Figure 8 saved: {save_path.name}")
