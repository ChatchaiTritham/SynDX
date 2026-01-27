"""
Comprehensive 2D/3D Visualization Generator for SynDX Framework

Generates publication-ready charts and graphs for all pipeline phases:
- Phase 1: Knowledge Extraction (Parameter space, NMF factors, SHAP importance)
- Phase 2: Synthesis (VAE latent space, DP privacy budget, Counterfactuals)
- Phase 3: Validation (Performance metrics, Fidelity scores, Statistical tests)

All visualizations meet Tier 1 journal standards (600 DPI, vector formats).

Author: PhD Candidate, Computer Science
Institution: [Your Institution]
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from mpl_toolkits.mplot3d import Axes3D
import plotly.graph_objects as go
import plotly.express as px
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import logging

# Configure plotting
plt.style.use('seaborn-v0_8-paper')
sns.set_context("paper", font_scale=1.5)
plt.rcParams['figure.dpi'] = 600
plt.rcParams['savefig.dpi'] = 600
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman']

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SynDXVisualizer:
    """
    Comprehensive visualization generator for SynDX framework.

    Produces 2D and 3D plots for:
    - Parameter space exploration
    - XAI-guided sampling strategies
    - Latent representations (NMF, VAE)
    - Performance metrics and validation
    """

    def __init__(self, output_dir: str = "outputs/visualizations"):
        """
        Initialize visualizer.

        Args:
            output_dir: Directory to save visualization outputs
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Create subdirectories
        (self.output_dir / "phase1").mkdir(exist_ok=True)
        (self.output_dir / "phase2").mkdir(exist_ok=True)
        (self.output_dir / "phase3").mkdir(exist_ok=True)
        (self.output_dir / "3d").mkdir(exist_ok=True)

    # ========== Phase 1: Knowledge Extraction Visualizations ==========

    def plot_parameter_space_2d(self,
                                archetypes_df: pd.DataFrame,
                                save_name: str = "parameter_space_2d"):
        """
        2D scatter plot of parameter space coverage.

        Visualizes archetype distribution across two principal dimensions.
        """
        fig, axes = plt.subplots(2, 2, figsize=(16, 14))

        # Age vs Diagnosis
        sns.scatterplot(
            data=archetypes_df,
            x='age',
            y='diagnosis',
            hue='urgency',
            style='timing',
            s=100,
            alpha=0.6,
            ax=axes[0, 0]
        )
        axes[0, 0].set_title('Parameter Space: Age × Diagnosis',
                             fontsize=16, fontweight='bold')
        axes[0, 0].set_xlabel('Age (years)', fontsize=14)
        axes[0, 0].set_ylabel('Diagnosis Category', fontsize=14)

        # Timing vs Trigger distribution
        timing_trigger = pd.crosstab(
            archetypes_df['timing'],
            archetypes_df['trigger'],
            normalize='index'
        ) * 100
        sns.heatmap(
            timing_trigger,
            annot=True,
            fmt='.1f',
            cmap='YlOrRd',
            ax=axes[0, 1],
            cbar_kws={'label': 'Percentage (%)'}
        )
        axes[0, 1].set_title('Timing × Trigger Distribution',
                             fontsize=16, fontweight='bold')
        axes[0, 1].set_xlabel('Trigger Type', fontsize=14)
        axes[0, 1].set_ylabel('Timing Pattern', fontsize=14)

        # Diagnosis distribution
        diagnosis_counts = archetypes_df['diagnosis'].value_counts()
        axes[1, 0].barh(diagnosis_counts.index,
                        diagnosis_counts.values, color='steelblue')
        axes[1, 0].set_title('Diagnosis Distribution',
                             fontsize=16, fontweight='bold')
        axes[1, 0].set_xlabel('Count', fontsize=14)
        axes[1, 0].set_ylabel('Diagnosis', fontsize=14)

        # Urgency vs Timing
        urgency_timing = pd.crosstab(
            archetypes_df['urgency'],
            archetypes_df['timing']
        )
        urgency_timing.plot(kind='bar', stacked=True, ax=axes[1, 1], color=[
                            '#1f77b4', '#ff7f0e', '#2ca02c'])
        axes[1, 1].set_title('Urgency × Timing Pattern',
                             fontsize=16, fontweight='bold')
        axes[1, 1].set_xlabel('Urgency Level', fontsize=14)
        axes[1, 1].set_ylabel('Count', fontsize=14)
        axes[1, 1].legend(title='Timing', fontsize=12)

        plt.tight_layout()
        plt.savefig(
            self.output_dir /
            "phase1" /
            f"{save_name}.png",
            dpi=600,
            bbox_inches='tight')
        plt.savefig(
            self.output_dir /
            "phase1" /
            f"{save_name}.pdf",
            bbox_inches='tight')
        plt.close()

        logger.info(f"Saved: {save_name}")

    def plot_parameter_space_3d(self,
                                archetypes_df: pd.DataFrame,
                                save_name: str = "parameter_space_3d"):
        """
        3D interactive visualization of parameter space.

        Uses age, symptom severity, and urgency as three dimensions.
        """
        fig = go.Figure(
            data=[
                go.Scatter3d(
                    x=archetypes_df['age'],
                    y=archetypes_df.get(
                        'symptom_severity',
                        np.random.randint(
                            1,
                            11,
                            len(archetypes_df))),
                    z=archetypes_df['urgency'],
                    mode='markers',
                    marker=dict(
                        size=5,
                        color=archetypes_df['urgency'],
                        colorscale='Viridis',
                        showscale=True,
                        colorbar=dict(
                            title="Urgency Level")),
                    text=archetypes_df['diagnosis'],
                    hovertemplate='<b>Age:</b> %{x}<br>' +
                    '<b>Severity:</b> %{y}<br>' +
                    '<b>Urgency:</b> %{z}<br>' +
                    '<b>Diagnosis:</b> %{text}<extra></extra>')])

        fig.update_layout(
            title='3D Parameter Space: Age × Severity × Urgency',
            scene=dict(
                xaxis_title='Age (years)',
                yaxis_title='Symptom Severity (1-10)',
                zaxis_title='Urgency Level (0-2)'
            ),
            font=dict(family="Times New Roman", size=14),
            width=1200,
            height=900
        )

        fig.write_html(self.output_dir / "3d" / f"{save_name}.html")
        logger.info(f"Saved: {save_name}.html")

    def plot_nmf_factors(self,
                         W: np.ndarray,
                         H: np.ndarray,
                         feature_names: List[str],
                         save_name: str = "nmf_factors"):
        """
        Visualize NMF latent factors (W and H matrices).

        Args:
            W: Archetype-to-latent weights (n_samples × r)
            H: Latent-to-feature basis (r × n_features)
            feature_names: List of feature names
        """
        r = W.shape[1]

        fig, axes = plt.subplots(2, 1, figsize=(18, 12))

        # H matrix heatmap (latent factors × features)
        top_features = 30  # Show top 30 most important features
        H_subset = H[:, :top_features]
        sns.heatmap(
            H_subset,
            cmap='RdBu_r',
            center=0,
            ax=axes[0],
            cbar_kws={'label': 'Factor Loading'},
            xticklabels=feature_names[:top_features] if len(feature_names) >= top_features else feature_names,
            yticklabels=[f'Factor {i + 1}' for i in range(r)]
        )
        axes[0].set_title(
            f'NMF Latent Factor Loadings (r={r})',
            fontsize=16,
            fontweight='bold')
        axes[0].set_xlabel('Features (Top 30)', fontsize=14)
        axes[0].set_ylabel('Latent Factors', fontsize=14)

        # W matrix distribution (archetype weights)
        axes[1].boxplot([W[:, i] for i in range(r)], tick_labels=[
                        f'F{i + 1}' for i in range(r)])
        axes[1].set_title(
            'Distribution of Archetype Weights Across Factors',
            fontsize=16,
            fontweight='bold')
        axes[1].set_xlabel('Latent Factor', fontsize=14)
        axes[1].set_ylabel('Weight', fontsize=14)
        axes[1].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(
            self.output_dir /
            "phase1" /
            f"{save_name}.png",
            dpi=600,
            bbox_inches='tight')
        plt.savefig(
            self.output_dir /
            "phase1" /
            f"{save_name}.pdf",
            bbox_inches='tight')
        plt.close()

        logger.info(f"Saved: {save_name}")

    def plot_shap_importance(self,
                             shap_values: np.ndarray,
                             feature_names: List[str],
                             save_name: str = "shap_importance"):
        """
        Visualize SHAP feature importance.

        Args:
            shap_values: SHAP values (n_samples × n_features)
            feature_names: List of feature names
        """
        # Calculate mean absolute SHAP values
        mean_abs_shap = np.abs(shap_values).mean(axis=0)

        # Sort by importance
        sorted_indices = np.argsort(mean_abs_shap)[-20:]  # Top 20

        fig, axes = plt.subplots(1, 2, figsize=(18, 8))

        # Bar plot
        axes[0].barh(
            range(len(sorted_indices)),
            mean_abs_shap[sorted_indices],
            color='steelblue'
        )
        axes[0].set_yticks(range(len(sorted_indices)))
        axes[0].set_yticklabels([feature_names[i] for i in sorted_indices])
        axes[0].set_title(
            'Top 20 Features by SHAP Importance',
            fontsize=16,
            fontweight='bold')
        axes[0].set_xlabel('Mean |SHAP Value|', fontsize=14)

        # Violin plot for top 10 features
        top_10_indices = sorted_indices[-10:]
        shap_df = pd.DataFrame(
            shap_values[:, top_10_indices],
            columns=[feature_names[i] for i in top_10_indices]
        )
        shap_df_melted = shap_df.melt(
            var_name='Feature', value_name='SHAP Value')

        sns.violinplot(
            data=shap_df_melted,
            y='Feature',
            x='SHAP Value',
            ax=axes[1]
        )
        axes[1].set_title(
            'SHAP Value Distribution (Top 10 Features)',
            fontsize=16,
            fontweight='bold')
        axes[1].axvline(
            x=0,
            color='red',
            linestyle='--',
            linewidth=2,
            alpha=0.5)

        plt.tight_layout()
        plt.savefig(
            self.output_dir /
            "phase1" /
            f"{save_name}.png",
            dpi=600,
            bbox_inches='tight')
        plt.savefig(
            self.output_dir /
            "phase1" /
            f"{save_name}.pdf",
            bbox_inches='tight')
        plt.close()

        logger.info(f"Saved: {save_name}")

    # ========== Phase 2: Synthesis Visualizations ==========

    def plot_vae_latent_space_2d(self,
                                 z_mean: np.ndarray,
                                 labels: np.ndarray,
                                 save_name: str = "vae_latent_2d"):
        """
        2D visualization of VAE latent space (first 2 dimensions).

        Args:
            z_mean: Latent representations (n_samples × latent_dim)
            labels: Diagnosis labels for coloring
        """
        fig, axes = plt.subplots(1, 2, figsize=(18, 8))

        # Scatter plot colored by diagnosis
        scatter = axes[0].scatter(
            z_mean[:, 0],
            z_mean[:, 1],
            c=labels,
            cmap='tab20',
            alpha=0.6,
            s=50
        )
        axes[0].set_title(
            'VAE Latent Space (2D Projection)',
            fontsize=16,
            fontweight='bold')
        axes[0].set_xlabel('Latent Dimension 1', fontsize=14)
        axes[0].set_ylabel('Latent Dimension 2', fontsize=14)
        axes[0].grid(True, alpha=0.3)
        plt.colorbar(scatter, ax=axes[0], label='Diagnosis')

        # Density plot
        from scipy.stats import gaussian_kde
        xy = np.vstack([z_mean[:, 0], z_mean[:, 1]])
        z = gaussian_kde(xy)(xy)
        scatter2 = axes[1].scatter(
            z_mean[:, 0],
            z_mean[:, 1],
            c=z,
            cmap='YlOrRd',
            alpha=0.6,
            s=50
        )
        axes[1].set_title(
            'VAE Latent Space Density',
            fontsize=16,
            fontweight='bold')
        axes[1].set_xlabel('Latent Dimension 1', fontsize=14)
        axes[1].set_ylabel('Latent Dimension 2', fontsize=14)
        axes[1].grid(True, alpha=0.3)
        plt.colorbar(scatter2, ax=axes[1], label='Density')

        plt.tight_layout()
        plt.savefig(
            self.output_dir /
            "phase2" /
            f"{save_name}.png",
            dpi=600,
            bbox_inches='tight')
        plt.savefig(
            self.output_dir /
            "phase2" /
            f"{save_name}.pdf",
            bbox_inches='tight')
        plt.close()

        logger.info(f"Saved: {save_name}")

    def plot_vae_latent_space_3d(self,
                                 z_mean: np.ndarray,
                                 labels: np.ndarray,
                                 label_names: List[str],
                                 save_name: str = "vae_latent_3d"):
        """
        3D interactive visualization of VAE latent space.

        Args:
            z_mean: Latent representations (n_samples × latent_dim)
            labels: Diagnosis labels
            label_names: Names of diagnosis categories
        """
        fig = go.Figure(data=[go.Scatter3d(
            x=z_mean[:, 0],
            y=z_mean[:, 1],
            z=z_mean[:, 2],
            mode='markers',
            marker=dict(
                size=4,
                color=labels,
                colorscale='Viridis',
                showscale=True,
                colorbar=dict(title="Diagnosis")
            ),
            text=[label_names[int(l)] if int(l) < len(label_names) else f"Label {int(l)}" for l in labels],
            hovertemplate='<b>Latent 1:</b> %{x:.3f}<br>' +
                          '<b>Latent 2:</b> %{y:.3f}<br>' +
                          '<b>Latent 3:</b> %{z:.3f}<br>' +
                          '<b>Diagnosis:</b> %{text}<extra></extra>'
        )])

        fig.update_layout(
            title='VAE Latent Space (3D Visualization)',
            scene=dict(
                xaxis_title='Latent Dimension 1',
                yaxis_title='Latent Dimension 2',
                zaxis_title='Latent Dimension 3'
            ),
            font=dict(family="Times New Roman", size=14),
            width=1200,
            height=900
        )

        fig.write_html(self.output_dir / "3d" / f"{save_name}.html")
        logger.info(f"Saved: {save_name}.html")

    def plot_privacy_budget_tracking(self,
                                     privacy_history: List[Dict],
                                     save_name: str = "privacy_budget"):
        """
        Visualize differential privacy budget consumption over epochs.

        Args:
            privacy_history: List of dicts with 'epoch', 'epsilon', 'delta'
        """
        epochs = [h['epoch'] for h in privacy_history]
        epsilons = [h['epsilon'] for h in privacy_history]
        deltas = [h['delta'] for h in privacy_history]

        fig, axes = plt.subplots(2, 1, figsize=(14, 10))

        # Epsilon over epochs
        axes[0].plot(epochs, epsilons, 'b-', linewidth=2, label='ε (epsilon)')
        axes[0].axhline(
            y=1.0,
            color='r',
            linestyle='--',
            linewidth=2,
            label='Target ε=1.0')
        axes[0].set_title(
            'Privacy Budget: Epsilon (ε) Consumption',
            fontsize=16,
            fontweight='bold')
        axes[0].set_xlabel('Epoch', fontsize=14)
        axes[0].set_ylabel('ε', fontsize=14)
        axes[0].legend(fontsize=12)
        axes[0].grid(True, alpha=0.3)

        # Delta over epochs
        axes[1].plot(epochs, deltas, 'g-', linewidth=2, label='δ (delta)')
        axes[1].axhline(
            y=1e-5,
            color='r',
            linestyle='--',
            linewidth=2,
            label='Target δ=10⁻⁵')
        axes[1].set_title(
            'Privacy Budget: Delta (δ) Consumption',
            fontsize=16,
            fontweight='bold')
        axes[1].set_xlabel('Epoch', fontsize=14)
        axes[1].set_ylabel('δ', fontsize=14)
        axes[1].set_yscale('log')
        axes[1].legend(fontsize=12)
        axes[1].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(
            self.output_dir /
            "phase2" /
            f"{save_name}.png",
            dpi=600,
            bbox_inches='tight')
        plt.savefig(
            self.output_dir /
            "phase2" /
            f"{save_name}.pdf",
            bbox_inches='tight')
        plt.close()

        logger.info(f"Saved: {save_name}")

    def plot_counterfactual_examples(
            self,
            original: np.ndarray,
            counterfactual: np.ndarray,
            feature_names: List[str],
            save_name: str = "counterfactual_examples"):
        """
        Visualize counterfactual examples showing feature changes.

        Args:
            original: Original patient features (n_features,)
            counterfactual: Counterfactual features (n_features,)
            feature_names: List of feature names
        """
        # Calculate differences
        diff = counterfactual - original
        changed_indices = np.where(np.abs(diff) > 0.01)[
            0][:15]  # Top 15 changes

        fig, ax = plt.subplots(figsize=(14, 8))

        x = np.arange(len(changed_indices))
        width = 0.35

        ax.barh(
            x - width / 2,
            original[changed_indices],
            width,
            label='Original',
            color='steelblue')
        ax.barh(
            x + width / 2,
            counterfactual[changed_indices],
            width,
            label='Counterfactual',
            color='coral')

        ax.set_yticks(x)
        ax.set_yticklabels([feature_names[i] for i in changed_indices])
        ax.set_xlabel('Feature Value', fontsize=14)
        ax.set_title(
            'Counterfactual Example: Feature Changes',
            fontsize=16,
            fontweight='bold')
        ax.legend(fontsize=12)
        ax.grid(True, alpha=0.3, axis='x')

        plt.tight_layout()
        plt.savefig(
            self.output_dir /
            "phase2" /
            f"{save_name}.png",
            dpi=600,
            bbox_inches='tight')
        plt.savefig(
            self.output_dir /
            "phase2" /
            f"{save_name}.pdf",
            bbox_inches='tight')
        plt.close()

        logger.info(f"Saved: {save_name}")

    # ========== Phase 3: Validation Visualizations ==========

    def plot_performance_comparison(self,
                                    metrics_dict: Dict[str, Dict[str, float]],
                                    save_name: str = "performance_comparison"):
        """
        Compare performance metrics across models.

        Args:
            metrics_dict: Dict of {model_name: {metric_name: value}}
        """
        df = pd.DataFrame(metrics_dict).T

        fig, axes = plt.subplots(2, 2, figsize=(16, 14))

        # Grouped bar chart
        df[['accuracy', 'precision', 'recall', 'f1_score']].plot(
            kind='bar',
            ax=axes[0, 0],
            color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
        )
        axes[0, 0].set_title(
            'Classification Metrics Comparison', fontsize=16, fontweight='bold')
        axes[0, 0].set_xlabel('Model', fontsize=14)
        axes[0, 0].set_ylabel('Score', fontsize=14)
        axes[0, 0].legend(title='Metric', fontsize=12)
        axes[0, 0].set_ylim([0, 1])
        axes[0, 0].grid(True, alpha=0.3, axis='y')

        # ROC-AUC comparison
        if 'roc_auc' in df.columns:
            df['roc_auc'].plot(kind='barh', ax=axes[0, 1], color='steelblue')
            axes[0, 1].set_title(
                'ROC-AUC Scores', fontsize=16, fontweight='bold')
            axes[0, 1].set_xlabel('ROC-AUC', fontsize=14)
            axes[0, 1].set_xlim([0, 1])
            axes[0, 1].grid(True, alpha=0.3, axis='x')

        # Heatmap of all metrics
        sns.heatmap(
            df,
            annot=True,
            fmt='.3f',
            cmap='RdYlGn',
            vmin=0,
            vmax=1,
            ax=axes[1, 0],
            cbar_kws={'label': 'Score'}
        )
        axes[1, 0].set_title('All Metrics Heatmap',
                             fontsize=16, fontweight='bold')
        axes[1, 0].set_xlabel('Metric', fontsize=14)
        axes[1, 0].set_ylabel('Model', fontsize=14)

        # Radar chart for first model
        if len(metrics_dict) > 0:
            first_model = list(metrics_dict.keys())[0]
            metrics = ['accuracy', 'precision', 'recall', 'f1_score']
            values = [metrics_dict[first_model].get(m, 0) for m in metrics]

            angles = np.linspace(
                0,
                2 * np.pi,
                len(metrics),
                endpoint=False).tolist()
            values += values[:1]
            angles += angles[:1]

            ax_polar = plt.subplot(224, projection='polar')
            ax_polar.plot(angles, values, 'o-', linewidth=2, color='steelblue')
            ax_polar.fill(angles, values, alpha=0.25, color='steelblue')
            ax_polar.set_xticks(angles[:-1])
            ax_polar.set_xticklabels(metrics, fontsize=12)
            ax_polar.set_ylim(0, 1)
            ax_polar.set_title(
                f'{first_model} Metrics',
                fontsize=14,
                fontweight='bold',
                pad=20)
            ax_polar.grid(True)

        plt.tight_layout()
        plt.savefig(
            self.output_dir /
            "phase3" /
            f"{save_name}.png",
            dpi=600,
            bbox_inches='tight')
        plt.savefig(
            self.output_dir /
            "phase3" /
            f"{save_name}.pdf",
            bbox_inches='tight')
        plt.close()

        logger.info(f"Saved: {save_name}")

    def plot_xai_fidelity_scores(self,
                                 fidelity_scores: Dict[str, float],
                                 save_name: str = "xai_fidelity"):
        """
        Visualize XAI fidelity metrics.

        Args:
            fidelity_scores: Dict of {metric_name: score}
        """
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))

        # Bar chart
        metrics = list(fidelity_scores.keys())
        values = list(fidelity_scores.values())

        colors = ['#1f77b4' if v >= 0.8 else '#ff7f0e' if v >=
                  0.6 else '#d62728' for v in values]

        axes[0].barh(metrics, values, color=colors)
        axes[0].set_title(
            'XAI Fidelity Scores',
            fontsize=16,
            fontweight='bold')
        axes[0].set_xlabel('Score', fontsize=14)
        axes[0].set_xlim([0, 1])
        axes[0].axvline(
            x=0.8,
            color='green',
            linestyle='--',
            linewidth=2,
            alpha=0.5,
            label='Excellent (≥0.8)')
        axes[0].axvline(
            x=0.6,
            color='orange',
            linestyle='--',
            linewidth=2,
            alpha=0.5,
            label='Good (≥0.6)')
        axes[0].legend(fontsize=11)
        axes[0].grid(True, alpha=0.3, axis='x')

        # Donut chart for overall fidelity
        if 'overall_fidelity' in fidelity_scores:
            overall = fidelity_scores['overall_fidelity']
            remaining = 1 - overall

            axes[1].pie(
                [overall, remaining],
                labels=['Fidelity Achieved', 'Gap'],
                autopct='%1.1f%%',
                startangle=90,
                colors=['steelblue', 'lightgray'],
                wedgeprops=dict(width=0.4)
            )
            axes[1].set_title(
                f'Overall XAI Fidelity: {
                    overall:.3f}',
                fontsize=16,
                fontweight='bold')

        plt.tight_layout()
        plt.savefig(
            self.output_dir /
            "phase3" /
            f"{save_name}.png",
            dpi=600,
            bbox_inches='tight')
        plt.savefig(
            self.output_dir /
            "phase3" /
            f"{save_name}.pdf",
            bbox_inches='tight')
        plt.close()

        logger.info(f"Saved: {save_name}")

    def plot_statistical_validation(self,
                                    chi_squared_results: Dict[str,
                                                              Tuple[float,
                                                                    float]],
                                    save_name: str = "statistical_validation"):
        """
        Visualize statistical validation results (chi-squared tests).

        Args:
            chi_squared_results: Dict of {feature: (chi2_stat, p_value)}
        """
        features = list(chi_squared_results.keys())
        chi2_stats = [chi_squared_results[f][0] for f in features]
        p_values = [chi_squared_results[f][1] for f in features]

        fig, axes = plt.subplots(2, 1, figsize=(14, 10))

        # Chi-squared statistics
        axes[0].barh(features, chi2_stats, color='steelblue')
        axes[0].set_title(
            'Chi-Squared Statistics by Feature',
            fontsize=16,
            fontweight='bold')
        axes[0].set_xlabel('χ² Statistic', fontsize=14)
        axes[0].grid(True, alpha=0.3, axis='x')

        # P-values
        colors = ['green' if p > 0.05 else 'red' for p in p_values]
        axes[1].barh(features, p_values, color=colors)
        axes[1].axvline(
            x=0.05,
            color='red',
            linestyle='--',
            linewidth=2,
            label='α=0.05')
        axes[1].set_title(
            'P-Values (Distribution Match Test)',
            fontsize=16,
            fontweight='bold')
        axes[1].set_xlabel('P-Value', fontsize=14)
        axes[1].set_xscale('log')
        axes[1].legend(fontsize=12)
        axes[1].grid(True, alpha=0.3, axis='x')

        plt.tight_layout()
        plt.savefig(
            self.output_dir /
            "phase3" /
            f"{save_name}.png",
            dpi=600,
            bbox_inches='tight')
        plt.savefig(
            self.output_dir /
            "phase3" /
            f"{save_name}.pdf",
            bbox_inches='tight')
        plt.close()

        logger.info(f"Saved: {save_name}")


def main():
    """
    Demonstration of visualization capabilities.
    """
    logger.info("=== SynDX Visualization Generator ===")
    logger.info("Generating demonstration plots...")

    visualizer = SynDXVisualizer(output_dir="outputs/demo_visualizations")

    # Generate sample data
    np.random.seed(42)
    n_samples = 1000

    # Sample archetype data
    archetypes_df = pd.DataFrame({
        'age': np.random.randint(18, 90, n_samples),
        'diagnosis': np.random.choice(['BPPV', 'Stroke', 'VM', 'VN'], n_samples),
        'timing': np.random.choice(['acute', 'episodic', 'chronic'], n_samples),
        'trigger': np.random.choice(['spontaneous', 'positional', 'head_movement'], n_samples),
        'urgency': np.random.choice([0, 1, 2], n_samples)
    })

    # Phase 1 visualizations
    logger.info("Generating Phase 1 visualizations...")
    visualizer.plot_parameter_space_2d(archetypes_df)
    visualizer.plot_parameter_space_3d(archetypes_df)

    # Sample NMF data
    W = np.random.rand(n_samples, 20)
    H = np.random.rand(20, 150)
    feature_names = [f'Feature_{i}' for i in range(150)]
    visualizer.plot_nmf_factors(W, H, feature_names)

    # Sample SHAP data
    shap_values = np.random.randn(n_samples, 150) * 0.1
    visualizer.plot_shap_importance(shap_values, feature_names)

    # Phase 2 visualizations
    logger.info("Generating Phase 2 visualizations...")
    z_mean = np.random.randn(n_samples, 20)
    labels = np.random.randint(0, 4, n_samples)
    label_names = ['BPPV', 'Stroke', 'VM', 'VN']
    visualizer.plot_vae_latent_space_2d(z_mean, labels)
    visualizer.plot_vae_latent_space_3d(z_mean, labels, label_names)

    # Privacy budget
    privacy_history = [
        {'epoch': i, 'epsilon': 0.1 * i, 'delta': 1e-5}
        for i in range(1, 11)
    ]
    visualizer.plot_privacy_budget_tracking(privacy_history)

    # Counterfactual
    original = np.random.rand(150)
    counterfactual = original + np.random.randn(150) * 0.2
    visualizer.plot_counterfactual_examples(
        original, counterfactual, feature_names)

    # Phase 3 visualizations
    logger.info("Generating Phase 3 visualizations...")
    metrics_dict = {
        'Archetype Model': {
            'accuracy': 0.92,
            'precision': 0.90,
            'recall': 0.88,
            'f1_score': 0.89,
            'roc_auc': 0.94
        },
        'Synthetic Model': {
            'accuracy': 0.89,
            'precision': 0.87,
            'recall': 0.85,
            'f1_score': 0.86,
            'roc_auc': 0.91
        }
    }
    visualizer.plot_performance_comparison(metrics_dict)

    # XAI fidelity
    fidelity_scores = {
        'shap_correlation': 0.87,
        'rank_agreement': 0.82,
        'top_k_overlap': 0.85,
        'overall_fidelity': 0.84
    }
    visualizer.plot_xai_fidelity_scores(fidelity_scores)

    # Statistical validation
    chi_squared_results = {
        f'Feature_{i}': (np.random.rand() * 10, np.random.rand())
        for i in range(10)
    }
    visualizer.plot_statistical_validation(chi_squared_results)

    logger.info("=== All visualizations generated successfully ===")
    logger.info(f"Output directory: outputs/demo_visualizations")


if __name__ == "__main__":
    main()
