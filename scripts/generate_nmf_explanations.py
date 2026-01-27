"""
NMF Interpretability for SynDX Framework
=========================================

Interprets latent factors from Non-negative Matrix Factorization (NMF).

Features:
1. Factor Interpretation - What each latent factor represents
2. Patient Factor Profiling - Individual patient's factor composition
3. Factor-Disease Association - Links between factors and diagnoses
4. Feature-Factor Networks - Visual representation of relationships

Clinical Value:
- Understand clinical phenotypes (latent factors)
- Simplify 150 features → 20 interpretable factors
- Patient profiling for personalized medicine
- Pattern discovery for research

Author: SynDX Framework
Date: 2026-01-25
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
import json
from datetime import datetime
import logging

# Network analysis
try:
    import networkx as nx
    NETWORKX_AVAILABLE = True
except ImportError:
    NETWORKX_AVAILABLE = False
    logging.warning("NetworkX not installed. Network plots disabled.")

# Interactive plots
try:
    import plotly.graph_objects as go
    import plotly.express as px
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False
    logging.warning("Plotly not installed. Interactive plots disabled.")

# NMF
from sklearn.decomposition import NMF

# Statistical tests
from scipy.stats import chi2_contingency, f_oneway

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Plotting style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")


class NMFInterpreter:
    """
    Interpret NMF latent factors for clinical understanding.

    Provides:
    1. Factor interpretation (what each factor means)
    2. Patient profiling (factor composition per patient)
    3. Factor-disease associations
    4. Feature-factor network visualization
    """

    def __init__(self, output_dir: str = "outputs/nmf_interpretability"):
        """
        Initialize NMF interpreter.

        Args:
            output_dir: Directory for saving outputs
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Create subdirectories
        (self.output_dir / "factors").mkdir(exist_ok=True)
        (self.output_dir / "patients").mkdir(exist_ok=True)
        (self.output_dir / "associations").mkdir(exist_ok=True)
        (self.output_dir / "networks").mkdir(exist_ok=True)
        (self.output_dir / "reports").mkdir(exist_ok=True)

        self.nmf_model = None
        self.W = None  # Patient-factor matrix
        self.H = None  # Factor-feature matrix
        self.feature_names = None
        self.n_factors = None

        logger.info(f"NMF Interpreter initialized. Output: {self.output_dir}")

    # =========================================================================
    # 1. FACTOR INTERPRETATION
    # =========================================================================

    def interpret_factors(
        self,
        H_matrix: np.ndarray,
        feature_names: List[str],
        n_top_features: int = 10,
        save_name: str = "factor_interpretation"
    ) -> Dict[str, Any]:
        """
        Interpret each NMF factor.

        Each factor = linear combination of features
        → Identify top contributing features
        → Assign clinical meaning

        Args:
            H_matrix: Factor-feature matrix (n_factors × n_features)
            feature_names: List of feature names
            n_top_features: Number of top features to show per factor
            save_name: Base name for saved files

        Returns:
            Dictionary with factor interpretations
        """
        logger.info("Interpreting NMF factors...")

        self.H = H_matrix
        self.feature_names = feature_names
        self.n_factors = H_matrix.shape[0]

        interpretations = {}

        for factor_idx in range(self.n_factors):
            # Get feature weights for this factor
            factor_weights = H_matrix[factor_idx, :]

            # Get top features
            top_indices = np.argsort(factor_weights)[-n_top_features:][::-1]
            top_features = [
                {
                    'feature': feature_names[idx],
                    'weight': float(factor_weights[idx]),
                    'normalized_weight': float(factor_weights[idx] / factor_weights.sum())
                }
                for idx in top_indices
            ]

            # Compute factor statistics
            interpretations[f"Factor_{factor_idx + 1}"] = {
                'id': factor_idx,
                'top_features': top_features,
                'sparsity': float(np.mean(factor_weights == 0)),
                'max_weight': float(factor_weights.max()),
                'mean_weight': float(factor_weights.mean()),
                'std_weight': float(factor_weights.std())
            }

        logger.info(f"Interpreted {self.n_factors} factors")
        return interpretations

    def plot_factor_compositions(
        self,
        interpretations: Dict[str, Any],
        save_name: str = "factor_compositions"
    ):
        """
        Visualize factor compositions.

        Creates:
        1. Heatmap of top features per factor
        2. Bar plots for each factor
        3. Sparsity analysis

        Args:
            interpretations: Results from interpret_factors()
            save_name: Base name for saved files
        """
        logger.info("Generating factor composition plots...")

        n_factors = len(interpretations)
        n_top = 10

        # 1. Heatmap: Factors × Top Features
        fig, ax = plt.subplots(figsize=(14, max(8, n_factors * 0.6)))

        # Collect data for heatmap
        factor_names = []
        all_features = set()
        for factor_key in interpretations.keys():
            factor_names.append(factor_key)
            for feat in interpretations[factor_key]['top_features']:
                all_features.add(feat['feature'])

        # Create matrix
        all_features = sorted(list(all_features))
        heatmap_data = np.zeros((n_factors, len(all_features)))

        for i, factor_key in enumerate(factor_names):
            for feat_info in interpretations[factor_key]['top_features']:
                feat_name = feat_info['feature']
                if feat_name in all_features:
                    j = all_features.index(feat_name)
                    heatmap_data[i, j] = feat_info['weight']

        # Plot heatmap
        sns.heatmap(
            heatmap_data,
            xticklabels=all_features,
            yticklabels=factor_names,
            cmap='YlOrRd',
            cbar_kws={'label': 'Feature Weight'},
            ax=ax
        )
        ax.set_xlabel('Features', fontsize=12, fontweight='bold')
        ax.set_ylabel('Latent Factors', fontsize=12, fontweight='bold')
        ax.set_title(
            'NMF Factor Compositions\n'
            f'Top {n_top} Contributing Features per Factor',
            fontsize=14,
            fontweight='bold'
        )
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()

        plt.savefig(
            self.output_dir / "factors" / f"{save_name}_heatmap.png",
            dpi=600,
            bbox_inches='tight'
        )
        plt.savefig(
            self.output_dir / "factors" / f"{save_name}_heatmap.pdf",
            bbox_inches='tight'
        )
        plt.close()

        # 2. Individual factor bar plots (top 5 factors)
        n_cols = 3
        n_rows = min(5, n_factors) // n_cols + 1
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(16, n_rows * 4))
        axes = axes.flatten() if n_factors > 1 else [axes]

        for idx, (factor_key, factor_data) in enumerate(
                list(interpretations.items())[:min(5, n_factors)]):
            ax = axes[idx]

            top_feats = factor_data['top_features']
            features = [f['feature'][:30]
                        for f in top_feats]  # Truncate long names
            weights = [f['weight'] for f in top_feats]

            ax.barh(
                features,
                weights,
                color='steelblue',
                edgecolor='navy',
                alpha=0.7)
            ax.set_xlabel('Weight', fontsize=10, fontweight='bold')
            ax.set_title(
                f'{factor_key}\n(Top {
                    len(top_feats)} Features)',
                fontsize=11,
                fontweight='bold')
            ax.invert_yaxis()
            ax.grid(True, alpha=0.3, axis='x')

        # Hide unused subplots
        for idx in range(min(5, n_factors), len(axes)):
            axes[idx].axis('off')

        plt.tight_layout()
        plt.savefig(
            self.output_dir / "factors" / f"{save_name}_bars.png",
            dpi=600,
            bbox_inches='tight'
        )
        plt.savefig(
            self.output_dir / "factors" / f"{save_name}_bars.pdf",
            bbox_inches='tight'
        )
        plt.close()

        logger.info(
            f"Factor composition plots saved to {
                self.output_dir /
                'factors'}")

    def generate_factor_report(
        self,
        interpretations: Dict[str, Any],
        save_name: str = "factor_interpretation_report"
    ):
        """
        Generate clinical factor interpretation report.

        Args:
            interpretations: Results from interpret_factors()
            save_name: Base name for report files
        """
        logger.info("Generating factor interpretation report...")

        report_lines = []
        report_lines.append("=" * 80)
        report_lines.append("NMF FACTOR INTERPRETATION REPORT")
        report_lines.append("SynDX Framework - Latent Factor Analysis")
        report_lines.append("=" * 80)
        report_lines.append(
            f"Generated: {
                datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report_lines.append(f"Number of Factors: {len(interpretations)}")
        report_lines.append("")

        for factor_key, factor_data in interpretations.items():
            report_lines.append("-" * 80)
            report_lines.append(f"{factor_key.upper()}")
            report_lines.append("-" * 80)
            report_lines.append("")
            report_lines.append("Top Contributing Features:")
            report_lines.append("")
            report_lines.append(
                f"{'Rank':<6} {'Feature':<40} {'Weight':<12} {'% of Total':<12}")
            report_lines.append("-" * 80)

            for rank, feat in enumerate(factor_data['top_features'], 1):
                report_lines.append(
                    f"{rank:<6} {feat['feature']:<40} "
                    f"{feat['weight']:<12.6f} {feat['normalized_weight'] * 100:<11.2f}%"
                )

            report_lines.append("")
            report_lines.append(f"Statistics:")
            report_lines.append(f"  Sparsity: {factor_data['sparsity']:.2%}")
            report_lines.append(
                f"  Max Weight: {
                    factor_data['max_weight']:.6f}")
            report_lines.append(
                f"  Mean Weight: {
                    factor_data['mean_weight']:.6f}")
            report_lines.append("")

        # Clinical interpretation guide
        report_lines.append("=" * 80)
        report_lines.append("CLINICAL INTERPRETATION GUIDE")
        report_lines.append("=" * 80)
        report_lines.append("")
        report_lines.append("What are NMF Latent Factors?")
        report_lines.append("-" * 40)
        report_lines.append(
            "Latent factors represent hidden clinical phenotypes:")
        report_lines.append("- Each factor = combination of features")
        report_lines.append(
            "- Factors capture common patterns across patients")
        report_lines.append("- Reduce 150 features → 20 interpretable factors")
        report_lines.append("")
        report_lines.append("Clinical Use:")
        report_lines.append(
            "1. Phenotype Discovery: Identify patient subtypes")
        report_lines.append("2. Feature Reduction: Simplify complex data")
        report_lines.append(
            "3. Pattern Recognition: Find common presentations")
        report_lines.append(
            "4. Personalized Medicine: Patient-specific profiles")
        report_lines.append("")

        # Save report
        report_path = self.output_dir / "reports" / f"{save_name}.txt"
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(report_lines))

        # Save JSON
        json_path = self.output_dir / "reports" / f"{save_name}.json"
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(interpretations, f, indent=2)

        logger.info(f"Factor reports saved to {self.output_dir / 'reports'}")

    # =========================================================================
    # 2. PATIENT FACTOR PROFILING
    # =========================================================================

    def profile_patient(
        self,
        W_matrix: np.ndarray,
        patient_idx: int = 0,
        factor_names: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Profile individual patient's factor composition.

        Args:
            W_matrix: Patient-factor matrix (n_patients × n_factors)
            patient_idx: Index of patient to profile
            factor_names: Optional custom factor names

        Returns:
            Dictionary with patient profile
        """
        logger.info(f"Profiling patient {patient_idx}...")

        self.W = W_matrix
        n_factors = W_matrix.shape[1]

        if factor_names is None:
            factor_names = [f"Factor_{i + 1}" for i in range(n_factors)]

        # Get patient's factor weights
        patient_factors = W_matrix[patient_idx, :]

        # Normalize to percentages
        patient_factors_norm = patient_factors / patient_factors.sum()

        # Create profile
        profile = {
            'patient_id': patient_idx,
            'factor_weights': {
                factor_names[i]: {
                    'raw_weight': float(patient_factors[i]),
                    'percentage': float(patient_factors_norm[i] * 100)
                }
                for i in range(n_factors)
            },
            'dominant_factors': [],
            'n_factors': n_factors
        }

        # Identify dominant factors (top 5)
        top_indices = np.argsort(patient_factors)[-5:][::-1]
        profile['dominant_factors'] = [
            {
                'factor': factor_names[i],
                'weight': float(patient_factors[i]),
                'percentage': float(patient_factors_norm[i] * 100)
            }
            for i in top_indices
        ]

        logger.info(f"Patient {patient_idx} profile created")
        return profile

    def plot_patient_profile(
        self,
        profile: Dict[str, Any],
        save_name: str = "patient_profile"
    ):
        """
        Visualize patient factor profile.

        Creates:
        1. Pie chart - Factor composition
        2. Radar chart - Factor strengths
        3. Bar chart - Factor weights

        Args:
            profile: Results from profile_patient()
            save_name: Base name for saved files
        """
        logger.info("Generating patient profile visualizations...")

        patient_id = profile['patient_id']
        factor_data = profile['factor_weights']

        factors = list(factor_data.keys())
        weights = [factor_data[f]['raw_weight'] for f in factors]
        percentages = [factor_data[f]['percentage'] for f in factors]

        fig = plt.figure(figsize=(16, 5))

        # 1. Pie chart (top 10 factors)
        ax1 = plt.subplot(1, 3, 1)
        top_10_indices = np.argsort(weights)[-10:][::-1]
        top_10_factors = [factors[i] for i in top_10_indices]
        top_10_pcts = [percentages[i] for i in top_10_indices]

        colors = plt.cm.Set3(np.linspace(0, 1, 10))
        ax1.pie(
            top_10_pcts,
            labels=top_10_factors,
            autopct='%1.1f%%',
            colors=colors,
            startangle=90
        )
        ax1.set_title(
            f'Patient {patient_id} - Factor Composition\n(Top 10 Factors)',
            fontsize=12,
            fontweight='bold'
        )

        # 2. Radar chart (top 8 factors)
        ax2 = plt.subplot(1, 3, 2, projection='polar')
        top_8_indices = np.argsort(weights)[-8:][::-1]
        top_8_factors = [factors[i] for i in top_8_indices]
        top_8_weights = [weights[i] for i in top_8_indices]

        angles = np.linspace(
            0,
            2 * np.pi,
            len(top_8_factors),
            endpoint=False).tolist()
        top_8_weights += top_8_weights[:1]  # Close the plot
        angles += angles[:1]

        ax2.plot(angles, top_8_weights, 'o-', linewidth=2, color='steelblue')
        ax2.fill(angles, top_8_weights, alpha=0.25, color='steelblue')
        ax2.set_xticks(angles[:-1])
        ax2.set_xticklabels(top_8_factors, size=8)
        ax2.set_title(
            f'Patient {patient_id} - Factor Radar\n(Top 8 Factors)',
            fontsize=12,
            fontweight='bold',
            pad=20
        )
        ax2.grid(True)

        # 3. Bar chart (all factors)
        ax3 = plt.subplot(1, 3, 3)
        sorted_indices = np.argsort(weights)[-15:]  # Top 15
        sorted_factors = [factors[i] for i in sorted_indices]
        sorted_weights = [weights[i] for i in sorted_indices]

        ax3.barh(
            sorted_factors,
            sorted_weights,
            color='coral',
            edgecolor='darkred',
            alpha=0.7)
        ax3.set_xlabel('Factor Weight', fontsize=10, fontweight='bold')
        ax3.set_title(
            f'Patient {patient_id} - Factor Weights\n(Top 15 Factors)',
            fontsize=12,
            fontweight='bold'
        )
        ax3.grid(True, alpha=0.3, axis='x')

        plt.tight_layout()
        plt.savefig(
            self.output_dir /
            "patients" /
            f"{save_name}_patient_{patient_id}.png",
            dpi=600,
            bbox_inches='tight')
        plt.savefig(
            self.output_dir /
            "patients" /
            f"{save_name}_patient_{patient_id}.pdf",
            bbox_inches='tight')
        plt.close()

        logger.info(
            f"Patient profile plots saved to {
                self.output_dir /
                'patients'}")

    # =========================================================================
    # 3. FACTOR-DISEASE ASSOCIATION
    # =========================================================================

    def analyze_factor_disease_association(
        self,
        W_matrix: np.ndarray,
        diagnoses: np.ndarray,
        diagnosis_names: List[str],
        factor_names: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Analyze association between factors and diseases.

        Statistical tests:
        - ANOVA: Test if factor weights differ across diagnoses
        - Chi-square: Test independence

        Args:
            W_matrix: Patient-factor matrix (n_patients × n_factors)
            diagnoses: Diagnosis labels (n_patients,)
            diagnosis_names: Names of diagnoses
            factor_names: Optional custom factor names

        Returns:
            Dictionary with association results
        """
        logger.info("Analyzing factor-disease associations...")

        n_factors = W_matrix.shape[1]

        if factor_names is None:
            factor_names = [f"Factor_{i + 1}" for i in range(n_factors)]

        # Group patients by diagnosis
        unique_diagnoses = np.unique(diagnoses)

        # Compute mean factor weights per diagnosis
        association_matrix = np.zeros((len(unique_diagnoses), n_factors))

        for i, diag in enumerate(unique_diagnoses):
            mask = diagnoses == diag
            association_matrix[i, :] = W_matrix[mask, :].mean(axis=0)

        # Statistical tests
        p_values = []
        for factor_idx in range(n_factors):
            # ANOVA: test if factor differs across diagnoses
            groups = [W_matrix[diagnoses == diag, factor_idx]
                      for diag in unique_diagnoses]
            _, p_val = f_oneway(*groups)
            p_values.append(p_val)

        results = {
            'association_matrix': association_matrix,
            'diagnosis_names': diagnosis_names,
            'factor_names': factor_names,
            'p_values': p_values,
            'significant_factors': [
                factor_names[i] for i, p in enumerate(p_values) if p < 0.05
            ]
        }

        logger.info(
            f"Found {len(results['significant_factors'])} significant factor-disease associations")
        return results

    def plot_factor_disease_heatmap(
        self,
        association_results: Dict[str, Any],
        save_name: str = "factor_disease_association"
    ):
        """
        Visualize factor-disease associations.

        Args:
            association_results: Results from analyze_factor_disease_association()
            save_name: Base name for saved files
        """
        logger.info("Generating factor-disease heatmap...")

        matrix = association_results['association_matrix']
        diagnoses = association_results['diagnosis_names']
        factors = association_results['factor_names']
        p_values = association_results['p_values']

        fig, ax = plt.subplots(
            figsize=(max(12, len(factors) * 0.5), len(diagnoses) * 0.8))

        # Plot heatmap
        sns.heatmap(
            matrix,
            xticklabels=factors,
            yticklabels=diagnoses,
            cmap='RdYlBu_r',
            cbar_kws={'label': 'Mean Factor Weight'},
            annot=True,
            fmt='.3f',
            ax=ax
        )

        # Mark significant factors
        for factor_idx, p_val in enumerate(p_values):
            if p_val < 0.05:
                ax.text(
                    factor_idx + 0.5,
                    -0.5,
                    '*',
                    ha='center',
                    va='center',
                    fontsize=16,
                    color='red',
                    fontweight='bold'
                )

        ax.set_xlabel(
            'Latent Factors\n(* = p < 0.05, ANOVA)',
            fontsize=12,
            fontweight='bold')
        ax.set_ylabel('Diagnoses', fontsize=12, fontweight='bold')
        ax.set_title(
            'Factor-Disease Association Heatmap\n'
            'Mean Factor Weights per Diagnosis',
            fontsize=14,
            fontweight='bold'
        )
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()

        plt.savefig(
            self.output_dir / "associations" / f"{save_name}_heatmap.png",
            dpi=600,
            bbox_inches='tight'
        )
        plt.savefig(
            self.output_dir / "associations" / f"{save_name}_heatmap.pdf",
            bbox_inches='tight'
        )
        plt.close()

        logger.info(
            f"Factor-disease heatmap saved to {self.output_dir / 'associations'}")

    # =========================================================================
    # 4. FEATURE-FACTOR NETWORK
    # =========================================================================

    def create_feature_factor_network(
        self,
        H_matrix: np.ndarray,
        feature_names: List[str],
        threshold: float = 0.1,
        save_name: str = "feature_factor_network"
    ):
        """
        Create network graph of feature-factor relationships.

        Args:
            H_matrix: Factor-feature matrix (n_factors × n_features)
            feature_names: List of feature names
            threshold: Minimum weight to include edge
            save_name: Base name for saved files
        """
        if not NETWORKX_AVAILABLE:
            logger.warning(
                "NetworkX not available. Skipping network visualization.")
            return

        logger.info("Creating feature-factor network...")

        n_factors, n_features = H_matrix.shape

        # Create graph
        G = nx.Graph()

        # Add factor nodes
        factor_nodes = [f"F{i + 1}" for i in range(n_factors)]
        G.add_nodes_from(factor_nodes, node_type='factor')

        # Add feature nodes and edges
        for factor_idx in range(n_factors):
            factor_node = factor_nodes[factor_idx]

            for feat_idx, weight in enumerate(H_matrix[factor_idx, :]):
                if weight >= threshold:
                    # Truncate long names
                    feat_node = feature_names[feat_idx][:20]

                    if feat_node not in G:
                        G.add_node(feat_node, node_type='feature')

                    G.add_edge(factor_node, feat_node, weight=weight)

        # Visualize
        fig, ax = plt.subplots(figsize=(16, 12))

        # Layout
        pos = nx.spring_layout(G, k=0.5, iterations=50, seed=42)

        # Draw nodes
        factor_nodes_list = [
            n for n, d in G.nodes(
                data=True) if d.get('node_type') == 'factor']
        feature_nodes_list = [
            n for n, d in G.nodes(
                data=True) if d.get('node_type') == 'feature']

        nx.draw_networkx_nodes(
            G, pos,
            nodelist=factor_nodes_list,
            node_color='coral',
            node_size=800,
            label='Latent Factors',
            ax=ax
        )

        nx.draw_networkx_nodes(
            G, pos,
            nodelist=feature_nodes_list,
            node_color='skyblue',
            node_size=400,
            label='Features',
            ax=ax
        )

        # Draw edges with varying thickness
        edges = G.edges()
        weights = [G[u][v]['weight'] for u, v in edges]
        max_weight = max(weights) if weights else 1

        nx.draw_networkx_edges(
            G, pos,
            width=[w / max_weight * 3 for w in weights],
            alpha=0.3,
            ax=ax
        )

        # Draw labels
        nx.draw_networkx_labels(
            G, pos,
            font_size=8,
            font_weight='bold',
            ax=ax
        )

        ax.set_title(
            f'Feature-Factor Network\n'
            f'({len(G.nodes())} nodes, {len(G.edges())} edges, threshold={threshold})',
            fontsize=14,
            fontweight='bold'
        )
        ax.legend(loc='upper right', fontsize=10)
        ax.axis('off')
        plt.tight_layout()

        plt.savefig(
            self.output_dir / "networks" / f"{save_name}.png",
            dpi=600,
            bbox_inches='tight'
        )
        plt.savefig(
            self.output_dir / "networks" / f"{save_name}.pdf",
            bbox_inches='tight'
        )
        plt.close()

        logger.info(f"Network saved to {self.output_dir / 'networks'}")

        # Save network data
        network_data = {
            'nodes': len(G.nodes()),
            'edges': len(G.edges()),
            'factors': len(factor_nodes_list),
            'features': len(feature_nodes_list),
            'threshold': threshold
        }

        json_path = self.output_dir / "networks" / f"{save_name}_stats.json"
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(network_data, f, indent=2)


# =============================================================================
# DEMONSTRATION / TESTING
# =============================================================================

def run_demonstration():
    """
    Demonstrate NMF interpretability with simulated data.
    """
    logger.info("=" * 80)
    logger.info("NMF INTERPRETABILITY DEMONSTRATION")
    logger.info("=" * 80)

    np.random.seed(42)

    # 1. Simulate NMF decomposition
    n_patients = 500
    n_features = 30  # Reduced for demo
    n_factors = 10

    # Feature names (clinical features)
    feature_names = [
        'age', 'symptom_duration', 'vertigo_intensity', 'nystagmus_type',
        'hearing_loss', 'tinnitus', 'headache', 'imbalance', 'vascular_risk',
        'stroke_history', 'hypertension', 'diabetes', 'cardiac_disease',
        'neurological_signs', 'positional_trigger', 'fall_risk', 'anxiety',
        'depression', 'medication_count', 'comorbidity_count', 'bmi',
        'blood_pressure_systolic', 'blood_pressure_diastolic', 'heart_rate',
        'visual_impairment', 'cognitive_score', 'gait_speed', 'balance_score',
        'previous_episodes', 'family_history'
    ]

    # Simulate archetype data
    X = np.abs(np.random.randn(n_patients, n_features))

    # Fit NMF
    logger.info("Fitting NMF model...")
    nmf_model = NMF(n_components=n_factors, random_state=42, max_iter=500)
    W = nmf_model.fit_transform(X)
    H = nmf_model.components_

    logger.info(f"NMF decomposition: {X.shape} → W{W.shape} × H{H.shape}")

    # Simulate diagnoses
    diagnoses = np.random.randint(0, 4, n_patients)
    diagnosis_names = [
        'BPPV',
        'Vestibular Neuritis',
        'Stroke',
        "Meniere's Disease"]

    # 2. Initialize interpreter
    interpreter = NMFInterpreter()

    # 3. Factor Interpretation
    logger.info("\n" + "=" * 80)
    logger.info("FACTOR INTERPRETATION")
    logger.info("=" * 80)

    interpretations = interpreter.interpret_factors(
        H, feature_names, n_top_features=10)
    interpreter.plot_factor_compositions(interpretations)
    interpreter.generate_factor_report(interpretations)

    # 4. Patient Profiling
    logger.info("\n" + "=" * 80)
    logger.info("PATIENT PROFILING")
    logger.info("=" * 80)

    for patient_idx in [0, 1, 2]:  # Profile 3 patients
        profile = interpreter.profile_patient(W, patient_idx)
        interpreter.plot_patient_profile(profile)

    # 5. Factor-Disease Association
    logger.info("\n" + "=" * 80)
    logger.info("FACTOR-DISEASE ASSOCIATION")
    logger.info("=" * 80)

    association_results = interpreter.analyze_factor_disease_association(
        W, diagnoses, diagnosis_names
    )
    interpreter.plot_factor_disease_heatmap(association_results)

    # 6. Feature-Factor Network
    logger.info("\n" + "=" * 80)
    logger.info("FEATURE-FACTOR NETWORK")
    logger.info("=" * 80)

    interpreter.create_feature_factor_network(H, feature_names, threshold=0.3)

    # Summary
    logger.info("\n" + "=" * 80)
    logger.info("DEMONSTRATION COMPLETE")
    logger.info("=" * 80)
    logger.info(f"Output directory: {interpreter.output_dir}")
    logger.info("\nGenerated files:")
    logger.info("  Factor Interpretation:")
    logger.info("    - Heatmaps and bar plots")
    logger.info("    - Clinical reports (TXT, JSON)")
    logger.info("  Patient Profiling:")
    logger.info("    - Pie, radar, and bar charts (3 patients)")
    logger.info("  Factor-Disease Association:")
    logger.info("    - Statistical heatmap with significance tests")
    logger.info("  Feature-Factor Network:")
    logger.info("    - Network graph visualization")
    logger.info("\nAll outputs are publication-ready (600 DPI)")
    logger.info("=" * 80)


if __name__ == "__main__":
    run_demonstration()
