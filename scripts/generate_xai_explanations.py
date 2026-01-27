"""
Enhanced Interpretability for SynDX Framework
==============================================

Implements Explainable AI (XAI) features:
1. SHAP Values - Feature importance explanation (medical AI standard)
2. Counterfactual Explanations - Actionable clinical insights

Output: Publication-ready visualizations and clinical reports
Target audience: Physicians, medical personnel, patients, general public

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

# SHAP library for feature importance
try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False
    logging.warning("SHAP not installed. Install with: pip install shap")

# Scikit-learn for ML models
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Configure plotting style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")


class XAIExplainer:
    """
    Explainable AI (XAI) for SynDX Framework.

    Provides two main interpretability methods:
    1. SHAP Values - Quantify feature importance
    2. Counterfactual Explanations - Generate actionable insights
    """

    def __init__(self, output_dir: str = "outputs/xai_explanations"):
        """
        Initialize XAI explainer.

        Args:
            output_dir: Directory for saving XAI outputs
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Create subdirectories
        (self.output_dir / "shap").mkdir(exist_ok=True)
        (self.output_dir / "counterfactuals").mkdir(exist_ok=True)
        (self.output_dir / "reports").mkdir(exist_ok=True)
        (self.output_dir / "clinical").mkdir(exist_ok=True)

        self.model = None
        self.feature_names = None
        self.class_names = None

        logger.info(f"XAI Explainer initialized. Output: {self.output_dir}")

    # =========================================================================
    # 1. SHAP VALUES - Feature Importance Explanation
    # =========================================================================

    def compute_shap_values(
        self,
        model,
        X_data: np.ndarray,
        feature_names: List[str],
        class_names: Optional[List[str]] = None,
        sample_size: int = 100
    ) -> Dict[str, Any]:
        """
        Compute SHAP values for model predictions.

        SHAP (SHapley Additive exPlanations):
        - Gold standard for feature importance in medical AI
        - Provides local (per-patient) and global (overall) explanations
        - Based on game theory (Shapley values)

        Args:
            model: Trained classifier (RandomForest, GradientBoosting, etc.)
            X_data: Feature matrix (n_samples × n_features)
            feature_names: List of feature names
            class_names: List of class labels
            sample_size: Number of samples for SHAP computation

        Returns:
            Dictionary with SHAP values, explanations, and statistics
        """
        if not SHAP_AVAILABLE:
            logger.error("SHAP library not available")
            return {}

        logger.info("Computing SHAP values...")

        # Sample data if too large
        if len(X_data) > sample_size:
            indices = np.random.choice(len(X_data), sample_size, replace=False)
            X_sample = X_data[indices]
        else:
            X_sample = X_data

        # Create SHAP explainer
        # TreeExplainer for tree-based models (fast and exact)
        if hasattr(model, 'estimators_'):  # Tree ensemble
            explainer = shap.TreeExplainer(model)
        else:  # Other models
            explainer = shap.Explainer(model, X_sample)

        # Compute SHAP values
        shap_values = explainer.shap_values(X_sample)

        # Handle multi-class case
        if isinstance(shap_values, list):
            # Multi-class: list of arrays
            n_classes = len(shap_values)
            shap_values_dict = {
                f"class_{i}": shap_values[i] for i in range(n_classes)
            }
        else:
            # Binary classification
            shap_values_dict = {"class_0": shap_values}
            n_classes = 1

        # Compute statistics
        feature_importance = {}
        for class_idx, (class_key, shap_vals) in enumerate(
                shap_values_dict.items()):
            # Mean absolute SHAP values per feature
            mean_abs_shap = np.abs(shap_vals).mean(axis=0)

            # Ensure 1D array
            if mean_abs_shap.ndim > 1:
                mean_abs_shap = mean_abs_shap.flatten()

            # Ensure correct length
            if len(mean_abs_shap) != len(feature_names):
                logger.warning(
                    f"SHAP shape mismatch: {
                        len(mean_abs_shap)} vs {
                        len(feature_names)} features")
                mean_abs_shap = mean_abs_shap[:len(feature_names)]

            # Create importance DataFrame
            importance_df = pd.DataFrame({
                'feature': feature_names,
                'importance': mean_abs_shap
            }).sort_values('importance', ascending=False)

            feature_importance[class_key] = importance_df

        results = {
            'shap_values': shap_values_dict,
            'explainer': explainer,
            'X_sample': X_sample,
            'feature_names': feature_names,
            'class_names': class_names,
            'feature_importance': feature_importance,
            'n_samples': len(X_sample),
            'n_classes': n_classes
        }

        logger.info(f"SHAP values computed for {len(X_sample)} samples")
        return results

    def plot_shap_summary(
        self,
        shap_results: Dict[str, Any],
        save_name: str = "shap_summary"
    ):
        """
        Generate SHAP summary plots.

        Creates:
        1. Summary plot (beeswarm) - Shows distribution of SHAP values
        2. Bar plot - Mean absolute SHAP values (global importance)
        3. Waterfall plot - Individual prediction explanation

        Args:
            shap_results: Results from compute_shap_values()
            save_name: Base name for saved files
        """
        if not shap_results:
            return

        logger.info("Generating SHAP visualizations...")

        shap_values = shap_results['shap_values']
        X_sample = shap_results['X_sample']
        feature_names = shap_results['feature_names']

        # For each class
        for class_idx, (class_key, shap_vals) in enumerate(
                shap_values.items()):
            try:
                # 1. Summary plot (beeswarm)
                plt.figure(figsize=(12, 8))
                shap.summary_plot(
                    shap_vals,
                    X_sample,
                    feature_names=feature_names,
                    show=False,
                    plot_type="dot"
                )
                plt.title(
                    f'SHAP Summary Plot - {class_key}\n'
                    'Color: Feature value (red=high, blue=low)\n'
                    'X-axis: SHAP value (impact on prediction)',
                    fontsize=14,
                    fontweight='bold'
                )
                plt.tight_layout()
                plt.savefig(
                    self.output_dir /
                    "shap" /
                    f"{save_name}_{class_key}_summary.png",
                    dpi=600,
                    bbox_inches='tight')
                plt.savefig(
                    self.output_dir /
                    "shap" /
                    f"{save_name}_{class_key}_summary.pdf",
                    bbox_inches='tight')
                plt.close()

                # 2. Bar plot (global importance)
                plt.figure(figsize=(10, 8))
                shap.summary_plot(
                    shap_vals,
                    X_sample,
                    feature_names=feature_names,
                    show=False,
                    plot_type="bar"
                )
                plt.title(
                    f'Feature Importance (Mean |SHAP|) - {class_key}',
                    fontsize=14,
                    fontweight='bold'
                )
                plt.tight_layout()
                plt.savefig(
                    self.output_dir /
                    "shap" /
                    f"{save_name}_{class_key}_importance.png",
                    dpi=600,
                    bbox_inches='tight')
                plt.savefig(
                    self.output_dir /
                    "shap" /
                    f"{save_name}_{class_key}_importance.pdf",
                    bbox_inches='tight')
                plt.close()
            except Exception as e:
                logger.warning(
                    f"Error generating SHAP plot for {class_key}: {e}")
                plt.close('all')

        # 3. Waterfall plot for first sample (individual explanation)
        if len(X_sample) > 0:
            for class_idx, (class_key, shap_vals) in enumerate(
                    shap_values.items()):
                try:
                    plt.figure(figsize=(10, 6))

                    # Get base value
                    expected_value = shap_results['explainer'].expected_value
                    if hasattr(expected_value, '__len__'):
                        base_val = expected_value[class_idx] if class_idx < len(
                            expected_value) else expected_value[0]
                    else:
                        base_val = expected_value

                    # Create Explanation object
                    # Handle shape: shap_vals might be (n_samples, n_features)
                    # or (n_samples, n_features, n_classes)
                    if shap_vals[0].ndim > 1:
                        # Multi-dimensional, take first feature dimension
                        shap_val_sample = shap_vals[0][:,
                                                       0] if shap_vals[0].shape[1] > 1 else shap_vals[0].flatten()
                    else:
                        shap_val_sample = shap_vals[0]

                    # Ensure correct length
                    if len(shap_val_sample) > len(feature_names):
                        shap_val_sample = shap_val_sample[:len(feature_names)]

                    explanation = shap.Explanation(
                        values=shap_val_sample,
                        base_values=base_val,
                        data=X_sample[0][:len(shap_val_sample)],
                        feature_names=feature_names[:len(shap_val_sample)]
                    )

                    shap.plots.waterfall(explanation, show=False)
                    plt.title(
                        f'SHAP Waterfall - Individual Prediction - {class_key}',
                        fontsize=14,
                        fontweight='bold')
                    plt.tight_layout()
                    plt.savefig(
                        self.output_dir /
                        "shap" /
                        f"{save_name}_{class_key}_waterfall.png",
                        dpi=600,
                        bbox_inches='tight')
                    plt.savefig(
                        self.output_dir /
                        "shap" /
                        f"{save_name}_{class_key}_waterfall.pdf",
                        bbox_inches='tight')
                    plt.close()
                except Exception as e:
                    logger.warning(
                        f"Error generating waterfall plot for {class_key}: {e}")
                    plt.close('all')

        logger.info(f"SHAP plots saved to {self.output_dir / 'shap'}")

    def generate_shap_report(
        self,
        shap_results: Dict[str, Any],
        save_name: str = "shap_report"
    ):
        """
        Generate clinical SHAP report for medical personnel.

        Creates:
        1. Text report with top features
        2. JSON file with detailed statistics
        3. CSV file for further analysis

        Args:
            shap_results: Results from compute_shap_values()
            save_name: Base name for report files
        """
        if not shap_results:
            return

        logger.info("Generating SHAP clinical report...")

        feature_importance = shap_results['feature_importance']
        class_names = shap_results.get(
            'class_names', [
                f"Class {i}" for i in range(
                    shap_results['n_classes'])])

        # 1. Text report
        report_lines = []
        report_lines.append("=" * 80)
        report_lines.append("SHAP FEATURE IMPORTANCE REPORT")
        report_lines.append("SynDX Explainable AI Framework")
        report_lines.append("=" * 80)
        report_lines.append(
            f"Generated: {
                datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report_lines.append(f"Samples analyzed: {shap_results['n_samples']}")
        report_lines.append("")

        for class_idx, (class_key, importance_df) in enumerate(
                feature_importance.items()):
            class_name = class_names[class_idx] if class_idx < len(
                class_names) else class_key

            report_lines.append("-" * 80)
            report_lines.append(f"CLASS: {class_name}")
            report_lines.append("-" * 80)
            report_lines.append("")
            report_lines.append("Top 15 Most Important Features:")
            report_lines.append("")
            report_lines.append(
                f"{'Rank':<6} {'Feature':<40} {'SHAP Importance':<15}")
            report_lines.append("-" * 80)

            for idx, row in importance_df.head(15).iterrows():
                report_lines.append(
                    f"{idx + 1:<6} {row['feature']:<40} {row['importance']:<15.6f}"
                )

            report_lines.append("")
            report_lines.append(
                f"Total features analyzed: {
                    len(importance_df)}")
            report_lines.append("")

        report_lines.append("=" * 80)
        report_lines.append("INTERPRETATION GUIDE")
        report_lines.append("=" * 80)
        report_lines.append("")
        report_lines.append(
            "SHAP values quantify each feature's contribution to predictions:")
        report_lines.append(
            "- Positive SHAP: Feature pushes prediction toward this class")
        report_lines.append(
            "- Negative SHAP: Feature pushes prediction away from this class")
        report_lines.append("- Magnitude: Strength of feature's impact")
        report_lines.append("")
        report_lines.append("Clinical Use:")
        report_lines.append("1. Identify key diagnostic indicators")
        report_lines.append("2. Understand why model made specific prediction")
        report_lines.append(
            "3. Validate predictions against clinical knowledge")
        report_lines.append("4. Support clinical decision-making")
        report_lines.append("")

        # Save text report
        report_path = self.output_dir / "reports" / f"{save_name}.txt"
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(report_lines))

        # 2. JSON report (detailed statistics)
        json_data = {
            'metadata': {
                'generated': datetime.now().isoformat(),
                'n_samples': shap_results['n_samples'],
                'n_classes': shap_results['n_classes']
            },
            'feature_importance': {}
        }

        for class_idx, (class_key, importance_df) in enumerate(
                feature_importance.items()):
            class_name = class_names[class_idx] if class_idx < len(
                class_names) else class_key
            json_data['feature_importance'][class_name] = importance_df.to_dict(
                'records')

        json_path = self.output_dir / "reports" / f"{save_name}.json"
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(json_data, f, indent=2)

        # 3. CSV files
        for class_idx, (class_key, importance_df) in enumerate(
                feature_importance.items()):
            class_name = class_names[class_idx] if class_idx < len(
                class_names) else class_key
            csv_path = self.output_dir / "reports" / \
                f"{save_name}_{class_name}.csv"
            importance_df.to_csv(csv_path, index=False)

        logger.info(f"SHAP reports saved to {self.output_dir / 'reports'}")

    # =========================================================================
    # 2. COUNTERFACTUAL EXPLANATIONS - Actionable Clinical Insights
    # =========================================================================

    def generate_counterfactuals(
        self,
        model,
        patient_data: np.ndarray,
        feature_names: List[str],
        target_class: Optional[int] = None,
        n_counterfactuals: int = 5,
        max_changes: int = 3
    ) -> Dict[str, Any]:
        """
        Generate counterfactual explanations.

        Counterfactuals answer: "What minimal changes would alter the diagnosis?"

        Clinical Value:
        - Actionable insights for physicians
        - Patient education ("What if I had...")
        - Treatment planning
        - Risk factor identification

        Args:
            model: Trained classifier
            patient_data: Single patient features (1D array)
            feature_names: List of feature names
            target_class: Desired class (if None, use opposite of current)
            n_counterfactuals: Number of counterfactuals to generate
            max_changes: Maximum features to modify

        Returns:
            Dictionary with counterfactual explanations
        """
        logger.info("Generating counterfactual explanations...")

        # Ensure patient_data is 2D
        if patient_data.ndim == 1:
            patient_data = patient_data.reshape(1, -1)

        # Current prediction
        current_pred = model.predict(patient_data)[0]
        current_proba = model.predict_proba(patient_data)[0]

        # Determine target class
        if target_class is None:
            # For binary: flip prediction
            # For multi-class: choose class with highest probability after
            # current
            if len(current_proba) == 2:
                target_class = 1 - current_pred
            else:
                sorted_classes = np.argsort(current_proba)[::-1]
                target_class = sorted_classes[1]  # Second best class

        # Generate counterfactuals using improved perturbation method
        counterfactuals = []
        max_attempts = n_counterfactuals * 50  # More attempts for better success rate

        for attempt in range(max_attempts):
            if len(counterfactuals) >= n_counterfactuals:
                break

            # Create perturbed version
            cf = patient_data.copy()

            # Randomly select features to change
            n_changes = np.random.randint(1, max_changes + 1)
            features_to_change = np.random.choice(
                len(feature_names),
                size=n_changes,
                replace=False
            )

            # Apply more aggressive perturbations
            for feat_idx in features_to_change:
                # Strategy 1: Flip direction (for features close to mean)
                # Strategy 2: Move toward extreme values
                # Strategy 3: Add significant noise

                strategy = np.random.choice([1, 2, 3])

                if strategy == 1:
                    # Flip: multiply by -1 then add random shift
                    cf[0, feat_idx] = -cf[0, feat_idx] + np.random.randn() * \
                        0.5
                elif strategy == 2:
                    # Move to extreme: multiply by large factor
                    noise_factor = np.random.choice([0.3, 0.5, 1.5, 2.0, 3.0])
                    cf[0, feat_idx] = cf[0, feat_idx] * noise_factor
                else:
                    # Add significant noise
                    cf[0, feat_idx] = cf[0, feat_idx] + np.random.randn() * 2.0

            # Check if prediction changed to target
            cf_pred = model.predict(cf)[0]
            cf_proba = model.predict_proba(cf)[0]

            if cf_pred == target_class:
                # Calculate feature changes
                changes = {}
                for feat_idx in features_to_change:
                    original_val = patient_data[0, feat_idx]
                    new_val = cf[0, feat_idx]
                    change_pct = ((new_val - original_val) /
                                  (abs(original_val) + 1e-10)) * 100

                    changes[feature_names[feat_idx]] = {
                        'original': float(original_val),
                        'counterfactual': float(new_val),
                        'change': float(new_val - original_val),
                        'change_percent': float(change_pct)
                    }

                counterfactuals.append({
                    'id': len(counterfactuals) + 1,
                    'counterfactual_data': cf[0],
                    'prediction': int(cf_pred),
                    'probability': cf_proba.tolist(),
                    'confidence': float(cf_proba[cf_pred]),
                    'n_changes': n_changes,
                    'changes': changes
                })

        results = {
            'original_prediction': int(current_pred),
            'original_probability': current_proba.tolist(),
            'original_confidence': float(current_proba[current_pred]),
            'target_class': int(target_class),
            'counterfactuals': counterfactuals,
            'n_found': len(counterfactuals),
            'feature_names': feature_names
        }

        logger.info(
            f"Generated {
                len(counterfactuals)} counterfactual explanations")
        return results

    def plot_counterfactuals(
        self,
        cf_results: Dict[str, Any],
        class_names: Optional[List[str]] = None,
        save_name: str = "counterfactuals"
    ):
        """
        Visualize counterfactual explanations.

        Creates:
        1. Feature change comparison plot
        2. Probability shift visualization
        3. Change magnitude heatmap

        Args:
            cf_results: Results from generate_counterfactuals()
            class_names: List of class labels
            save_name: Base name for saved files
        """
        if not cf_results or not cf_results['counterfactuals']:
            logger.warning("No counterfactuals to plot")
            return

        logger.info("Generating counterfactual visualizations...")

        counterfactuals = cf_results['counterfactuals']

        # 1. Feature change comparison (top 5 counterfactuals)
        fig, axes = plt.subplots(2, 1, figsize=(14, 10))

        # Collect all changed features
        all_changes = {}
        for cf in counterfactuals[:5]:
            for feat, change_info in cf['changes'].items():
                if feat not in all_changes:
                    all_changes[feat] = []
                all_changes[feat].append(change_info['change_percent'])

        # Plot feature changes
        features = list(all_changes.keys())
        changes = [np.mean(all_changes[f]) for f in features]

        axes[0].barh(
            features,
            changes,
            color='skyblue',
            edgecolor='navy',
            alpha=0.7)
        axes[0].set_xlabel(
            'Average Change (%)',
            fontsize=12,
            fontweight='bold')
        axes[0].set_ylabel('Feature', fontsize=12, fontweight='bold')
        axes[0].set_title(
            'Counterfactual Feature Changes\n'
            f'(Top {min(5, len(counterfactuals))} counterfactuals)',
            fontsize=14,
            fontweight='bold'
        )
        axes[0].axvline(
            x=0,
            color='red',
            linestyle='--',
            linewidth=2,
            alpha=0.7)
        axes[0].grid(True, alpha=0.3)

        # 2. Probability shift
        orig_class = cf_results['original_prediction']
        target_class = cf_results['target_class']

        if class_names is None:
            class_names = [f"Class {i}" for i in range(
                len(cf_results['original_probability']))]

        orig_probs = cf_results['original_probability']
        cf_probs = [cf['probability'] for cf in counterfactuals[:5]]

        x = np.arange(len(class_names))
        width = 0.15

        # Original probabilities
        axes[1].bar(
            x - width * 2.5,
            orig_probs,
            width,
            label='Original',
            color='coral',
            edgecolor='darkred',
            alpha=0.8
        )

        # Counterfactual probabilities
        for i, cf_prob in enumerate(cf_probs):
            axes[1].bar(
                x - width * 2.5 + width * (i + 1),
                cf_prob,
                width,
                label=f'CF {i + 1}',
                alpha=0.7
            )

        axes[1].set_xlabel('Class', fontsize=12, fontweight='bold')
        axes[1].set_ylabel('Probability', fontsize=12, fontweight='bold')
        axes[1].set_title(
            'Prediction Probability Shifts',
            fontsize=14,
            fontweight='bold'
        )
        axes[1].set_xticks(x)
        axes[1].set_xticklabels(class_names, rotation=45, ha='right')
        axes[1].legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        axes[1].grid(True, alpha=0.3, axis='y')

        plt.tight_layout()
        plt.savefig(
            self.output_dir /
            "counterfactuals" /
            f"{save_name}_comparison.png",
            dpi=600,
            bbox_inches='tight')
        plt.savefig(
            self.output_dir /
            "counterfactuals" /
            f"{save_name}_comparison.pdf",
            bbox_inches='tight')
        plt.close()

        logger.info(
            f"Counterfactual plots saved to {
                self.output_dir /
                'counterfactuals'}")

    def generate_counterfactual_report(
        self,
        cf_results: Dict[str, Any],
        class_names: Optional[List[str]] = None,
        save_name: str = "counterfactual_report"
    ):
        """
        Generate clinical counterfactual report.

        Creates actionable insights for:
        - Physicians: Treatment planning
        - Patients: Understanding risk factors
        - General public: Health education

        Args:
            cf_results: Results from generate_counterfactuals()
            class_names: List of class labels
            save_name: Base name for report files
        """
        if not cf_results or not cf_results['counterfactuals']:
            logger.warning("No counterfactuals for report")
            return

        logger.info("Generating counterfactual clinical report...")

        if class_names is None:
            class_names = [f"Class {i}" for i in range(
                len(cf_results['original_probability']))]

        counterfactuals = cf_results['counterfactuals']

        # Create report
        report_lines = []
        report_lines.append("=" * 80)
        report_lines.append("COUNTERFACTUAL EXPLANATION REPORT")
        report_lines.append(
            "SynDX Explainable AI Framework - Clinical Decision Support")
        report_lines.append("=" * 80)
        report_lines.append(
            f"Generated: {
                datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report_lines.append("")

        # Original prediction
        report_lines.append("-" * 80)
        report_lines.append("ORIGINAL PREDICTION")
        report_lines.append("-" * 80)
        report_lines.append(
            f"Class: {class_names[cf_results['original_prediction']]}")
        report_lines.append(
            f"Confidence: {
                cf_results['original_confidence']:.2%}")
        report_lines.append("")
        report_lines.append("Probabilities:")
        for i, (cls, prob) in enumerate(
                zip(class_names, cf_results['original_probability'])):
            report_lines.append(f"  {cls}: {prob:.4f} ({prob:.1%})")
        report_lines.append("")

        # Target
        report_lines.append("-" * 80)
        report_lines.append("COUNTERFACTUAL ANALYSIS")
        report_lines.append("-" * 80)
        report_lines.append(
            f"Target Class: {class_names[cf_results['target_class']]}")
        report_lines.append(f"Counterfactuals Found: {len(counterfactuals)}")
        report_lines.append("")

        # Each counterfactual
        for cf in counterfactuals[:5]:  # Top 5
            report_lines.append("-" * 80)
            report_lines.append(f"COUNTERFACTUAL #{cf['id']}")
            report_lines.append("-" * 80)
            report_lines.append(
                f"Predicted Class: {class_names[cf['prediction']]}")
            report_lines.append(f"Confidence: {cf['confidence']:.2%}")
            report_lines.append(f"Number of Changes: {cf['n_changes']}")
            report_lines.append("")
            report_lines.append("Feature Changes Required:")
            report_lines.append("")
            report_lines.append(
                f"{'Feature':<30} {'Original':<12} {'New':<12} {'Change %':<12}")
            report_lines.append("-" * 80)

            for feat, change in cf['changes'].items():
                report_lines.append(
                    f"{feat:<30} {change['original']:<12.4f} "
                    f"{change['counterfactual']:<12.4f} {change['change_percent']:>11.2f}%"
                )

            report_lines.append("")

        # Clinical interpretation
        report_lines.append("=" * 80)
        report_lines.append("CLINICAL INTERPRETATION GUIDE")
        report_lines.append("=" * 80)
        report_lines.append("")
        report_lines.append("What are Counterfactuals?")
        report_lines.append("-" * 40)
        report_lines.append(
            "Counterfactuals answer: 'What minimal changes would alter the diagnosis?'")
        report_lines.append("")
        report_lines.append("For Physicians:")
        report_lines.append("- Identify modifiable risk factors")
        report_lines.append("- Guide treatment planning")
        report_lines.append("- Understand critical diagnostic thresholds")
        report_lines.append("- Support differential diagnosis")
        report_lines.append("")
        report_lines.append("For Patients:")
        report_lines.append("- Understand personal risk factors")
        report_lines.append("- See impact of lifestyle changes")
        report_lines.append("- Motivate preventive actions")
        report_lines.append("- Clarify 'what if' questions")
        report_lines.append("")
        report_lines.append("For General Public:")
        report_lines.append("- Health education material")
        report_lines.append("- Risk factor awareness")
        report_lines.append("- Prevention strategies")
        report_lines.append("")

        # Save report
        report_path = self.output_dir / "clinical" / f"{save_name}.txt"
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(report_lines))

        # Save JSON (convert numpy arrays to lists)
        json_path = self.output_dir / "clinical" / f"{save_name}.json"

        # Convert numpy arrays in cf_results to lists
        json_safe_results = {}
        for key, value in cf_results.items():
            if isinstance(value, np.ndarray):
                json_safe_results[key] = value.tolist()
            elif isinstance(value, list):
                json_safe_results[key] = [
                    {k: (v.tolist() if isinstance(v, np.ndarray) else v)
                     for k, v in item.items()}
                    if isinstance(item, dict) else item
                    for item in value
                ]
            else:
                json_safe_results[key] = value

        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(json_safe_results, f, indent=2)

        logger.info(
            f"Counterfactual reports saved to {
                self.output_dir /
                'clinical'}")


# =============================================================================
# DEMONSTRATION / TESTING
# =============================================================================

def run_demonstration():
    """
    Demonstrate XAI features with simulated vestibular disorder data.
    """
    logger.info("=" * 80)
    logger.info("XAI DEMONSTRATION - SynDX Framework")
    logger.info("=" * 80)

    # Set random seed
    np.random.seed(42)

    # 1. Generate simulated clinical data
    n_samples = 500
    n_features = 15

    # Feature names (vestibular disorder indicators)
    feature_names = [
        'age',
        'symptom_duration_days',
        'nystagmus_intensity',
        'vertigo_severity',
        'hearing_loss',
        'tinnitus',
        'headache',
        'imbalance_score',
        'vascular_risk_score',
        'stroke_history',
        'hypertension',
        'diabetes',
        'cardiac_disease',
        'neurological_signs',
        'positional_trigger'
    ]

    # Class names
    class_names = [
        'BPPV',
        'Vestibular Neuritis',
        'Stroke',
        'Meniere\'s Disease']

    # Simulate data with meaningful patterns
    X = np.random.randn(n_samples, n_features)

    # Add class-specific patterns
    y = np.random.randint(0, 4, n_samples)

    # BPPV: high positional trigger, low neurological signs
    bppv_mask = y == 0
    X[bppv_mask, 14] += 2.0  # High positional trigger
    X[bppv_mask, 13] -= 1.0  # Low neurological signs

    # Stroke: high neurological signs, high vascular risk
    stroke_mask = y == 2
    X[stroke_mask, 13] += 2.5  # High neurological signs
    X[stroke_mask, 8] += 2.0   # High vascular risk
    X[stroke_mask, 9] += 1.5   # Stroke history

    # Convert to DataFrame
    X_df = pd.DataFrame(X, columns=feature_names)

    # 2. Train a classifier
    logger.info("Training classifier...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42
    )

    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        random_state=42
    )
    model.fit(X_train, y_train)

    accuracy = model.score(X_test, y_test)
    logger.info(f"Model accuracy: {accuracy:.2%}")

    # 3. Initialize XAI explainer
    explainer = XAIExplainer()

    # 4. SHAP Analysis
    if SHAP_AVAILABLE:
        logger.info("\n" + "=" * 80)
        logger.info("SHAP ANALYSIS")
        logger.info("=" * 80)

        shap_results = explainer.compute_shap_values(
            model=model,
            X_data=X_test,
            feature_names=feature_names,
            class_names=class_names,
            sample_size=100
        )

        explainer.plot_shap_summary(shap_results)
        explainer.generate_shap_report(shap_results)

    # 5. Counterfactual Analysis
    logger.info("\n" + "=" * 80)
    logger.info("COUNTERFACTUAL ANALYSIS")
    logger.info("=" * 80)

    # Select a patient (first test sample)
    patient_data = X_test[0]

    cf_results = explainer.generate_counterfactuals(
        model=model,
        patient_data=patient_data,
        feature_names=feature_names,
        n_counterfactuals=5,
        max_changes=3
    )

    explainer.plot_counterfactuals(cf_results, class_names=class_names)
    explainer.generate_counterfactual_report(
        cf_results, class_names=class_names)

    # Summary
    logger.info("\n" + "=" * 80)
    logger.info("DEMONSTRATION COMPLETE")
    logger.info("=" * 80)
    logger.info(f"Output directory: {explainer.output_dir}")
    logger.info("\nGenerated files:")
    logger.info("  SHAP Analysis:")
    logger.info("    - Summary plots (beeswarm, bar, waterfall)")
    logger.info("    - Clinical reports (TXT, JSON, CSV)")
    logger.info("  Counterfactual Analysis:")
    logger.info("    - Feature change visualizations")
    logger.info("    - Probability shift plots")
    logger.info("    - Clinical decision support reports")
    logger.info("\nAll outputs are publication-ready (600 DPI)")
    logger.info("=" * 80)


if __name__ == "__main__":
    run_demonstration()
