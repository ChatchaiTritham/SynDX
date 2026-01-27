"""
Comprehensive Performance Metrics Calculator for SynDX Framework

Computes all standard classification and validation metrics:
- Confusion Matrix
- Precision, Recall, F1-Score (per-class and weighted)
- ROC-AUC, PR-AUC
- Specificity, Sensitivity
- Matthews Correlation Coefficient (MCC)
- Cohen's Kappa
- Calibration metrics (Brier score, ECE)

Designed for multi-class diagnostic classification tasks.
All metrics follow scikit-learn conventions and Tier 1 journal standards.

Author: PhD Candidate, Computer Science
Institution: [Your Institution]
"""

import numpy as np
import pandas as pd
from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    matthews_corrcoef,
    cohen_kappa_score,
    brier_score_loss,
    log_loss
)
from sklearn.preprocessing import label_binarize
from typing import Dict, List, Tuple, Optional
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import logging
import json

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class PerformanceMetricsCalculator:
    """
    Comprehensive metrics calculator for diagnostic classification.

    Implements all standard evaluation metrics for multi-class
    classification tasks in medical AI systems.
    """

    def __init__(self, class_names: Optional[List[str]] = None):
        """
        Initialize metrics calculator.

        Args:
            class_names: List of class names for labeling
        """
        self.class_names = class_names
        self.metrics = {}

    def compute_all_metrics(self,
                            y_true: np.ndarray,
                            y_pred: np.ndarray,
                            y_pred_proba: Optional[np.ndarray] = None) -> Dict:
        """
        Compute comprehensive performance metrics.

        Args:
            y_true: True labels (n_samples,)
            y_pred: Predicted labels (n_samples,)
            y_pred_proba: Predicted probabilities (n_samples, n_classes)

        Returns:
            Dictionary containing all computed metrics
        """
        logger.info("Computing performance metrics...")

        metrics = {}

        # Basic metrics
        metrics['accuracy'] = accuracy_score(y_true, y_pred)

        # Per-class and averaged metrics
        metrics['precision_macro'] = precision_score(
            y_true, y_pred, average='macro', zero_division=0)
        metrics['precision_weighted'] = precision_score(
            y_true, y_pred, average='weighted', zero_division=0)
        metrics['recall_macro'] = recall_score(
            y_true, y_pred, average='macro', zero_division=0)
        metrics['recall_weighted'] = recall_score(
            y_true, y_pred, average='weighted', zero_division=0)
        metrics['f1_macro'] = f1_score(
            y_true, y_pred, average='macro', zero_division=0)
        metrics['f1_weighted'] = f1_score(
            y_true, y_pred, average='weighted', zero_division=0)

        # Per-class metrics
        precision_per_class = precision_score(
            y_true, y_pred, average=None, zero_division=0)
        recall_per_class = recall_score(
            y_true, y_pred, average=None, zero_division=0)
        f1_per_class = f1_score(y_true, y_pred, average=None, zero_division=0)

        for i, (p, r, f) in enumerate(
                zip(precision_per_class, recall_per_class, f1_per_class)):
            class_name = self.class_names[i] if self.class_names else f'Class_{i}'
            metrics[f'precision_{class_name}'] = p
            metrics[f'recall_{class_name}'] = r
            metrics[f'f1_{class_name}'] = f

        # Confusion matrix
        cm = confusion_matrix(y_true, y_pred)
        metrics['confusion_matrix'] = cm.tolist()

        # Specificity and Sensitivity (per-class)
        specificity_list, sensitivity_list = self._compute_specificity_sensitivity(
            cm)
        metrics['specificity_macro'] = np.mean(specificity_list)
        metrics['sensitivity_macro'] = np.mean(sensitivity_list)

        for i, (spec, sens) in enumerate(
                zip(specificity_list, sensitivity_list)):
            class_name = self.class_names[i] if self.class_names else f'Class_{i}'
            metrics[f'specificity_{class_name}'] = spec
            metrics[f'sensitivity_{class_name}'] = sens

        # Matthews Correlation Coefficient
        metrics['mcc'] = matthews_corrcoef(y_true, y_pred)

        # Cohen's Kappa
        metrics['cohen_kappa'] = cohen_kappa_score(y_true, y_pred)

        # Probability-based metrics (if probabilities provided)
        if y_pred_proba is not None:
            # ROC-AUC (one-vs-rest)
            try:
                n_classes = y_pred_proba.shape[1]
                y_true_bin = label_binarize(y_true, classes=range(n_classes))

                # Macro average
                roc_auc_macro = roc_auc_score(
                    y_true_bin, y_pred_proba, average='macro', multi_class='ovr')
                metrics['roc_auc_macro'] = roc_auc_macro

                # Weighted average
                roc_auc_weighted = roc_auc_score(
                    y_true_bin, y_pred_proba, average='weighted', multi_class='ovr')
                metrics['roc_auc_weighted'] = roc_auc_weighted

                # Per-class ROC-AUC
                for i in range(n_classes):
                    class_name = self.class_names[i] if self.class_names else f'Class_{i}'
                    try:
                        roc_auc_class = roc_auc_score(
                            y_true_bin[:, i], y_pred_proba[:, i])
                        metrics[f'roc_auc_{class_name}'] = roc_auc_class
                    except ValueError:
                        metrics[f'roc_auc_{class_name}'] = np.nan

            except Exception as e:
                logger.warning(f"Could not compute ROC-AUC: {e}")
                metrics['roc_auc_macro'] = np.nan
                metrics['roc_auc_weighted'] = np.nan

            # PR-AUC (Precision-Recall AUC)
            try:
                pr_auc_macro = average_precision_score(
                    y_true_bin, y_pred_proba, average='macro')
                metrics['pr_auc_macro'] = pr_auc_macro

                pr_auc_weighted = average_precision_score(
                    y_true_bin, y_pred_proba, average='weighted')
                metrics['pr_auc_weighted'] = pr_auc_weighted

                # Per-class PR-AUC
                for i in range(n_classes):
                    class_name = self.class_names[i] if self.class_names else f'Class_{i}'
                    try:
                        pr_auc_class = average_precision_score(
                            y_true_bin[:, i], y_pred_proba[:, i])
                        metrics[f'pr_auc_{class_name}'] = pr_auc_class
                    except ValueError:
                        metrics[f'pr_auc_{class_name}'] = np.nan

            except Exception as e:
                logger.warning(f"Could not compute PR-AUC: {e}")
                metrics['pr_auc_macro'] = np.nan
                metrics['pr_auc_weighted'] = np.nan

            # Brier score (calibration)
            try:
                brier_scores = []
                for i in range(n_classes):
                    brier = brier_score_loss(
                        y_true_bin[:, i], y_pred_proba[:, i])
                    brier_scores.append(brier)
                    class_name = self.class_names[i] if self.class_names else f'Class_{i}'
                    metrics[f'brier_score_{class_name}'] = brier

                metrics['brier_score_mean'] = np.mean(brier_scores)
            except Exception as e:
                logger.warning(f"Could not compute Brier score: {e}")
                metrics['brier_score_mean'] = np.nan

            # Log loss
            try:
                logloss = log_loss(y_true, y_pred_proba)
                metrics['log_loss'] = logloss
            except Exception as e:
                logger.warning(f"Could not compute log loss: {e}")
                metrics['log_loss'] = np.nan

            # Expected Calibration Error (ECE)
            try:
                ece = self._compute_ece(y_true, y_pred_proba, n_bins=10)
                metrics['expected_calibration_error'] = ece
            except Exception as e:
                logger.warning(f"Could not compute ECE: {e}")
                metrics['expected_calibration_error'] = np.nan

        self.metrics = metrics
        logger.info("Metrics computation complete.")
        return metrics

    def _compute_specificity_sensitivity(
            self, cm: np.ndarray) -> Tuple[List[float], List[float]]:
        """
        Compute per-class specificity and sensitivity from confusion matrix.

        Args:
            cm: Confusion matrix (n_classes, n_classes)

        Returns:
            Tuple of (specificity_list, sensitivity_list)
        """
        n_classes = cm.shape[0]
        specificity_list = []
        sensitivity_list = []

        for i in range(n_classes):
            # True Positives, False Positives, False Negatives, True Negatives
            tp = cm[i, i]
            fn = np.sum(cm[i, :]) - tp
            fp = np.sum(cm[:, i]) - tp
            tn = np.sum(cm) - tp - fn - fp

            # Sensitivity (Recall) = TP / (TP + FN)
            sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            sensitivity_list.append(sensitivity)

            # Specificity = TN / (TN + FP)
            specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
            specificity_list.append(specificity)

        return specificity_list, sensitivity_list

    def _compute_ece(
            self,
            y_true: np.ndarray,
            y_pred_proba: np.ndarray,
            n_bins: int = 10) -> float:
        """
        Compute Expected Calibration Error (ECE).

        Args:
            y_true: True labels
            y_pred_proba: Predicted probabilities (n_samples, n_classes)
            n_bins: Number of bins for calibration

        Returns:
            Expected Calibration Error
        """
        # Get predicted class and confidence
        y_pred_class = np.argmax(y_pred_proba, axis=1)
        confidences = np.max(y_pred_proba, axis=1)
        accuracies = (y_pred_class == y_true).astype(float)

        # Create bins
        bin_boundaries = np.linspace(0, 1, n_bins + 1)
        ece = 0.0

        for i in range(n_bins):
            # Find samples in this bin
            in_bin = (confidences >= bin_boundaries[i]) & (
                confidences < bin_boundaries[i + 1])

            if np.sum(in_bin) > 0:
                # Average confidence and accuracy in bin
                avg_confidence = np.mean(confidences[in_bin])
                avg_accuracy = np.mean(accuracies[in_bin])
                bin_size = np.sum(in_bin)

                # ECE contribution
                ece += (bin_size / len(y_true)) * \
                    np.abs(avg_confidence - avg_accuracy)

        return ece

    def plot_confusion_matrix(self,
                              save_path: Optional[str] = None,
                              figsize: Tuple[int, int] = (12, 10)) -> None:
        """
        Plot confusion matrix heatmap.

        Args:
            save_path: Path to save figure (optional)
            figsize: Figure size
        """
        if 'confusion_matrix' not in self.metrics:
            raise ValueError(
                "Metrics not computed yet. Call compute_all_metrics() first.")

        cm = np.array(self.metrics['confusion_matrix'])

        fig, axes = plt.subplots(1, 2, figsize=(figsize[0] * 2, figsize[1]))

        # Raw counts
        sns.heatmap(
            cm,
            annot=True,
            fmt='d',
            cmap='Blues',
            ax=axes[0],
            xticklabels=self.class_names if self.class_names else range(
                cm.shape[0]),
            yticklabels=self.class_names if self.class_names else range(
                cm.shape[0]),
            cbar_kws={
                'label': 'Count'})
        axes[0].set_title(
            'Confusion Matrix (Counts)',
            fontsize=16,
            fontweight='bold')
        axes[0].set_xlabel('Predicted Label', fontsize=14)
        axes[0].set_ylabel('True Label', fontsize=14)

        # Normalized (row-wise)
        cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        sns.heatmap(
            cm_normalized,
            annot=True,
            fmt='.2f',
            cmap='Blues',
            ax=axes[1],
            xticklabels=self.class_names if self.class_names else range(
                cm.shape[0]),
            yticklabels=self.class_names if self.class_names else range(
                cm.shape[0]),
            cbar_kws={
                'label': 'Proportion'})
        axes[1].set_title(
            'Confusion Matrix (Normalized)',
            fontsize=16,
            fontweight='bold')
        axes[1].set_xlabel('Predicted Label', fontsize=14)
        axes[1].set_ylabel('True Label', fontsize=14)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=600, bbox_inches='tight')
            plt.savefig(save_path.replace('.png', '.pdf'), bbox_inches='tight')
            logger.info(f"Saved confusion matrix to {save_path}")

        plt.close()

    def plot_per_class_metrics(self,
                               save_path: Optional[str] = None,
                               figsize: Tuple[int, int] = (14, 8)) -> None:
        """
        Plot per-class precision, recall, F1-score.

        Args:
            save_path: Path to save figure (optional)
            figsize: Figure size
        """
        if not self.metrics:
            raise ValueError(
                "Metrics not computed yet. Call compute_all_metrics() first.")

        # Extract per-class metrics
        class_names = self.class_names if self.class_names else [
            f'Class_{i}' for i in range(len(self.metrics['confusion_matrix']))]

        precision_vals = [
            self.metrics.get(
                f'precision_{cn}',
                0) for cn in class_names]
        recall_vals = [
            self.metrics.get(
                f'recall_{cn}',
                0) for cn in class_names]
        f1_vals = [self.metrics.get(f'f1_{cn}', 0) for cn in class_names]

        x = np.arange(len(class_names))
        width = 0.25

        fig, ax = plt.subplots(figsize=figsize)

        ax.bar(
            x - width,
            precision_vals,
            width,
            label='Precision',
            color='steelblue')
        ax.bar(x, recall_vals, width, label='Recall', color='coral')
        ax.bar(x + width, f1_vals, width, label='F1-Score', color='seagreen')

        ax.set_xlabel('Class', fontsize=14)
        ax.set_ylabel('Score', fontsize=14)
        ax.set_title(
            'Per-Class Performance Metrics',
            fontsize=16,
            fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(class_names, rotation=45, ha='right')
        ax.legend(fontsize=12)
        ax.set_ylim([0, 1.0])
        ax.grid(True, alpha=0.3, axis='y')

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=600, bbox_inches='tight')
            plt.savefig(save_path.replace('.png', '.pdf'), bbox_inches='tight')
            logger.info(f"Saved per-class metrics to {save_path}")

        plt.close()

    def save_metrics_to_json(self, save_path: str) -> None:
        """
        Save metrics to JSON file.

        Args:
            save_path: Path to save JSON file
        """
        if not self.metrics:
            raise ValueError(
                "Metrics not computed yet. Call compute_all_metrics() first.")

        # Convert numpy types to Python native types
        metrics_serializable = {}
        for key, value in self.metrics.items():
            if isinstance(value, np.ndarray):
                metrics_serializable[key] = value.tolist()
            elif isinstance(value, (np.int64, np.int32)):
                metrics_serializable[key] = int(value)
            elif isinstance(value, (np.float64, np.float32)):
                metrics_serializable[key] = float(value)
            else:
                metrics_serializable[key] = value

        with open(save_path, 'w') as f:
            json.dump(metrics_serializable, f, indent=2)

        logger.info(f"Saved metrics to {save_path}")

    def generate_latex_table(self, save_path: Optional[str] = None) -> str:
        """
        Generate LaTeX table of key metrics for publication.

        Args:
            save_path: Path to save LaTeX file (optional)

        Returns:
            LaTeX table string
        """
        if not self.metrics:
            raise ValueError(
                "Metrics not computed yet. Call compute_all_metrics() first.")

        latex = "\\begin{table}[htbp]\n"
        latex += "\\centering\n"
        latex += "\\caption{Classification Performance Metrics}\n"
        latex += "\\label{tab:performance_metrics}\n"
        latex += "\\begin{tabular}{lc}\n"
        latex += "\\toprule\n"
        latex += "\\textbf{Metric} & \\textbf{Value} \\\\\n"
        latex += "\\midrule\n"

        # Key metrics
        key_metrics = [
            ('Accuracy', 'accuracy'),
            ('Precision (Macro)', 'precision_macro'),
            ('Precision (Weighted)', 'precision_weighted'),
            ('Recall (Macro)', 'recall_macro'),
            ('Recall (Weighted)', 'recall_weighted'),
            ('F1-Score (Macro)', 'f1_macro'),
            ('F1-Score (Weighted)', 'f1_weighted'),
            ('ROC-AUC (Macro)', 'roc_auc_macro'),
            ('ROC-AUC (Weighted)', 'roc_auc_weighted'),
            ('MCC', 'mcc'),
            ("Cohen's Kappa", 'cohen_kappa'),
        ]

        for metric_name, metric_key in key_metrics:
            if metric_key in self.metrics and not np.isnan(
                    self.metrics[metric_key]):
                value = self.metrics[metric_key]
                latex += f"{metric_name} & {value:.3f} \\\\\n"

        latex += "\\bottomrule\n"
        latex += "\\end{tabular}\n"
        latex += "\\end{table}\n"

        if save_path:
            with open(save_path, 'w') as f:
                f.write(latex)
            logger.info(f"Saved LaTeX table to {save_path}")

        return latex

    def print_summary(self) -> None:
        """
        Print summary of computed metrics to console.
        """
        if not self.metrics:
            raise ValueError(
                "Metrics not computed yet. Call compute_all_metrics() first.")

        print("\n" + "=" * 80)
        print("PERFORMANCE METRICS SUMMARY")
        print("=" * 80)

        print("\n--- Overall Metrics ---")
        print(f"Accuracy:            {self.metrics.get('accuracy', 0):.4f}")
        print(
            f"Precision (Macro):   {
                self.metrics.get(
                    'precision_macro',
                    0):.4f}")
        print(
            f"Recall (Macro):      {
                self.metrics.get(
                    'recall_macro',
                    0):.4f}")
        print(f"F1-Score (Macro):    {self.metrics.get('f1_macro', 0):.4f}")
        print(
            f"ROC-AUC (Macro):     {self.metrics.get('roc_auc_macro', np.nan):.4f}")
        print(f"MCC:                 {self.metrics.get('mcc', 0):.4f}")
        print(f"Cohen's Kappa:       {self.metrics.get('cohen_kappa', 0):.4f}")

        if 'brier_score_mean' in self.metrics:
            print(
                f"Brier Score (Mean):  {
                    self.metrics['brier_score_mean']:.4f}")

        if 'expected_calibration_error' in self.metrics:
            print(
                f"ECE:                 {
                    self.metrics['expected_calibration_error']:.4f}")

        print("\n--- Per-Class Metrics ---")
        if self.class_names:
            for cn in self.class_names:
                print(f"\n{cn}:")
                print(
                    f"  Precision:   {
                        self.metrics.get(
                            f'precision_{cn}',
                            0):.4f}")
                print(
                    f"  Recall:      {
                        self.metrics.get(
                            f'recall_{cn}',
                            0):.4f}")
                print(f"  F1-Score:    {self.metrics.get(f'f1_{cn}', 0):.4f}")
                print(
                    f"  Sensitivity: {
                        self.metrics.get(
                            f'sensitivity_{cn}',
                            0):.4f}")
                print(
                    f"  Specificity: {
                        self.metrics.get(
                            f'specificity_{cn}',
                            0):.4f}")

        print("\n" + "=" * 80 + "\n")


def main():
    """
    Demonstration of performance metrics calculation.
    """
    logger.info("=== SynDX Performance Metrics Calculator ===")

    # Generate sample data
    np.random.seed(42)
    n_samples = 1000
    n_classes = 4

    class_names = ['BPPV', 'Stroke', 'VM', 'VN']

    # Simulate predictions
    y_true = np.random.randint(0, n_classes, n_samples)
    y_pred = y_true.copy()
    # Add some errors
    error_indices = np.random.choice(
        n_samples, size=int(
            n_samples * 0.15), replace=False)
    y_pred[error_indices] = np.random.randint(0, n_classes, len(error_indices))

    # Simulate probabilities
    y_pred_proba = np.random.dirichlet(np.ones(n_classes), size=n_samples)
    # Make probabilities align somewhat with predictions
    for i in range(n_samples):
        y_pred_proba[i, y_pred[i]] += 0.5
    y_pred_proba = y_pred_proba / y_pred_proba.sum(axis=1, keepdims=True)

    # Compute metrics
    calculator = PerformanceMetricsCalculator(class_names=class_names)
    metrics = calculator.compute_all_metrics(y_true, y_pred, y_pred_proba)

    # Print summary
    calculator.print_summary()

    # Save outputs
    output_dir = Path("outputs/performance_metrics")
    output_dir.mkdir(parents=True, exist_ok=True)

    calculator.plot_confusion_matrix(
        save_path=str(
            output_dir /
            "confusion_matrix.png"))
    calculator.plot_per_class_metrics(
        save_path=str(
            output_dir /
            "per_class_metrics.png"))
    calculator.save_metrics_to_json(save_path=str(output_dir / "metrics.json"))
    calculator.generate_latex_table(
        save_path=str(
            output_dir /
            "metrics_table.tex"))

    logger.info("=== Metrics computation complete ===")
    logger.info(f"Outputs saved to: {output_dir}")


if __name__ == "__main__":
    main()
