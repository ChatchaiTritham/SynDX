"""
Comprehensive Evaluation Metrics

This is the kitchen sink of validation - every metric we can think of to
prove the synthetic data is good enough for training ML models.

Phase 4 metrics (Sub-Phases 4.1, 4.2, 4.3):

Statistical Realism (4.1):
- KL Divergence < 0.05 - synthetic matches archetype distribution
- Feature correlation analysis - relationships preserved

Predictive Performance (4.2):
- ROC-AUC > 0.90 - models trained on synthetic data actually work
- Precision, Recall, F1 - the usual suspects

Clinical Impact (4.3):
- Decision Curve Analysis - does this help clinical decisions?
- Net Benefit - quantifies actual utility
- CT Scan Reduction Rate - can we reduce unnecessary imaging?

Built by: Chatchai Tritham & Chakkrit Snae Namahoot
Where: Naresuan University, Thailand
"""

import numpy as np
import pandas as pd
from scipy.stats import entropy
from sklearn.metrics import roc_auc_score, roc_curve, precision_recall_fscore_support
from typing import Dict


class EvaluationMetrics:
    """
    Evaluation Metrics Calculator for SynDX Framework.

    Implements Sub-Phases 4.1, 4.2, 4.3:
    - Statistical realism (KL divergence)
    - Predictive performance (ROC-AUC, precision, recall)
    - Clinical impact (DCA, resource utilization)
    """
    def __init__(self):
        pass

    def calculate_kl_divergence(self, synthetic_df, expert_target_df, key_features=None):
        """
        Calculates the average KL Divergence between distributions of key features
        in synthetic data and expert-defined target distributions.
        This is a conceptual implementation for demonstration.

        Args:
            synthetic_df (pd.DataFrame): DataFrame of synthetic data.
            expert_target_df (pd.DataFrame): DataFrame representing expert-defined target distributions
                                             (e.g., historical distributions, clinical prevalence rates).
            key_features (list, optional): List of feature names to compare. If None,
                                           it tries to compare common columns.
        Returns:
            float: Average KL Divergence across specified features.
        """
        if key_features is None:
            # Attempt to find common numerical columns for comparison
            common_cols = list(set(synthetic_df.columns) & set(expert_target_df.columns))
            numerical_cols = [col for col in common_cols if pd.api.types.is_numeric_dtype(synthetic_df[col]) and pd.api.types.is_numeric_dtype(expert_target_df[col])]
            if not numerical_cols:
                print("Warning: No common numerical features found for KL divergence. Returning NaN.")
                return np.nan
            key_features = numerical_cols
        
        kl_divergences = []
        for feature in key_features:
            if feature in synthetic_df.columns and feature in expert_target_df.columns:
                # For numerical features, discretize and calculate probability distributions
                if pd.api.types.is_numeric_dtype(synthetic_df[feature]):
                    # Create bins that cover the range of both synthetic and target data
                    min_val = min(synthetic_df[feature].min(), expert_target_df[feature].min())
                    max_val = max(synthetic_df[feature].max(), expert_target_df[feature].max())
                    bins = np.linspace(min_val, max_val, num=20) # 20 bins for discretization

                    p_synth, _ = np.histogram(synthetic_df[feature], bins=bins, density=True)
                    q_target, _ = np.histogram(expert_target_df[feature], bins=bins, density=True)

                    # Add a small epsilon to avoid log(0)
                    p_synth = p_synth + 1e-10
                    q_target = q_target + 1e-10

                    kl_div = entropy(p_synth, q_target)
                    kl_divergences.append(kl_div)
                elif pd.api.types.is_categorical_dtype(synthetic_df[feature]) or pd.api.types.is_object_dtype(synthetic_df[feature]):
                    # For categorical features, calculate probability mass functions
                    p_synth = synthetic_df[feature].value_counts(normalize=True)
                    q_target = expert_target_df[feature].value_counts(normalize=True)
                    
                    # Align indices and fill missing categories with 0
                    common_index = p_synth.index.union(q_target.index)
                    p_synth = p_synth.reindex(common_index, fill_value=0) + 1e-10
                    q_target = q_target.reindex(common_index, fill_value=0) + 1e-10
                    
                    kl_div = entropy(p_synth, q_target)
                    kl_divergences.append(kl_div)
            else:
                print(f"Warning: Feature '{feature}' not found in both dataframes for KL divergence.")

        if not kl_divergences:
            return np.nan # Return NaN if no features could be compared
        return np.mean(kl_divergences)

    def calculate_roc_auc(self, y_true, y_pred_proba):
        """
        Calculates the Receiver Operating Characteristic Area Under Curve (ROC-AUC).
        
        Args:
            y_true (array-like): True binary labels.
            y_pred_proba (array-like): Predicted probabilities for the positive class.
        Returns:
            float: ROC-AUC score.
        """
        try:
            # Ensure y_true contains at least two classes
            if len(np.unique(y_true)) < 2:
                print("Warning: y_true contains fewer than 2 unique classes. ROC-AUC is not well-defined. Returning NaN.")
                return np.nan
            return roc_auc_score(y_true, y_pred_proba)
        except ValueError as e:
            print(f"Error calculating ROC-AUC: {e}. Ensure y_true and y_pred_proba are valid for ROC-AUC calculation.")
            return np.nan 

    def calculate_precision_recall_f1(self, y_true, y_pred, average='weighted'):
        """
        Calculates Precision, Recall, and F1-score.
        
        Args:
            y_true (array-like): True binary or multiclass labels.
            y_pred (array-like): Predicted labels.
            average (str): Type of averaging to perform ('binary', 'micro', 'macro', 'weighted').
        Returns:
            tuple: (precision, recall, f1_score, support).
        """
        precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average=average, zero_division=0)
        return precision, recall, f1

    def calculate_net_benefit(self, y_true, y_pred_proba, threshold_range=np.arange(0.01, 1.0, 0.01)):
        """
        Calculates Net Benefit for Decision Curve Analysis (DCA).

        Based on Vickers et al., 2006: Decision Curve Analysis - A Novel Method
        for Evaluating Prediction Models.

        Net Benefit = (TP/N) - (FP/N) × [pt/(1-pt)]

        Where:
        - TP = True Positives
        - FP = False Positives
        - N = Total samples
        - pt = Probability threshold

        Args:
            y_true (array-like): True binary labels (1=event, 0=no event)
            y_pred_proba (array-like): Predicted probabilities for positive class
            threshold_range (array-like): Array of probability thresholds to evaluate

        Returns:
            dict: Contains 'thresholds', 'net_benefit', 'treat_all', 'treat_none'
        """
        y_true = np.array(y_true)
        y_pred_proba = np.array(y_pred_proba)
        n = len(y_true)

        if n == 0:
            return {
                'thresholds': threshold_range,
                'net_benefit': np.zeros_like(threshold_range),
                'treat_all': np.zeros_like(threshold_range),
                'treat_none': np.zeros_like(threshold_range)
            }

        prevalence = np.mean(y_true)

        net_benefits = []
        treat_all_benefits = []
        treat_none_benefits = []

        for pt in threshold_range:
            # Model strategy: treat if predicted probability >= threshold
            y_pred_binary = (y_pred_proba >= pt).astype(int)
            tp = np.sum((y_pred_binary == 1) & (y_true == 1))
            fp = np.sum((y_pred_binary == 1) & (y_true == 0))

            # Net benefit of model
            nb_model = (tp / n) - (fp / n) * (pt / (1 - pt + 1e-10))
            net_benefits.append(nb_model)

            # Net benefit of "treat all" strategy
            # All patients treated: TP = all positive, FP = all negative
            nb_all = prevalence - (1 - prevalence) * (pt / (1 - pt + 1e-10))
            treat_all_benefits.append(nb_all)

            # Net benefit of "treat none" strategy
            nb_none = 0.0  # No intervention = no benefit
            treat_none_benefits.append(nb_none)

        return {
            'thresholds': threshold_range,
            'net_benefit': np.array(net_benefits),
            'treat_all': np.array(treat_all_benefits),
            'treat_none': np.array(treat_none_benefits),
            'max_net_benefit': np.max(net_benefits) if net_benefits else 0.0
        }

    def calculate_ct_scan_reduction(self, y_true, y_pred_proba,
                                   baseline_strategy='high_sensitivity',
                                   model_threshold=0.5,
                                   stroke_prevalence=0.15):
        """
        Calculates reduction in unnecessary CT scans using AI-driven triage.

        The model compares:
        1. Baseline strategy (e.g., CT all high-risk patients, ~40% of ED visits)
        2. AI-optimized strategy (CT only high-probability stroke cases)

        Clinical Context (Vestibular/Dizziness ED presentations):
        - Stroke prevalence: ~15% (Newman-Toker 2013, Stroke)
        - Baseline CT rate in practice: 30-50% of dizziness patients
        - Goal: Reduce unnecessary CTs while maintaining stroke detection

        Args:
            y_true (array-like): True labels (1=stroke, 0=benign)
            y_pred_proba (array-like): Model's predicted stroke probability
            baseline_strategy (str): 'high_sensitivity', 'moderate', or 'guideline'
            model_threshold (float): Probability threshold for recommending CT
            stroke_prevalence (float): Expected stroke prevalence (~0.15)

        Returns:
            dict: CT scan metrics including reduction rate, sensitivity, specificity
        """
        y_true = np.array(y_true)
        y_pred_proba = np.array(y_pred_proba)
        n = len(y_true)

        if n == 0:
            return {'error': 'Empty data'}

        n_stroke = np.sum(y_true == 1)
        n_benign = np.sum(y_true == 0)

        # =====================================================================
        # Baseline Strategy: Current clinical practice
        # =====================================================================
        if baseline_strategy == 'high_sensitivity':
            # Conservative: CT for all high-risk presentations
            # Includes: age>60, vascular risk factors, acute onset
            # Typically captures ~40% of ED dizziness patients
            baseline_ct_rate = 0.40
            baseline_ct_mask = np.random.rand(n) < baseline_ct_rate  # Simulated
            baseline_sensitivity = 0.95  # High sensitivity, low specificity

        elif baseline_strategy == 'moderate':
            # Moderate approach: ABCD² or similar score
            # ~30% of patients get CT
            baseline_ct_rate = 0.30
            baseline_ct_mask = np.random.rand(n) < baseline_ct_rate
            baseline_sensitivity = 0.85

        elif baseline_strategy == 'guideline':
            # Guideline-based (e.g., HINTS plus)
            # ~25% get CT in optimized centers
            baseline_ct_rate = 0.25
            baseline_ct_mask = np.random.rand(n) < baseline_ct_rate
            baseline_sensitivity = 0.90

        else:
            # Default to high sensitivity
            baseline_ct_rate = 0.40
            baseline_ct_mask = np.random.rand(n) < baseline_ct_rate
            baseline_sensitivity = 0.95

        baseline_n_ct = int(baseline_ct_rate * n)
        baseline_unnecessary_ct = int(baseline_ct_rate * n_benign)

        # =====================================================================
        # AI-Optimized Strategy: Use model predictions
        # =====================================================================
        model_ct_mask = y_pred_proba >= model_threshold
        model_n_ct = np.sum(model_ct_mask)

        # Calculate true/false positives and negatives
        tp = np.sum(model_ct_mask & (y_true == 1))
        fp = np.sum(model_ct_mask & (y_true == 0))
        tn = np.sum(~model_ct_mask & (y_true == 0))
        fn = np.sum(~model_ct_mask & (y_true == 1))

        model_sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
        model_specificity = tn / (tn + fp) if (tn + fp) > 0 else 0

        model_unnecessary_ct = fp  # CTs given to benign cases

        # =====================================================================
        # Calculate Reduction
        # =====================================================================
        absolute_reduction = baseline_n_ct - model_n_ct
        relative_reduction = absolute_reduction / baseline_n_ct if baseline_n_ct > 0 else 0

        unnecessary_ct_reduction = baseline_unnecessary_ct - model_unnecessary_ct
        unnecessary_ct_reduction_rate = (unnecessary_ct_reduction / baseline_unnecessary_ct
                                        if baseline_unnecessary_ct > 0 else 0)

        # =====================================================================
        # Clinical Impact Metrics
        # =====================================================================
        # Cost savings (assuming $1000 per CT scan)
        cost_per_ct = 1000  # USD
        cost_savings = absolute_reduction * cost_per_ct

        # Time savings (assuming 60 min per CT: scan + read + report)
        time_per_ct = 60  # minutes
        time_savings = absolute_reduction * time_per_ct

        return {
            'baseline_ct_rate': baseline_ct_rate,
            'baseline_n_ct': baseline_n_ct,
            'baseline_sensitivity': baseline_sensitivity,
            'baseline_unnecessary_ct': baseline_unnecessary_ct,

            'model_ct_rate': model_n_ct / n,
            'model_n_ct': model_n_ct,
            'model_sensitivity': model_sensitivity,
            'model_specificity': model_specificity,
            'model_unnecessary_ct': model_unnecessary_ct,

            'absolute_ct_reduction': absolute_reduction,
            'relative_ct_reduction': relative_reduction,
            'unnecessary_ct_reduction': unnecessary_ct_reduction,
            'unnecessary_ct_reduction_rate': unnecessary_ct_reduction_rate,

            'missed_strokes': fn,
            'missed_stroke_rate': fn / n_stroke if n_stroke > 0 else 0,

            'cost_savings_usd': cost_savings,
            'time_savings_minutes': time_savings,

            'tp': int(tp), 'fp': int(fp), 'tn': int(tn), 'fn': int(fn)
        } 
    
    def simulate_diagnosis_time_reduction(self):
        """
        Simulates reduction in diagnosis time due to an optimized patient pathway.
        This is a conceptual simulation for demonstration purposes.
        Returns:
            float: Simulated time reduction in minutes (e.g., 20 minutes).
        """
        # This function directly returns the projected value from the paper for demonstration.
        return 20 # Corresponds to the ~20 minutes in the paper

    def compute_all_metrics(self, real_data: pd.DataFrame, synthetic_data: pd.DataFrame) -> Dict:
        """
        Compute all evaluation metrics for synthetic data validation.

        This comprehensive method calculates:
        - Statistical realism (KL divergence, distribution similarity)
        - Data quality metrics (completeness, consistency)
        - Feature-wise comparison

        Args:
            real_data: DataFrame of real/original data
            synthetic_data: DataFrame of synthetic data

        Returns:
            Dictionary containing all computed metrics

        Example:
            >>> evaluator = EvaluationMetrics()
            >>> metrics = evaluator.compute_all_metrics(real_df, synthetic_df)
            >>> print(f"KL Divergence: {metrics['statistical']['mean_kl_divergence']}")
        """
        metrics = {
            'statistical': {},
            'quality': {},
            'features': {}
        }

        # 1. Statistical Realism
        try:
            kl_div = self.calculate_kl_divergence(synthetic_data, real_data)
            metrics['statistical']['mean_kl_divergence'] = float(kl_div) if not np.isnan(kl_div) else None
        except Exception as e:
            metrics['statistical']['mean_kl_divergence'] = None
            metrics['statistical']['kl_error'] = str(e)

        # 2. Data Quality Metrics
        metrics['quality']['real_samples'] = len(real_data)
        metrics['quality']['synthetic_samples'] = len(synthetic_data)
        metrics['quality']['real_features'] = real_data.shape[1]
        metrics['quality']['synthetic_features'] = synthetic_data.shape[1]
        metrics['quality']['feature_match'] = (real_data.shape[1] == synthetic_data.shape[1])

        # Missing values
        metrics['quality']['real_missing_pct'] = (real_data.isnull().sum().sum() /
                                                  (real_data.shape[0] * real_data.shape[1]) * 100)
        metrics['quality']['synthetic_missing_pct'] = (synthetic_data.isnull().sum().sum() /
                                                       (synthetic_data.shape[0] * synthetic_data.shape[1]) * 100)

        # 3. Feature-wise Statistics
        common_features = list(set(real_data.columns) & set(synthetic_data.columns))
        numeric_features = [col for col in common_features
                           if pd.api.types.is_numeric_dtype(real_data[col])]

        for feature in numeric_features[:10]:  # First 10 numeric features
            try:
                metrics['features'][feature] = {
                    'real_mean': float(real_data[feature].mean()),
                    'synthetic_mean': float(synthetic_data[feature].mean()),
                    'real_std': float(real_data[feature].std()),
                    'synthetic_std': float(synthetic_data[feature].std()),
                    'mean_diff': abs(float(real_data[feature].mean()) -
                                   float(synthetic_data[feature].mean()))
                }
            except Exception:
                continue

        # 4. Overall Assessment
        metrics['summary'] = {
            'total_metrics_computed': sum(1 for v in metrics.values() if isinstance(v, dict) for _ in v.keys()),
            'evaluation_complete': True,
            'timestamp': pd.Timestamp.now().isoformat()
        }

        return metrics

