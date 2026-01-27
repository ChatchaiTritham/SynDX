"""
Diagnostic Performance Evaluator for Synthetic Data Validation

Evaluates downstream diagnostic task performance to validate synthetic data utility.

Compares diagnostic classifiers trained on synthetic vs archetype data across
multiple metrics to ensure synthetic data maintains clinical utility.

Key Metrics:
- Accuracy, Precision, Recall, F1-Score (per-class and macro/micro)
- ROC-AUC (one-vs-rest for multi-class)
- Confusion matrices
- Utility Gap (performance difference synthetic vs archetype)
- Statistical significance testing (McNemar's test)

Author: Chatchai Tritham
Date: 2026-01-25
"""

import numpy as np
import xgboost as xgb
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_auc_score,
    roc_curve, auc
)
from scipy.stats import mcnemar
from typing import Dict, List, Tuple, Optional, Any
import logging

logger = logging.getLogger(__name__)


class DiagnosticEvaluator:
    """
    Evaluate diagnostic performance of synthetic vs archetype-trained models.

    Validates that synthetic data maintains utility for downstream diagnostic
    tasks by comparing classifier performance across multiple metrics.

    Workflow:
    1. Train classifier on archetype data (baseline)
    2. Train classifier on synthetic data
    3. Test both on held-out archetype test set
    4. Compare performance metrics
    5. Compute utility gap and statistical significance

    Example:
        >>> evaluator = DiagnosticEvaluator(model_type='xgboost')
        >>> evaluator.fit_archetype_model(X_arch_train, y_arch_train)
        >>> evaluator.fit_synthetic_model(X_synth_train, y_synth_train)
        >>> results = evaluator.evaluate(X_arch_test, y_arch_test)
        >>> print(f"Utility Gap: {results['utility_gap']:.2%}")
    """

    def __init__(self,
                 model_type: str = 'xgboost',
                 cv_folds: int = 5,
                 random_state: int = 42):
        """
        Initialize diagnostic evaluator.

        Args:
            model_type: Classifier type ('xgboost' or 'random_forest')
            cv_folds: Number of cross-validation folds
            random_state: Random seed
        """
        self.model_type = model_type
        self.cv_folds = cv_folds
        self.random_state = random_state

        # Models
        self.archetype_model = None
        self.synthetic_model = None

        # Evaluation results
        self.results = {}

        logger.info(
            f"Initialized DiagnosticEvaluator "
            f"(model={model_type}, cv_folds={cv_folds})"
        )

    def _create_classifier(self):
        """Create a fresh classifier instance."""
        if self.model_type == 'xgboost':
            return xgb.XGBClassifier(
                n_estimators=100,
                max_depth=6,
                learning_rate=0.1,
                random_state=self.random_state,
                eval_metric='mlogloss',
                use_label_encoder=False
            )
        elif self.model_type == 'random_forest':
            return RandomForestClassifier(
                n_estimators=100,
                max_depth=10,
                random_state=self.random_state
            )
        else:
            raise ValueError(f"Unknown model type: {self.model_type}")

    def fit_archetype_model(self, X_train: np.ndarray, y_train: np.ndarray):
        """
        Train diagnostic classifier on archetype data (baseline).

        Args:
            X_train: Archetype training features
            y_train: Archetype training labels
        """
        n_samples, n_features = X_train.shape
        n_classes = len(np.unique(y_train))

        logger.info(
            f"Training archetype model on {n_samples} samples, "
            f"{n_features} features, {n_classes} classes..."
        )

        self.archetype_model = self._create_classifier()
        self.archetype_model.fit(X_train, y_train)

        train_accuracy = self.archetype_model.score(X_train, y_train)
        logger.info(
            f"✓ Archetype model trained. Train accuracy: {
                train_accuracy:.3f}")

    def fit_synthetic_model(self, X_train: np.ndarray, y_train: np.ndarray):
        """
        Train diagnostic classifier on synthetic data.

        Args:
            X_train: Synthetic training features
            y_train: Synthetic training labels
        """
        n_samples, n_features = X_train.shape
        n_classes = len(np.unique(y_train))

        logger.info(
            f"Training synthetic model on {n_samples} samples, "
            f"{n_features} features, {n_classes} classes..."
        )

        self.synthetic_model = self._create_classifier()
        self.synthetic_model.fit(X_train, y_train)

        train_accuracy = self.synthetic_model.score(X_train, y_train)
        logger.info(
            f"✓ Synthetic model trained. Train accuracy: {
                train_accuracy:.3f}")

    def evaluate(self, X_test: np.ndarray, y_test: np.ndarray) -> Dict:
        """
        Evaluate both models on held-out test set.

        Args:
            X_test: Test features (typically held-out archetypes)
            y_test: Test labels

        Returns:
            Dictionary with comprehensive evaluation results
        """
        if self.archetype_model is None or self.synthetic_model is None:
            raise ValueError("Must fit both models first")

        logger.info("=" * 80)
        logger.info("Evaluating diagnostic performance...")
        logger.info("=" * 80)

        # Get predictions
        y_pred_arch = self.archetype_model.predict(X_test)
        y_pred_synth = self.synthetic_model.predict(X_test)

        # Get prediction probabilities for AUC
        y_proba_arch = self.archetype_model.predict_proba(X_test)
        y_proba_synth = self.synthetic_model.predict_proba(X_test)

        # Compute metrics for archetype model
        arch_metrics = self._compute_metrics(
            y_test, y_pred_arch, y_proba_arch, "Archetype")

        # Compute metrics for synthetic model
        synth_metrics = self._compute_metrics(
            y_test, y_pred_synth, y_proba_synth, "Synthetic")

        # Compute utility gap
        utility_gap = arch_metrics['accuracy'] - synth_metrics['accuracy']

        # Statistical significance (McNemar's test for paired predictions)
        mcnemar_result = self._mcnemar_test(y_test, y_pred_arch, y_pred_synth)

        # Store results
        self.results = {
            'archetype': arch_metrics,
            'synthetic': synth_metrics,
            'utility_gap': utility_gap,
            'mcnemar': mcnemar_result,
            'n_test_samples': len(y_test)
        }

        # Log summary
        logger.info("=" * 80)
        logger.info("DIAGNOSTIC PERFORMANCE SUMMARY")
        logger.info("=" * 80)
        logger.info(
            f"Archetype Model Accuracy:  {
                arch_metrics['accuracy']:.3f}")
        logger.info(
            f"Synthetic Model Accuracy:  {
                synth_metrics['accuracy']:.3f}")
        logger.info(
            f"Utility Gap:               {utility_gap:+.3f} ({utility_gap * 100:+.1f}%)")
        logger.info(
            f"McNemar p-value:           {mcnemar_result['p_value']:.4f}")
        logger.info("=" * 80)

        return self.results

    def _compute_metrics(self,
                         y_true: np.ndarray,
                         y_pred: np.ndarray,
                         y_proba: np.ndarray,
                         model_name: str) -> Dict:
        """
        Compute comprehensive classification metrics.

        Args:
            y_true: True labels
            y_pred: Predicted labels
            y_proba: Prediction probabilities
            model_name: Model identifier for logging

        Returns:
            Dictionary with all metrics
        """
        logger.info(f"\nComputing metrics for {model_name} model...")

        # Basic metrics
        accuracy = accuracy_score(y_true, y_pred)

        # Multi-class metrics (macro and weighted averages)
        precision_macro = precision_score(
            y_true, y_pred, average='macro', zero_division=0)
        recall_macro = recall_score(
            y_true, y_pred, average='macro', zero_division=0)
        f1_macro = f1_score(y_true, y_pred, average='macro', zero_division=0)

        precision_weighted = precision_score(
            y_true, y_pred, average='weighted', zero_division=0)
        recall_weighted = recall_score(
            y_true, y_pred, average='weighted', zero_division=0)
        f1_weighted = f1_score(
            y_true,
            y_pred,
            average='weighted',
            zero_division=0)

        # Per-class metrics
        precision_per_class = precision_score(
            y_true, y_pred, average=None, zero_division=0)
        recall_per_class = recall_score(
            y_true, y_pred, average=None, zero_division=0)
        f1_per_class = f1_score(y_true, y_pred, average=None, zero_division=0)

        # Confusion matrix
        cm = confusion_matrix(y_true, y_pred)

        # AUC (one-vs-rest for multi-class)
        try:
            if len(np.unique(y_true)) > 2:
                # Multi-class: macro and weighted AUC
                auc_macro = roc_auc_score(
                    y_true, y_proba, average='macro', multi_class='ovr')
                auc_weighted = roc_auc_score(
                    y_true, y_proba, average='weighted', multi_class='ovr')
            else:
                # Binary case
                auc_macro = roc_auc_score(y_true, y_proba[:, 1])
                auc_weighted = auc_macro
        except Exception as e:
            logger.warning(f"Could not compute AUC: {e}")
            auc_macro = 0.0
            auc_weighted = 0.0

        metrics = {
            'accuracy': float(accuracy),
            'precision_macro': float(precision_macro),
            'recall_macro': float(recall_macro),
            'f1_macro': float(f1_macro),
            'precision_weighted': float(precision_weighted),
            'recall_weighted': float(recall_weighted),
            'f1_weighted': float(f1_weighted),
            'auc_macro': float(auc_macro),
            'auc_weighted': float(auc_weighted),
            'precision_per_class': precision_per_class.tolist(),
            'recall_per_class': recall_per_class.tolist(),
            'f1_per_class': f1_per_class.tolist(),
            'confusion_matrix': cm.tolist()
        }

        # Log key metrics
        logger.info(f"  Accuracy:         {accuracy:.3f}")
        logger.info(f"  F1 (macro):       {f1_macro:.3f}")
        logger.info(f"  F1 (weighted):    {f1_weighted:.3f}")
        logger.info(f"  AUC (macro):      {auc_macro:.3f}")

        return metrics

    def _mcnemar_test(self,
                      y_true: np.ndarray,
                      y_pred1: np.ndarray,
                      y_pred2: np.ndarray) -> Dict:
        """
        McNemar's test for statistical significance between two models.

        Tests null hypothesis: both models have equal error rates.

        Args:
            y_true: True labels
            y_pred1: Predictions from model 1 (archetype)
            y_pred2: Predictions from model 2 (synthetic)

        Returns:
            Dictionary with test results
        """
        # Create contingency table
        # [both_correct, model1_correct_only]
        # [model2_correct_only, both_wrong]
        correct1 = (y_pred1 == y_true)
        correct2 = (y_pred2 == y_true)

        both_correct = np.sum(correct1 & correct2)
        both_wrong = np.sum(~correct1 & ~correct2)
        model1_only = np.sum(correct1 & ~correct2)
        model2_only = np.sum(~correct1 & correct2)

        # McNemar's test focuses on discordant pairs
        contingency_table = np.array([[both_correct, model1_only],
                                      [model2_only, both_wrong]])

        try:
            result = mcnemar(contingency_table, exact=False, correction=True)
            p_value = result.pvalue
            statistic = result.statistic
        except Exception as e:
            logger.warning(f"McNemar test failed: {e}")
            p_value = 1.0
            statistic = 0.0

        significance = "significant" if p_value < 0.05 else "not significant"

        logger.info(f"\nMcNemar's Test:")
        logger.info(f"  Both correct:        {both_correct}")
        logger.info(f"  Archetype only:      {model1_only}")
        logger.info(f"  Synthetic only:      {model2_only}")
        logger.info(f"  Both wrong:          {both_wrong}")
        logger.info(f"  Statistic:           {statistic:.3f}")
        logger.info(f"  p-value:             {p_value:.4f} ({significance})")

        return {
            'statistic': float(statistic),
            'p_value': float(p_value),
            'both_correct': int(both_correct),
            'both_wrong': int(both_wrong),
            'archetype_only_correct': int(model1_only),
            'synthetic_only_correct': int(model2_only),
            'is_significant': p_value < 0.05
        }

    def cross_validate(self,
                       X: np.ndarray,
                       y: np.ndarray,
                       data_type: str = 'archetype') -> Dict:
        """
        Perform k-fold cross-validation.

        Args:
            X: Feature matrix
            y: Labels
            data_type: 'archetype' or 'synthetic'

        Returns:
            Cross-validation results
        """
        logger.info(f"Running {self.cv_folds}-fold CV on {data_type} data...")

        clf = self._create_classifier()
        skf = StratifiedKFold(
            n_splits=self.cv_folds,
            shuffle=True,
            random_state=self.random_state)

        # Compute multiple metrics via CV
        cv_accuracy = cross_val_score(clf, X, y, cv=skf, scoring='accuracy')
        cv_f1_macro = cross_val_score(clf, X, y, cv=skf, scoring='f1_macro')
        cv_f1_weighted = cross_val_score(
            clf, X, y, cv=skf, scoring='f1_weighted')

        results = {
            'accuracy_mean': float(np.mean(cv_accuracy)),
            'accuracy_std': float(np.std(cv_accuracy)),
            'f1_macro_mean': float(np.mean(cv_f1_macro)),
            'f1_macro_std': float(np.std(cv_f1_macro)),
            'f1_weighted_mean': float(np.mean(cv_f1_weighted)),
            'f1_weighted_std': float(np.std(cv_f1_weighted)),
            'cv_folds': self.cv_folds
        }

        logger.info(
            f"✓ CV Accuracy:     {
                results['accuracy_mean']:.3f} ± {
                results['accuracy_std']:.3f}")
        logger.info(
            f"✓ CV F1 (macro):   {
                results['f1_macro_mean']:.3f} ± {
                results['f1_macro_std']:.3f}")

        return results

    def get_feature_importance(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get feature importance from both models.

        Returns:
            (archetype_importance, synthetic_importance)
        """
        if self.archetype_model is None or self.synthetic_model is None:
            raise ValueError("Must fit both models first")

        if hasattr(self.archetype_model, 'feature_importances_'):
            arch_importance = self.archetype_model.feature_importances_
            synth_importance = self.synthetic_model.feature_importances_
        else:
            logger.warning("Model does not support feature importance")
            return None, None

        return arch_importance, synth_importance

    def summary(self) -> Dict:
        """
        Get summary of diagnostic evaluation.

        Returns:
            Dictionary with evaluation summary and interpretation
        """
        if not self.results:
            return {'status': 'not_evaluated'}

        utility_gap = self.results['utility_gap']
        is_significant = self.results['mcnemar']['is_significant']

        # Interpret utility gap
        if abs(utility_gap) < 0.02:
            interpretation = "Excellent - Synthetic data matches archetype performance"
        elif abs(utility_gap) < 0.05:
            interpretation = "Good - Minimal utility gap, synthetic data is acceptable"
        elif abs(utility_gap) < 0.10:
            interpretation = "Fair - Noticeable utility gap, consider improvements"
        else:
            interpretation = "Poor - Large utility gap, synthetic data needs improvement"

        # Recommendation
        if abs(utility_gap) < 0.05 and not is_significant:
            recommendation = "Synthetic data approved for diagnostic applications"
        elif abs(utility_gap) < 0.10:
            recommendation = "Synthetic data acceptable with caution"
        else:
            recommendation = "Improve synthetic data generation before deployment"

        return {
            'status': 'evaluated',
            'results': self.results,
            'utility_gap': utility_gap,
            'interpretation': interpretation,
            'recommendation': recommendation,
            'archetype_accuracy': self.results['archetype']['accuracy'],
            'synthetic_accuracy': self.results['synthetic']['accuracy']
        }

    def __repr__(self) -> str:
        """String representation"""
        if not self.results:
            return f"DiagnosticEvaluator(model_type='{
                self.model_type}', evaluated=False)"
        else:
            utility_gap = self.results['utility_gap']
            return (f"DiagnosticEvaluator(model_type='{self.model_type}', "
                    f"utility_gap={utility_gap:+.3f})")


# Main demonstration
if __name__ == '__main__':
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    logger.info("DiagnosticEvaluator - Demo Mode")

    # Generate mock archetype and synthetic datasets
    np.random.seed(42)

    n_arch_train = 800
    n_arch_test = 200
    n_synth_train = 1000
    n_features = 50
    n_classes = 4  # BPPV, VM, VN, Stroke

    # Create archetype data with structure
    X_arch_train = np.random.randn(n_arch_train, n_features)
    important_features = [0, 5, 10, 15, 20]
    for feat_idx in important_features:
        X_arch_train[:, feat_idx] *= 3.0

    # Create labels based on features
    y_arch_train = np.zeros(n_arch_train, dtype=int)
    for i in range(n_arch_train):
        score = np.sum(X_arch_train[i, important_features])
        if score > 4:
            y_arch_train[i] = 3  # Stroke
        elif score > 2:
            y_arch_train[i] = 2  # VN
        elif score > 0:
            y_arch_train[i] = 1  # VM
        else:
            y_arch_train[i] = 0  # BPPV

    # Create test set (held-out archetypes)
    X_arch_test = np.random.randn(n_arch_test, n_features)
    for feat_idx in important_features:
        X_arch_test[:, feat_idx] *= 3.0

    y_arch_test = np.zeros(n_arch_test, dtype=int)
    for i in range(n_arch_test):
        score = np.sum(X_arch_test[i, important_features])
        if score > 4:
            y_arch_test[i] = 3
        elif score > 2:
            y_arch_test[i] = 2
        elif score > 0:
            y_arch_test[i] = 1
        else:
            y_arch_test[i] = 0

    # Create synthetic data (similar but with slightly degraded signal)
    X_synth_train = np.random.randn(n_synth_train, n_features)
    for feat_idx in important_features:
        X_synth_train[:, feat_idx] *= 2.7  # Slightly weaker signal

    y_synth_train = np.zeros(n_synth_train, dtype=int)
    for i in range(n_synth_train):
        score = np.sum(X_synth_train[i, important_features])
        if score > 3.5:  # Slightly different thresholds
            y_synth_train[i] = 3
        elif score > 1.5:
            y_synth_train[i] = 2
        elif score > -0.5:
            y_synth_train[i] = 1
        else:
            y_synth_train[i] = 0

    logger.info(f"\nDataset sizes:")
    logger.info(
        f"  Archetype train: {
            X_arch_train.shape}, classes: {
            np.bincount(y_arch_train)}")
    logger.info(
        f"  Archetype test:  {
            X_arch_test.shape}, classes: {
            np.bincount(y_arch_test)}")
    logger.info(
        f"  Synthetic train: {
            X_synth_train.shape}, classes: {
            np.bincount(y_synth_train)}")

    # Initialize evaluator
    evaluator = DiagnosticEvaluator(model_type='xgboost', cv_folds=5)

    # Train both models
    logger.info("\n" + "=" * 80)
    logger.info("Training Models")
    logger.info("=" * 80)

    evaluator.fit_archetype_model(X_arch_train, y_arch_train)
    evaluator.fit_synthetic_model(X_synth_train, y_synth_train)

    # Evaluate on test set
    logger.info("\n" + "=" * 80)
    logger.info("Evaluation on Test Set")
    logger.info("=" * 80)

    results = evaluator.evaluate(X_arch_test, y_arch_test)

    # Print detailed results
    logger.info("\n" + "=" * 80)
    logger.info("DETAILED RESULTS")
    logger.info("=" * 80)

    logger.info("\nArchetype Model:")
    logger.info(f"  Accuracy:    {results['archetype']['accuracy']:.3f}")
    logger.info(f"  F1 (macro):  {results['archetype']['f1_macro']:.3f}")
    logger.info(f"  AUC (macro): {results['archetype']['auc_macro']:.3f}")

    logger.info("\nSynthetic Model:")
    logger.info(f"  Accuracy:    {results['synthetic']['accuracy']:.3f}")
    logger.info(f"  F1 (macro):  {results['synthetic']['f1_macro']:.3f}")
    logger.info(f"  AUC (macro): {results['synthetic']['auc_macro']:.3f}")

    logger.info(
        f"\nUtility Gap: {results['utility_gap']:+.3f} ({results['utility_gap'] * 100:+.1f}%)")

    # Summary
    summary = evaluator.summary()
    logger.info(f"\nInterpretation: {summary['interpretation']}")
    logger.info(f"Recommendation: {summary['recommendation']}")

    logger.info("\n✓ Demo complete!")
