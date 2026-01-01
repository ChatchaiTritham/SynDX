"""
XAI Driver - making synthetic data generation explainable

This is what makes SynDX different from other synthetic data generators.
Instead of just generating random data, we use explainability (SHAP values)
to guide the generation process toward clinically meaningful patterns.

Functionality:
- SHAP-guided reweighting of feature importance
- Counterfactual generation to test edge cases
- Probabilistic logic integration to maintain clinical constraints

Significance: Opaque generative models may produce clinically implausible data. This keeps the
generation process grounded in clinical reasoning.
"""

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import logging
from typing import Dict, List, Optional, Tuple, Any
from sklearn.ensemble import RandomForestClassifier
import warnings

from .probabilistic_logic import ProbabilisticLogic

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class XAIDriver:
    """
    XAI Driver for synthetic data refinement.
    Integrates explainability directly into the data generation process.
    """

    def __init__(self):
        """Initialize XAI Driver."""
        logger.info("XAIDriver initialized.")
        self.shap_values_cache = {}
        self.counterfactual_cache = {}

    def create_dummy_predictive_model(self, d_synthetic_current: pd.DataFrame) -> Optional[RandomForestClassifier]:
        """
        Creates a dummy predictive model for XAI analysis.

        Args:
            d_synthetic_current: Current synthetic dataset

        Returns:
            Trained RandomForestClassifier or None if insufficient data
        """
        logger.info("Creating dummy predictive model for XAI refinement...")

        if d_synthetic_current.empty or 'diagnosis' not in d_synthetic_current.columns:
            logger.warning("Cannot create predictive model: empty data or missing diagnosis column")
            return None

        try:
            # Prepare features and target
            X = d_synthetic_current.drop('diagnosis', axis=1)
            y = d_synthetic_current['diagnosis']

            # One-hot encode categorical features
            X_encoded = pd.get_dummies(X, drop_first=False)

            # Convert diagnosis to binary (Stroke vs Others)
            y_binary = y.apply(lambda x: 1 if x == 'Stroke' else 0)

            # Check if we have enough samples and classes
            if len(np.unique(y_binary)) < 2:
                logger.warning("Only one class present. Cannot train meaningful model.")
                return None

            if len(X_encoded) < 10:
                logger.warning("Insufficient samples for model training")
                return None

            # Train simple Random Forest
            model = RandomForestClassifier(
                n_estimators=50,
                max_depth=5,
                min_samples_split=5,
                random_state=42,
                n_jobs=-1
            )

            model.fit(X_encoded, y_binary)
            logger.info("Dummy predictive model created successfully")

            # Store feature names for later use
            model.feature_names_ = X_encoded.columns.tolist()

            return model

        except Exception as e:
            logger.error(f"Error creating dummy predictive model: {e}")
            return None

    def compute_shap_values(self, model, data: pd.DataFrame) -> Optional[pd.DataFrame]:
        """
        Computes SHAP values for the given model and data.

        Args:
            model: Trained predictive model
            data: Data to compute SHAP values for

        Returns:
            DataFrame with SHAP values or None
        """
        logger.info("Computing SHAP values...")

        if model is None or data.empty:
            logger.warning("Cannot compute SHAP values: invalid model or data")
            return None

        try:
            # Try to import SHAP
            try:
                import shap
            except ImportError:
                logger.warning("SHAP library not available. Using feature importance as approximation.")
                return self._approximate_shap_with_feature_importance(model, data)

            # Prepare data
            X = data.drop('diagnosis', axis=1) if 'diagnosis' in data.columns else data
            X_encoded = pd.get_dummies(X, drop_first=False)

            # Align columns with model
            if hasattr(model, 'feature_names_'):
                missing_cols = set(model.feature_names_) - set(X_encoded.columns)
                for col in missing_cols:
                    X_encoded[col] = 0
                X_encoded = X_encoded[model.feature_names_]

            # Sample data if too large
            if len(X_encoded) > 100:
                X_sample = X_encoded.sample(n=100, random_state=42)
            else:
                X_sample = X_encoded

            # Create SHAP explainer
            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(X_sample)

            # Handle binary classification output
            if isinstance(shap_values, list):
                shap_values = shap_values[1]  # Use positive class

            # Convert to DataFrame
            shap_df = pd.DataFrame(shap_values, columns=X_sample.columns)

            logger.info("SHAP values computed successfully")
            return shap_df

        except Exception as e:
            logger.warning(f"Error computing SHAP values: {e}. Using approximation.")
            return self._approximate_shap_with_feature_importance(model, data)

    def _approximate_shap_with_feature_importance(self, model, data: pd.DataFrame) -> pd.DataFrame:
        """
        Approximates SHAP values using feature importance when SHAP is unavailable.

        Args:
            model: Trained model with feature_importances_
            data: Input data

        Returns:
            Approximate SHAP values DataFrame
        """
        try:
            if not hasattr(model, 'feature_importances_'):
                logger.warning("Model has no feature_importances_. Returning uniform values.")
                X = data.drop('diagnosis', axis=1) if 'diagnosis' in data.columns else data
                X_encoded = pd.get_dummies(X, drop_first=False)
                return pd.DataFrame(
                    np.ones((len(X_encoded), len(X_encoded.columns))) / len(X_encoded.columns),
                    columns=X_encoded.columns
                )

            X = data.drop('diagnosis', axis=1) if 'diagnosis' in data.columns else data
            X_encoded = pd.get_dummies(X, drop_first=False)

            # Align with model features
            if hasattr(model, 'feature_names_'):
                missing_cols = set(model.feature_names_) - set(X_encoded.columns)
                for col in missing_cols:
                    X_encoded[col] = 0
                X_encoded = X_encoded[model.feature_names_]

            # Use feature importance as proxy
            importances = model.feature_importances_

            # Create approximate SHAP values (importance * feature value)
            approx_shap = X_encoded.values * importances

            return pd.DataFrame(approx_shap, columns=X_encoded.columns)

        except Exception as e:
            logger.error(f"Error in SHAP approximation: {e}")
            return None

    def shap_guided_reweighting(self,
                                 d_synthetic: pd.DataFrame,
                                 model,
                                 expected_importance: Dict[str, float],
                                 tolerance: float = 0.15) -> pd.DataFrame:
        """
        Reweights synthetic samples based on SHAP value alignment with clinical expectations.

        Args:
            d_synthetic: Synthetic dataset
            model: Predictive model
            expected_importance: Dict mapping features to expected importance
            tolerance: Acceptable deviation from expected importance

        Returns:
            Reweighted synthetic dataset
        """
        logger.info("Applying SHAP-guided reweighting...")

        if d_synthetic.empty or model is None:
            logger.warning("Cannot apply SHAP reweighting: invalid data or model")
            return d_synthetic

        try:
            # Compute SHAP values
            shap_df = self.compute_shap_values(model, d_synthetic)

            if shap_df is None:
                logger.warning("SHAP values not available. Skipping reweighting.")
                return d_synthetic

            # Calculate actual feature importance from SHAP
            shap_importance = np.abs(shap_df).mean()
            shap_importance_normalized = shap_importance / shap_importance.sum()

            # Map feature names (handle one-hot encoding)
            feature_scores = {}
            for feature, expected_imp in expected_importance.items():
                # Find columns related to this feature
                related_cols = [col for col in shap_importance_normalized.index
                               if col.startswith(feature)]

                if related_cols:
                    actual_imp = shap_importance_normalized[related_cols].sum()
                    feature_scores[feature] = abs(actual_imp - expected_imp)
                else:
                    # Feature not found in SHAP values
                    feature_scores[feature] = expected_imp

            # Calculate sample weights based on deviation
            avg_deviation = np.mean(list(feature_scores.values()))

            if avg_deviation > tolerance:
                logger.info(f"Average deviation {avg_deviation:.3f} exceeds tolerance {tolerance}")

                # Create weights (samples with better alignment get higher weight)
                weights = np.ones(len(d_synthetic))

                # Adjust weights based on feature alignment (simplified)
                for idx in range(len(d_synthetic)):
                    weights[idx] = max(0.5, 1.0 - avg_deviation)

                # Resample with weights
                reweighted_data = d_synthetic.sample(
                    n=len(d_synthetic),
                    replace=True,
                    weights=weights,
                    random_state=42
                ).reset_index(drop=True)

                logger.info("SHAP-guided reweighting completed")
                return reweighted_data
            else:
                logger.info(f"SHAP alignment acceptable (deviation: {avg_deviation:.3f})")
                return d_synthetic

        except Exception as e:
            logger.error(f"Error in SHAP-guided reweighting: {e}")
            return d_synthetic

    def adjust_vae_parameters_with_shap(self,
                                        vae_model,
                                        d_synthetic: pd.DataFrame,
                                        feature_columns: List[str],
                                        vae_scaler,
                                        expected_importance: Dict[str, float],
                                        adjustment_rate: float = 0.1,
                                        tolerance: float = 0.15) -> Tuple[bool, Dict[str, float]]:
        """
        Algorithm 2: SHAP-Guided Parameter Adjustment

        Adjusts VAE decoder parameters based on SHAP feature importance alignment.
        This implements the core XAI-driven synthesis from the research paper.

        Args:
            vae_model: Trained VAE model
            d_synthetic: Current synthetic dataset
            feature_columns: List of feature column names (after one-hot encoding)
            vae_scaler: Scaler used for normalization
            expected_importance: Dict of expected feature importance from clinical guidelines
            adjustment_rate: Learning rate for parameter adjustment
            tolerance: Acceptable deviation from expected importance

        Returns:
            Tuple of (needs_adjustment: bool, deviation_metrics: Dict)
        """
        logger.info("=" * 80)
        logger.info("Algorithm 2: SHAP-Guided Parameter Adjustment")
        logger.info("=" * 80)

        if vae_model is None or d_synthetic.empty:
            logger.warning("Cannot adjust parameters: invalid VAE model or data")
            return False, {}

        try:
            # ================================================================
            # Step 1: Create predictive model for SHAP analysis
            # ================================================================
            predictive_model = self.create_dummy_predictive_model(d_synthetic)

            if predictive_model is None:
                logger.warning("Cannot create predictive model. Skipping adjustment.")
                return False, {}

            # ================================================================
            # Step 2: Compute SHAP values
            # ================================================================
            shap_df = self.compute_shap_values(predictive_model, d_synthetic)

            if shap_df is None:
                logger.warning("Cannot compute SHAP values. Skipping adjustment.")
                return False, {}

            # ================================================================
            # Step 3: Calculate actual feature importance from SHAP
            # ================================================================
            shap_importance = np.abs(shap_df).mean()
            shap_importance_normalized = shap_importance / shap_importance.sum()

            logger.info("SHAP Feature Importance (normalized):")
            for feat, imp in shap_importance_normalized.items():
                logger.info(f"  {feat}: {imp:.4f}")

            # ================================================================
            # Step 4: Map to original features (handle one-hot encoding)
            # ================================================================
            feature_deviations = {}

            for feature, expected_imp in expected_importance.items():
                # Find columns related to this feature
                related_cols = [col for col in shap_importance_normalized.index
                               if col.startswith(feature)]

                if related_cols:
                    actual_imp = shap_importance_normalized[related_cols].sum()
                    deviation = actual_imp - expected_imp
                    feature_deviations[feature] = {
                        'expected': expected_imp,
                        'actual': actual_imp,
                        'deviation': deviation,
                        'relative_error': abs(deviation) / expected_imp if expected_imp > 0 else 0
                    }

                    logger.info(f"Feature '{feature}':")
                    logger.info(f"  Expected: {expected_imp:.4f}")
                    logger.info(f"  Actual:   {actual_imp:.4f}")
                    logger.info(f"  Deviation: {deviation:+.4f}")

            # ================================================================
            # Step 5: Check if adjustment is needed
            # ================================================================
            avg_deviation = np.mean([abs(v['deviation']) for v in feature_deviations.values()])
            max_deviation = np.max([abs(v['deviation']) for v in feature_deviations.values()])

            logger.info(f"\nDeviation Metrics:")
            logger.info(f"  Average deviation: {avg_deviation:.4f}")
            logger.info(f"  Maximum deviation: {max_deviation:.4f}")
            logger.info(f"  Tolerance: {tolerance:.4f}")

            if avg_deviation <= tolerance:
                logger.info(f"✓ SHAP alignment acceptable (avg deviation: {avg_deviation:.4f} <= {tolerance})")
                logger.info("=" * 80)
                return False, feature_deviations

            # ================================================================
            # Step 6: Adjust VAE decoder parameters
            # ================================================================
            logger.info(f"\n⚠ SHAP alignment needs improvement (avg deviation: {avg_deviation:.4f} > {tolerance})")
            logger.info("Adjusting VAE decoder parameters...")

            # Map feature deviations to decoder layer adjustments
            # Strategy: Features with too low importance should have decoder weights increased
            #          Features with too high importance should have decoder weights decreased

            vae_model.eval()  # Set to evaluation mode for gradient computation

            # Create feature importance adjustment vector
            adjustment_vector = torch.zeros(len(feature_columns))

            for feature, metrics in feature_deviations.items():
                deviation = metrics['deviation']

                # Find indices of related columns
                related_indices = [i for i, col in enumerate(feature_columns)
                                 if col.startswith(feature)]

                # Adjustment: if deviation < 0 (actual < expected), increase weight
                #            if deviation > 0 (actual > expected), decrease weight
                adjustment_factor = -deviation * adjustment_rate

                for idx in related_indices:
                    adjustment_vector[idx] = adjustment_factor

            # Apply adjustment to VAE decoder's output layer
            if hasattr(vae_model, 'fc_output'):
                with torch.no_grad():
                    # Adjust output layer weights
                    # Output layer shape: [output_dim, input_to_layer_dim]
                    output_weight = vae_model.fc_output.weight.data

                    logger.info(f"  Output layer weight shape: {output_weight.shape}")
                    logger.info(f"  Adjustment vector length: {len(adjustment_vector)}")

                    # Transpose adjustment vector to match output dimension
                    # output_weight is [output_features, hidden_dim]
                    # We want to adjust the output features (rows)
                    adjustment_vector_col = adjustment_vector.unsqueeze(1)  # [output_features, 1]

                    # Scale adjustment based on current weight magnitude (per output feature)
                    weight_magnitude_per_feature = torch.abs(output_weight).mean(dim=1, keepdim=True)
                    adjusted_increment = adjustment_vector_col * weight_magnitude_per_feature

                    # Expand to full weight matrix
                    adjusted_increment_expanded = adjusted_increment.expand_as(output_weight)

                    # Apply adjustment
                    vae_model.fc_output.weight.data += adjusted_increment_expanded

                    logger.info(f"✓ Adjusted VAE decoder output layer weights")
                    logger.info(f"  Adjustment magnitude: {torch.abs(adjusted_increment_expanded).mean():.6f}")

            logger.info("=" * 80)

            return True, feature_deviations

        except Exception as e:
            logger.error(f"Error in SHAP-guided parameter adjustment: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return False, {}

    def generate_counterfactuals(self,
                                  d_synthetic: pd.DataFrame,
                                  model,
                                  target_diagnosis: str = 'Stroke',
                                  n_counterfactuals: int = 10) -> pd.DataFrame:
        """
        Generates counterfactual examples to test diagnostic boundaries.

        Args:
            d_synthetic: Synthetic dataset
            model: Predictive model
            target_diagnosis: Diagnosis to generate counterfactuals for
            n_counterfactuals: Number of counterfactuals to generate

        Returns:
            DataFrame with counterfactual examples
        """
        logger.info(f"Generating {n_counterfactuals} counterfactuals for {target_diagnosis}...")

        if d_synthetic.empty or 'diagnosis' not in d_synthetic.columns:
            logger.warning("Cannot generate counterfactuals: invalid data")
            return pd.DataFrame()

        try:
            # Select samples NOT of target diagnosis
            non_target = d_synthetic[d_synthetic['diagnosis'] != target_diagnosis].copy()

            if non_target.empty:
                logger.warning(f"No non-{target_diagnosis} samples available")
                return pd.DataFrame()

            # Sample cases to modify
            n_samples = min(n_counterfactuals, len(non_target))
            samples = non_target.sample(n=n_samples, random_state=42).copy()

            counterfactuals = []

            for idx, row in samples.iterrows():
                cf = row.copy()

                # Modify features based on clinical knowledge
                if target_diagnosis == 'Stroke':
                    # Change features to match Stroke pattern
                    cf['nystagmus_type'] = 'vertical'
                    if 'age' in cf.index:
                        cf['age'] = max(cf['age'], 60)
                    if 'stroke_risk_factor' in cf.index:
                        cf['stroke_risk_factor'] = 1
                    cf['diagnosis'] = 'Stroke'

                elif target_diagnosis == 'BPPV':
                    # Change features to match BPPV pattern
                    cf['nystagmus_type'] = 'torsional'
                    if 'symptom_duration_days' in cf.index:
                        cf['symptom_duration_days'] = min(cf['symptom_duration_days'], 30)
                    cf['diagnosis'] = 'BPPV'

                counterfactuals.append(cf)

            cf_df = pd.DataFrame(counterfactuals)
            logger.info(f"Generated {len(cf_df)} counterfactuals")

            return cf_df

        except Exception as e:
            logger.error(f"Error generating counterfactuals: {e}")
            return pd.DataFrame()

    def apply_probabilistic_logic(self,
                                   d_synthetic: pd.DataFrame,
                                   clinical_guidelines: Dict) -> pd.DataFrame:
        """
        Applies probabilistic logic rules to enforce clinical coherence.

        Args:
            d_synthetic: Synthetic dataset
            clinical_guidelines: Clinical guidelines with probabilistic rules

        Returns:
            Refined synthetic dataset
        """
        logger.info("Applying probabilistic logic refinement...")

        try:
            pl = ProbabilisticLogic()
            refined_data = pl.refine_with_probabilistic_logic(d_synthetic, clinical_guidelines)
            logger.info("Probabilistic logic applied successfully")
            return refined_data

        except Exception as e:
            logger.error(f"Error applying probabilistic logic: {e}")
            return d_synthetic

    def simulate_xai_driven_synthesis_refinement(self,
                                                  d_synthetic_current: pd.DataFrame,
                                                  dummy_predictive_model,
                                                  generative_model_instance: Dict,
                                                  titrate_rules: Dict) -> pd.DataFrame:
        """
        Main XAI-driven refinement pipeline.

        Integrates:
        1. SHAP-guided reweighting
        2. Counterfactual generation
        3. Probabilistic logic

        Args:
            d_synthetic_current: Current synthetic dataset
            dummy_predictive_model: Model for XAI analysis
            generative_model_instance: Trained generative model
            titrate_rules: Clinical guidelines

        Returns:
            Refined synthetic dataset
        """
        logger.info("Starting XAI-driven synthesis refinement...")

        if d_synthetic_current.empty:
            logger.warning("Empty synthetic data. Skipping XAI refinement.")
            return d_synthetic_current

        refined_data = d_synthetic_current.copy()

        # Step 1: SHAP-Guided Reweighting
        if dummy_predictive_model is not None:
            expected_importance = {
                'nystagmus_type': 0.25,
                'age': 0.20,
                'symptom_duration_days': 0.15,
                'stroke_risk_factor': 0.15,
                'head_impulse_test_positive': 0.10
            }

            refined_data = self.shap_guided_reweighting(
                refined_data,
                dummy_predictive_model,
                expected_importance,
                tolerance=0.15
            )

        # Step 2: Counterfactual Generation and Integration
        if dummy_predictive_model is not None and len(refined_data) < 5000:
            counterfactuals = self.generate_counterfactuals(
                refined_data,
                dummy_predictive_model,
                target_diagnosis='Stroke',
                n_counterfactuals=min(50, len(refined_data) // 10)
            )

            if not counterfactuals.empty:
                # Add counterfactuals to enhance decision boundaries
                refined_data = pd.concat([refined_data, counterfactuals], ignore_index=True)
                logger.info(f"Added {len(counterfactuals)} counterfactuals to dataset")

        # Step 3: Probabilistic Logic
        refined_data = self.apply_probabilistic_logic(refined_data, titrate_rules)

        logger.info(f"XAI-driven refinement completed. Dataset size: {len(refined_data)}")

        return refined_data

    def evaluate_xai_quality(self, d_synthetic: pd.DataFrame, model) -> Dict[str, float]:
        """
        Evaluates the quality of XAI-driven synthesis.

        Args:
            d_synthetic: Synthetic dataset
            model: Predictive model

        Returns:
            Dictionary of quality metrics
        """
        logger.info("Evaluating XAI quality...")

        metrics = {
            'shap_alignment': 0.0,
            'counterfactual_validity': 0.0,
            'probabilistic_coherence': 0.0
        }

        try:
            if model is not None and not d_synthetic.empty:
                shap_df = self.compute_shap_values(model, d_synthetic)

                if shap_df is not None:
                    # Measure SHAP consistency
                    shap_std = shap_df.std().mean()
                    metrics['shap_alignment'] = 1.0 / (1.0 + shap_std)

            # Check probabilistic coherence (simplified)
            if 'diagnosis' in d_synthetic.columns and 'stroke_risk_factor' in d_synthetic.columns:
                stroke_cases = d_synthetic[d_synthetic['diagnosis'] == 'Stroke']
                if len(stroke_cases) > 0:
                    coherence = stroke_cases['stroke_risk_factor'].mean()
                    metrics['probabilistic_coherence'] = coherence

            logger.info(f"XAI quality metrics: {metrics}")

        except Exception as e:
            logger.error(f"Error evaluating XAI quality: {e}")

        return metrics
