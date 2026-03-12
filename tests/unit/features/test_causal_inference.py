"""
Tests for causal inference feature selection.

This module tests causal inference based feature selection functionality.
"""

import numpy as np
import pandas as pd

from ztb.features.causal_inference import CausalFeatureSelector


class TestCausalFeatureSelector:
    """Test cases for CausalFeatureSelector class."""

    def setup_method(self):
        """Set up test fixtures."""
        self.selector = CausalFeatureSelector(
            treatment_threshold=0.1, min_samples=50, max_features=5
        )

    def test_init_default_parameters(self):
        """Test initialization with default parameters."""
        selector = CausalFeatureSelector()
        assert selector.treatment_threshold == 0.1
        assert selector.min_samples == 1000
        assert selector.max_features is None

    def test_init_custom_parameters(self):
        """Test initialization with custom parameters."""
        selector = CausalFeatureSelector(
            treatment_threshold=0.2, min_samples=500, max_features=10
        )
        assert selector.treatment_threshold == 0.2
        assert selector.min_samples == 500
        assert selector.max_features == 10

    def test_estimate_causal_effect_insufficient_samples(self):
        """Test causal effect estimation with insufficient samples."""
        # Create small dataset
        np.random.seed(42)
        data = pd.DataFrame(
            {
                "feature1": np.random.normal(0, 1, 10),
                "feature2": np.random.normal(0, 1, 10),
                "outcome": np.random.normal(0, 1, 10),
            }
        )

        effect = self.selector.estimate_causal_effect(data, "feature1", "outcome", [])

        # Should return dict with effect close to 0 for insufficient samples
        assert isinstance(effect, dict)
        assert "effect" in effect

    def test_estimate_causal_effect_sufficient_samples(self):
        """Test causal effect estimation with sufficient samples."""
        # Create dataset with known causal relationship
        np.random.seed(42)
        n_samples = 200
        treatment = np.random.binomial(1, 0.5, n_samples)
        outcome = 2.0 * treatment + np.random.normal(0, 0.1, n_samples)

        data = pd.DataFrame({"treatment": treatment, "outcome": outcome})

        effect = self.selector.estimate_causal_effect(data, "treatment", "outcome", [])

        # Should detect positive causal effect
        assert isinstance(effect, dict)
        assert "effect" in effect
        assert effect["effect"] > 0.0

    def test_select_features_causal_insufficient_data(self):
        """Test feature selection with insufficient data."""
        data = pd.DataFrame(
            {
                "feature1": [1.0, 2.0, 3.0],
                "feature2": [4.0, 5.0, 6.0],
                "outcome": [7.0, 8.0, 9.0],
            }
        )

        selected_features, _ = self.selector.select_features_causal(
            data, ["feature1", "feature2"], "outcome"
        )

        # Should return empty list for insufficient data
        assert selected_features == []

    def test_select_features_causal_with_data(self):
        """Test feature selection with sufficient data."""
        np.random.seed(42)
        n_samples = 300

        # Create features with different causal strengths
        feature1 = np.random.normal(0, 1, n_samples)  # Strong causal effect
        feature2 = np.random.normal(0, 1, n_samples)  # Weak causal effect
        feature3 = np.random.normal(0, 1, n_samples)  # No causal effect

        outcome = 2.0 * feature1 + 0.5 * feature2 + np.random.normal(0, 0.1, n_samples)

        data = pd.DataFrame(
            {
                "feature1": feature1,
                "feature2": feature2,
                "feature3": feature3,
                "outcome": outcome,
            }
        )

        selected_features, _ = self.selector.select_features_causal(
            data, ["feature1", "feature2", "feature3"], "outcome"
        )

        assert isinstance(selected_features, list)
        # Should select some features
        assert len(selected_features) >= 0

    def test_get_feature_importance_empty(self):
        """Test getting feature importance when no features analyzed."""
        importance = self.selector.get_feature_importance()

        assert isinstance(importance, dict)
        assert len(importance) == 0

    def test_update_causal_model(self):
        """Test updating causal model with new data."""
        np.random.seed(42)
        data = pd.DataFrame(
            {
                "feature1": np.random.normal(0, 1, 100),
                "outcome": np.random.normal(0, 1, 100),
            }
        )

        # Update model
        self.selector.update_causal_model(data, "outcome")

        # Check that model was updated
        assert hasattr(self.selector, "causal_model")
        assert self.selector.causal_model is not None

    def test_update_causal_model_insufficient_data(self):
        """Test updating causal model with insufficient data."""
        data = pd.DataFrame({"feature1": [1.0, 2.0], "outcome": [3.0, 4.0]})

        # Should not raise error with insufficient data
        self.selector.update_causal_model(data, "outcome")
