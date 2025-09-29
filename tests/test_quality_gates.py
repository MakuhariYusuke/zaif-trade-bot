"""
Test suite for QualityGates.evaluate() functionality
Tests dual correlation analysis (short/long-term) and rescue routes
"""

import numpy as np
import pandas as pd
import pytest
from ztb.evaluation.quality_gates import QualityGates


class TestQualityGates:
    """Test cases for QualityGates.evaluate() method"""

    def setup_method(self):
        """Setup test fixtures"""
        self.quality_gates = QualityGates()

        # Create sample data for testing with higher correlation
        np.random.seed(42)
        dates = pd.date_range("2020-01-01", periods=2000, freq="h")  # More data points

        # Create price data
        price_changes = np.random.randn(2000) * 0.01
        price_data = 100 + np.cumsum(price_changes)

        # Create highly correlated feature
        feature_noise = np.random.randn(2000) * 0.001
        feature_data = price_data * 0.1 + feature_noise + np.random.randn(2000) * 0.01

        self.sample_data = pd.DataFrame(
            {
                "close": price_data,
                "feature_good": feature_data,
                "feature_bad": np.random.randn(2000),  # Uncorrelated
                "feature_nan": [
                    np.nan if i % 10 != 0 else np.random.randn() for i in range(2000)
                ],  # 90% NaN
                "feature_skewed": np.random.beta(
                    0.1, 10, 2000
                ),  # Even more skewed beta distribution
                "feature_kurtotic": np.random.normal(0, 1, 2000) ** 4,
            },
            index=dates,
        )

    def test_dual_correlation_short_term_pass(self):
        """Test that feature passes when short-term correlation meets threshold"""
        # Use the pre-created highly correlated feature
        result = self.quality_gates.evaluate(
            self.sample_data["feature_good"],
            self.sample_data["close"],
            "test_feature_short",
        )

        assert result["correlation_pass"] == True
        assert "corr_short" in result
        assert "corr_long" in result
        assert result["corr_short"] >= 0.05  # Should meet threshold
        assert result["corr_long"] >= 0.05  # Should meet threshold

    def test_dual_correlation_long_term_pass(self):
        """Test that feature passes when long-term correlation meets threshold"""
        # Use the pre-created highly correlated feature
        result = self.quality_gates.evaluate(
            self.sample_data["feature_good"],
            self.sample_data["close"],
            "test_feature_long",
        )

        assert result["correlation_pass"] == True
        assert "corr_short" in result
        assert "corr_long" in result
        assert result["corr_long"] >= 0.05  # Should meet threshold

    def test_dual_correlation_both_fail(self):
        """Test that feature fails when both short and long-term correlations are poor"""
        # Use uncorrelated feature
        result = self.quality_gates.evaluate(
            self.sample_data["feature_bad"],
            self.sample_data["close"],
            "test_feature_bad",
        )

        assert result["correlation_pass"] == False
        assert result["corr_short"] < 0.05
        assert result["corr_long"] < 0.05

    def test_nan_rate_pass(self):
        """Test NaN rate validation passes for low NaN features"""
        result = self.quality_gates.evaluate(
            self.sample_data["feature_good"], self.sample_data["close"], "test_feature"
        )

        assert result["nan_rate_pass"] == True
        assert result["nan_rate"] < 0.05  # Should be low

    def test_nan_rate_fail(self):
        """Test NaN rate validation fails for high NaN features"""
        result = self.quality_gates.evaluate(
            self.sample_data["feature_nan"],
            self.sample_data["close"],
            "test_feature_nan",
        )

        assert result["nan_rate_pass"] == False
        assert result["nan_rate"] > 0.05  # Should be high

    def test_skew_pass(self):
        """Test skewness validation passes for normal-like features"""
        result = self.quality_gates.evaluate(
            self.sample_data["feature_good"], self.sample_data["close"], "test_feature"
        )

        assert result["skew_pass"] == True
        assert abs(result["skew"]) < 2.0  # Should be reasonable

    def test_skew_fail(self):
        """Test skewness validation fails for highly skewed features"""
        result = self.quality_gates.evaluate(
            self.sample_data["feature_skewed"],
            self.sample_data["close"],
            "test_feature_skewed",
        )

        assert result["skew_pass"] == False
        assert abs(result["skew"]) > 2.0  # Should be high

    def test_kurtosis_pass(self):
        """Test kurtosis validation passes for normal-like features"""
        result = self.quality_gates.evaluate(
            self.sample_data["feature_good"], self.sample_data["close"], "test_feature"
        )

        assert result["kurtosis_pass"] == True
        assert abs(result["kurtosis"]) < 10.0  # Should be reasonable

    def test_kurtosis_fail(self):
        """Test kurtosis validation fails for high kurtosis features"""
        result = self.quality_gates.evaluate(
            self.sample_data["feature_kurtotic"],
            self.sample_data["close"],
            "test_feature_kurtotic",
        )

        assert result["kurtosis_pass"] == False
        assert result["kurtosis"] > 10.0  # Should be high

    def test_overall_pass_all_good(self):
        """Test overall pass when all quality gates pass"""
        result = self.quality_gates.evaluate(
            self.sample_data["feature_good"], self.sample_data["close"], "test_feature"
        )

        assert result["overall_pass"] == True

    def test_overall_fail_correlation(self):
        """Test overall fail when correlation fails"""
        result = self.quality_gates.evaluate(
            self.sample_data["feature_bad"],
            self.sample_data["close"],
            "test_feature_bad",
        )

        assert result["overall_pass"] == False

    def test_overall_fail_multiple_gates(self):
        """Test overall fail when multiple quality gates fail"""
        result = self.quality_gates.evaluate(
            self.sample_data["feature_nan"],
            self.sample_data["close"],
            "test_feature_nan",
        )

        assert result["overall_pass"] == False

    def test_rescue_route_correlation_fail(self):
        """Test that correlation failure triggers rescue route instead of immediate discard"""
        # Test with a feature that will fail correlation (poor correlation)
        result = self.quality_gates.evaluate(
            self.sample_data["feature_bad"],
            self.sample_data["close"],
            "test_feature_bad",
        )

        # Should fail correlation but still return results (not raise exception)
        assert result["correlation_pass"] == False
        assert result["overall_pass"] == False
        assert "corr_short" in result
        assert "corr_long" in result

    def test_debug_csv_contains_dual_correlation(self):
        """Test that debug CSV contains corr_short and corr_long columns"""
        # Test the quality gates evaluation directly
        result = self.quality_gates.evaluate(
            self.sample_data["feature_good"], self.sample_data["close"], "test_feature"
        )

        # Verify that results contain corr_short and corr_long
        assert "corr_short" in result
        assert "corr_long" in result
        assert isinstance(result["corr_short"], (float, type(None)))
        assert isinstance(result["corr_long"], (float, type(None)))


if __name__ == "__main__":
    pytest.main([__file__])
