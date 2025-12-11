"""
Unit tests for ztb.metrics.statistical_validator module.
"""

import numpy as np
import pandas as pd

from ztb.metrics.statistical_validator import StatisticalValidator


class TestStatisticalValidator:
    """Test cases for StatisticalValidator class."""

    def setup_method(self):
        """Setup test data."""
        self.config = {
            "multiple_test_method": "bonferroni",
            "alpha_level": 0.05,
            "confidence_level": 0.95,
            "bootstrap_samples": 100,  # Small number for fast testing
            "min_sample_size": 10,
        }
        self.validator = StatisticalValidator(self.config)

        # Sample returns data (positive mean)
        np.random.seed(42)
        self.returns = np.random.normal(0.001, 0.02, 100)
        self.returns_pd = pd.Series(self.returns)

    def test_initialization(self):
        """Test initialization."""
        assert self.validator.multiple_test_method == "bonferroni"
        assert self.validator.alpha_level == 0.05

    def test_validate_performance_metrics(self):
        """Test validate_performance_metrics."""
        # Test with list
        results = self.validator.validate_performance_metrics(self.returns.tolist())
        assert results["valid"] is True
        assert "sharpe_ratio" in results
        assert "basic_stats" in results

        # Test with numpy array
        results_np = self.validator.validate_performance_metrics(self.returns)
        assert results_np["valid"] is True

        # Test with pandas Series
        # Note: The implementation converts to numpy array, so this should work
        results_pd = self.validator.validate_performance_metrics(self.returns_pd)
        assert results_pd["valid"] is True

    def test_validate_performance_metrics_insufficient_data(self):
        """Test validate_performance_metrics with insufficient data."""
        short_data = [0.01, 0.02, 0.03]
        results = self.validator.validate_performance_metrics(short_data)
        assert results["valid"] is False
        assert "Insufficient sample size" in results["error"]

    def test_validate_multiple_strategies(self):
        """Test validate_multiple_strategies."""
        strategy_returns = {
            "strat1": np.random.normal(0.001, 0.02, 100).tolist(),
            "strat2": np.random.normal(0.002, 0.02, 100).tolist(),
        }
        results = self.validator.validate_multiple_strategies(strategy_returns)
        assert results["valid"] is True
        assert "strategy_comparison" in results
        assert "best_strategy" in results["strategy_comparison"]

    def test_calculate_confidence_intervals(self):
        """Test calculate_confidence_intervals (indirectly via validate_performance_metrics)."""
        results = self.validator.validate_performance_metrics(self.returns.tolist())
        ci = results["sharpe_ratio"]["confidence_interval"]
        assert len(ci) == 2
        assert ci[0] < ci[1]
