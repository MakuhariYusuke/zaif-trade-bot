#!/usr/bin/env python3
"""
Unit tests for statistical significance calculations.

Tests DSR and bootstrap p-values with synthetic data.
"""

import numpy as np
import pytest

from ztb.trading.backtest.metrics import MetricsCalculator

# 年間取引日数（一般的に252日）
TRADING_DAYS_PER_YEAR = 252


class TestStatisticalSignificance:
    """Test statistical significance calculations."""

    @pytest.fixture
    def fixed_seed(self):
        """Set fixed random seed for reproducible tests."""
        np.random.seed(42)
        return 42

    def test_iid_gaussian_returns_dsr_near_zero(self, fixed_seed: int) -> None:
        """Test that IID Gaussian returns produce DSR near zero."""
        # Generate IID Gaussian returns
        n_periods = 1000
        returns = np.random.normal(0.001, 0.02, n_periods)  # 0.1% mean, 2% vol

        # Calculate DSR
        dsr = MetricsCalculator.calculate_deflated_sharpe_ratio(returns)

        # For truly random returns, DSR should be close to zero
        assert dsr is not None
        assert -0.5 <= dsr <= 0.5, f"DSR {dsr} should be near zero for IID returns"

    def test_iid_gaussian_bootstrap_p_near_half(self, fixed_seed: int) -> None:
        """Test that bootstrap p-value is near 0.5 for identical strategies."""
        # Generate two identical IID return series
        n_periods = 500
        returns_a = np.random.normal(0.001, 0.02, n_periods)
        returns_b = np.random.normal(0.001, 0.02, n_periods)  # Same distribution

        # Calculate bootstrap p-value
        p_value = calculate_bootstrap_pvalue(returns_a, returns_b, n_bootstrap=200)

        # For identical distributions, p-value should be around 0.5
        assert p_value is not None
        assert (
            0.3 <= p_value <= 0.7
        ), f"Bootstrap p-value {p_value} should be near 0.5 for identical distributions"

    def test_dominant_rl_series_low_p_value(self, fixed_seed: int) -> None:
        """Test that dominant RL series produces p < 0.05."""
        # Generate returns: RL with consistent outperformance
        n_periods = 500
        base_returns = np.random.normal(0.001, 0.02, n_periods)

        # RL outperforms by 0.5% per period consistently
        rl_returns = base_returns + 0.005
        buy_hold_returns = base_returns

        # Calculate bootstrap p-value
        p_value = calculate_bootstrap_pvalue(
            rl_returns, buy_hold_returns, n_bootstrap=200
        )

        # RL should significantly outperform
        assert p_value is not None
        assert (
            p_value < 0.05
        ), f"Bootstrap p-value {p_value} should be < 0.05 for dominant RL"

    def test_bootstrap_p_value_consistency(self, fixed_seed: int) -> None:
        """Test that bootstrap p-values are consistent with fixed seed."""
        # Generate test data
        n_periods = 300
        returns_a = np.random.normal(0.002, 0.015, n_periods)
        returns_b = np.random.normal(0.001, 0.015, n_periods)

        # Calculate p-values multiple times with same seed
        p_values = []
        for _ in range(3):
            np.random.seed(42)  # Reset seed
            p_val = calculate_bootstrap_pvalue(returns_a, returns_b, n_bootstrap=100)
            p_values.append(p_val)

        # All p-values should be identical with fixed seed
        assert all(
            p == p_values[0] for p in p_values
        ), f"P-values not consistent: {p_values}"

    def test_dsr_with_extreme_skewness(self, fixed_seed: int) -> None:
        """Test DSR calculation with skewed returns."""
        # Generate returns with negative skewness (crash-like)
        n_periods = 1000
        normal_returns = np.random.normal(0.001, 0.02, n_periods)

        # Add occasional large negative returns
        crash_indices = np.random.choice(n_periods, size=10, replace=False)
        skewed_returns = normal_returns.copy()
        skewed_returns[crash_indices] -= 0.10  # -10% crashes

        # Calculate DSR
        dsr = calculate_deflated_sharpe_ratio(skewed_returns)

        # DSR should account for skewness
        assert dsr is not None
        # With negative skewness, DSR should be lower than regular Sharpe
        regular_sharpe = (
            np.mean(skewed_returns)
            / np.std(skewed_returns)
            * np.sqrt(TRADING_DAYS_PER_YEAR)
        )
        assert (
            dsr < regular_sharpe
        ), f"DSR {dsr} should be lower than regular Sharpe {regular_sharpe} for negatively skewed returns"

    def test_bootstrap_with_small_sample(self, fixed_seed: int) -> None:
        """Test bootstrap with small sample sizes."""
        # Small sample for fast CI testing
        n_periods = 50
        returns_a = np.random.normal(0.002, 0.03, n_periods)
        returns_b = np.random.normal(0.001, 0.03, n_periods)

        # Use small bootstrap count for CI
        p_value = calculate_bootstrap_pvalue(returns_a, returns_b, n_bootstrap=50)

        assert p_value is not None
        assert 0.0 <= p_value <= 1.0, f"P-value {p_value} should be in [0,1]"

    def test_dsr_with_zero_volatility(self, fixed_seed):
        """Test DSR handling of zero volatility."""
        # Constant returns (zero volatility)
        returns = np.full(100, 0.001)  # 0.1% constant return

        dsr = calculate_deflated_sharpe_ratio(returns)

        # Should handle zero volatility gracefully
        assert dsr is not None
        # With zero volatility, higher moments are undefined, but should not crash
        assert isinstance(dsr, (int, float))

    def test_bootstrap_identical_returns(self, fixed_seed: int) -> None:
        """Test bootstrap with identical return series."""
        returns = np.random.normal(0.001, 0.02, 200)

        p_value = calculate_bootstrap_pvalue(returns, returns, n_bootstrap=100)

        # Identical series should give p-value very close to 0.5
        assert p_value is not None
        assert (
            0.4 <= p_value <= 0.6
        ), f"P-value {p_value} should be near 0.5 for identical series"


if __name__ == "__main__":
    pytest.main([__file__])
