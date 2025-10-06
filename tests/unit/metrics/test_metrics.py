"""
Unit tests for ztb.metrics.metrics module.
"""

import numpy as np
import pandas as pd
import pytest

from ztb.metrics.metrics import (
    calmar_ratio,
    calculate_all_metrics,
    max_drawdown,
    profit_factor,
    sharpe_ratio,
    sortino_ratio,
    win_rate,
)


class TestMetrics:
    """Test cases for metrics functions."""

    def test_sharpe_ratio_normal(self):
        """Test Sharpe ratio calculation with normal returns."""
        returns = pd.Series([0.01, 0.02, -0.01, 0.03, -0.02])
        result = sharpe_ratio(returns)
        
        assert isinstance(result, float)
        assert result > 0  # Should be positive for this data

    def test_sharpe_ratio_zero_volatility(self):
        """Test Sharpe ratio with zero volatility."""
        returns = pd.Series([0.01, 0.01, 0.01, 0.01, 0.01])
        result = sharpe_ratio(returns)
        
        assert isinstance(result, float)
        assert result == 0.0  # Zero Sharpe with zero volatility

    def test_sharpe_ratio_with_risk_free(self):
        """Test Sharpe ratio with risk-free rate."""
        returns = pd.Series([0.02, 0.03, 0.01, 0.04, 0.02])
        rf = 0.01  # 1% risk-free rate
        result = sharpe_ratio(returns, rf=rf)
        
        assert isinstance(result, float)

    def test_sharpe_ratio_empty(self):
        """Test Sharpe ratio with empty returns."""
        returns = pd.Series([])
        result = sharpe_ratio(returns)
        
        assert result == 0.0

    def test_sortino_ratio_normal(self):
        """Test Sortino ratio calculation."""
        returns = pd.Series([0.01, 0.02, -0.01, 0.03, -0.02])
        result = sortino_ratio(returns)
        
        assert isinstance(result, float)
        assert result > 0

    def test_sortino_ratio_no_downside(self):
        """Test Sortino ratio with no downside returns."""
        returns = pd.Series([0.01, 0.02, 0.03, 0.04, 0.05])
        result = sortino_ratio(returns)
        
        assert isinstance(result, float)
        assert np.isinf(result)  # Infinite Sortino with no downside

    def test_max_drawdown_normal(self):
        """Test maximum drawdown calculation."""
        equity = pd.Series([100, 105, 102, 108, 95, 110])
        result = max_drawdown(equity)
        
        assert isinstance(result, float)
        assert result < 0  # Should be negative (drawdown magnitude)
        assert result >= -1.0  # Should be >= -100%

    def test_max_drawdown_no_drawdown(self):
        """Test maximum drawdown with increasing equity."""
        equity = pd.Series([100, 105, 110, 115, 120])
        result = max_drawdown(equity)
        
        assert isinstance(result, float)
        assert result == 0.0

    def test_calmar_ratio_normal(self):
        """Test Calmar ratio calculation."""
        returns = pd.Series([0.01, 0.02, -0.01, 0.03, -0.02])
        result = calmar_ratio(returns)
        
        assert isinstance(result, float)

    def test_win_rate_normal(self):
        """Test win rate calculation."""
        returns = pd.Series([0.01, -0.02, 0.03, -0.01, 0.02])
        result = win_rate(returns)
        
        assert isinstance(result, float)
        assert 0 <= result <= 1

    def test_win_rate_all_wins(self):
        """Test win rate with all positive returns."""
        returns = pd.Series([0.01, 0.02, 0.03, 0.04, 0.05])
        result = win_rate(returns)
        
        assert result == 1.0

    def test_win_rate_all_losses(self):
        """Test win rate with all negative returns."""
        returns = pd.Series([-0.01, -0.02, -0.03, -0.04, -0.05])
        result = win_rate(returns)
        
        assert result == 0.0

    def test_profit_factor_normal(self):
        """Test profit factor calculation."""
        returns = pd.Series([0.01, -0.02, 0.03, -0.01, 0.02])
        result = profit_factor(returns)
        
        assert isinstance(result, float)
        assert result > 0

    def test_profit_factor_no_losses(self):
        """Test profit factor with no losses."""
        returns = pd.Series([0.01, 0.02, 0.03, 0.04, 0.05])
        result = profit_factor(returns)
        
        assert np.isinf(result)

    def test_profit_factor_no_wins(self):
        """Test profit factor with no wins."""
        returns = pd.Series([-0.01, -0.02, -0.03, -0.04, -0.05])
        result = profit_factor(returns)
        
        assert result == 0.0

    def test_calculate_all_metrics_normal(self):
        """Test calculation of all metrics."""
        returns = pd.Series([0.01, 0.02, -0.01, 0.03, -0.02])
        
        result = calculate_all_metrics(returns)
        
        assert isinstance(result, dict)
        expected_keys = [
            "total_return", "annual_return", "volatility", "sharpe_ratio",
            "sortino_ratio", "calmar_ratio", "max_drawdown", "win_rate",
            "profit_factor", "num_periods"
        ]
        
        for key in expected_keys:
            assert key in result
            assert isinstance(result[key], (int, float))

    def test_calculate_all_metrics_empty(self):
        """Test calculation of all metrics with empty data."""
        returns = pd.Series([])
        
        result = calculate_all_metrics(returns)
        
        assert isinstance(result, dict)
        # Should return default values for empty data