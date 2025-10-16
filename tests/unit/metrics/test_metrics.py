"""
Unit tests for ztb.metrics.metrics module.
"""

import numpy as np
import pandas as pd
import pytest

from ztb.metrics.metrics import (
    calmar_ratio,
    calculate_all_metrics,
    classify_market_regime,
    drawdown_analysis,
    expected_value,
    max_drawdown,
    multi_market_backtest_analysis,
    p_mean_method,
    perform_statistical_tests,
    profit_factor,
    recovery_factor,
    rolling_analysis,
    seasonality_analysis,
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

    def test_expected_value_normal(self):
        """Test expected value calculation with normal returns."""
        returns = pd.Series([0.01, -0.005, 0.02, -0.01, 0.015])
        result = expected_value(returns)
        
        assert isinstance(result, float)
        # Expected value should be positive for this profitable series
        assert result > 0

    def test_expected_value_mixed(self):
        """Test expected value with mixed positive/negative returns."""
        returns = pd.Series([0.02, -0.03, 0.01, -0.02, 0.025])
        result = expected_value(returns)
        
        assert isinstance(result, float)
        # Should calculate: (win_rate * avg_win) - ((1-win_rate) * avg_loss)

    def test_expected_value_all_losses(self):
        """Test expected value with all negative returns."""
        returns = pd.Series([-0.01, -0.02, -0.005, -0.015])
        result = expected_value(returns)
        
        assert isinstance(result, float)
        assert result < 0  # Should be negative for all losses

    def test_expected_value_empty(self):
        """Test expected value with empty returns."""
        returns = pd.Series([])
        result = expected_value(returns)
        
        assert isinstance(result, float)
        assert result == 0.0

    def test_recovery_factor_normal(self):
        """Test recovery factor calculation with normal returns."""
        returns = pd.Series([0.01, 0.02, -0.01, 0.03, -0.02, 0.015])
        result = recovery_factor(returns)
        
        assert isinstance(result, float)
        assert result > 0

    def test_recovery_factor_with_risk_free(self):
        """Test recovery factor with risk-free rate."""
        returns = pd.Series([0.02, 0.03, 0.01, 0.04, 0.02])
        rf = 0.01
        result = recovery_factor(returns, rf=rf)
        
        assert isinstance(result, float)

    def test_recovery_factor_empty(self):
        """Test recovery factor with empty returns."""
        returns = pd.Series([])
        result = recovery_factor(returns)
        
        assert isinstance(result, float)
        assert result == 0.0

    def test_rolling_analysis_normal(self):
        """Test rolling analysis with normal returns."""
        returns = pd.Series([0.01, -0.005, 0.02, 0.01, -0.01, 0.005, -0.02, 0.015])
        result = rolling_analysis(returns, window=5, step=1)
        
        assert isinstance(result, pd.DataFrame)
        assert len(result) > 0
        assert 'sharpe_ratio' in result.columns
        assert 'expected_value' in result.columns
        assert 'recovery_factor' in result.columns

    def test_rolling_analysis_small_window(self):
        """Test rolling analysis with window larger than data."""
        returns = pd.Series([0.01, 0.02, 0.03])
        result = rolling_analysis(returns, window=5)
        
        assert isinstance(result, pd.DataFrame)
        assert len(result) == 0  # Should return empty DataFrame

    def test_rolling_analysis_empty(self):
        """Test rolling analysis with empty returns."""
        returns = pd.Series([])
        result = rolling_analysis(returns, window=5)
        
        assert isinstance(result, pd.DataFrame)
        assert len(result) == 0

    def test_drawdown_analysis_normal(self):
        """Test drawdown analysis with equity curve containing drawdowns."""
        # Create equity curve with drawdowns
        returns = np.array([0.01, -0.02, 0.015, -0.03, 0.025, -0.01, 0.02])
        equity = np.cumprod(1 + returns)
        
        result = drawdown_analysis(equity)
        
        assert isinstance(result, dict)
        assert 'max_drawdown' in result
        assert 'num_drawdowns' in result
        assert 'drawdown_periods' in result
        assert result['num_drawdowns'] > 0
        assert result['max_drawdown'] < 0  # Drawdown should be negative

    def test_drawdown_analysis_no_drawdowns(self):
        """Test drawdown analysis with no drawdowns."""
        # Create equity curve with only gains
        returns = np.array([0.01, 0.02, 0.015, 0.03, 0.025])
        equity = np.cumprod(1 + returns)
        
        result = drawdown_analysis(equity)
        
        assert isinstance(result, dict)
        assert result['num_drawdowns'] == 0
        assert result['max_drawdown'] == 0.0

    def test_drawdown_analysis_empty(self):
        """Test drawdown analysis with empty equity curve."""
        equity = np.array([])
        result = drawdown_analysis(equity)
        
        assert isinstance(result, dict)
        assert result['num_drawdowns'] == 0
        assert result['max_drawdown'] == 0.0

    def test_calculate_all_metrics_with_new_fields(self):
        """Test that calculate_all_metrics includes new fields."""
        returns = pd.Series([0.01, -0.005, 0.02, -0.01, 0.015])
        result = calculate_all_metrics(returns)
        
        assert isinstance(result, dict)
        
        # Check that new fields are present
        new_fields = ['expected_value', 'recovery_factor']
        for field in new_fields:
            assert field in result
            assert isinstance(result[field], (int, float))

    def test_seasonality_analysis_basic(self):
        """Test seasonality analysis with basic data."""
        # Create sample returns for a year (365 days)
        np.random.seed(42)
        returns = np.random.normal(0.001, 0.02, 365)
        dates = pd.date_range(start='2023-01-01', periods=365, freq='D')

        result = seasonality_analysis(returns, dates)

        assert isinstance(result, dict)
        assert 'monthly_analysis' in result
        assert 'quarterly_analysis' in result
        assert 'seasonality_assessment' in result

        # Check monthly analysis structure
        monthly = result['monthly_analysis']
        assert 'stats' in monthly
        assert 'sharpe_ratios' in monthly
        assert 'best_month' in monthly
        assert 'worst_month' in monthly

        # Check quarterly analysis structure
        quarterly = result['quarterly_analysis']
        assert 'stats' in quarterly
        assert 'sharpe_ratios' in quarterly
        assert 'best_quarter' in quarterly
        assert 'worst_quarter' in quarterly

    def test_seasonality_analysis_without_dates(self):
        """Test seasonality analysis without providing dates."""
        returns = np.array([0.01, -0.005, 0.02, 0.01, -0.01] * 20)  # 100 periods

        result = seasonality_analysis(returns)

        assert isinstance(result, dict)
        assert 'monthly_analysis' in result
        assert 'quarterly_analysis' in result

    def test_seasonality_analysis_empty(self):
        """Test seasonality analysis with empty returns."""
        returns = np.array([])
        result = seasonality_analysis(returns)

        assert isinstance(result, dict)
        assert len(result) == 0

    def test_seasonality_analysis_single_period(self):
        """Test seasonality analysis with single period data."""
        returns = np.array([0.01])
        dates = pd.date_range(start='2023-01-01', periods=1, freq='D')

        result = seasonality_analysis(returns, dates)

        assert isinstance(result, dict)
        # Should still have basic structure even with limited data
        assert 'monthly_analysis' in result

    def test_classify_market_regime_basic(self):
        """Test market regime classification with basic price data."""
        # Create sample price data with different trends
        np.random.seed(42)
        base_price = 100
        prices = [base_price]

        # Add sideways period
        for _ in range(10):
            prices.append(prices[-1] + np.random.normal(0, 0.5))

        # Add bull period
        for _ in range(10):
            prices.append(prices[-1] + np.random.normal(2, 0.5))

        # Add bear period
        for _ in range(10):
            prices.append(prices[-1] + np.random.normal(-2, 0.5))

        prices = pd.Series(prices)

        regimes = classify_market_regime(prices, window=5)

        assert isinstance(regimes, pd.Series)
        assert len(regimes) == len(prices)
        assert all(regime in ['unknown', 'bull', 'bear', 'sideways', 'weak_bull', 'weak_bear', 'volatile_sideways']
                  for regime in regimes.unique())

    def test_multi_market_backtest_analysis(self):
        """Test multi-market backtest analysis."""
        # Create sample data
        np.random.seed(42)
        prices = pd.Series(np.cumprod(1 + np.random.normal(0.001, 0.02, 100)))
        returns = prices.pct_change().fillna(0)

        result = multi_market_backtest_analysis(returns, prices, regime_window=10)

        assert isinstance(result, dict)
        assert 'regime_performance' in result
        assert 'regime_distribution' in result
        assert 'regime_transitions' in result

        # Check that we have some regime analysis
        assert isinstance(result['regime_performance'], dict)
        assert isinstance(result['regime_distribution'], dict)


class TestStatisticalTests:
    """Test cases for statistical testing functions."""

    def test_p_mean_method_arithmetic(self):
        """Test p-mean method with arithmetic mean."""
        p_values = [0.03, 0.07, 0.02]
        result = p_mean_method(p_values, 'arithmetic')
        
        assert isinstance(result, float)
        assert 0.0 <= result <= 1.0
        # Arithmetic mean should be (0.03 + 0.07 + 0.02) / 3 = 0.04
        assert abs(result - 0.04) < 1e-6

    def test_p_mean_method_geometric(self):
        """Test p-mean method with geometric mean."""
        p_values = [0.03, 0.07, 0.02]
        result = p_mean_method(p_values, 'geometric')
        
        assert isinstance(result, float)
        assert 0.0 <= result <= 1.0
        # Should be less than arithmetic mean due to geometric averaging
        arithmetic_mean = p_mean_method(p_values, 'arithmetic')
        assert result <= arithmetic_mean

    def test_p_mean_method_empty_list(self):
        """Test p-mean method with empty list."""
        result = p_mean_method([])
        assert result == 1.0

    def test_p_mean_method_single_value(self):
        """Test p-mean method with single value."""
        p_values = [0.05]
        result_arithmetic = p_mean_method(p_values, 'arithmetic')
        result_geometric = p_mean_method(p_values, 'geometric')
        
        assert result_arithmetic == 0.05
        assert abs(result_geometric - 0.05) < 1e-10  # Allow for floating point precision

    def test_perform_statistical_tests_normal_case(self):
        """Test statistical tests between two datasets."""
        data_a = [0.01, 0.02, -0.01, 0.03, -0.02, 0.01, 0.02]
        data_b = [0.02, 0.01, 0.00, 0.04, -0.01, 0.02, 0.01]
        
        result = perform_statistical_tests(data_a, data_b)
        
        assert isinstance(result, dict)
        assert 't_statistic' in result
        assert 'p_value' in result
        assert 'significant' in result
        assert 'mean_a' in result
        assert 'mean_b' in result
        assert 'effect_size' in result
        
        assert isinstance(result['t_statistic'], float)
        assert isinstance(result['p_value'], float)
        assert isinstance(result['significant'], bool)
        assert isinstance(result['mean_a'], float)
        assert isinstance(result['mean_b'], float)
        assert isinstance(result['effect_size'], float)
        
        assert 0.0 <= result['p_value'] <= 1.0

    def test_perform_statistical_tests_insufficient_data(self):
        """Test statistical tests with insufficient data."""
        data_a = [0.01]
        data_b = [0.02]
        
        result = perform_statistical_tests(data_a, data_b)
        
        # Should return default values
        assert result['t_statistic'] == 0.0
        assert result['p_value'] == 1.0
        assert result['significant'] == False

    def test_multi_market_backtest_with_statistical_tests(self):
        """Test that multi-market backtest includes statistical tests."""
        # Create synthetic data with different regimes
        dates = pd.date_range('2020-01-01', periods=100, freq='D')
        prices = pd.Series(np.random.randn(100).cumsum() + 100, index=dates)
        returns = pd.Series(np.random.randn(100) * 0.01, index=dates)
        
        result = multi_market_backtest_analysis(returns, prices)
        
        assert 'statistical_tests' in result
        assert isinstance(result['statistical_tests'], dict)
        
        # If we have multiple regimes, we should have statistical tests
        if len(result['regime_performance']) >= 2:
            assert len(result['statistical_tests']) > 0