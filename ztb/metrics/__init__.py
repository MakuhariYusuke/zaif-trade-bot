"""
Metrics module for trading performance calculations.

This module provides centralized, robust implementations of statistical and
trading performance metrics with comprehensive error handling.
"""

from .metrics import (
    autocorrelation,
    BacktestMetrics,
    calculate_all_metrics,
    classify_market_regime,
    coefficient_of_variation,
    kurtosis,
    MetricsCalculator,
    max_drawdown,
    multi_market_backtest_analysis,
    profit_factor,
    seasonality_analysis,
    sharpe_ratio,
    skewness,
    sortino_ratio,
    test_normality,
)

__all__ = [
    "sharpe_ratio",
    "max_drawdown",
    "sortino_ratio",
    "profit_factor",
    "coefficient_of_variation",
    "skewness",
    "kurtosis",
    "test_normality",
    "autocorrelation",
    "BacktestMetrics",
    "calculate_all_metrics",
    "classify_market_regime",
    "multi_market_backtest_analysis",
    "seasonality_analysis",
    "MetricsCalculator",
]
