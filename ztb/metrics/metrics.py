#!/usr/bin/env python3
"""
metrics.py
Robust implementation of trading performance metrics
"""

from typing import Any, TypedDict, Union, cast

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from ztb.utils.metrics.trading_metrics import sharpe_ratio as _sharpe_ratio

from ztb.utils.errors import safe_operation

# 年間取引日数（一般的に252日）
TRADING_DAYS_PER_YEAR = 252


class MetricsResult(TypedDict):
    """Type definition for metrics calculation results"""

    total_return: float
    annual_return: float
    volatility: float
    sharpe_ratio: float
    sortino_ratio: float
    calmar_ratio: float
    max_drawdown: float
    win_rate: float
    profit_factor: float
    num_periods: int


def sharpe_ratio(
    returns: Union[pd.Series, NDArray[Any]],
    rf: float = 0.0,
    period_per_year: int = TRADING_DAYS_PER_YEAR,
) -> float:
    """
    Calculate Sharpe ratio with robust error handling

    Args:
        returns: Return series
        rf: Risk-free rate (annual)
        period_per_year: Number of periods per year (252 for daily, 365 for crypto)

    Returns:
        Sharpe ratio
    """
    return safe_operation(
        logger=None,  # Use default logger
        operation=lambda: _sharpe_ratio(np.asarray(returns), rf, period_per_year),
        context="sharpe_ratio_calculation",
        default_result=0.0,  # Return 0.0 on failure
    )


def sortino_ratio(
    returns: Union[pd.Series, NDArray[Any]],
    rf: float = 0.0,
    period_per_year: int = TRADING_DAYS_PER_YEAR,
    downside_floor: float = 0.0,
) -> float:
    """
    Calculate Sortino ratio (downside deviation instead of total volatility)

    Args:
        returns: Return series
        rf: Risk-free rate (annual)
        period_per_year: Number of periods per year
        downside_floor: Minimum acceptable return (default: 0)

    Returns:
        Sortino ratio
    """
    return safe_operation(
        logger=None,  # Use default logger
        operation=lambda: _sortino_ratio_impl(returns, rf, period_per_year, downside_floor),
        context="sortino_ratio_calculation",
        default_result=0.0,  # Return 0.0 on failure
    )


def _sortino_ratio_impl(
    returns: Union[pd.Series, NDArray[Any]],
    rf: float = 0.0,
    period_per_year: int = TRADING_DAYS_PER_YEAR,
    downside_floor: float = 0.0,
) -> float:
    """Implementation of Sortino ratio calculation."""
    returns = np.asarray(returns)

    if len(returns) == 0:
        return 0.0

    # Remove NaN values
    returns = returns[~np.isnan(returns)]

    if len(returns) == 0:
        return 0.0

    # Calculate excess returns
    excess_returns = returns - (rf / period_per_year)

    # Calculate downside returns (below the floor)
    downside_returns = excess_returns - downside_floor
    downside_returns = downside_returns[downside_returns < 0]

    if len(downside_returns) == 0:
        return np.inf if np.mean(excess_returns) > 0 else 0.0

    # Calculate downside deviation
    downside_std = np.std(downside_returns, ddof=1)

    if downside_std == 0 or np.isnan(downside_std):
        return 0.0

    mean_return = np.mean(excess_returns)
    return (mean_return / downside_std) * np.sqrt(period_per_year)  # type: ignore


def max_drawdown(equity_curve: Union[pd.Series, NDArray[Any]]) -> float:
    """
    Calculate maximum drawdown from equity curve

    Args:
        equity_curve: Cumulative returns or equity values

    Returns:
        Maximum drawdown (negative value)
    """
    return safe_operation(
        logger=None,  # Use default logger
        operation=lambda: _max_drawdown_impl(equity_curve),
        context="max_drawdown_calculation",
        default_result=0.0,  # Return 0.0 on failure
    )


def _max_drawdown_impl(equity_curve: Union[pd.Series, NDArray[Any]]) -> float:
    """Implementation of maximum drawdown calculation."""
    equity_curve = np.asarray(equity_curve)

    if len(equity_curve) == 0:
        return 0.0

    # Remove NaN values
    equity_curve = equity_curve[~np.isnan(equity_curve)]

    if len(equity_curve) == 0:
        return 0.0

    # Calculate running maximum (peak)
    running_max = np.maximum.accumulate(equity_curve)

    # Calculate drawdown
    drawdown = (equity_curve - running_max) / running_max

    # Return maximum drawdown (most negative value)
    return float(np.min(drawdown))


def calmar_ratio(
    returns: Union[pd.Series, NDArray[Any]],
    rf: float = 0.0,
    period_per_year: int = TRADING_DAYS_PER_YEAR,
) -> float:
    """
    Calculate Calmar ratio (Annual return / Max Drawdown)

    Args:
        returns: Return series
        rf: Risk-free rate (annual)
        period_per_year: Number of periods per year

    Returns:
        Calmar ratio
    """
    return safe_operation(
        logger=None,  # Use default logger
        operation=lambda: _calmar_ratio_impl(returns, rf, period_per_year),
        context="calmar_ratio_calculation",
        default_result=0.0,  # Return 0.0 on failure
    )


def _calmar_ratio_impl(
    returns: Union[pd.Series, NDArray[Any]],
    rf: float = 0.0,
    period_per_year: int = TRADING_DAYS_PER_YEAR,
) -> float:
    """Implementation of Calmar ratio calculation."""
    returns = np.asarray(returns)

    if len(returns) == 0:
        return 0.0

    # Remove NaN values
    returns = returns[~np.isnan(returns)]

    if len(returns) == 0:
        return 0.0

    # Calculate annual return
    total_return = np.prod(1 + returns) - 1
    periods = len(returns)
    annual_return = (1 + total_return) ** (period_per_year / periods) - 1

    # Calculate maximum drawdown
    equity_curve = np.cumprod(1 + returns)
    mdd = max_drawdown(equity_curve)

    if mdd == 0 or np.isnan(mdd):
        return np.inf if annual_return > rf else 0.0

    return float((annual_return - rf) / abs(mdd))


def win_rate(returns: Union[pd.Series, NDArray[Any]]) -> float:
    """
    Calculate win rate (percentage of positive returns)

    Args:
        returns: Return series

    Returns:
        Win rate (0 to 1)
    """
    return safe_operation(
        logger=None,  # Use default logger
        operation=lambda: _win_rate_impl(returns),
        context="win_rate_calculation",
        default_result=0.0,  # Return 0.0 on failure
    )


def _win_rate_impl(returns: Union[pd.Series, NDArray[Any]]) -> float:
    """Implementation of win rate calculation."""
    returns = np.asarray(returns)

    if len(returns) == 0:
        return 0.0

    # Remove NaN values
    returns = returns[~np.isnan(returns)]

    if len(returns) == 0:
        return 0.0

    positive_returns = returns > 0
    return float(np.mean(positive_returns))


def profit_factor(returns: Union[pd.Series, NDArray[Any]]) -> float:
    """
    Calculate profit factor (gross profit / gross loss)

    Args:
        returns: Return series

    Returns:
        Profit factor
    """
    return safe_operation(
        logger=None,  # Use default logger
        operation=lambda: _profit_factor_impl(returns),
        context="profit_factor_calculation",
        default_result=1.0,  # Return 1.0 on failure (neutral)
    )


def _profit_factor_impl(returns: Union[pd.Series, NDArray[Any]]) -> float:
    """Implementation of profit factor calculation."""
    returns = np.asarray(returns)

    if len(returns) == 0:
        return 1.0

    # Remove NaN values
    returns = returns[~np.isnan(returns)]

    if len(returns) == 0:
        return 1.0

    gross_profit = np.sum(returns[returns > 0])
    gross_loss = np.sum(np.abs(returns[returns < 0]))

    if gross_loss == 0:
        return float(np.inf if gross_profit > 0 else 1.0)

    return float(gross_profit / gross_loss)


def calculate_all_metrics(
    returns: Union[pd.Series, NDArray[Any]],
    rf: float = 0.0,
    period_per_year: int = TRADING_DAYS_PER_YEAR,
) -> MetricsResult:
    """
    Calculate all performance metrics at once

    Args:
        returns: Return series
        rf: Risk-free rate (annual)
        period_per_year: Number of periods per year

    Returns:
        Dictionary with all metrics
    """
    return safe_operation(
        logger=None,  # Use default logger
        operation=lambda: _calculate_all_metrics_impl(returns, rf, period_per_year),
        context="all_metrics_calculation",
        default_result=MetricsResult(
            total_return=0.0,
            annual_return=0.0,
            volatility=0.0,
            sharpe_ratio=0.0,
            sortino_ratio=0.0,
            calmar_ratio=0.0,
            max_drawdown=0.0,
            win_rate=0.0,
            profit_factor=1.0,
            num_periods=0,
        ),  # Return default metrics on failure
    )


def _calculate_all_metrics_impl(
    returns: Union[pd.Series, NDArray[Any]],
    rf: float = 0.0,
    period_per_year: int = TRADING_DAYS_PER_YEAR,
) -> MetricsResult:
    """Implementation of all metrics calculation."""
    returns = np.asarray(returns)

    # Basic statistics
    total_return = np.prod(1 + returns) - 1 if len(returns) > 0 else 0.0
    annual_return = (
        (1 + total_return) ** (period_per_year / len(returns)) - 1
        if len(returns) > 0
        else 0.0
    )
    volatility = (
        np.std(returns, ddof=1) * np.sqrt(period_per_year) if len(returns) > 1 else 0.0
    )

    # Performance metrics
    sharpe = sharpe_ratio(returns, rf, period_per_year)
    sortino = sortino_ratio(returns, rf, period_per_year)
    calmar = calmar_ratio(returns, rf, period_per_year)

    # Risk metrics
    equity_curve = np.cumprod(1 + returns) if len(returns) > 0 else np.array([1.0])
    mdd = max_drawdown(equity_curve)

    # Trade statistics
    win_pct = win_rate(returns)
    pf = profit_factor(returns)

    return {
        "total_return": float(total_return),
        "annual_return": float(annual_return),
        "volatility": float(volatility),
        "sharpe_ratio": float(sharpe),
        "sortino_ratio": float(sortino),
        "calmar_ratio": float(calmar),
        "max_drawdown": float(mdd),
        "win_rate": float(win_pct),
        "profit_factor": float(pf),
        "num_periods": int(len(returns)),
    }


if __name__ == "__main__":
    # Test with synthetic data
    np.random.seed(42)

    # Generate synthetic return series
    n_periods = TRADING_DAYS_PER_YEAR
    returns = np.random.normal(
        0.001, 0.02, n_periods
    )  # Daily returns: 0.1% mean, 2% std

    print("Testing metrics with synthetic data:")
    print(f"Returns shape: {returns.shape}")
    print(f"Mean return: {np.mean(returns):.4f}")
    print(f"Std return: {np.std(returns):.4f}")

    # Calculate all metrics
    metrics = calculate_all_metrics(returns)

    print("\nCalculated metrics:")
    for key, value in metrics.items():
        if isinstance(value, float):
            print(f"{key}: {value:.4f}")
        else:
            print(f"{key}: {value}")

    # Test edge cases
    print("\nTesting edge cases:")

    # Empty returns
    empty_metrics = calculate_all_metrics(np.array([]))
    print(f"Empty returns Sharpe: {empty_metrics['sharpe_ratio']}")

    # Constant returns
    constant_returns = np.full(100, 0.01)
    constant_metrics = calculate_all_metrics(constant_returns)
    print(f"Constant returns Sharpe: {constant_metrics['sharpe_ratio']}")

    # All negative returns
    negative_returns = np.random.normal(-0.001, 0.02, 100)
    negative_metrics = calculate_all_metrics(negative_returns)
    print(f"Negative returns Sharpe: {negative_metrics['sharpe_ratio']}")

    print("All tests completed successfully!")
