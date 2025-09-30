#!/usr/bin/env python3
"""
metrics.py
Robust implementation of trading performance metrics
"""

from typing import TypedDict, Union

import numpy as np
import pandas as pd

from ztb.utils.metrics.trading_metrics import sharpe_ratio as _sharpe_ratio
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
    returns: Union[pd.Series, np.ndarray], rf: float = 0.0, period_per_year: int = TRADING_DAYS_PER_YEAR
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
    return _sharpe_ratio(np.asarray(returns), rf, period_per_year)


def sortino_ratio(
    returns: Union[pd.Series, np.ndarray],
    rf: float = 0.0,
    period_per_year: int = 252,
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
    return (mean_return / downside_std) * np.sqrt(period_per_year)


def max_drawdown(equity_curve: Union[pd.Series, np.ndarray]) -> float:
    """
    Calculate maximum drawdown from equity curve

    Args:
        equity_curve: Cumulative returns or equity values

    Returns:
        Maximum drawdown (negative value)
    """
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
    returns: Union[pd.Series, np.ndarray], rf: float = 0.0, period_per_year: int = TRADING_DAYS_PER_YEAR
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


def win_rate(returns: Union[pd.Series, np.ndarray]) -> float:
    """
    Calculate win rate (percentage of positive returns)

    Args:
        returns: Return series

    Returns:
        Win rate (0 to 1)
    """
    returns = np.asarray(returns)

    if len(returns) == 0:
        return 0.0

    # Remove NaN values
    returns = returns[~np.isnan(returns)]

    if len(returns) == 0:
        return 0.0

    positive_returns = returns > 0
    return float(np.mean(positive_returns))


def profit_factor(returns: Union[pd.Series, np.ndarray]) -> float:
    """
    Calculate profit factor (gross profit / gross loss)

    Args:
        returns: Return series

    Returns:
        Profit factor
    """
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
    returns: Union[pd.Series, np.ndarray], rf: float = 0.0, period_per_year: int = TRADING_DAYS_PER_YEAR
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
    n_periods = 252
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
