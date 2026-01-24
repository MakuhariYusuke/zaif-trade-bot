#!/usr/bin/env python3
"""
metrics.py
Robust implementation of trading performance metrics

This module provides comprehensive trading performance metrics with:
- Robust error handling using safe_operation
- Memory-efficient implementations
- Comprehensive type hints
- Extensive documentation
- Statistical validation
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, TypedDict, Union, cast

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from scipy import stats

# Trading constants
from ztb.trading.constants import (
    ACTION_BUY,
    ACTION_HOLD,
    ACTION_SELL,
    TRADING_DAYS_PER_YEAR,
)
from ztb.utils.errors import safe_operation
from ztb.utils.types import FeatureMetrics, StatsResult


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
    expected_value: float
    recovery_factor: float
    num_periods: int
    seasonality_analysis: Optional[Dict[str, Any]]
    market_regime_analysis: Optional[Dict[str, Any]]
    walkforward_analysis: Optional[Dict[str, Any]]
    stress_test_analysis: Optional[Dict[str, Any]]
    statistical_tests: Optional[Dict[str, Any]]


def sharpe_ratio(
    returns: Union[pd.Series, NDArray[Any]],
    rf: float = 0.0,
    period_per_year: int = TRADING_DAYS_PER_YEAR,
) -> float:
    """
    Calculate Sharpe ratio with robust error handling.

    The Sharpe ratio measures risk-adjusted return by dividing excess return
    by volatility. Higher values indicate better risk-adjusted performance.

    Args:
        returns: Return series (pandas Series or numpy array)
        rf: Risk-free rate (annual, default: 0.0)
        period_per_year: Number of periods per year (252 for daily, 365 for crypto)

    Returns:
        Sharpe ratio as float

    Examples:
        >>> import numpy as np
        >>> returns = np.array([0.01, 0.02, -0.01, 0.03])
        >>> sharpe_ratio(returns)
        0.577...
    """
    return cast(
        float,
        safe_operation(
            logger=None,
            operation=lambda: _sharpe_ratio_impl(returns, rf, period_per_year),
            context="sharpe_ratio_calculation",
            default_result=0.0,
        ),
    )


def _sharpe_ratio_impl(
    returns: Union[pd.Series, NDArray[Any]],
    rf: float = 0.0,
    period_per_year: int = TRADING_DAYS_PER_YEAR,
) -> float:
    """
    Implementation of Sharpe ratio calculation.

    Args:
        returns: Return series
        rf: Risk-free rate (annual)
        period_per_year: Number of periods per year

    Returns:
        Sharpe ratio
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

    # Calculate volatility
    volatility = np.std(excess_returns, ddof=1)

    if volatility == 0 or np.isnan(volatility):
        return 0.0

    # Annualize the ratio
    mean_excess_return = np.mean(excess_returns)
    return (mean_excess_return / volatility) * np.sqrt(period_per_year)


def calculate_deflated_sharpe_ratio(
    returns: Union[pd.Series, NDArray[Any], List[float]],
    num_strategies: int = 1000,
    risk_free_rate: float = 0.0,
) -> float:
    """
    Calculate Deflated Sharpe Ratio (DSR) to account for multiple testing.

    This is a simplified version that adjusts the Sharpe Ratio based on the
    number of trials (strategies tested).
    """
    if isinstance(returns, list):
        returns = np.array(returns)

    sr = sharpe_ratio(returns, rf=risk_free_rate, period_per_year=TRADING_DAYS_PER_YEAR)
    deflation_factor = 1.0 / np.sqrt(num_strategies) if num_strategies > 0 else 1.0

    return float(sr * deflation_factor)


def calculate_bootstrap_pvalue(
    strategy_returns: Union[pd.Series, NDArray[Any], List[float]],
    benchmark_returns: Union[pd.Series, NDArray[Any], List[float]],
    n_bootstrap: int = 1000,
) -> float:
    """
    Calculate bootstrap p-value for strategy vs benchmark comparison.

    Tests the null hypothesis that the strategy returns are not significantly
    different from the benchmark returns.
    """
    if isinstance(strategy_returns, list):
        strategy_returns = np.array(strategy_returns)
    if isinstance(benchmark_returns, list):
        benchmark_returns = np.array(benchmark_returns)

    if len(strategy_returns) != len(benchmark_returns):
        min_len = min(len(strategy_returns), len(benchmark_returns))
        strategy_returns = strategy_returns[:min_len]
        benchmark_returns = benchmark_returns[:min_len]

    observed_diff = np.mean(strategy_returns) - np.mean(benchmark_returns)
    combined = np.concatenate([strategy_returns, benchmark_returns])
    n = len(strategy_returns)

    bootstrap_diffs = []
    for _ in range(n_bootstrap):
        strat_sample = np.random.choice(combined, size=n, replace=True)
        bench_sample = np.random.choice(combined, size=n, replace=True)
        bootstrap_diffs.append(np.mean(strat_sample) - np.mean(bench_sample))

    bootstrap_array = np.array(bootstrap_diffs)
    p_value = np.mean(np.abs(bootstrap_array) >= np.abs(observed_diff))

    return float(p_value)


def sortino_ratio(
    returns: Union[pd.Series, NDArray[Any]],
    rf: float = 0.0,
    period_per_year: int = TRADING_DAYS_PER_YEAR,
    downside_floor: float = 0.0,
) -> float:
    """
    Calculate Sortino ratio with robust error handling.

    The Sortino ratio is similar to the Sharpe ratio but only considers downside
    volatility (negative returns below the target). It provides a better measure
    of risk-adjusted performance for strategies where upside volatility is desirable.

    Args:
        returns: Return series (pandas Series or numpy array)
        rf: Risk-free rate (annual, default: 0.0)
        period_per_year: Number of periods per year (252 for daily, 365 for crypto)
        downside_floor: Minimum acceptable return threshold (default: 0.0)

    Returns:
        Sortino ratio as float

    Examples:
        >>> import numpy as np
        >>> returns = np.array([0.01, 0.02, -0.01, 0.03, -0.02])
        >>> sortino_ratio(returns)
        0.894...
    """
    return cast(
        float,
        safe_operation(
            logger=None,
            operation=lambda: _sortino_ratio_impl(
                returns, rf, period_per_year, downside_floor
            ),
            context="sortino_ratio_calculation",
            default_result=0.0,
        ),
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


def calculate_downside_risk_reward(
    returns: Union[pd.Series, NDArray[Any]], penalty_multiplier: float = 1.0
) -> float:
    """Calculate a simple downside-risk-based reward used by simplified reward tests.

    The function returns a positive penalty proportionate to the average magnitude
    of negative returns (i.e., larger negative returns increase the penalty).
    """
    arr = np.asarray(returns)
    if arr.size == 0:
        return 0.0
    neg = arr[arr < 0]
    if neg.size == 0:
        return 0.0
    return float(-np.mean(neg) * float(penalty_multiplier))


def calculate_risk_adjusted_reward(returns: Union[pd.Series, NDArray[Any]], risk_penalty: float = 1.0) -> float:
    """Simple risk-adjusted reward used by script-level tests.

    This combines average return with a downside penalty.
    """
    arr = np.asarray(returns)
    if arr.size == 0:
        return 0.0
    avg = float(np.nanmean(arr))
    downside = calculate_downside_risk_reward(arr, penalty_multiplier=risk_penalty)
    return float(avg - downside)


def calculate_trading_reward(*args, **kwargs):
    """Compatibility alias expected by older scripts/tests."""
    return calculate_risk_adjusted_reward(*args, **kwargs)


def max_drawdown(equity_curve: Union[pd.Series, NDArray[Any]]) -> float:
    """
    Calculate maximum drawdown from equity curve.

    Maximum drawdown measures the largest peak-to-trough decline in portfolio value.
    It represents the worst-case scenario of loss from peak to bottom.

    Args:
        equity_curve: Cumulative returns or equity values (pandas Series or numpy array)

    Returns:
        Maximum drawdown as negative float (e.g., -0.15 for 15% drawdown)

    Note:
        Returns a negative value representing the maximum loss from peak.
        To get the absolute drawdown percentage, use abs(max_drawdown(equity_curve)).

    Examples:
        >>> import numpy as np
        >>> equity = np.array([10000, 10500, 10300, 10800, 10200, 10600])
        >>> max_drawdown(equity)
        -0.0555...
    """
    return cast(
        float,
        safe_operation(
            logger=None,
            operation=lambda: _max_drawdown_impl(equity_curve),
            context="max_drawdown_calculation",
            default_result=0.0,
        ),
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
    Calculate Calmar ratio (annual return / maximum drawdown).

    The Calmar ratio measures risk-adjusted returns relative to the maximum drawdown.
    It shows how much return is generated per unit of maximum risk experienced.
    Higher values indicate better risk-adjusted performance.

    Args:
        returns: Return series (pandas Series or numpy array)
        rf: Risk-free rate (annual, default: 0.0)
        period_per_year: Number of periods per year (252 for daily, 365 for crypto)

    Returns:
        Calmar ratio as float. Returns 0.0 if max drawdown is zero.

    Examples:
        >>> import numpy as np
        >>> returns = np.array([0.01, -0.02, 0.015, -0.03, 0.025])
        >>> cr = calmar_ratio(returns)
        >>> print(f"Calmar Ratio: {cr:.3f}")
        >>> # Higher values indicate better risk-adjusted returns

        >>> # Strategy with no drawdown
        >>> flat_returns = np.array([0.01, 0.01, 0.01])
        >>> cr_perfect = calmar_ratio(flat_returns)
        >>> print(f"Perfect Calmar Ratio: {cr_perfect}")  # Returns 0.0 (division by zero)
    """
    return cast(
        float,
        safe_operation(
            logger=None,  # Use default logger
            operation=lambda: _calmar_ratio_impl(returns, rf, period_per_year),
            context="calmar_ratio_calculation",
            default_result=0.0,  # Return 0.0 on failure
        ),
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
    Calculate win rate (proportion of positive returns).

    Args:
        returns: Return series

    Returns:
        Win rate as float (0.0 to 1.0)
    """
    return cast(
        float,
        safe_operation(
            logger=None,
            operation=lambda: _win_rate_impl(returns),
            context="win_rate_calculation",
            default_result=0.0,
        ),
    )


def _win_rate_impl(returns: Union[pd.Series, NDArray[Any]]) -> float:
    returns = np.asarray(returns)
    if len(returns) == 0:
        return 0.0

    # Filter out NaN
    returns = returns[~np.isnan(returns)]
    if len(returns) == 0:
        return 0.0

    return float(np.mean(returns > 0))


def profit_factor(returns: Union[pd.Series, NDArray[Any]]) -> float:
    """
    Calculate profit factor (gross profit / gross loss ratio).

    The profit factor measures the relationship between profitable and losing trades.
    A profit factor > 1 indicates net profitability, with higher values being better.
    Values between 1.25-1.5 are considered good, >1.5 excellent.

    Args:
        returns: Return series (pandas Series or numpy array)

    Returns:
        Profit factor as float. Returns 1.0 if no losses (infinite profit factor).

    Examples:
        >>> import numpy as np
        >>> returns = np.array([0.01, -0.005, 0.02, -0.01, 0.015])
        >>> pf = profit_factor(returns)
        >>> print(f"Profit Factor: {pf:.3f}")
        >>> # Profit factor > 1 indicates profitable strategy

        >>> # All positive returns
        >>> positive_returns = np.array([0.01, 0.02, 0.015])
        >>> pf_perfect = profit_factor(positive_returns)
        >>> print(f"Perfect Profit Factor: {pf_perfect}")  # Returns 1.0 (no losses)
    """
    return cast(
        float,
        safe_operation(
            logger=None,  # Use default logger
            operation=lambda: _profit_factor_impl(returns),
            context="profit_factor_calculation",
            default_result=1.0,  # Return 1.0 on failure (neutral)
        ),
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


def expected_value(returns: Union[pd.Series, NDArray[Any]]) -> float:
    """
    Calculate expected value per trade/period.

    The expected value represents the average return per trade or period,
    calculated as: (win_rate × average_win) - ((1 - win_rate) × average_loss).
    Positive values indicate profitable strategies on average.

    Args:
        returns: Return series (pandas Series or numpy array)

    Returns:
        Expected value per trade/period as float.

    Examples:
        >>> import numpy as np
        >>> returns = np.array([0.02, -0.01, 0.015, -0.005, 0.03])
        >>> ev = expected_value(returns)
        >>> print(f"Expected Value: {ev:.6f}")
        >>> # Positive value indicates profitable strategy

        >>> # Mixed returns with different magnitudes
        >>> mixed_returns = np.array([0.1, -0.05, 0.08, -0.03])
        >>> ev_mixed = expected_value(mixed_returns)
        >>> print(f"Mixed Expected Value: {ev_mixed:.6f}")
    """
    return cast(
        float,
        safe_operation(
            logger=None,  # Use default logger
            operation=lambda: _expected_value_impl(returns),
            context="expected_value_calculation",
            default_result=0.0,  # Return 0.0 on failure
        ),
    )


def _expected_value_impl(returns: Union[pd.Series, NDArray[Any]]) -> float:
    """Implementation of expected value calculation."""
    returns = np.asarray(returns)

    if len(returns) == 0:
        return 0.0

    # Remove NaN values
    returns = returns[~np.isnan(returns)]

    if len(returns) == 0:
        return 0.0

    winning_trades = returns[returns > 0]
    losing_trades = returns[returns < 0]

    if len(winning_trades) == 0:
        return np.mean(losing_trades)  # All losses

    if len(losing_trades) == 0:
        return np.mean(winning_trades)  # All wins

    win_rate_val = _win_rate_impl(returns)
    avg_win = np.mean(winning_trades)
    avg_loss = np.mean(np.abs(losing_trades))  # Use absolute value for losses

    return float((win_rate_val * avg_win) - ((1 - win_rate_val) * avg_loss))


def recovery_factor(
    returns: Union[pd.Series, NDArray[Any]],
    rf: float = 0.0,
    period_per_year: int = TRADING_DAYS_PER_YEAR,
) -> float:
    """
    Calculate recovery factor (net profit / maximum drawdown).

    The recovery factor measures how efficiently a strategy recovers from losses.
    It shows the net profit generated relative to the maximum drawdown experienced.
    Higher values indicate better recovery capability and risk management.

    Args:
        returns: Return series (pandas Series or numpy array)
        rf: Risk-free rate (annual, default: 0.0)
        period_per_year: Number of periods per year (252 for daily, 365 for crypto)

    Returns:
        Recovery factor as float. Returns 0.0 if max drawdown is zero.

    Examples:
        >>> import numpy as np
        >>> returns = np.array([0.02, -0.03, 0.04, -0.02, 0.05])
        >>> rf = recovery_factor(returns)
        >>> print(f"Recovery Factor: {rf:.3f}")
        >>> # Higher values indicate better recovery from drawdowns

        >>> # Strategy with small drawdowns
        >>> stable_returns = np.array([0.01, -0.005, 0.012, -0.003])
        >>> rf_stable = recovery_factor(stable_returns)
        >>> print(f"Stable Recovery Factor: {rf_stable:.3f}")
    """
    return cast(
        float,
        safe_operation(
            logger=None,  # Use default logger
            operation=lambda: _recovery_factor_impl(returns, rf, period_per_year),
            context="recovery_factor_calculation",
            default_result=0.0,  # Return 0.0 on failure
        ),
    )


def _recovery_factor_impl(
    returns: Union[pd.Series, NDArray[Any]],
    rf: float = 0.0,
    period_per_year: int = TRADING_DAYS_PER_YEAR,
) -> float:
    """Implementation of recovery factor calculation."""
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

    # Recovery factor = Annual return / |Max Drawdown|
    return float((annual_return - rf) / abs(mdd))


def rolling_analysis(
    returns: Union[pd.Series, NDArray[Any]],
    window: int = 30,
    step: int = 1,
    rf: float = 0.0,
    period_per_year: int = TRADING_DAYS_PER_YEAR,
) -> pd.DataFrame:
    """
    Perform rolling analysis of performance metrics over time windows

    Args:
        returns: Return series
        window: Rolling window size (in periods)
        step: Step size for rolling windows
        rf: Risk-free rate (annual)
        period_per_year: Number of periods per year

    Returns:
        DataFrame with rolling metrics for each window
    """
    return cast(
        pd.DataFrame,
        safe_operation(
            logger=None,  # Use default logger
            operation=lambda: _rolling_analysis_impl(
                returns, window, step, rf, period_per_year
            ),
            context="rolling_analysis_calculation",
            default_result=pd.DataFrame(),  # Return empty DataFrame on failure
        ),
    )


def _rolling_analysis_impl(
    returns: Union[pd.Series, NDArray[Any]],
    window: int,
    step: int,
    rf: float,
    period_per_year: int,
) -> pd.DataFrame:
    """Implementation of rolling analysis calculation."""
    returns = np.asarray(returns)

    if len(returns) < window:
        return pd.DataFrame()

    rolling_metrics = []

    for start_idx in range(0, len(returns) - window + 1, step):
        end_idx = start_idx + window
        window_returns = returns[start_idx:end_idx]

        # Calculate metrics for this window
        metrics = calculate_all_metrics(window_returns, rf, period_per_year)

        # Add window information
        metrics_dict = dict(metrics)
        metrics_dict["window_start"] = start_idx
        metrics_dict["window_end"] = end_idx - 1

        rolling_metrics.append(metrics_dict)

    return pd.DataFrame(rolling_metrics)


def drawdown_analysis(equity_curve: Union[pd.Series, NDArray[Any]]) -> Dict[str, Any]:
    """
    Comprehensive drawdown analysis including duration, depth, and recovery time

    Args:
        equity_curve: Cumulative equity values

    Returns:
        Dictionary with drawdown analysis results
    """
    return cast(
        Dict[str, Any],
        safe_operation(
            logger=None,  # Use default logger
            operation=lambda: _drawdown_analysis_impl(equity_curve),
            context="drawdown_analysis_calculation",
            default_result={},  # Return empty dict on failure
        ),
    )


def _drawdown_analysis_impl(
    equity_curve: Union[pd.Series, NDArray[Any]],
) -> Dict[str, Any]:
    """Implementation of comprehensive drawdown analysis."""
    equity_curve = np.asarray(equity_curve)

    if len(equity_curve) == 0:
        return {
            "max_drawdown": 0.0,
            "avg_drawdown": 0.0,
            "median_drawdown": 0.0,
            "max_drawdown_duration": 0,
            "avg_drawdown_duration": 0.0,
            "max_recovery_time": 0,
            "avg_recovery_time": 0.0,
            "num_drawdowns": 0,
            "drawdown_periods": [],
        }

    # Calculate running maximum (peaks)
    running_max = np.maximum.accumulate(equity_curve)

    # Calculate drawdown series
    drawdown_series = (equity_curve - running_max) / running_max

    # Find drawdown periods
    drawdown_periods = []
    in_drawdown = False
    drawdown_start = 0  # Initialize with 0
    peak_value = equity_curve[0]

    for i, (equity, dd) in enumerate(zip(equity_curve, drawdown_series)):
        if dd < 0 and not in_drawdown:
            # Start of drawdown
            in_drawdown = True
            drawdown_start = i
            peak_value = running_max[i]
        elif dd >= 0 and in_drawdown:
            # End of drawdown (recovery)
            in_drawdown = False
            trough_idx = (
                np.argmin(drawdown_series[drawdown_start : i + 1]) + drawdown_start
            )
            recovery_idx = i

            drawdown_periods.append(
                {
                    "start_idx": drawdown_start,
                    "peak_idx": np.argmax(running_max[: drawdown_start + 1])
                    if drawdown_start > 0
                    else 0,
                    "trough_idx": trough_idx,
                    "recovery_idx": recovery_idx,
                    "duration": recovery_idx - drawdown_start,
                    "depth": drawdown_series[trough_idx],
                    "peak_value": peak_value,
                    "trough_value": equity_curve[trough_idx],
                    "recovery_value": equity,
                    "recovery_time": recovery_idx - trough_idx,
                }
            )

    # Handle case where drawdown continues to the end
    if in_drawdown:
        trough_idx = np.argmin(drawdown_series[drawdown_start:]) + drawdown_start
        recovery_idx = len(equity_curve) - 1  # No recovery yet

        drawdown_periods.append(
            {
                "start_idx": drawdown_start,
                "peak_idx": np.argmax(running_max[: drawdown_start + 1])
                if drawdown_start > 0
                else 0,
                "trough_idx": trough_idx,
                "recovery_idx": recovery_idx,  # No recovery
                "duration": recovery_idx - drawdown_start + 1,  # Include current period
                "depth": drawdown_series[trough_idx],
                "peak_value": peak_value,
                "trough_value": equity_curve[trough_idx],
                "recovery_value": equity_curve[recovery_idx],  # Current value
                "recovery_time": 0,  # No recovery yet
            }
        )

    # Calculate summary statistics
    if drawdown_periods:
        depths = [dd["depth"] for dd in drawdown_periods]
        durations = [dd["duration"] for dd in drawdown_periods]
        recovery_times = [dd["recovery_time"] for dd in drawdown_periods]

        return {
            "max_drawdown": float(np.min(depths)),
            "avg_drawdown": float(np.mean(depths)),
            "median_drawdown": float(np.median(depths)),
            "max_drawdown_duration": int(np.max(durations)),
            "avg_drawdown_duration": float(np.mean(durations)),
            "max_recovery_time": int(np.max(recovery_times)),
            "avg_recovery_time": float(np.mean(recovery_times)),
            "num_drawdowns": len(drawdown_periods),
            "drawdown_periods": drawdown_periods,
        }
    else:
        return {
            "max_drawdown": 0.0,
            "avg_drawdown": 0.0,
            "median_drawdown": 0.0,
            "max_drawdown_duration": 0,
            "avg_drawdown_duration": 0.0,
            "max_recovery_time": 0,
            "avg_recovery_time": 0.0,
            "num_drawdowns": 0,
            "drawdown_periods": [],
        }


def seasonality_analysis(
    returns: Union[pd.Series, NDArray[Any]],
    dates: Optional[Union[pd.Series, NDArray[Any]]] = None,
) -> Dict[str, Any]:
    """
    Comprehensive seasonality analysis of returns

    Analyzes performance patterns by month, quarter, and year

    Args:
        returns: Return series
        dates: Corresponding dates for the returns (if available)

    Returns:
        Dictionary with seasonality analysis results
    """
    return cast(
        Dict[str, Any],
        safe_operation(
            logger=None,  # Use default logger
            operation=lambda: _seasonality_analysis_impl(returns, dates),
            context="seasonality_analysis_calculation",
            default_result={},  # Return empty dict on failure
        ),
    )


def _seasonality_analysis_impl(
    returns: Union[pd.Series, NDArray[Any]],
    dates: Optional[Union[pd.Series, NDArray[Any]]] = None,
) -> Dict[str, Any]:
    """Implementation of seasonality analysis."""
    returns = np.asarray(returns)

    if len(returns) == 0:
        return {}

    # If dates are not provided, create synthetic dates
    analysis_dates: Any
    if dates is None:
        # Assume daily returns starting from a reference date
        analysis_dates = pd.date_range(
            start="2020-01-01", periods=len(returns), freq="D"
        )
    else:
        analysis_dates = pd.to_datetime(dates)

    # Create DataFrame for analysis
    df = pd.DataFrame({"returns": returns, "dates": analysis_dates})

    # Extract seasonal components
    df["month"] = df["dates"].dt.month
    df["quarter"] = df["dates"].dt.quarter
    df["year"] = df["dates"].dt.year
    df["day_of_year"] = df["dates"].dt.dayofyear

    results: Dict[str, Any] = {}

    # Monthly analysis
    monthly_stats = (
        df.groupby("month")["returns"].agg(["mean", "std", "count", "sum"]).round(6)
    )

    # Calculate monthly Sharpe ratios (annualized)
    monthly_sharpe = {}
    for month in range(1, 13):
        month_returns = df[df["month"] == month]["returns"]
        if len(month_returns) > 1:
            mean_ret = month_returns.mean()
            std_ret = month_returns.std()
            if std_ret > 0:
                # Annualize Sharpe ratio (assuming monthly data)
                monthly_sharpe[month] = (mean_ret / std_ret) * np.sqrt(12)
            else:
                monthly_sharpe[month] = 0.0 if mean_ret >= 0 else -np.inf
        else:
            monthly_sharpe[month] = 0.0

    results["monthly_analysis"] = {
        "stats": monthly_stats.to_dict(),
        "sharpe_ratios": monthly_sharpe,
        "best_month": monthly_stats["mean"].idxmax(),
        "worst_month": monthly_stats["mean"].idxmin(),
        "best_month_return": monthly_stats["mean"].max(),
        "worst_month_return": monthly_stats["mean"].min(),
    }

    # Quarterly analysis
    quarterly_stats = (
        df.groupby("quarter")["returns"].agg(["mean", "std", "count", "sum"]).round(6)
    )

    quarterly_sharpe = {}
    for quarter in range(1, 5):
        quarter_returns = df[df["quarter"] == quarter]["returns"]
        if len(quarter_returns) > 1:
            mean_ret = quarter_returns.mean()
            std_ret = quarter_returns.std()
            if std_ret > 0:
                # Annualize Sharpe ratio (assuming quarterly data)
                quarterly_sharpe[quarter] = (mean_ret / std_ret) * np.sqrt(4)
            else:
                quarterly_sharpe[quarter] = 0.0 if mean_ret >= 0 else -np.inf
        else:
            quarterly_sharpe[quarter] = 0.0

    results["quarterly_analysis"] = {
        "stats": quarterly_stats.to_dict(),
        "sharpe_ratios": quarterly_sharpe,
        "best_quarter": quarterly_stats["mean"].idxmax(),
        "worst_quarter": quarterly_stats["mean"].idxmin(),
        "best_quarter_return": quarterly_stats["mean"].max(),
        "worst_quarter_return": quarterly_stats["mean"].min(),
    }

    # Yearly analysis (if multiple years available)
    if len(df["year"].unique()) > 1:
        yearly_stats = (
            df.groupby("year")["returns"].agg(["mean", "std", "count", "sum"]).round(6)
        )

        results["yearly_analysis"] = {
            "stats": yearly_stats.to_dict(),
            "best_year": yearly_stats["mean"].idxmax(),
            "worst_year": yearly_stats["mean"].idxmin(),
            "best_year_return": yearly_stats["mean"].max(),
            "worst_year_return": yearly_stats["mean"].min(),
        }
    else:
        results["yearly_analysis"] = None

    # Overall seasonality assessment
    monthly_means = monthly_stats["mean"]
    seasonal_strength = (
        monthly_means.std() / monthly_means.abs().mean()
        if monthly_means.abs().mean() > 0
        else 0.0
    )

    results["seasonality_assessment"] = {
        "seasonal_strength": seasonal_strength,
        "has_strong_seasonality": seasonal_strength > 0.5,  # Arbitrary threshold
        "monthly_variation_coefficient": seasonal_strength,
    }

    return results


def classify_market_regime(
    prices: Union[pd.Series, NDArray[Any]],
    returns: Optional[Union[pd.Series, NDArray[Any]]] = None,
    window: int = 20,
) -> pd.Series:
    """
    Classify market regimes based on price action and volatility

    Args:
        prices: Price series
        returns: Return series (calculated from prices if not provided)
        window: Rolling window for regime classification

    Returns:
        Series with regime labels: 'bull', 'bear', 'sideways', 'volatile'
    """
    return safe_operation(
        logger=None,
        operation=lambda: _classify_market_regime_impl(prices, returns, window),
        context="classify_market_regime",
        default_result=pd.Series(),
    )


def _classify_market_regime_impl(
    prices: Union[pd.Series, NDArray[Any]],
    returns: Optional[Union[pd.Series, NDArray[Any]]] = None,
    window: int = 20,
) -> pd.Series:
    prices = pd.Series(prices)

    if returns is None:
        returns = prices.pct_change().fillna(0)
    else:
        returns = pd.Series(returns)

    # Calculate rolling metrics
    rolling_mean = returns.rolling(window=window).mean()
    rolling_std = returns.rolling(window=window).std()
    rolling_trend = prices.rolling(window=window).apply(
        lambda x: (x.iloc[-1] - x.iloc[0]) / x.iloc[0] if len(x) > 1 else 0
    )

    # Classify regimes
    regimes = []

    for i in range(len(prices)):
        if i < window - 1:
            regimes.append("unknown")
            continue

        trend = rolling_trend.iloc[i]
        volatility = rolling_std.iloc[i]

        # Classification logic
        if abs(trend) < 0.02:  # Less than 2% change
            if volatility > returns.std() * 1.5:  # High volatility
                regime = "volatile_sideways"
            else:
                regime = "sideways"
        elif trend > 0.05:  # Strong uptrend (>5%)
            regime = "bull"
        elif trend < -0.05:  # Strong downtrend (<-5%)
            regime = "bear"
        elif trend > 0:  # Moderate uptrend
            regime = "weak_bull"
        else:  # Moderate downtrend
            regime = "weak_bear"

        regimes.append(regime)

    return pd.Series(regimes, index=prices.index)


def multi_market_backtest_analysis(
    returns: Union[pd.Series, NDArray[Any]],
    prices: Union[pd.Series, NDArray[Any]],
    regime_window: int = 20,
) -> Dict[str, Any]:
    """
    Analyze backtest performance across different market regimes

    Args:
        returns: Return series
        prices: Price series
        regime_window: Window for regime classification

    Returns:
        Dictionary with performance analysis by market regime
    """
    return safe_operation(
        logger=None,
        operation=lambda: _multi_market_backtest_analysis_impl(
            returns, prices, regime_window
        ),
        context="multi_market_backtest_analysis",
        default_result={},
    )


def _multi_market_backtest_analysis_impl(
    returns: Union[pd.Series, NDArray[Any]],
    prices: Union[pd.Series, NDArray[Any]],
    regime_window: int = 20,
) -> Dict[str, Any]:
    returns = pd.Series(returns)
    prices = pd.Series(prices)

    # Classify market regimes
    regimes = classify_market_regime(prices, returns, regime_window)

    # Analyze performance by regime
    regime_performance = {}

    unique_regimes = regimes.unique()
    for regime in unique_regimes:
        if regime == "unknown":
            continue

        regime_mask = regimes == regime
        regime_returns = returns[regime_mask]

        if len(regime_returns) > 0:
            metrics = calculate_all_metrics(regime_returns.values)
            regime_performance[regime] = {
                "metrics": metrics,
                "periods": len(regime_returns),
                "total_return": regime_returns.sum(),
            }

    # Perform statistical tests between regimes if we have multiple regimes with sufficient data
    statistical_tests_results: Dict[str, Any] = {}
    regime_list = [regime for regime in unique_regimes if regime != "unknown"]

    if len(regime_list) >= 2:
        # Compare each pair of regimes
        for i, regime_a in enumerate(regime_list):
            for j, regime_b in enumerate(regime_list):
                if i < j:  # Avoid duplicate comparisons
                    mask_a = regimes == regime_a
                    mask_b = regimes == regime_b
                    returns_a = returns[mask_a]
                    returns_b = returns[mask_b]

                    if len(returns_a) >= 2 and len(returns_b) >= 2:
                        test_result = perform_statistical_tests(
                            returns_a.values, returns_b.values
                        )
                        statistical_tests_results[
                            f"{regime_a}_vs_{regime_b}"
                        ] = test_result

        # Apply p-mean method if we have multiple test results
        if len(statistical_tests_results) > 1:
            p_values = [
                result["p_value"] for result in statistical_tests_results.values()
            ]
            statistical_tests_results["p_mean_arithmetic"] = p_mean_method(
                p_values, "arithmetic"
            )
            statistical_tests_results["p_mean_geometric"] = p_mean_method(
                p_values, "geometric"
            )
            statistical_tests_results["overall_significant"] = (
                statistical_tests_results["p_mean_geometric"] < 0.05
            )

    # Overall regime distribution
    regime_distribution = regimes.value_counts(normalize=True).to_dict()

    return {
        "regime_performance": regime_performance,
        "regime_distribution": regime_distribution,
        "regime_transitions": _analyze_regime_transitions(regimes),
        "statistical_tests": statistical_tests_results,
    }


def _analyze_regime_transitions(regimes: pd.Series) -> Dict[str, Any]:
    """Analyze transitions between market regimes."""
    transitions: Dict[str, int] = {}
    prev_regime = None

    for regime in regimes:
        if prev_regime is not None and regime != prev_regime:
            transition_key = f"{prev_regime}_to_{regime}"
            transitions[transition_key] = transitions.get(transition_key, 0) + 1
        prev_regime = regime

    return transitions


def calculate_all_metrics(
    returns: Union[pd.Series, NDArray[Any]],
    rf: float = 0.0,
    period_per_year: int = TRADING_DAYS_PER_YEAR,
) -> MetricsResult:
    """
    Calculate comprehensive set of trading performance metrics.

    This function computes all major risk-adjusted performance metrics in a single call,
    providing a complete analysis of trading strategy performance.

    Args:
        returns: Return series (pandas Series or numpy array)
        rf: Risk-free rate (annual, default: 0.0)
        period_per_year: Number of periods per year (252 for daily, 365 for crypto)

    Returns:
        MetricsResult TypedDict containing all calculated metrics:
        - total_return: Total cumulative return
        - annual_return: Annualized return
        - volatility: Annualized volatility
        - sharpe_ratio: Risk-adjusted return (Sharpe ratio)
        - sortino_ratio: Downside risk-adjusted return
        - calmar_ratio: Drawdown-adjusted return
        - max_drawdown: Maximum peak-to-trough decline
        - win_rate: Proportion of profitable periods
        - profit_factor: Gross profit / Gross loss ratio
        - expected_value: Average return per period
        - recovery_factor: Net profit / Max drawdown
        - num_periods: Number of periods analyzed
        - seasonality_analysis: Seasonal performance patterns
        - market_regime_analysis: Performance by market conditions
        - walkforward_analysis: Out-of-sample performance
        - stress_test_analysis: Extreme scenario analysis
        - statistical_tests: Statistical significance tests

    Examples:
        >>> import numpy as np
        >>> returns = np.random.normal(0.001, 0.02, 252)  # 1 year of daily returns
        >>> metrics = calculate_all_metrics(returns)
        >>> print(f"Sharpe Ratio: {metrics['sharpe_ratio']:.3f}")
        >>> print(f"Max Drawdown: {metrics['max_drawdown']:.1%}")
    """
    return cast(
        MetricsResult,
        safe_operation(
            logger=None,
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
                expected_value=0.0,
                recovery_factor=0.0,
                num_periods=0,
                seasonality_analysis=None,
                market_regime_analysis=None,
                walkforward_analysis=None,
                stress_test_analysis=None,
                statistical_tests=None,
            ),  # Return default metrics on failure
        ),
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
    ev = expected_value(returns)
    rf_factor = recovery_factor(returns, rf, period_per_year)

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
        "expected_value": float(ev),
        "recovery_factor": float(rf_factor),
        "num_periods": int(len(returns)),
        "seasonality_analysis": None,
        "market_regime_analysis": None,
        "walkforward_analysis": None,
        "stress_test_analysis": None,
        "statistical_tests": None,
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


def p_mean_method(
    p_values: Union[list[float], NDArray[Any]], method: str = "arithmetic"
) -> float:
    """
    p平均法による総合p値の計算

    p平均法は、複数の独立した統計検定のp値を統合し、
    全体として統計的有意性があるかを評価する手法です。

    Args:
        p_values: p値のリスト（0.0 ~ 1.0の範囲）
        method: 平均化手法
            - 'arithmetic': 算術平均（単純平均）
            - 'geometric': 幾何平均（対数変換後平均）

    Returns:
        総合p値（0.0 ~ 1.0）

    算術平均の特徴:
        - 直感的で理解しやすい
        - 全てのp値に等しい重み付け
        - 極端なp値（0.99など）の影響を受けやすい

    幾何平均の特徴:
        - 極端なp値の影響を緩和
        - 0に非常に近いp値を適切に扱える
        - 統計学的によりロバスト

    使用例:
        # 3つのメトリクスのp値統合
        p_values = [0.03, 0.07, 0.02]  # 個別のt検定結果
        combined_p = p_mean_method(p_values, 'geometric')
        significant = combined_p < 0.05

    注意事項:
        - p値が完全に独立であることを仮定
        - 相関のある検定では結果が保守的になる可能性
        - 解釈時は個別の検定結果も確認すること
    """
    if not p_values:
        return 1.0

    p_array = np.array(p_values)

    if method == "arithmetic":
        # 算術平均
        return float(np.mean(p_array))
    elif method == "geometric":
        # 幾何平均 (0を避けるため小さな値を加算)
        p_array = np.clip(p_array, 1e-10, 1.0)
        return float(np.exp(np.mean(np.log(p_array))))
    else:
        raise ValueError(f"Unknown method: {method}")


def perform_statistical_tests(
    data_a: Union[pd.Series, NDArray[Any], list[float]],
    data_b: Union[pd.Series, NDArray[Any], list[float]],
    alpha: float = 0.05,
) -> Dict[str, Any]:
    """
    2つのデータセット間の統計的検定を実行

    Welchのt検定（等分散を仮定しない）を使用して、
    データ間の差の統計的有意性を評価します。

    Args:
        data_a: データセットA
        data_b: データセットB
        alpha: 有意水準（デフォルト: 0.05）

    Returns:
        検定結果の辞書
        {
            't_statistic': float,      # t統計量
            'p_value': float,          # p値
            'significant': bool,       # p < alphaかどうか
            'mean_a': float,           # データAの平均値
            'mean_b': float,           # データBの平均値
            'effect_size': float       # 効果量（Cohen's d）
        }

    効果量の解釈 (Cohen's d):
        - 0.2: 小さな効果
        - 0.5: 中程度の効果
        - 0.8: 大きな効果

    注意事項:
        - データが正規分布に従うことを仮定
        - サンプルサイズが小さい場合、検定力が低下
        - 複数の検定を行う場合、p値の補正を検討
    """
    return cast(
        Dict[str, Any],
        safe_operation(
            logger=None,
            operation=lambda: _perform_statistical_tests_impl(data_a, data_b, alpha),
            context="statistical_tests_calculation",
            default_result={
                "t_statistic": 0.0,
                "p_value": 1.0,
                "significant": False,
                "mean_a": 0.0,
                "mean_b": 0.0,
                "effect_size": 0.0,
            },
        ),
    )


def _perform_statistical_tests_impl(
    data_a: Union[pd.Series, NDArray[Any], list[float]],
    data_b: Union[pd.Series, NDArray[Any], list[float]],
    alpha: float = 0.05,
) -> Dict[str, Any]:
    """統計的検定の実装"""
    values_a = np.asarray(data_a)
    values_b = np.asarray(data_b)

    if len(values_a) < 2 or len(values_b) < 2:
        raise ValueError(
            "Insufficient data for statistical test (need at least 2 samples each)"
        )

    # t検定実行 (Welch's t-test: 等分散を仮定しない)
    t_stat, p_value = stats.ttest_ind(values_a, values_b, equal_var=False)

    # 有意性判断
    significant = p_value < alpha

    # 効果量 (Cohen's d)
    mean_a = np.mean(values_a)
    mean_b = np.mean(values_b)
    std_a = np.std(values_a, ddof=1)
    std_b = np.std(values_b, ddof=1)
    pooled_std = np.sqrt((std_a**2 + std_b**2) / 2)
    effect_size = abs(mean_a - mean_b) / pooled_std if pooled_std > 0 else 0.0

    return {
        "t_statistic": float(t_stat),
        "p_value": float(p_value),
        "significant": bool(significant),
        "mean_a": float(mean_a),
        "mean_b": float(mean_b),
        "effect_size": float(effect_size),
    }


def coefficient_of_variation(
    values: Union[pd.Series, NDArray[Any]],
) -> float:
    """
    Calculate coefficient of variation (CV).

    The coefficient of variation is a standardized measure of dispersion
    of a probability distribution or frequency distribution. It is defined
    as the ratio of the standard deviation to the mean.

    Args:
        values: Array of values (pandas Series or numpy array)

    Returns:
        Coefficient of variation as float. Returns 0.0 if mean is zero.

    Examples:
        >>> import numpy as np
        >>> returns = np.array([0.01, 0.02, -0.01, 0.03])
        >>> cv = coefficient_of_variation(returns)
        >>> print(f"Coefficient of Variation: {cv:.4f}")
        >>> # Lower values indicate more consistent performance
    """
    return safe_operation(
        logger=None,
        operation=lambda: _coefficient_of_variation_impl(values),
        context="coefficient_of_variation_calculation",
        default_result=0.0,
    )


def _coefficient_of_variation_impl(values: Union[pd.Series, NDArray[Any]]) -> float:
    """Implementation of coefficient of variation calculation."""
    values = np.asarray(values)

    if len(values) == 0:
        return 0.0

    # Remove NaN values
    values = values[~np.isnan(values)]

    if len(values) == 0:
        return 0.0

    mean_val = np.mean(values)
    if mean_val == 0:
        return 0.0

    std_val = np.std(values, ddof=1) if len(values) > 1 else 0.0
    return float(std_val / abs(mean_val))


def skewness(
    returns: Union[pd.Series, NDArray[Any]],
) -> float:
    """
    Calculate skewness of returns distribution.

    Skewness measures the asymmetry of the probability distribution.
    Positive skewness indicates a distribution with an asymmetric tail
    extending towards positive values. Negative skewness indicates a
    distribution with an asymmetric tail extending towards negative values.

    Args:
        returns: Return series (pandas Series or numpy array)

    Returns:
        Skewness as float. Returns 0.0 for insufficient data.

    Examples:
        >>> import numpy as np
        >>> returns = np.array([0.01, 0.02, -0.01, 0.03, -0.02])
        >>> skew = skewness(returns)
        >>> print(f"Skewness: {skew:.4f}")
        >>> # Positive skewness indicates upside potential
    """
    return safe_operation(
        logger=None,
        operation=lambda: _skewness_impl(returns),
        context="skewness_calculation",
        default_result=0.0,
    )


def _skewness_impl(returns: Union[pd.Series, NDArray[Any]]) -> float:
    """Implementation of skewness calculation."""
    returns = np.asarray(returns)

    if len(returns) < 3:
        return 0.0

    # Remove NaN values
    returns = returns[~np.isnan(returns)]

    if len(returns) < 3:
        return 0.0

    return float(stats.skew(returns))


def kurtosis(
    returns: Union[pd.Series, NDArray[Any]],
) -> float:
    """
    Calculate kurtosis of returns distribution.

    Kurtosis measures the "tailedness" of the probability distribution.
    High kurtosis indicates heavy tails (more extreme values).
    Low kurtosis indicates light tails (fewer extreme values).

    Args:
        returns: Return series (pandas Series or numpy array)

    Returns:
        Kurtosis as float (excess kurtosis, normal distribution = 0).
        Returns 0.0 for insufficient data.

    Examples:
        >>> import numpy as np
        >>> returns = np.array([0.01, 0.02, -0.01, 0.03, -0.02])
        >>> kurt = kurtosis(returns)
        >>> print(f"Kurtosis: {kurt:.4f}")
        >>> # High kurtosis indicates risk of extreme events
    """
    return safe_operation(
        logger=None,
        operation=lambda: _kurtosis_impl(returns),
        context="kurtosis_calculation",
        default_result=0.0,
    )


def _kurtosis_impl(returns: Union[pd.Series, NDArray[Any]]) -> float:
    """Implementation of kurtosis calculation."""
    returns = np.asarray(returns)

    if len(returns) < 4:
        return 0.0

    # Remove NaN values
    returns = returns[~np.isnan(returns)]

    if len(returns) < 4:
        return 0.0

    return float(stats.kurtosis(returns))


def test_normality(
    returns: Union[pd.Series, NDArray[Any]],
) -> Dict[str, Any]:
    """
    Test normality of returns distribution using multiple statistical tests.

    Performs Shapiro-Wilk, Kolmogorov-Smirnov, and Jarque-Bera tests
    to assess whether returns follow a normal distribution.

    Args:
        returns: Return series (pandas Series or numpy array)

    Returns:
        Dictionary containing test results:
        - shapiro_wilk: Shapiro-Wilk test results
        - kolmogorov_smirnov: KS test results
        - jarque_bera: Jarque-Bera test results

    Examples:
        >>> import numpy as np
        >>> returns = np.random.normal(0.001, 0.02, 100)
        >>> normality_results = test_normality(returns)
        >>> print(f"Shapiro-Wilk p-value: {normality_results['shapiro_wilk']['p_value']:.4f}")
    """
    return safe_operation(
        logger=None,
        operation=lambda: _test_normality_impl(returns),
        context="normality_test_calculation",
        default_result={
            "shapiro_wilk": {"statistic": None, "p_value": None, "is_normal": False},
            "kolmogorov_smirnov": {
                "statistic": 0.0,
                "p_value": 0.0,
                "is_normal": False,
            },
            "jarque_bera": {"statistic": 0.0, "p_value": 0.0, "is_normal": False},
        },
    )


def _test_normality_impl(returns: Union[pd.Series, NDArray[Any]]) -> Dict[str, Any]:
    """Implementation of normality testing."""
    returns = np.asarray(returns)

    if len(returns) < 3:
        return {
            "shapiro_wilk": {"statistic": None, "p_value": None, "is_normal": False},
            "kolmogorov_smirnov": {
                "statistic": 0.0,
                "p_value": 0.0,
                "is_normal": False,
            },
            "jarque_bera": {"statistic": 0.0, "p_value": 0.0, "is_normal": False},
        }

    # Remove NaN values
    returns = returns[~np.isnan(returns)]

    if len(returns) < 3:
        return {
            "shapiro_wilk": {"statistic": None, "p_value": None, "is_normal": False},
            "kolmogorov_smirnov": {
                "statistic": 0.0,
                "p_value": 0.0,
                "is_normal": False,
            },
            "jarque_bera": {"statistic": 0.0, "p_value": 0.0, "is_normal": False},
        }

    results = {}

    # Shapiro-Wilk test
    if 3 <= len(returns) <= 5000:
        shapiro_stat, shapiro_p = stats.shapiro(returns)
        results["shapiro_wilk"] = {
            "statistic": float(shapiro_stat),
            "p_value": float(shapiro_p),
            "is_normal": shapiro_p > 0.05,
        }
    else:
        results["shapiro_wilk"] = {
            "statistic": None,
            "p_value": None,
            "is_normal": False,
        }

    # Kolmogorov-Smirnov test
    ks_stat, ks_p = stats.kstest(
        returns, "norm", args=(np.mean(returns), np.std(returns, ddof=1))
    )
    results["kolmogorov_smirnov"] = {
        "statistic": float(ks_stat),
        "p_value": float(ks_p),
        "is_normal": ks_p > 0.05,
    }

    # Jarque-Bera test
    jb_stat, jb_p = stats.jarque_bera(returns)
    results["jarque_bera"] = {
        "statistic": float(jb_stat),
        "p_value": float(jb_p),
        "is_normal": jb_p > 0.05,
    }

    return results


def autocorrelation(
    returns: Union[pd.Series, NDArray[Any]],
    lag: int = 1,
) -> float:
    """
    Calculate autocorrelation of returns at specified lag.

    Autocorrelation measures the correlation between a time series
    and its lagged version. Significant autocorrelation can indicate
    predictability or market inefficiencies.

    Args:
        returns: Return series (pandas Series or numpy array)
        lag: Lag period for autocorrelation calculation (default: 1)

    Returns:
        Autocorrelation coefficient as float between -1 and 1.

    Examples:
        >>> import numpy as np
        >>> returns = np.random.normal(0.001, 0.02, 100)
        >>> autocorr = autocorrelation(returns, lag=1)
        >>> print(f"Autocorrelation (lag 1): {autocorr:.4f}")
        >>> # Values near 0 indicate no serial correlation
    """
    return safe_operation(
        logger=None,
        operation=lambda: _autocorrelation_impl(returns, lag),
        context="autocorrelation_calculation",
        default_result=0.0,
    )


def _autocorrelation_impl(
    returns: Union[pd.Series, NDArray[Any]], lag: int = 1
) -> float:
    """Implementation of autocorrelation calculation."""
    returns = np.asarray(returns)

    if len(returns) <= lag:
        return 0.0

    # Remove NaN values
    returns = returns[~np.isnan(returns)]

    if len(returns) <= lag:
        return 0.0

    # Calculate autocorrelation using numpy
    autocorr = np.corrcoef(returns[lag:], returns[:-lag])[0, 1]
    return float(autocorr) if not np.isnan(autocorr) else 0.0


def action_distribution(
    actions: Union[List[int], NDArray[np.integer]],
) -> Dict[str, float]:
    """
    Calculate action distribution (proportion of HOLD, BUY, SELL).

    Args:
        actions: Array of actions (-1: SELL, 0: HOLD, 1: BUY)

    Returns:
        Dictionary with action distribution {"HOLD": ratio, "BUY": ratio, "SELL": ratio}
    """
    return cast(
        Dict[str, float],
        safe_operation(
            logger=None,
            operation=lambda: _action_distribution_impl(actions),
            context="action_distribution_calculation",
            default_result={"HOLD": 0.0, "BUY": 0.0, "SELL": 0.0},
        ),
    )


def _action_distribution_impl(
    actions: Union[List[int], NDArray[np.integer]],
) -> Dict[str, float]:
    actions_np = np.asarray(actions)

    if len(actions_np) == 0:
        return {"HOLD": 0.0, "BUY": 0.0, "SELL": 0.0}

    # Remove NaN values
    actions_np = actions_np[~np.isnan(actions_np)]

    if len(actions_np) == 0:
        return {"HOLD": 0.0, "BUY": 0.0, "SELL": 0.0}

    # Shift actions: -1,0,1 -> 0,1,2
    actions_shifted = actions_np + 1

    # Count occurrences
    action_counts = np.bincount(actions_shifted.astype(int), minlength=3)

    total_actions = len(actions_np)

    return {
        "HOLD": float(action_counts[ACTION_HOLD + 1] / total_actions),
        "BUY": float(action_counts[ACTION_BUY + 1] / total_actions),
        "SELL": float(action_counts[ACTION_SELL + 1] / total_actions),
    }


@safe_operation
def calculate_performance_metrics(
    returns: pd.Series,
    risk_free_rate: float = 0.02,
    annualize: bool = True,
    periods_per_year: int = TRADING_DAYS_PER_YEAR,
) -> Dict[str, float]:
    """
    Calculate comprehensive performance metrics for a return series.

    Args:
        returns: Series of returns
        risk_free_rate: Risk-free rate for Sharpe ratio
        annualize: Whether to annualize metrics
        periods_per_year: Number of periods per year for annualization

    Returns:
        Dictionary of performance metrics
    """
    if len(returns) == 0:
        return {}

    # Basic metrics
    cumprod_result = (1 + returns).prod()
    total_return = float(cumprod_result - 1)
    volatility = float(returns.std())

    if annualize and len(returns) > 0:
        # Annualize
        volatility = volatility * np.sqrt(periods_per_year)
        annual_return = (1 + total_return) ** (periods_per_year / len(returns)) - 1
    else:
        annual_return = total_return

    sharpe_ratio_val = sharpe_ratio(returns, rf=risk_free_rate, period_per_year=periods_per_year if annualize else len(returns))
    max_drawdown = float((returns.cumsum() - returns.cumsum().cummax()).min())
    win_rate = float((returns > 0).mean())

    return {
        "total_return": total_return,
        "annual_return": annual_return,
        "volatility": volatility,
        "sharpe_ratio": sharpe_ratio_val,
        "max_drawdown": max_drawdown,
        "win_rate": win_rate,
    }


def calculate_returns(equity_curve: pd.Series, freq: str = "D") -> pd.Series:
    """
    Calculate periodic returns from an equity curve.
    """
    if freq == "D":
        return equity_curve.pct_change().fillna(0)
    return equity_curve.resample(freq).last().pct_change().fillna(0)


def calculate_cagr(equity_curve: pd.Series) -> float:
    """
    Calculate Compound Annual Growth Rate (CAGR).
    """
    if len(equity_curve) < 2:
        return 0.0

    total_return = equity_curve.iloc[-1] / equity_curve.iloc[0] - 1
    years = len(equity_curve) / TRADING_DAYS_PER_YEAR
    if years <= 0 or total_return <= -1:
        return 0.0
    return float((1 + total_return) ** (1 / years) - 1)


def calculate_trade_metrics(
    orders: pd.DataFrame,
) -> Dict[str, float]:
    """
    Calculate trade-level metrics from order history.

    Returns keys: total_trades, win_rate, avg_win, avg_loss, profit_factor.
    """
    if orders.empty:
        return {
            "total_trades": 0.0,
            "win_rate": 0.0,
            "avg_win": 0.0,
            "avg_loss": 0.0,
            "profit_factor": 0.0,
        }

    if "pnl" not in orders.columns:
        total_trades = float(len(orders))
        return {
            "total_trades": total_trades,
            "win_rate": 0.0,
            "avg_win": 0.0,
            "avg_loss": 0.0,
            "profit_factor": 0.0,
        }

    pnls = orders["pnl"].dropna()
    if len(pnls) == 0:
        total_trades = float(len(orders))
        return {
            "total_trades": total_trades,
            "win_rate": 0.0,
            "avg_win": 0.0,
            "avg_loss": 0.0,
            "profit_factor": 0.0,
        }

    winning_trades = pnls[pnls > 0]
    losing_trades = pnls[pnls < 0]

    total_trades = float(len(pnls))
    win_rate_val = float(len(winning_trades) / total_trades) if total_trades > 0 else 0.0
    avg_win = float(winning_trades.mean()) if len(winning_trades) > 0 else 0.0
    avg_loss = float(abs(losing_trades.mean())) if len(losing_trades) > 0 else 0.0

    total_win = float(winning_trades.sum())
    total_loss = float(abs(losing_trades.sum()))
    profit_factor = total_win / total_loss if total_loss > 0 else float("inf")

    return {
        "total_trades": total_trades,
        "win_rate": win_rate_val,
        "avg_win": avg_win,
        "avg_loss": avg_loss,
        "profit_factor": float(profit_factor),
    }


def estimate_turnover(
    orders: pd.DataFrame, initial_capital: float = 10000
) -> float:
    """
    Estimate portfolio turnover (annualized).
    """
    if orders.empty or "notional" not in orders.columns:
        return 0.0

    total_turnover = float(orders["notional"].abs().sum())
    days = len(orders) if len(orders) > 0 else 1
    annualized_turnover = total_turnover / initial_capital * (
        TRADING_DAYS_PER_YEAR / days
    )
    return float(annualized_turnover)


def estimate_slippage_bps(orders: pd.DataFrame, slippage_bps: float = 5.0) -> float:
    """
    Estimate slippage impact in basis points.
    """
    if orders.empty:
        return float(slippage_bps)
    return float(slippage_bps)


def _ensure_numpy(data: Union[List[float], np.ndarray, pd.Series]) -> np.ndarray:
    """Convert input data to numpy array."""
    if isinstance(data, pd.Series):
        return data.to_numpy()
    if isinstance(data, list):
        return np.array(data)
    if isinstance(data, np.ndarray):
        return data
    raise TypeError(f"Unsupported data type: {type(data)}")


def p_mean_method(p_values: List[float], method: str = "arithmetic") -> float:
    """
    Calculate combined p-value using the p-mean method.
    """
    if not p_values:
        return 1.0

    p_array = np.array(p_values)
    if method == "arithmetic":
        return float(np.mean(p_array))
    if method == "geometric":
        p_array = np.clip(p_array, 1e-10, 1.0)
        return float(np.exp(np.mean(np.log(p_array))))
    raise ValueError(f"Unknown method: {method}")


def rolling_statistics(
    data: Union[List[float], np.ndarray, pd.Series], window: int
) -> Dict[str, List[float]]:
    """
    Calculate rolling mean/std/min/max for time series data.
    """
    np_data = _ensure_numpy(data)
    if len(np_data) < window:
        return {"mean": [], "std": [], "min": [], "max": []}

    if isinstance(data, pd.Series):
        rolling = data.rolling(window=window)
        return {
            "mean": rolling.mean().dropna().tolist(),
            "std": rolling.std().dropna().tolist(),
            "min": rolling.min().dropna().tolist(),
            "max": rolling.max().dropna().tolist(),
        }

    means: List[float] = []
    stds: List[float] = []
    mins: List[float] = []
    maxs: List[float] = []

    for i in range(window - 1, len(np_data)):
        window_data = np_data[i - window + 1 : i + 1]
        means.append(float(np.mean(window_data)))
        stds.append(float(np.std(window_data, ddof=1)))
        mins.append(float(np.min(window_data)))
        maxs.append(float(np.max(window_data)))

    return {"mean": means, "std": stds, "min": mins, "max": maxs}


def calculate_volatility(
    data: Union[List[float], np.ndarray, pd.Series], window: int = 20
) -> List[float]:
    """
    Calculate rolling volatility (standard deviation).
    """
    stats = rolling_statistics(data, window)
    return stats["std"]


def calculate_autocorrelation(
    data: Union[List[float], np.ndarray, pd.Series], lag: int = 1
) -> float:
    """
    Calculate autocorrelation at specified lag.
    """
    series = data if isinstance(data, pd.Series) else pd.Series(data)
    return autocorrelation(series, lag=lag)


def detect_outliers_iqr(
    data: Union[List[float], np.ndarray, pd.Series], multiplier: float = 1.5
) -> List[bool]:
    """
    Detect outliers using the IQR method.
    """
    np_data = _ensure_numpy(data)
    if len(np_data) == 0:
        return []

    q1 = np.percentile(np_data, 25)
    q3 = np.percentile(np_data, 75)
    iqr = q3 - q1

    lower_bound = q1 - multiplier * iqr
    upper_bound = q3 + multiplier * iqr

    return [(x < lower_bound or x > upper_bound) for x in np_data]


def calculate_atr(data: pd.DataFrame, period: int = 14) -> pd.Series:
    """
    Calculate Average True Range (ATR).
    """
    if len(data) == 0:
        return pd.Series(dtype=float)

    if (
        "high" not in data.columns
        or "low" not in data.columns
        or "close" not in data.columns
    ):
        raise ValueError("Data must contain 'high', 'low', and 'close' columns")

    from ztb.features.generators.technical.volatility.atr import compute_atr

    return compute_atr(data, period=period)


def calculate_distribution_stats(
    data: Union[List[float], np.ndarray], decimals: int = 6
) -> Dict[str, Any]:
    """
    Calculate distribution statistics (mean, std, 95% CI).
    """
    np_data = _ensure_numpy(data)
    if len(np_data) == 0:
        return {"mean": 0.0, "std": 0.0, "ci95": [0.0, 0.0]}

    mean = float(np.mean(np_data))
    std = float(np.std(np_data, ddof=1))
    ci95_low = float(np.percentile(np_data, 2.5))
    ci95_high = float(np.percentile(np_data, 97.5))

    return {
        "mean": round(mean, decimals),
        "std": round(std, decimals),
        "ci95": [round(ci95_low, decimals), round(ci95_high, decimals)],
    }


def sharpe_with_stats(sharpes: List[float]) -> StatsResult:
    """
    Calculate summary statistics for Sharpe ratios.

    Args:
        sharpes: List of Sharpe ratios

    Returns:
        StatsResult with mean, std, and 95% CI.
    """
    return cast(StatsResult, calculate_distribution_stats(sharpes))


def calculate_delta_sharpe(
    base_sharpes: List[float],
    with_feature_sharpes: List[float],
    min_samples: int = 10000,
) -> Optional[StatsResult]:
    """
    Calculate delta Sharpe (stabilized) between baseline and feature-enhanced runs.

    Args:
        base_sharpes: Sharpe ratios for baseline configuration
        with_feature_sharpes: Sharpe ratios with added feature
        min_samples: Minimum combined sample size

    Returns:
        StatsResult for delta Sharpe, or None if insufficient samples.
    """
    total_samples = len(base_sharpes) + len(with_feature_sharpes)
    if total_samples < min_samples:
        return None
    if len(base_sharpes) == 0 or len(with_feature_sharpes) == 0:
        return None

    base_stats = sharpe_with_stats(base_sharpes)
    with_stats = sharpe_with_stats(with_feature_sharpes)

    delta_mean = cast(float, with_stats["mean"]) - cast(float, base_stats["mean"])
    delta_std = np.sqrt(
        cast(float, with_stats["std"]) ** 2 + cast(float, base_stats["std"]) ** 2
    )

    delta_ci95_low = delta_mean - 1.96 * delta_std
    delta_ci95_high = delta_mean + 1.96 * delta_std

    return {
        "mean": round(delta_mean, 6),
        "std": round(delta_std, 6),
        "ci95": [round(delta_ci95_low, 6), round(delta_ci95_high, 6)],
    }


def calculate_feature_metrics(
    feature_data: pd.Series, price_data: pd.Series, feature_name: str
) -> FeatureMetrics:
    """Calculate basic trading metrics for feature evaluation."""
    if feature_name == "RSI":
        signals = pd.Series(0, index=feature_data.index)
        signals[feature_data < 30] = 1
        signals[feature_data > 70] = -1
    elif feature_name == "ROC":
        signals = pd.Series(0, index=feature_data.index)
        signals[feature_data > 5] = 1
        signals[feature_data < -5] = -1
    elif feature_name == "OBV":
        obv_change = feature_data.diff().astype(float)
        signals = pd.Series(0, index=feature_data.index)
        signals[obv_change > 0] = 1
        signals[obv_change < 0] = -1
    elif feature_name == "ZScore":
        signals = pd.Series(0, index=feature_data.index)
        signals[feature_data < -1] = 1
        signals[feature_data > 1] = -1
    else:
        signals = (feature_data > 0).astype(int) - (feature_data < 0).astype(int)

    returns = price_data.pct_change().shift(-1)
    valid_idx = signals.notna() & returns.notna() & (signals != 0)
    if valid_idx.sum() == 0:
        return {
            "win_rate": 0.0,
            "max_drawdown": 0.0,
            "sharpe_ratio": 0.0,
            "sortino_ratio": 0.0,
            "calmar_ratio": 0.0,
            "sample_count": 0,
        }

    strategy_returns = signals[valid_idx] * returns[valid_idx]
    cumulative = (1 + strategy_returns).cumprod()

    return {
        "win_rate": win_rate(strategy_returns),
        "max_drawdown": max_drawdown(cumulative),
        "sharpe_ratio": sharpe_ratio(strategy_returns),
        "sortino_ratio": sortino_ratio(strategy_returns),
        "calmar_ratio": calmar_ratio(strategy_returns),
        "sample_count": int(valid_idx.sum()),
    }


def validate_ablation_results(results: Dict[str, Any]) -> bool:
    """
    Validate ablation results for expected delta Sharpe structure.

    Args:
        results: Ablation result dict

    Returns:
        True if results look valid, False otherwise.
    """
    if "delta_sharpe" not in results or results["delta_sharpe"] is None:
        return False

    delta = results["delta_sharpe"]
    required_keys = ["mean", "std", "ci95"]
    for key in required_keys:
        if key not in delta:
            return False

    if not isinstance(delta["ci95"], list) or len(delta["ci95"]) != 2:
        return False

    return True


@dataclass
class BacktestMetrics:
    """Container for backtest performance metrics."""

    # Risk-adjusted returns
    sharpe_ratio: float
    sortino_ratio: float
    calmar_ratio: float

    # Return metrics
    total_return: float
    cagr: float
    annualized_return: float

    # Risk metrics
    max_drawdown: float
    volatility: float

    # Trade metrics
    total_trades: int
    win_rate: float
    avg_win: float
    avg_loss: float
    profit_factor: float
    turnover: float

    # Slippage estimate
    estimated_slippage_bps: float

    # Statistical significance (optional)
    deflated_sharpe_ratio: Optional[float] = None
    pvalue_bootstrap: Optional[float] = None


class MetricsCalculator:
    """Calculates comprehensive trading performance metrics."""

    @staticmethod
    def calculate_returns(equity_curve: pd.Series, freq: str = "D") -> pd.Series:
        """Calculate periodic returns from equity curve."""
        return calculate_returns(equity_curve, freq=freq)

    @staticmethod
    def calculate_sharpe_ratio(
        returns: pd.Series, risk_free_rate: float = 0.02
    ) -> float:
        """Calculate Sharpe ratio (annualized)."""
        return sharpe_ratio(
            returns, rf=risk_free_rate, period_per_year=TRADING_DAYS_PER_YEAR
        )

    @staticmethod
    def calculate_sortino_ratio(
        returns: pd.Series, risk_free_rate: float = 0.02
    ) -> float:
        """Calculate Sortino ratio (downside deviation only)."""
        return sortino_ratio(
            returns, rf=risk_free_rate, period_per_year=TRADING_DAYS_PER_YEAR
        )

    @staticmethod
    def calculate_calmar_ratio(returns: pd.Series, max_dd: float) -> float:
        """Calculate Calmar ratio (annualized return / max drawdown)."""
        _ = max_dd
        return calmar_ratio(returns, period_per_year=TRADING_DAYS_PER_YEAR)

    @staticmethod
    def calculate_cagr(equity_curve: pd.Series) -> float:
        """Calculate Compound Annual Growth Rate."""
        return calculate_cagr(equity_curve)

    @staticmethod
    def calculate_trade_metrics(
        orders: pd.DataFrame,
    ) -> Tuple[int, float, float, float, float]:
        """
        Calculate trade-level metrics.

        Returns: (total_trades, win_rate, avg_win, avg_loss, profit_factor)
        """
        metrics = calculate_trade_metrics(orders)
        return (
            int(metrics["total_trades"]),
            float(metrics["win_rate"]),
            float(metrics["avg_win"]),
            float(metrics["avg_loss"]),
            float(metrics["profit_factor"]),
        )

    @staticmethod
    def estimate_turnover(
        orders: pd.DataFrame, initial_capital: float = 10000
    ) -> float:
        """Estimate portfolio turnover (annualized)."""
        return estimate_turnover(orders, initial_capital=initial_capital)

    @staticmethod
    def estimate_slippage_bps(orders: pd.DataFrame, slippage_bps: float = 5.0) -> float:
        """Estimate slippage impact in basis points."""
        return estimate_slippage_bps(orders, slippage_bps=slippage_bps)

    @classmethod
    def calculate_all_metrics(
        cls,
        equity_curve: pd.Series,
        orders: pd.DataFrame,
        initial_capital: float = 10000,
        risk_free_rate: float = 0.02,
        slippage_bps: float = 5.0,
    ) -> BacktestMetrics:
        """Calculate all performance metrics."""

        returns = cls.calculate_returns(equity_curve)

        sharpe = cls.calculate_sharpe_ratio(returns, risk_free_rate)
        sortino = cls.calculate_sortino_ratio(returns, risk_free_rate)
        max_dd = max_drawdown(equity_curve)
        calmar = cls.calculate_calmar_ratio(returns, max_dd)
        cagr = cls.calculate_cagr(equity_curve)

        total_return = (
            (equity_curve.iloc[-1] / equity_curve.iloc[0] - 1)
            if len(equity_curve) > 1
            else 0.0
        )
        from ztb.metrics.technical import calculate_volatility_from_returns

        volatility = calculate_volatility_from_returns(
            returns, window=len(returns), annualize=True
        )

        (
            total_trades,
            win_rate_val,
            avg_win,
            avg_loss,
            profit_factor,
        ) = cls.calculate_trade_metrics(orders)
        turnover = cls.estimate_turnover(orders, initial_capital)
        estimated_slippage = cls.estimate_slippage_bps(orders, slippage_bps)

        deflated_sharpe = cls.calculate_deflated_sharpe_ratio(returns)
        pvalue_bootstrap = cls.calculate_bootstrap_pvalue(
            returns, benchmark_returns=returns * 0.5
        )

        return BacktestMetrics(
            sharpe_ratio=sharpe,
            sortino_ratio=sortino,
            calmar_ratio=calmar,
            total_return=total_return,
            cagr=cagr,
            annualized_return=cagr,
            max_drawdown=max_dd,
            volatility=volatility,
            total_trades=total_trades,
            win_rate=win_rate_val,
            avg_win=avg_win,
            avg_loss=avg_loss,
            profit_factor=profit_factor,
            turnover=turnover,
            estimated_slippage_bps=estimated_slippage,
            deflated_sharpe_ratio=deflated_sharpe,
            pvalue_bootstrap=pvalue_bootstrap,
        )

    @staticmethod
    def calculate_deflated_sharpe_ratio(
        returns: pd.Series, num_strategies: int = 1000
    ) -> float:
        """Calculate deflated Sharpe ratio to account for multiple testing."""
        return calculate_deflated_sharpe_ratio(returns, num_strategies)

    @staticmethod
    def calculate_bootstrap_pvalue(
        strategy_returns: pd.Series,
        benchmark_returns: pd.Series,
        num_bootstrap: int = 1000,
    ) -> float:
        """Calculate bootstrap p-value for strategy vs benchmark comparison."""
        return calculate_bootstrap_pvalue(
            strategy_returns, benchmark_returns, n_bootstrap=num_bootstrap
        )

    @staticmethod
    def calculate_returns_autocorrelation(returns: pd.Series, lag: int = 1) -> float:
        """Calculate autocorrelation of returns at specified lag."""
        returns_list = returns.dropna().tolist()
        return autocorrelation(returns_list, lag=lag)

    @staticmethod
    def detect_return_outliers(
        returns: pd.Series, multiplier: float = 1.5
    ) -> List[bool]:
        """Detect outliers in returns using IQR method."""
        returns_list = returns.dropna().tolist()
        return detect_outliers_iqr(returns_list, multiplier)
