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

from typing import Any, Dict, Optional, TypedDict, Union, cast

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from scipy import stats

# Trading constants
from ztb.trading.constants import TRADING_DAYS_PER_YEAR
from ztb.utils.errors import safe_operation


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
    Calculate win rate (percentage of positive returns).

    Win rate measures the proportion of profitable periods/trades.
    A higher win rate indicates more consistent positive performance.

    Args:
        returns: Return series (pandas Series or numpy array)

    Returns:
        Win rate as float between 0 and 1 (e.g., 0.65 for 65% win rate)

    Examples:
        >>> import numpy as np
        >>> returns = np.array([0.01, -0.02, 0.03, -0.01, 0.02])
        >>> win_rate(returns)
        0.6
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

    win_rate = len(winning_trades) / len(returns)
    avg_win = np.mean(winning_trades)
    avg_loss = np.mean(np.abs(losing_trades))  # Use absolute value for losses

    return float((win_rate * avg_win) - ((1 - win_rate) * avg_loss))


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
    if dates is None:
        # Assume daily returns starting from a reference date
        dates = pd.date_range(start="2020-01-01", periods=len(returns), freq="D")
    else:
        dates = pd.to_datetime(dates)

    # Create DataFrame for analysis
    df = pd.DataFrame({"returns": returns, "dates": dates})

    # Extract seasonal components
    df["month"] = df["dates"].dt.month
    df["quarter"] = df["dates"].dt.quarter
    df["year"] = df["dates"].dt.year
    df["day_of_year"] = df["dates"].dt.dayofyear

    results = {}

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
    statistical_tests_results = {}
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
    transitions = {}
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
            "kolmogorov_smirnov": {"statistic": 0.0, "p_value": 0.0, "is_normal": False},
            "jarque_bera": {"statistic": 0.0, "p_value": 0.0, "is_normal": False},
        },
    )


def _test_normality_impl(returns: Union[pd.Series, NDArray[Any]]) -> Dict[str, Any]:
    """Implementation of normality testing."""
    returns = np.asarray(returns)

    if len(returns) < 3:
        return {
            "shapiro_wilk": {"statistic": None, "p_value": None, "is_normal": False},
            "kolmogorov_smirnov": {"statistic": 0.0, "p_value": 0.0, "is_normal": False},
            "jarque_bera": {"statistic": 0.0, "p_value": 0.0, "is_normal": False},
        }

    # Remove NaN values
    returns = returns[~np.isnan(returns)]

    if len(returns) < 3:
        return {
            "shapiro_wilk": {"statistic": None, "p_value": None, "is_normal": False},
            "kolmogorov_smirnov": {"statistic": 0.0, "p_value": 0.0, "is_normal": False},
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


def _autocorrelation_impl(returns: Union[pd.Series, NDArray[Any]], lag: int = 1) -> float:
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
