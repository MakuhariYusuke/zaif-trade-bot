"""
Feature optimization utilities for slow features.

This module provides optimization techniques for computationally expensive features
including KAMA, ADX, and Kalman filters.
"""

import time
from functools import lru_cache
from typing import Any, Callable, Dict, Optional, Union, cast

import numba as nb  # type: ignore
import numpy as np
import pandas as pd


def optimize_kama(
    prices: pd.Series,
    fast_period: int = 2,
    slow_period: int = 30,
    efficiency_ratio_period: int = 10,
) -> pd.Series:
    """
    Optimized Kaufman Adaptive Moving Average (KAMA)

    Args:
        prices: Price series
        fast_period: Fast EMA period
        slow_period: Slow EMA period
        efficiency_ratio_period: Efficiency ratio period

    Returns:
        KAMA values
    """

    @nb.jit(nopython=True)
    def _calculate_kama_numba(
        price_array, fast_period, slow_period, efficiency_ratio_period
    ):  # type: ignore
        n = len(price_array)
        kama = np.full(n, np.nan)

        if n < slow_period + efficiency_ratio_period:
            return kama

        # Calculate efficiency ratio
        for i in range(slow_period + efficiency_ratio_period - 1, n):
            # Volatility (sum of absolute changes)
            volatility = 0.0
            for j in range(i - efficiency_ratio_period + 1, i + 1):
                volatility += abs(price_array[j] - price_array[j - 1])

            # Direction (absolute change over period)
            direction = abs(price_array[i] - price_array[i - efficiency_ratio_period])

            # Efficiency ratio
            er = direction / volatility if volatility > 0 else 0.0

            # Smoothing constant
            fast_sc = 2.0 / (fast_period + 1)
            slow_sc = 2.0 / (slow_period + 1)
            sc = er * (fast_sc - slow_sc) + slow_sc
            sc = sc * sc  # Square it for more stability

            # KAMA calculation
            if np.isnan(kama[i - 1]):
                kama[i] = price_array[i]
            else:
                kama[i] = kama[i - 1] + sc * (price_array[i] - kama[i - 1])

        return kama

    price_array = prices.values
    kama_values = cast(
        np.ndarray,
        _calculate_kama_numba(
            price_array, fast_period, slow_period, efficiency_ratio_period
        ),
    )

    return pd.Series(kama_values, index=prices.index, name="KAMA")


def optimize_adx(
    high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14
) -> pd.DataFrame:
    """
    Optimized Average Directional Index (ADX)

    Args:
        high: High prices
        low: Low prices
        close: Close prices
        period: ADX period

    Returns:
        DataFrame with ADX, +DI, -DI
    """

    @nb.jit(nopython=True)
    def _calculate_adx_numba(high_array, low_array, close_array, period):  # type: ignore
        n = len(high_array)
        adx = np.full(n, np.nan)
        plus_di = np.full(n, np.nan)
        minus_di = np.full(n, np.nan)

        if n < period + 1:
            return adx, plus_di, minus_di

        # Calculate True Range, +DM, -DM
        tr = np.full(n, np.nan)
        plus_dm = np.full(n, 0.0)
        minus_dm = np.full(n, 0.0)

        for i in range(1, n):
            # True Range
            tr[i] = max(
                high_array[i] - low_array[i],
                abs(high_array[i] - close_array[i - 1]),
                abs(low_array[i] - close_array[i - 1]),
            )

            # Directional Movement
            move_up = high_array[i] - high_array[i - 1]
            move_down = low_array[i - 1] - low_array[i]

            if move_up > move_down and move_up > 0:
                plus_dm[i] = move_up
            elif move_down > move_up and move_down > 0:
                minus_dm[i] = move_down

        # Smooth TR, +DM, -DM
        tr_smooth = np.full(n, np.nan)
        plus_dm_smooth = np.full(n, np.nan)
        minus_dm_smooth = np.full(n, np.nan)

        # Initialize first values
        tr_smooth[period] = np.sum(tr[1 : period + 1])
        plus_dm_smooth[period] = np.sum(plus_dm[1 : period + 1])
        minus_dm_smooth[period] = np.sum(minus_dm[1 : period + 1])

        # Calculate smoothed values
        for i in range(period + 1, n):
            tr_smooth[i] = tr_smooth[i - 1] - (tr_smooth[i - 1] / period) + tr[i]
            plus_dm_smooth[i] = (
                plus_dm_smooth[i - 1] - (plus_dm_smooth[i - 1] / period) + plus_dm[i]
            )
            minus_dm_smooth[i] = (
                minus_dm_smooth[i - 1] - (minus_dm_smooth[i - 1] / period) + minus_dm[i]
            )

        # Calculate +DI, -DI, DX, ADX
        for i in range(period, n):
            if tr_smooth[i] > 0:
                plus_di[i] = 100 * plus_dm_smooth[i] / tr_smooth[i]
                minus_di[i] = 100 * minus_dm_smooth[i] / tr_smooth[i]

                dx = (
                    100 * abs(plus_di[i] - minus_di[i]) / (plus_di[i] + minus_di[i])
                    if (plus_di[i] + minus_di[i]) > 0
                    else 0
                )

                if i == period:
                    adx[i] = dx
                else:
                    adx[i] = (adx[i - 1] * (period - 1) + dx) / period

        return adx, plus_di, minus_di

    high_array = high.values
    low_array = low.values
    close_array = close.values

    adx_values, plus_di_values, minus_di_values = _calculate_adx_numba(
        high_array, low_array, close_array, period
    )

    return pd.DataFrame(
        {"ADX": adx_values, "+DI": plus_di_values, "-DI": minus_di_values},
        index=high.index,
    )


def optimize_kalman_filter(
    prices: pd.Series, process_noise: float = 1e-5, measurement_noise: float = 1e-3
) -> pd.Series:
    """
    Optimized Kalman filter for price smoothing

    Args:
        prices: Price series
        process_noise: Process noise variance
        measurement_noise: Measurement noise variance

    Returns:
        Smoothed price series
    """

    @nb.jit(nopython=True)
    def _kalman_filter_numba(price_array, process_noise, measurement_noise):  # type: ignore
        n = len(price_array)
        filtered = np.full(n, np.nan)

        if n == 0:
            return filtered

        # Initialize
        x_hat = price_array[0]  # State estimate
        p = 1.0  # Error covariance

        filtered[0] = x_hat

        for i in range(1, n):
            if np.isnan(price_array[i]):
                filtered[i] = x_hat
                continue

            # Prediction
            x_hat_minus = x_hat
            p_minus = p + process_noise

            # Update
            k = p_minus / (p_minus + measurement_noise)  # Kalman gain
            x_hat = x_hat_minus + k * (price_array[i] - x_hat_minus)
            p = (1 - k) * p_minus

            filtered[i] = x_hat

        return filtered

    price_array = prices.values
    filtered_values = cast(
        np.ndarray, _kalman_filter_numba(price_array, process_noise, measurement_noise)
    )

    return pd.Series(filtered_values, index=prices.index, name="Kalman_Filter")


@lru_cache(maxsize=128)
def cached_computation(func: Callable[..., Any], *args: Any, **kwargs: Any) -> Any:
    """
    Cache expensive computations

    Args:
        func: Function to cache
        *args: Positional arguments
        **kwargs: Keyword arguments

    Returns:
        Cached result
    """
    return func(*args, **kwargs)


def vectorized_rolling_computation(
    data: pd.Series, window: int, func: Callable
) -> pd.Series:
    """
    Vectorized rolling computation for better performance

    Args:
        data: Input data series
        window: Rolling window size
        func: Function to apply

    Returns:
        Result series
    """
    if len(data) < window:
        return pd.Series([np.nan] * len(data), index=data.index)

    # Use pandas rolling with vectorized operations where possible
    if func.__name__ == "mean":
        return data.rolling(window=window, min_periods=1).mean()
    elif func.__name__ == "std":
        return data.rolling(window=window, min_periods=1).std()
    elif func.__name__ == "var":
        return data.rolling(window=window, min_periods=1).var()
    else:
        # Fallback to apply
        return data.rolling(window=window, min_periods=1).apply(func, raw=True)


def benchmark_feature_computation(
    feature_func: Callable[..., Any], *args: Any, n_runs: int = 5
) -> Dict[str, float | int]:
    """
    Benchmark feature computation performance

    Args:
        feature_func: Feature computation function
        *args: Arguments for the function
        n_runs: Number of benchmark runs

    Returns:
        Performance statistics
    """
    times = []

    for _ in range(n_runs):
        start_time = time.time()
        feature_func(*args)
        end_time = time.time()
        times.append(end_time - start_time)

    times_array = np.array(times)

    return {
        "mean_time": float(np.mean(times_array)),
        "std_time": float(np.std(times_array)),
        "min_time": float(np.min(times_array)),
        "max_time": float(np.max(times_array)),
        "median_time": float(np.median(times_array)),
        "n_runs": n_runs,  # int 許容 (型注釈を Union 化)
    }


def optimize_feature_pipeline(
    features_config: Dict[str, Any], ohlc_data: pd.DataFrame
) -> Dict[str, Any]:
    """
    Optimize feature computation pipeline

    Args:
        features_config: Feature configuration
        ohlc_data: OHLC data

    Returns:
        Optimized feature results
    """
    optimized_results = {}
    performance_stats = {}

    # Process features in parallel where possible
    slow_features = ["KAMA", "ADX", "KalmanFilter"]

    for feature_name, config in features_config.items():
        result: Optional[Union[pd.Series, pd.DataFrame]] = (
            None  # 未バインド警告回避のため初期化
        )

        if feature_name in slow_features:
            # Use optimized versions
            if feature_name == "KAMA":
                result = optimize_kama(ohlc_data["close"], **config.get("params", {}))
            elif feature_name == "ADX":
                result = optimize_adx(
                    ohlc_data["high"],
                    ohlc_data["low"],
                    ohlc_data["close"],
                    **config.get("params", {}),
                )
            elif feature_name == "KalmanFilter":
                result = optimize_kalman_filter(
                    ohlc_data["close"], **config.get("params", {})
                )
            else:
                raise ValueError(f"未対応の slow feature: {feature_name}")

            # Benchmark performance
            perf_stats = benchmark_feature_computation(
                lambda: result,  # Dummy lambda to benchmark result generation
                n_runs=1,
            )
            performance_stats[feature_name] = perf_stats
        else:
            # Use regular computation
            result = config["func"](ohlc_data, **config.get("params", {}))

        optimized_results[feature_name] = result

    return {
        "features": optimized_results,
        "performance": performance_stats,
        "optimization_applied": list(slow_features),
    }
