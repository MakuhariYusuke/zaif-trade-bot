"""
supertrend_ext.py
Extended Supertrend features with trend strength, reversal signals, and multi-timeframe analysis
"""

from typing import Any

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from ztb.features.registry import FeatureRegistry


@FeatureRegistry.register("Supertrend_Strength")
def compute_supertrend_strength(df: pd.DataFrame) -> pd.Series:
    """Supertrend Trend Strength (normalized distance from bands)"""
    extended_features = calculate_supertrend_extended(df)
    return extended_features["supertrend_strength"]


@FeatureRegistry.register("Supertrend_Reversal_Signal")
def compute_supertrend_reversal_signal(df: pd.DataFrame) -> pd.Series:
    """Supertrend Reversal Signal (1=buy, -1=sell, 0=no signal)"""
    extended_features = calculate_supertrend_extended(df)
    return extended_features["supertrend_reversal_signal"]


@FeatureRegistry.register("Supertrend_Trend_Duration")
def compute_supertrend_trend_duration(df: pd.DataFrame) -> pd.Series:
    """Supertrend Trend Duration (bars since last reversal)"""
    extended_features = calculate_supertrend_extended(df)
    return extended_features["supertrend_trend_duration"]


@FeatureRegistry.register("Supertrend_Volatility_Filter")
def compute_supertrend_volatility_filter(df: pd.DataFrame) -> pd.Series:
    """Supertrend Volatility Filter (ATR-normalized trend strength)"""
    extended_features = calculate_supertrend_extended(df)
    return extended_features["supertrend_volatility_filter"]


def calculate_supertrend_extended(
    df: pd.DataFrame,
    period: int = 10,
    multiplier: float = 3.0,
) -> pd.DataFrame:
    """
    Calculate extended Supertrend features including:
    - Trend strength (normalized distance from bands)
    - Reversal signals
    - Trend duration
    - Volatility-adjusted signals

    Args:
        df: DataFrame with OHLC data and ATR column
        period: ATR calculation period (default: 10)
        multiplier: ATR multiplier for band calculation (default: 3.0)

    Returns:
        DataFrame with extended Supertrend features
    """
    from .supertrend import Supertrend

    # Ensure ATR is available
    atr_col = f"atr_{period}"
    if atr_col not in df.columns:
        from ztb.features.volatility.atr import compute_atr_simplified

        atr_series = compute_atr_simplified(df, period)
        df = df.copy()
        df[atr_col] = atr_series

    # Get basic Supertrend calculation
    st_feature = Supertrend()
    st_result = st_feature.compute(df, period=period, multiplier=multiplier)

    result = pd.DataFrame(index=df.index)
    result["supertrend"] = st_result["supertrend"]
    result["supertrend_direction"] = st_result["supertrend_direction"]

    # Calculate extended features using Numba-optimized function
    high = np.asarray(df["high"])
    low = np.asarray(df["low"])
    close = np.asarray(df["close"])
    atr_col = f"atr_{period}"
    atr = np.asarray(df[atr_col])

    strength, reversal_signal, trend_duration = _compute_supertrend_extended(
        high, low, close, atr, np.asarray(result["supertrend_direction"]), multiplier
    )

    result["supertrend_strength"] = strength
    result["supertrend_reversal_signal"] = reversal_signal
    result["supertrend_trend_duration"] = trend_duration

    # Volatility filter: strength adjusted by recent ATR changes
    atr_ma = pd.Series(atr).rolling(window=period).mean()
    atr_ratio = atr / (np.asarray(atr_ma) + 1e-8)
    result["supertrend_volatility_filter"] = strength * atr_ratio

    # Handle NaN values
    result = result.fillna(0)

    return result


def _compute_supertrend_extended(
    high: NDArray[np.floating[Any]],
    low: NDArray[np.floating[Any]],
    close: NDArray[np.floating[Any]],
    atr: NDArray[np.floating[Any]],
    direction: NDArray[np.floating[Any]],
    multiplier: float
) -> tuple[NDArray[np.floating[Any]], NDArray[np.int32], NDArray[np.floating[Any]]]:
    """
    Calculate extended Supertrend features using pure numpy (no numba).
    """
    n = len(close)
    strength = np.zeros(n, dtype=np.float64)
    reversal_signal = np.zeros(n, dtype=np.int32)
    trend_duration = np.zeros(n, dtype=np.float64)

    current_trend_start = 0

    for i in range(n):
        if i == 0:
            continue

        # Calculate current bands (avoid recalculating in each iteration)
        hl2 = (high[i] + low[i]) / 2
        upperband = hl2 + (multiplier * atr[i])
        lowerband = hl2 - (multiplier * atr[i])

        # Trend strength: normalized distance from current band
        if direction[i] == 1:  # Uptrend
            distance = abs(close[i] - lowerband)
            strength[i] = distance / (atr[i] + 1e-8)  # Normalize by ATR
        else:  # Downtrend
            distance = abs(close[i] - upperband)
            strength[i] = distance / (atr[i] + 1e-8)

        # Trend duration
        if direction[i] != direction[i - 1]:
            current_trend_start = i
        trend_duration[i] = i - current_trend_start

        # Reversal signals (look for 3-bar confirmation)
        if i >= 2:
            prev2_direction = direction[i - 2]
            prev_direction = direction[i - 1]
            current_direction = direction[i]

            # Bullish reversal: downtrend -> uptrend
            if (
                prev2_direction == -1
                and prev_direction == -1
                and current_direction == 1
            ):
                reversal_signal[i] = 1
            # Bearish reversal: uptrend -> downtrend
            elif (
                prev2_direction == 1 and prev_direction == 1 and current_direction == -1
            ):
                reversal_signal[i] = -1

    return strength, reversal_signal, trend_duration


def supertrend_feature_summary() -> dict[str, str]:
    """
    Returns a dictionary summarizing each extended Supertrend feature.
    """
    return {
        "supertrend": "Basic Supertrend value",
        "supertrend_direction": "Trend direction (1=uptrend, -1=downtrend)",
        "supertrend_strength": "Normalized distance from trend bands (higher = stronger trend)",
        "supertrend_reversal_signal": "Reversal signals (1=buy, -1=sell, 0=no signal)",
        "supertrend_trend_duration": "Bars since last trend reversal",
        "supertrend_volatility_filter": "Volatility-adjusted trend strength",
    }


if __name__ == "__main__":
    # Simple test
    np.random.seed(42)
    n = 1000

    # Generate test OHLC data
    test_data = pd.DataFrame(
        {
            "high": np.random.uniform(100, 200, n),
            "low": np.random.uniform(50, 150, n),
            "close": np.random.uniform(75, 175, n),
            "volume": np.random.uniform(1000, 5000, n),
        }
    )

    # Ensure high >= close >= low
    test_data["high"] = np.maximum(test_data["high"], test_data["close"])
    test_data["low"] = np.minimum(test_data["low"], test_data["close"])

    # Add ATR column (required for Supertrend)
    from ztb.features.volatility.atr import compute_atr

    test_data["atr_10"] = compute_atr(test_data, period=10)

    # Calculate extended features
    features = calculate_supertrend_extended(test_data)

    print("Supertrend Extended Features:")
    print(features.head(10))
    print(f"\nFeature columns: {list(features.columns)}")
    print(f"NaN count per column:\n{features.isnull().sum()}")

    # Summary
    summary = supertrend_feature_summary()
    print(f"\nFeature summary:\n{summary}")
