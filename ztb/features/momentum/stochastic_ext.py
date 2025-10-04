"""
stochastic_ext.py
Extended Stochastic Oscillator features with divergence and signal analysis
"""

from typing import Any

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from ztb.features.registry import FeatureRegistry


@FeatureRegistry.register("Stochastic_Divergence")
def compute_stochastic_divergence(df: pd.DataFrame) -> pd.Series:
    """Stochastic Divergence Signal (1=bullish divergence, -1=bearish divergence, 0=no divergence)"""
    extended_features = calculate_stochastic_extended(df)
    return extended_features["stochastic_divergence"]


@FeatureRegistry.register("Stochastic_Signal_Strength")
def compute_stochastic_signal_strength(df: pd.DataFrame) -> pd.Series:
    """Stochastic Signal Strength (normalized momentum)"""
    extended_features = calculate_stochastic_extended(df)
    return extended_features["stochastic_signal_strength"]


@FeatureRegistry.register("Stochastic_Trend_Alignment")
def compute_stochastic_trend_alignment(df: pd.DataFrame) -> pd.Series:
    """Stochastic Trend Alignment (1=aligned with trend, -1=against trend)"""
    extended_features = calculate_stochastic_extended(df)
    return extended_features["stochastic_trend_alignment"]


def calculate_stochastic_extended(
    df: pd.DataFrame,
    k_period: int = 14,
    d_period: int = 3,
    slowing_period: int = 3,
) -> pd.DataFrame:
    """
    Calculate extended Stochastic features including:
    - Basic %K and %D lines
    - Divergence signals
    - Signal strength
    - Trend alignment

    Args:
        df: DataFrame with OHLC data
        k_period: Lookback period for %K calculation
        d_period: Smoothing period for %D
        slowing_period: Additional smoothing for %K

    Returns:
        DataFrame with extended Stochastic features
    """
    result = pd.DataFrame(index=df.index)

    # Basic Stochastic calculation
    high_max = df["high"].rolling(window=k_period).max()
    low_min = df["low"].rolling(window=k_period).min()

    # Raw %K
    raw_k = 100 * ((df["close"] - low_min) / (high_max - low_min + 1e-8))

    # Smoothed %K (using simple moving average)
    result["stoch_k"] = raw_k.rolling(window=slowing_period).mean()

    # %D (signal line)
    result["stoch_d"] = result["stoch_k"].rolling(window=d_period).mean()

    # Calculate extended features using Numba
    divergence, signal_strength, trend_alignment = _compute_stochastic_extended(
        np.asarray(df["close"]), np.asarray(result["stoch_k"]), k_period
    )

    result["stochastic_divergence"] = divergence
    result["stochastic_signal_strength"] = signal_strength
    result["stochastic_trend_alignment"] = trend_alignment

    # Handle NaN values
    result = result.fillna(0)

    return result


def _compute_stochastic_extended(
    close: NDArray[np.floating[Any]],
    stoch_k: NDArray[np.floating[Any]],
    k_period: int
) -> tuple[NDArray[np.int32], NDArray[np.floating[Any]], NDArray[np.floating[Any]]]:
    """
    Calculate extended Stochastic features using pure numpy (no numba).
    """
    n = len(close)
    divergence = np.zeros(n, dtype=np.int32)
    signal_strength = np.zeros(n, dtype=np.float64)
    trend_alignment = np.zeros(n, dtype=np.float64)

    # Look for divergences over the last 20-30 periods
    lookback_period = min(30, n // 3)

    for i in range(lookback_period, n):
        # Get recent data
        start_idx = i - lookback_period
        recent_prices = close[start_idx : i + 1]
        recent_k = stoch_k[start_idx : i + 1]

        # Calculate trends using linear regression approximation
        price_trend = recent_prices[-1] - recent_prices[0]
        stoch_trend = recent_k[-1] - recent_k[0]

        # Check for bullish divergence: price makes lower low, stochastic makes higher low
        if price_trend < 0 and stoch_trend > 0:
            divergence[i] = 1
        # Bearish divergence: price makes higher high, stochastic makes lower high
        elif price_trend > 0 and stoch_trend < 0:
            divergence[i] = -1

    # Signal strength: distance from overbought/oversold levels
    for i in range(n):
        if not np.isnan(stoch_k[i]):
            distance_from_center = abs(stoch_k[i] - 50) / 50  # Normalized 0-1 scale
            signal_strength[i] = distance_from_center

    # Trend alignment: compare %K position with price trend
    for i in range(k_period, n):
        # Simple trend calculation using recent closes
        recent_trend = (
            close[i] - close[i - k_period // 2]
            if i >= k_period // 2
            else close[i] - close[max(0, i - 5)]
        )

        # Stochastic trend: above/below 50
        stoch_position = 1 if stoch_k[i] > 50 else -1
        price_trend = 1 if recent_trend > 0 else -1

        # Alignment: 1 if both agree, -1 if they disagree
        trend_alignment[i] = 1 if stoch_position == price_trend else -1

    return divergence, signal_strength, trend_alignment


def stochastic_feature_summary() -> dict[str, str]:
    """
    Returns a dictionary summarizing each extended Stochastic feature.
    """
    return {
        "stoch_k": "Stochastic %K line (main oscillator)",
        "stoch_d": "Stochastic %D line (signal/ smoothed %K)",
        "stochastic_divergence": "Divergence signals (1=bullish, -1=bearish, 0=none)",
        "stochastic_signal_strength": "Normalized distance from neutral (50)",
        "stochastic_trend_alignment": "Alignment with price trend (1=aligned, -1=against)",
    }


if __name__ == "__main__":
    # Simple test
    np.random.seed(42)
    n = 500

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

    # Calculate extended features
    features = calculate_stochastic_extended(test_data)

    print("Stochastic Extended Features:")
    print(features.head(10))
    print(f"\nFeature columns: {list(features.columns)}")
    print(f"NaN count per column:\n{features.isnull().sum()}")

    # Summary
    summary = stochastic_feature_summary()
    print(f"\nFeature summary:\n{summary}")
