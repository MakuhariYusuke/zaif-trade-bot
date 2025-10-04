"""
bollinger_ext.py
Extended Bollinger Bands features with squeeze detection and bandwidth analysis
"""

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from typing import Any

from ztb.features.registry import FeatureRegistry


@FeatureRegistry.register("Bollinger_Squeeze")
def compute_bollinger_squeeze(df: pd.DataFrame) -> pd.Series:
    """Bollinger Bands Squeeze (1=squeeze, 0=no squeeze)"""
    extended_features = calculate_bollinger_extended(df)
    return extended_features["bollinger_squeeze"]


@FeatureRegistry.register("Bollinger_Bandwidth")
def compute_bollinger_bandwidth(df: pd.DataFrame) -> pd.Series:
    """Bollinger Bands Bandwidth (volatility measure)"""
    extended_features = calculate_bollinger_extended(df)
    return extended_features["bollinger_bandwidth"]


@FeatureRegistry.register("Bollinger_Percent_B")
def compute_bollinger_percent_b(df: pd.DataFrame) -> pd.Series:
    """Bollinger Bands %B (position within bands)"""
    extended_features = calculate_bollinger_extended(df)
    return extended_features["bollinger_percent_b"]


@FeatureRegistry.register("Bollinger_Band_Expansion")
def compute_bollinger_expansion(df: pd.DataFrame) -> pd.Series:
    """Bollinger Bands Expansion Rate"""
    extended_features = calculate_bollinger_extended(df)
    return extended_features["bollinger_expansion"]


def calculate_bollinger_extended(
    df: pd.DataFrame,
    period: int = 20,
    std_dev: float = 2.0,
) -> pd.DataFrame:
    """
    Calculate extended Bollinger Bands features including:
    - Basic upper/middle/lower bands
    - Bandwidth (volatility measure)
    - %B (position within bands)
    - Squeeze detection
    - Band expansion rate

    Args:
        df: DataFrame with OHLC data
        period: Moving average period
        std_dev: Standard deviation multiplier

    Returns:
        DataFrame with extended Bollinger Bands features
    """
    result = pd.DataFrame(index=df.index)

    # Basic Bollinger Bands calculation using vectorized operations
    middle_band = df["close"].rolling(window=period).mean()
    std = df["close"].rolling(window=period).std()

    result["bollinger_upper"] = middle_band + (std_dev * std)
    result["bollinger_middle"] = middle_band
    result["bollinger_lower"] = middle_band - (std_dev * std)

    # Calculate extended features using Numba
    bandwidth, percent_b, squeeze, expansion = _compute_bollinger_extended(
        np.asarray(result["bollinger_upper"]),
        np.asarray(result["bollinger_middle"]),
        np.asarray(result["bollinger_lower"]),
        np.asarray(df["close"]),
        period,
    )

    result["bollinger_bandwidth"] = bandwidth
    result["bollinger_percent_b"] = percent_b
    result["bollinger_squeeze"] = squeeze
    result["bollinger_expansion"] = expansion

    # Handle NaN values
    result = result.fillna(0)

    return result


def _compute_bollinger_extended(
    upper: NDArray[np.floating[Any]],
    middle: NDArray[np.floating[Any]],
    lower: NDArray[np.floating[Any]],
    close: NDArray[np.floating[Any]],
    period: int
) -> tuple[NDArray[np.floating[Any]], NDArray[np.floating[Any]], NDArray[np.int32], NDArray[np.floating[Any]]]:
    """
    Calculate extended Bollinger Bands features using pure numpy (no numba).
    """
    n = len(upper)
    bandwidth = np.zeros(n, dtype=np.float64)
    percent_b = np.zeros(n, dtype=np.float64)
    squeeze = np.zeros(n, dtype=np.int32)
    expansion = np.zeros(n, dtype=np.float64)

    for i in range(n):
        if not (np.isnan(upper[i]) or np.isnan(middle[i]) or np.isnan(lower[i])):
            # Bandwidth: (Upper - Lower) / Middle
            bandwidth[i] = (upper[i] - lower[i]) / (middle[i] + 1e-8)

            # %B: (Price - Lower) / (Upper - Lower)
            percent_b[i] = (close[i] - lower[i]) / (upper[i] - lower[i] + 1e-8)

            # Squeeze detection: when bandwidth is below its moving average
            if i >= period - 1:
                # Calculate moving average of bandwidth
                start_idx = max(0, i - period + 1)
                bandwidth_window = bandwidth[start_idx : i + 1]
                bandwidth_ma = np.mean(bandwidth_window)
                squeeze[i] = 1 if bandwidth[i] < bandwidth_ma else 0

                # Band expansion rate: rate of change of bandwidth
                if i >= period:
                    prev_bandwidth = bandwidth[i - period]
                    if prev_bandwidth != 0:
                        expansion[i] = (bandwidth[i] - prev_bandwidth) / prev_bandwidth

    return bandwidth, percent_b, squeeze, expansion


def bollinger_feature_summary() -> dict[str, str]:
    """
    Returns a dictionary summarizing each extended Bollinger Bands feature.
    """
    return {
        "bollinger_upper": "Upper Bollinger Band",
        "bollinger_middle": "Middle Bollinger Band (SMA)",
        "bollinger_lower": "Lower Bollinger Band",
        "bollinger_bandwidth": "Band width (volatility measure)",
        "bollinger_percent_b": "Position within bands (0=lower, 0.5=middle, 1=upper)",
        "bollinger_squeeze": "Squeeze signal (1=squeeze, 0=no squeeze)",
        "bollinger_expansion": "Rate of band expansion/contraction",
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
    features = calculate_bollinger_extended(test_data)

    print("Bollinger Bands Extended Features:")
    print(features.head(10))
    print(f"\nFeature columns: {list(features.columns)}")
    print(f"NaN count per column:\n{features.isnull().sum()}")

    # Summary
    summary = bollinger_feature_summary()
    print(f"\nFeature summary:\n{summary}")
