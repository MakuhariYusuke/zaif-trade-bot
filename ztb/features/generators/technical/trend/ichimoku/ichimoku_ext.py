#!/usr/bin/env python3
"""
ichimoku_ext.py
Extended Ichimoku features with cloud thickness, price-cloud distance, and lagging span analysis
Multi-timeframe support: 1min to 1day equivalent timeframes
"""

from typing import Dict, Optional

import numpy as np
import pandas as pd

from ztb.features.core.registry import FeatureRegistry
from ztb.features.timeframe import Timeframe

# Traditional Ichimoku parameters for different timeframes
ICHIMOKU_PARAMS = {
    Timeframe.M1: {"tenkan": 1, "kijun": 4, "senkou_b": 8},  # 1-minute equivalent
    Timeframe.M5: {"tenkan": 2, "kijun": 7, "senkou_b": 14},  # 5-minute equivalent
    Timeframe.M15: {"tenkan": 4, "kijun": 13, "senkou_b": 26},  # 15-minute equivalent
    Timeframe.H1: {"tenkan": 7, "kijun": 22, "senkou_b": 44},  # 1-hour equivalent
    Timeframe.H4: {"tenkan": 14, "kijun": 44, "senkou_b": 88},  # 4-hour equivalent
    Timeframe.D1: {
        "tenkan": 9,
        "kijun": 26,
        "senkou_b": 52,
    },  # 1-day equivalent (traditional)
}


def get_ichimoku_params(timeframe: Timeframe) -> Dict[str, int]:
    """
    Get Ichimoku parameters for a specific timeframe.

    Args:
        timeframe: Timeframe enum value

    Returns:
        Dictionary with tenkan, kijun, and senkou_b periods
    """
    return ICHIMOKU_PARAMS[timeframe]


@FeatureRegistry.register("Ichimoku_Tenkan")
def compute_ichimoku_tenkan(
    df: pd.DataFrame, timeframe: Optional[Timeframe] = None
) -> pd.Series:
    """Ichimoku Tenkan-sen (Conversion Line)"""
    extended_features = calculate_ichimoku_extended(df, timeframe=timeframe)
    return extended_features["ichimoku_tenkan"]


@FeatureRegistry.register("Ichimoku_Kijun")
def compute_ichimoku_kijun(
    df: pd.DataFrame, timeframe: Optional[Timeframe] = None
) -> pd.Series:
    """Ichimoku Kijun-sen (Base Line)"""
    extended_features = calculate_ichimoku_extended(df, timeframe=timeframe)
    return extended_features["ichimoku_kijun"]


@FeatureRegistry.register("Ichimoku_Senkou_A")
def compute_ichimoku_senkou_a(
    df: pd.DataFrame, timeframe: Optional[Timeframe] = None
) -> pd.Series:
    """Ichimoku Senkou Span A (Leading Span A)"""
    extended_features = calculate_ichimoku_extended(df, timeframe=timeframe)
    return extended_features["ichimoku_senkou_a"]


@FeatureRegistry.register("Ichimoku_Senkou_B")
def compute_ichimoku_senkou_b(
    df: pd.DataFrame, timeframe: Optional[Timeframe] = None
) -> pd.Series:
    """Ichimoku Senkou Span B (Leading Span B)"""
    extended_features = calculate_ichimoku_extended(df, timeframe=timeframe)
    return extended_features["ichimoku_senkou_b"]


@FeatureRegistry.register("Ichimoku_Chikou")
def compute_ichimoku_chikou(
    df: pd.DataFrame, timeframe: Optional[Timeframe] = None
) -> pd.Series:
    """Ichimoku Chikou Span (Lagging Span)"""
    extended_features = calculate_ichimoku_extended(df, timeframe=timeframe)
    return extended_features["ichimoku_chikou"]


@FeatureRegistry.register("Ichimoku_Cloud_Thickness")
def compute_ichimoku_cloud_thickness(
    df: pd.DataFrame, timeframe: Optional[Timeframe] = None
) -> pd.Series:
    """Ichimoku Cloud Thickness (volatility measure)"""
    extended_features = calculate_ichimoku_extended(df, timeframe=timeframe)
    return extended_features["ichimoku_cloud_thickness"]


@FeatureRegistry.register("Ichimoku_Price_Cloud_Distance")
def compute_ichimoku_price_cloud_distance(
    df: pd.DataFrame, timeframe: Optional[Timeframe] = None
) -> pd.Series:
    """Ichimoku Price-Cloud Distance"""
    extended_features = calculate_ichimoku_extended(df, timeframe=timeframe)
    return extended_features["ichimoku_price_cloud_distance"]


@FeatureRegistry.register("Ichimoku_Composite_Signal")
def compute_ichimoku_composite_signal(
    df: pd.DataFrame, timeframe: Optional[Timeframe] = None
) -> pd.Series:
    """Ichimoku Composite Signal"""
    extended_features = calculate_ichimoku_extended(df, timeframe=timeframe)
    return extended_features["ichimoku_composite_signal"]


@FeatureRegistry.register("Ichimoku_Trend")
def compute_ichimoku_trend(
    df: pd.DataFrame, timeframe: Optional[Timeframe] = None
) -> pd.Series:
    """Ichimoku Trend Determination (1=bullish, -1=bearish, 0=neutral)"""
    extended_features = calculate_ichimoku_extended(df, timeframe=timeframe)
    return extended_features["ichimoku_trend"]


# === Multi-Timeframe Ichimoku Features ===


@FeatureRegistry.register("Ichimoku_Composite_Signal_M1")
def compute_ichimoku_composite_signal_m1(df: pd.DataFrame) -> pd.Series:
    """Ichimoku Composite Signal for 1-minute timeframe"""
    return compute_ichimoku_composite_signal(df, timeframe=Timeframe.M1)


@FeatureRegistry.register("Ichimoku_Composite_Signal_M5")
def compute_ichimoku_composite_signal_m5(df: pd.DataFrame) -> pd.Series:
    """Ichimoku Composite Signal for 5-minute timeframe"""
    return compute_ichimoku_composite_signal(df, timeframe=Timeframe.M5)


@FeatureRegistry.register("Ichimoku_Composite_Signal_M15")
def compute_ichimoku_composite_signal_m15(df: pd.DataFrame) -> pd.Series:
    """Ichimoku Composite Signal for 15-minute timeframe"""
    return compute_ichimoku_composite_signal(df, timeframe=Timeframe.M15)


@FeatureRegistry.register("Ichimoku_Composite_Signal_H1")
def compute_ichimoku_composite_signal_h1(df: pd.DataFrame) -> pd.Series:
    """Ichimoku Composite Signal for 1-hour timeframe"""
    return compute_ichimoku_composite_signal(df, timeframe=Timeframe.H1)


@FeatureRegistry.register("Ichimoku_Composite_Signal_H4")
def compute_ichimoku_composite_signal_h4(df: pd.DataFrame) -> pd.Series:
    """Ichimoku Composite Signal for 4-hour timeframe"""
    return compute_ichimoku_composite_signal(df, timeframe=Timeframe.H4)


@FeatureRegistry.register("Ichimoku_Composite_Signal_D1")
def compute_ichimoku_composite_signal_d1(df: pd.DataFrame) -> pd.Series:
    """Ichimoku Composite Signal for daily timeframe"""
    return compute_ichimoku_composite_signal(df, timeframe=Timeframe.D1)


@FeatureRegistry.register("Ichimoku_Trend_M1")
def compute_ichimoku_trend_m1(df: pd.DataFrame) -> pd.Series:
    """Ichimoku Trend for 1-minute timeframe"""
    return compute_ichimoku_trend(df, timeframe=Timeframe.M1)


@FeatureRegistry.register("Ichimoku_Trend_M5")
def compute_ichimoku_trend_m5(df: pd.DataFrame) -> pd.Series:
    """Ichimoku Trend for 5-minute timeframe"""
    return compute_ichimoku_trend(df, timeframe=Timeframe.M5)


@FeatureRegistry.register("Ichimoku_Trend_M15")
def compute_ichimoku_trend_m15(df: pd.DataFrame) -> pd.Series:
    """Ichimoku Trend for 15-minute timeframe"""
    return compute_ichimoku_trend(df, timeframe=Timeframe.M15)


@FeatureRegistry.register("Ichimoku_Trend_H1")
def compute_ichimoku_trend_h1(df: pd.DataFrame) -> pd.Series:
    """Ichimoku Trend for 1-hour timeframe"""
    return compute_ichimoku_trend(df, timeframe=Timeframe.H1)


@FeatureRegistry.register("Ichimoku_Trend_H4")
def compute_ichimoku_trend_h4(df: pd.DataFrame) -> pd.Series:
    """Ichimoku Trend for 4-hour timeframe"""
    return compute_ichimoku_trend(df, timeframe=Timeframe.H4)


@FeatureRegistry.register("Ichimoku_Trend_D1")
def compute_ichimoku_trend_d1(df: pd.DataFrame) -> pd.Series:
    """Ichimoku Trend for daily timeframe"""
    return compute_ichimoku_trend(df, timeframe=Timeframe.D1)


@FeatureRegistry.register("Ichimoku_Cloud_Thickness_M1")
def compute_ichimoku_cloud_thickness_m1(df: pd.DataFrame) -> pd.Series:
    """Ichimoku Cloud Thickness for 1-minute timeframe (volatility measure)"""
    return compute_ichimoku_cloud_thickness(df, timeframe=Timeframe.M1)


@FeatureRegistry.register("Ichimoku_Cloud_Thickness_M5")
def compute_ichimoku_cloud_thickness_m5(df: pd.DataFrame) -> pd.Series:
    """Ichimoku Cloud Thickness for 5-minute timeframe (volatility measure)"""
    return compute_ichimoku_cloud_thickness(df, timeframe=Timeframe.M5)


@FeatureRegistry.register("Ichimoku_Cloud_Thickness_M15")
def compute_ichimoku_cloud_thickness_m15(df: pd.DataFrame) -> pd.Series:
    """Ichimoku Cloud Thickness for 15-minute timeframe (volatility measure)"""
    return compute_ichimoku_cloud_thickness(df, timeframe=Timeframe.M15)


@FeatureRegistry.register("Ichimoku_Cloud_Thickness_H1")
def compute_ichimoku_cloud_thickness_h1(df: pd.DataFrame) -> pd.Series:
    """Ichimoku Cloud Thickness for 1-hour timeframe (volatility measure)"""
    return compute_ichimoku_cloud_thickness(df, timeframe=Timeframe.H1)


@FeatureRegistry.register("Ichimoku_Cloud_Thickness_H4")
def compute_ichimoku_cloud_thickness_h4(df: pd.DataFrame) -> pd.Series:
    """Ichimoku Cloud Thickness for 4-hour timeframe (volatility measure)"""
    return compute_ichimoku_cloud_thickness(df, timeframe=Timeframe.H4)


@FeatureRegistry.register("Ichimoku_Cloud_Thickness_D1")
def compute_ichimoku_cloud_thickness_d1(df: pd.DataFrame) -> pd.Series:
    """Ichimoku Cloud Thickness for daily timeframe (volatility measure)"""
    return compute_ichimoku_cloud_thickness(df, timeframe=Timeframe.D1)


@FeatureRegistry.register("Ichimoku_Price_Cloud_Distance_M1")
def compute_ichimoku_price_cloud_distance_m1(df: pd.DataFrame) -> pd.Series:
    """Ichimoku Price-Cloud Distance for 1-minute timeframe"""
    return compute_ichimoku_price_cloud_distance(df, timeframe=Timeframe.M1)


@FeatureRegistry.register("Ichimoku_Price_Cloud_Distance_M5")
def compute_ichimoku_price_cloud_distance_m5(df: pd.DataFrame) -> pd.Series:
    """Ichimoku Price-Cloud Distance for 5-minute timeframe"""
    return compute_ichimoku_price_cloud_distance(df, timeframe=Timeframe.M5)


@FeatureRegistry.register("Ichimoku_Price_Cloud_Distance_M15")
def compute_ichimoku_price_cloud_distance_m15(df: pd.DataFrame) -> pd.Series:
    """Ichimoku Price-Cloud Distance for 15-minute timeframe"""
    return compute_ichimoku_price_cloud_distance(df, timeframe=Timeframe.M15)


@FeatureRegistry.register("Ichimoku_Price_Cloud_Distance_H1")
def compute_ichimoku_price_cloud_distance_h1(df: pd.DataFrame) -> pd.Series:
    """Ichimoku Price-Cloud Distance for 1-hour timeframe"""
    return compute_ichimoku_price_cloud_distance(df, timeframe=Timeframe.H1)


@FeatureRegistry.register("Ichimoku_Price_Cloud_Distance_H4")
def compute_ichimoku_price_cloud_distance_h4(df: pd.DataFrame) -> pd.Series:
    """Ichimoku Price-Cloud Distance for 4-hour timeframe"""
    return compute_ichimoku_price_cloud_distance(df, timeframe=Timeframe.H4)


@FeatureRegistry.register("Ichimoku_Price_Cloud_Distance_D1")
def compute_ichimoku_price_cloud_distance_d1(df: pd.DataFrame) -> pd.Series:
    """Ichimoku Price-Cloud Distance for daily timeframe"""
    return compute_ichimoku_price_cloud_distance(df, timeframe=Timeframe.D1)


def calculate_ichimoku_extended(
    df: pd.DataFrame,
    timeframe: Optional[Timeframe] = None,
    tenkan_period: Optional[int] = None,
    kijun_period: Optional[int] = None,
    senkou_span_b_period: Optional[int] = None,
) -> pd.DataFrame:
    """
    Calculate extended Ichimoku features including:
    - Traditional lines (Tenkan, Kijun, Senkou Span A, B)
    - Cloud thickness
    - Price-cloud center distance
    - Lagging span analysis

    Args:
        df: DataFrame with OHLC data
        timeframe: Timeframe enum (takes precedence over individual parameters)
        tenkan_period: Tenkan-sen period (default: 9 for daily)
        kijun_period: Kijun-sen period (default: 26 for daily)
        senkou_span_b_period: Senkou Span B period (default: 52 for daily)

    Returns:
        DataFrame with extended Ichimoku features
    """

    # Resolve parameters: timeframe takes precedence, then individual params, then defaults
    if timeframe is not None:
        params = get_ichimoku_params(timeframe)
        tenkan_period = params["tenkan"]
        kijun_period = params["kijun"]
        senkou_span_b_period = params["senkou_b"]
    else:
        # Use provided parameters or defaults
        tenkan_period = tenkan_period or 9
        kijun_period = kijun_period or 26
        senkou_span_b_period = senkou_span_b_period or 52

    result = pd.DataFrame(index=df.index)

    # Basic Ichimoku lines
    # Tenkan-sen (Conversion Line): (9-period high + 9-period low) / 2
    result["ichimoku_tenkan"] = (
        df["high"].rolling(tenkan_period).max() + df["low"].rolling(tenkan_period).min()
    ) / 2

    # Kijun-sen (Base Line): (26-period high + 26-period low) / 2
    result["ichimoku_kijun"] = (
        df["high"].rolling(kijun_period).max() + df["low"].rolling(kijun_period).min()
    ) / 2

    # Senkou Span A (Leading Span A): (Tenkan + Kijun) / 2, shifted 26 periods ahead
    senkou_span_a = (result["ichimoku_tenkan"] + result["ichimoku_kijun"]) / 2
    result["ichimoku_senkou_a"] = senkou_span_a.shift(kijun_period)

    # Senkou Span B (Leading Span B): (52-period high + 52-period low) / 2, shifted 26 periods ahead
    senkou_span_b = (
        df["high"].rolling(senkou_span_b_period).max()
        + df["low"].rolling(senkou_span_b_period).min()
    ) / 2
    result["ichimoku_senkou_b"] = senkou_span_b.shift(kijun_period)

    # Chikou Span (Lagging Span): Close shifted 26 periods back
    result["ichimoku_chikou"] = df["close"].shift(-kijun_period)

    # === Extended Features ===

    # 1. Cloud thickness (absolute difference between Span A and B)
    result["ichimoku_cloud_thickness"] = abs(
        result["ichimoku_senkou_a"] - result["ichimoku_senkou_b"]
    )

    # 2. Price-cloud center distance
    cloud_center = (result["ichimoku_senkou_a"] + result["ichimoku_senkou_b"]) / 2
    result["ichimoku_price_cloud_distance"] = df["close"] - cloud_center

    # 3. Price-cloud distance normalized by cloud thickness (avoid division by zero)
    result["ichimoku_price_cloud_normalized"] = np.where(
        result["ichimoku_cloud_thickness"] > 0,
        result["ichimoku_price_cloud_distance"] / result["ichimoku_cloud_thickness"],
        0,
    )

    # 4. Price position relative to cloud (above=1, inside=0, below=-1)
    result["ichimoku_price_position"] = np.where(
        df["close"]
        > np.maximum(result["ichimoku_senkou_a"], result["ichimoku_senkou_b"]),
        1,
        np.where(
            df["close"]
            < np.minimum(result["ichimoku_senkou_a"], result["ichimoku_senkou_b"]),
            -1,
            0,
        ),
    )

    # 5. Tenkan-Kijun cross signal
    result["ichimoku_tk_cross"] = np.where(
        result["ichimoku_tenkan"] > result["ichimoku_kijun"], 1, -1
    )

    # 6. Chikou span vs price comparison (lagging span confirmation)
    # Compare current chikou with price 26 periods ago
    price_26_ago = df["close"].shift(kijun_period)
    result["ichimoku_chikou_confirmation"] = np.where(
        result["ichimoku_chikou"] > price_26_ago,
        1,
        np.where(result["ichimoku_chikou"] < price_26_ago, -1, 0),
    )

    # 7. Cloud color (green=1 when Span A > Span B, red=-1 otherwise)
    result["ichimoku_cloud_color"] = np.where(
        result["ichimoku_senkou_a"] > result["ichimoku_senkou_b"], 1, -1
    )

    # 8. Distance ratios (normalized by close price to make scale-invariant)
    close_price = df["close"]
    result["ichimoku_tenkan_ratio"] = (
        result["ichimoku_tenkan"] - close_price
    ) / close_price
    result["ichimoku_kijun_ratio"] = (
        result["ichimoku_kijun"] - close_price
    ) / close_price

    # 9. Multi-timeframe confirmation score (simple version)
    # This combines multiple signals: TK cross + price position + chikou confirmation
    result["ichimoku_composite_signal"] = (
        result["ichimoku_tk_cross"]
        + result["ichimoku_price_position"]
        + result["ichimoku_chikou_confirmation"]
    ) / 3

    # 10. Trend determination based on Ichimoku signals
    # Bullish trend: price above cloud, tenkan > kijun, chikou above price
    bullish_signals = (
        (result["ichimoku_price_position"] == 1)  # Price above cloud
        & (result["ichimoku_tenkan"] > result["ichimoku_kijun"])  # Tenkan > Kijun
        & (result["ichimoku_chikou_confirmation"] == 1)  # Chikou bullish
    )

    # Bearish trend: price below cloud, tenkan < kijun, chikou below price
    bearish_signals = (
        (result["ichimoku_price_position"] == -1)  # Price below cloud
        & (result["ichimoku_tenkan"] < result["ichimoku_kijun"])  # Tenkan < Kijun
        & (result["ichimoku_chikou_confirmation"] == -1)  # Chikou bearish
    )

    result["ichimoku_trend"] = np.where(
        bullish_signals, 1, np.where(bearish_signals, -1, 0)
    )

    # Handle NaN values - forward fill for the first few rows where calculations aren't possible
    result = result.bfill(limit=max(tenkan_period, kijun_period, senkou_span_b_period))

    # Fill remaining NaN with method='ffill' (forward fill), or leave as NaN for analysis
    # result = result.fillna(0)
    result = result.ffill()

    return result


def ichimoku_feature_summary() -> Dict[str, str]:
    """
    Returns a dictionary summarizing each extended Ichimoku feature.

    The returned dictionary maps feature column names to their descriptions,
    providing an overview of the meaning and intended usage of each feature.
    This is useful for documentation, feature selection, and understanding
    the output of `calculate_ichimoku_extended`.

    Returns:
        Dict[str, str]: A mapping from feature names to their descriptions.
    """
    return {
        "ichimoku_tenkan": "Tenkan-sen (Conversion Line) - short-term trend",
        "ichimoku_kijun": "Kijun-sen (Base Line) - medium-term trend",
        "ichimoku_senkou_a": "Senkou Span A (Leading Span A) - fast cloud edge",
        "ichimoku_senkou_b": "Senkou Span B (Leading Span B) - slow cloud edge",
        "ichimoku_chikou": "Chikou Span (Lagging Span) - momentum confirmation",
        "ichimoku_cloud_thickness": "Absolute thickness of the cloud (volatility measure)",
        "ichimoku_price_cloud_distance": "Distance from price to cloud center",
        "ichimoku_price_cloud_normalized": "Price-cloud distance normalized by thickness",
        "ichimoku_price_position": "Price position relative to cloud (-1/0/1)",
        "ichimoku_tk_cross": "Tenkan-Kijun cross signal (-1/1)",
        "ichimoku_chikou_confirmation": "Lagging span momentum confirmation (-1/0/1)",
        "ichimoku_cloud_color": "Cloud color: green=1, red=-1",
        "ichimoku_tenkan_ratio": "Tenkan distance ratio to price",
        "ichimoku_kijun_ratio": "Kijun distance ratio to price",
        "ichimoku_composite_signal": "Composite signal combining multiple Ichimoku elements",
        "ichimoku_trend": "Trend determination based on Ichimoku signals (1=bullish, -1=bearish, 0=neutral)",
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

    # Calculate features
    features = calculate_ichimoku_extended(test_data)

    print("Ichimoku Extended Features:")
    print(features.head(10))
    print(f"\nFeature columns: {list(features.columns)}")
    print(f"NaN count per column:\n{features.isnull().sum()}")

    # Summary
    summary = ichimoku_feature_summary()
    print(f"\nFeature summary:\n{summary}")
