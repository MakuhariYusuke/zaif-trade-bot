#!/usr/bin/env python3
"""
donchian_ext.py
Extended Donchian Channel analysis with breakout distance and channel width
"""

import numpy as np
import pandas as pd

from ztb.features.registry import FeatureRegistry


@FeatureRegistry.register("Donchian_Width")
def compute_donchian_width(df: pd.DataFrame) -> pd.Series:
    """Donchian Channel Width"""
    extended_features = calculate_donchian_extended(df)
    return (
        extended_features["donchian_width"]
        if "donchian_width" in extended_features.columns
        else pd.Series([0.0] * len(df), index=df.index)
    )


@FeatureRegistry.register("Donchian_Price_Position")
def compute_donchian_price_position(df: pd.DataFrame) -> pd.Series:
    """Donchian Channel Price Position (0-1)"""
    extended_features = calculate_donchian_extended(df)
    return (
        extended_features["donchian_price_position"]
        if "donchian_price_position" in extended_features.columns
        else pd.Series([0.0] * len(df), index=df.index)
    )


@FeatureRegistry.register("Donchian_Breakout_Strength")
def compute_donchian_breakout_strength(df: pd.DataFrame) -> pd.Series:
    """Donchian Channel Breakout Strength"""
    extended_features = calculate_donchian_extended(df)
    return (
        extended_features["donchian_breakout_strength"]
        if "donchian_breakout_strength" in extended_features.columns
        else pd.Series([0.0] * len(df), index=df.index)
    )


@FeatureRegistry.register("Donchian_Squeeze_Ratio")
def compute_donchian_squeeze_ratio(df: pd.DataFrame) -> pd.Series:
    """Donchian Channel Squeeze Ratio"""
    extended_features = calculate_donchian_extended(df)
    return (
        extended_features["donchian_squeeze_ratio"]
        if "donchian_squeeze_ratio" in extended_features.columns
        else pd.Series([0.0] * len(df), index=df.index)
    )


def calculate_donchian_extended(
    data: pd.DataFrame, periods: int = 20, include_basic: bool = True
) -> pd.DataFrame:
    """
    Calculate extended Donchian Channel features.

    Args:
        data: DataFrame with OHLCV data
        periods: Lookback period for channel calculation
        include_basic: Whether to include basic upper/lower lines

    Returns:
        DataFrame with extended Donchian features.
        NaN values are intentionally left as-is for initial periods and cases where calculation is not possible.
        This preserves the indicator's meaning and avoids introducing bias from imputation.
        If you need to fill NaN values, do so explicitly for specific columns as needed.
    """
    if len(data) < periods:
        return pd.DataFrame(index=data.index)

    # Basic Donchian Channel
    high_roll = data["high"].rolling(window=periods)
    low_roll = data["low"].rolling(window=periods)

    upper_channel = high_roll.max()
    lower_channel = low_roll.min()
    middle_channel = (upper_channel + lower_channel) / 2

    # Extended features
    features = {}

    if include_basic:
        features["donchian_upper"] = upper_channel
        features["donchian_lower"] = lower_channel
        features["donchian_middle"] = middle_channel

    # Channel width analysis
    channel_width = upper_channel - lower_channel
    features["donchian_width"] = channel_width
    features["donchian_width_pct"] = channel_width / data["close"]

    # Channel width percentile (relative to historical width)
    # Reduce window size for performance, or compute percentile less frequently
    reduced_window = max(periods, 30)  # Use a smaller window, e.g., max(periods, 30)
    width_percentile = channel_width.rolling(window=reduced_window).rank(pct=True)
    features["donchian_width_percentile"] = width_percentile

    # Price position within channel
    price_position = (data["close"] - lower_channel) / channel_width
    features["donchian_price_position"] = price_position

    # Distance from price to channel boundaries
    features["donchian_upper_distance"] = (upper_channel - data["close"]) / data[
        "close"
    ]
    features["donchian_lower_distance"] = (data["close"] - lower_channel) / data[
        "close"
    ]
    features["donchian_middle_distance"] = (data["close"] - middle_channel) / data[
        "close"
    ]

    # Breakout signals
    features["donchian_upper_breakout"] = (
        data["close"] > upper_channel.shift(1)
    ).astype(int)
    features["donchian_lower_breakout"] = (
        data["close"] < lower_channel.shift(1)
    ).astype(int)
    breakout_strength_upper = (
        (data["close"] - upper_channel.shift(1)) / upper_channel.shift(1)
    ).where(data["close"] > upper_channel.shift(1), 0)
    breakout_strength_lower = (
        (lower_channel.shift(1) - data["close"]) / lower_channel.shift(1)
    ).where(data["close"] < lower_channel.shift(1), 0)

    features["donchian_breakout_strength_upper"] = breakout_strength_upper
    features["donchian_breakout_strength_lower"] = breakout_strength_lower
    features["donchian_breakout_strength"] = (
        breakout_strength_upper + breakout_strength_lower
    )

    # Channel squeeze detection (narrow width compared to historical)
    width_ma = channel_width.rolling(window=periods).mean()
    squeeze_ratio = channel_width / width_ma
    features["donchian_squeeze_ratio"] = squeeze_ratio
    features["donchian_squeeze_signal"] = (squeeze_ratio < 0.8).astype(int)

    # Channel expansion detection
    width_change = channel_width.pct_change(periods=5)
    features["donchian_expansion_rate"] = width_change
    features["donchian_expansion_signal"] = (width_change > 0.1).astype(int)

    # Price momentum relative to channel
    price_momentum = data["close"].pct_change(periods=5)
    channel_momentum = middle_channel.pct_change(periods=5)
    features["donchian_relative_momentum"] = price_momentum - channel_momentum

    # Channel slope (trend direction)
    upper_slope = upper_channel.diff(periods=5) / upper_channel.shift(5)
    lower_slope = lower_channel.diff(periods=5) / lower_channel.shift(5)
    middle_slope = middle_channel.diff(periods=5) / middle_channel.shift(5)

    features["donchian_upper_slope"] = upper_slope
    features["donchian_lower_slope"] = lower_slope
    features["donchian_middle_slope"] = middle_slope

    # Channel trend consistency
    slope_consistency = np.sign(upper_slope) == np.sign(lower_slope)
    features["donchian_trend_consistency"] = slope_consistency.astype(int)

    # Volatility-adjusted features
    volatility = data["close"].rolling(window=periods).std()
    features["donchian_width_vol_adj"] = channel_width / volatility
    volatility = data["close"].rolling(window=periods).std()
    volatility_safe = volatility.replace(0, 1e-6)
    features["donchian_width_vol_adj"] = channel_width / volatility_safe
    features["donchian_breakout_vol_adj"] = (
        features["donchian_breakout_strength"] / volatility_safe
    )
    # Short-term position within long-term channel
    if len(data) >= periods * 2:
        long_upper = data["high"].rolling(window=periods * 2).max()
        long_lower = data["low"].rolling(window=periods * 2).min()
        long_position = (data["close"] - long_lower) / (long_upper - long_lower)
        features["donchian_long_position"] = long_position

        # Divergence between short and long term positions
        features["donchian_position_divergence"] = price_position - long_position

    # Create result DataFrame
    result_df = pd.DataFrame(features, index=data.index)

    # Leave NaN values as-is to preserve indicator meaning (especially for initial periods)
    # If you want to fill only specific columns, do so explicitly, e.g.:
    # result_df['donchian_price_position'] = result_df['donchian_price_position'].ffill().bfill()

    return result_df


def calculate_donchian_signals(
    extended_features: pd.DataFrame, confidence_threshold: float = 0.7
) -> pd.DataFrame:
    """
    Generate trading signals from extended Donchian features

    Args:
        extended_features: Output from calculate_donchian_extended
        confidence_threshold: Minimum confidence for signal generation

    Returns:
        DataFrame with trading signals
    """
    signals = {}

    # Basic breakout signals
    signals["donchian_bullish_breakout"] = (
        (extended_features["donchian_upper_breakout"] == 1)
        & (extended_features["donchian_breakout_strength_upper"] > 0.01)
    ).astype(int)

    signals["donchian_bearish_breakout"] = (
        (extended_features["donchian_lower_breakout"] == 1)
        & (extended_features["donchian_breakout_strength_lower"] > 0.01)
    ).astype(int)

    # Squeeze and expansion combination
    signals["donchian_squeeze_expansion"] = (
        (extended_features["donchian_squeeze_signal"].shift(1) == 1)
        & (extended_features["donchian_expansion_signal"] == 1)
    ).astype(int)

    # Trend-following signals
    trend_up = (
        (extended_features["donchian_middle_slope"] > 0)
        & (extended_features["donchian_trend_consistency"] == 1)
        & (extended_features["donchian_price_position"] > 0.5)
    )

    trend_down = (
        (extended_features["donchian_middle_slope"] < 0)
        & (extended_features["donchian_trend_consistency"] == 1)
        & (extended_features["donchian_price_position"] < 0.5)
    )

    signals["donchian_trend_bullish"] = trend_up.astype(int)
    signals["donchian_trend_bearish"] = trend_down.astype(int)

    # High-confidence composite signal
    composite_bullish = (
        (extended_features["donchian_upper_breakout"] == 1)
        & (extended_features["donchian_breakout_strength_upper"] > 0.005)
        & (extended_features["donchian_width_percentile"] > 0.3)
        & (extended_features["donchian_relative_momentum"] > 0)
    )

    composite_bearish = (
        (extended_features["donchian_lower_breakout"] == 1)
        & (extended_features["donchian_breakout_strength_lower"] > 0.005)
        & (extended_features["donchian_width_percentile"] > 0.3)
        & (extended_features["donchian_relative_momentum"] < 0)
    )

    signals["donchian_composite_bullish"] = composite_bullish.astype(int)
    signals["donchian_composite_bearish"] = composite_bearish.astype(int)

    return pd.DataFrame(signals, index=extended_features.index)


if __name__ == "__main__":
    # Test with synthetic data

    # Generate synthetic OHLCV data
    np.random.seed(42)
    dates = pd.date_range("2023-01-01", periods=500, freq="D")

    # Create trending price series with volatility
    base_price = 100
    trend = np.cumsum(np.random.normal(0.001, 0.02, 500))
    noise = np.random.normal(0, 0.01, 500)
    close_prices = base_price * np.exp(trend + noise)

    # Generate OHLC from close prices
    high_mult = 1 + np.abs(np.random.normal(0, 0.005, 500))
    low_mult = 1 - np.abs(np.random.normal(0, 0.005, 500))

    data = pd.DataFrame(
        {
            "open": close_prices * np.random.uniform(0.995, 1.005, 500),
            "high": close_prices * high_mult,
            "low": close_prices * low_mult,
            "close": close_prices,
            "volume": np.random.randint(1000, 10000, 500),
        },
        index=dates,
    )

    print("Testing Donchian Extended Features...")
    print(f"Data shape: {data.shape}")
    print(f"Date range: {data.index[0]} to {data.index[-1]}")

    # Calculate extended features
    extended_features = calculate_donchian_extended(data, periods=20)

    print(f"\nExtended features shape: {extended_features.shape}")
    print(f"Feature columns ({len(extended_features.columns)}):")
    for i, col in enumerate(extended_features.columns):
        print(f"  {i + 1:2d}. {col}")

    # Calculate signals
    signals = calculate_donchian_signals(extended_features)

    print(f"\nSignals shape: {signals.shape}")
    print("Signal columns:")
    for col in signals.columns:
        signal_count = signals[col].sum()
        print(f"  {col}: {signal_count} signals")

    # Basic statistics
    print(f"\nBasic feature statistics:")
    key_features = [
        "donchian_width",
        "donchian_price_position",
        "donchian_breakout_strength",
    ]
    for feature in key_features:
        if feature in extended_features.columns:
            series = extended_features[feature].dropna()
            print(f"  {feature}:")
            print(f"    Mean: {series.mean():.4f}")
            print(f"    Std:  {series.std():.4f}")
            print(f"    Range: [{series.min():.4f}, {series.max():.4f}]")

    print("Donchian extended features test completed successfully!")
