"""
Bollinger Bands implementation.
ボリンジャーバンド - ボラティリティ指標
"""

import numpy as np
import pandas as pd

from ztb.features.registry import FeatureRegistry


@FeatureRegistry.register("BB_Upper")
def compute_bb_upper(
    df: pd.DataFrame, period: int = 20, std_dev: float = 2.0
) -> pd.Series:
    """Compute Bollinger Band Upper"""
    mean = df["close"].rolling(period).mean()
    std = df["close"].rolling(period).std()
    return mean + std_dev * std


@FeatureRegistry.register("BB_Lower")
def compute_bb_lower(
    df: pd.DataFrame, period: int = 20, std_dev: float = 2.0
) -> pd.Series:
    """Compute Bollinger Band Lower"""
    mean = df["close"].rolling(period).mean()
    std = df["close"].rolling(period).std()
    return mean - std_dev * std


@FeatureRegistry.register("BB_Middle")
def compute_bb_middle(df: pd.DataFrame, period: int = 20) -> pd.Series:
    """Compute Bollinger Band Middle (SMA)"""
    return df["close"].rolling(period).mean()


@FeatureRegistry.register("BB_Width")
def compute_bb_width(
    df: pd.DataFrame, period: int = 20, std_dev: float = 2.0
) -> pd.Series:
    """Compute Bollinger Band Width - Optimized version"""
    # Calculate mean and std once, reuse for efficiency
    close_rolling = df["close"].rolling(period)
    mean = close_rolling.mean()
    std = close_rolling.std()

    # Width = (upper - lower) / mean = (4 * std_dev * std) / mean
    # Simplified calculation: width = 4 * std_dev * (std / mean)
    width = 4 * std_dev * (std / mean.where(mean != 0, np.nan))
    return width.fillna(0)


@FeatureRegistry.register("BB_Position")
def compute_bb_position(
    df: pd.DataFrame, period: int = 20, std_dev: float = 2.0
) -> pd.Series:
    """Compute Bollinger Band Position (%B)"""
    mean = df["close"].rolling(period).mean()
    std = df["close"].rolling(period).std()
    upper = mean + std_dev * std
    lower = mean - std_dev * std
    denominator = upper - lower
    position = (df["close"] - lower) / denominator
    # If the band width is zero, treat the position as the middle (0.5)
    return position.where(denominator != 0, 0.5)
