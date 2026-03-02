"""
Bollinger Bands implementation.
ボリンジャーバンド - ボラティリティ指標
"""

import numpy as np
import pandas as pd

from ztb.features.core.registry import FeatureRegistry
from ztb.utils.talib_wrapper import TaLibWrapper

@FeatureRegistry.register("BB_Upper")
def compute_bb_upper(
    df: pd.DataFrame, period: int = 20, std_dev: float = 2.0
) -> pd.Series:
    """Compute Bollinger Band Upper with Ta-Lib support"""
    close_prices = np.asarray(df["close"].values, dtype=float)
    upper, middle, lower = TaLibWrapper.bbands(close_prices, period, std_dev, std_dev)
    return pd.Series(upper, index=df.index).bfill()

@FeatureRegistry.register("BB_Lower")
def compute_bb_lower(
    df: pd.DataFrame, period: int = 20, std_dev: float = 2.0
) -> pd.Series:
    """Compute Bollinger Band Lower with Ta-Lib support"""
    close_prices = np.asarray(df["close"].values, dtype=float)
    upper, middle, lower = TaLibWrapper.bbands(close_prices, period, std_dev, std_dev)
    return pd.Series(lower, index=df.index).bfill()

@FeatureRegistry.register("BB_Middle")
def compute_bb_middle(df: pd.DataFrame, period: int = 20) -> pd.Series:
    """Compute Bollinger Band Middle (SMA) with Ta-Lib support"""
    close_prices = np.asarray(df["close"].values, dtype=float)
    upper, middle, lower = TaLibWrapper.bbands(close_prices, period, 2.0, 2.0)
    return pd.Series(middle, index=df.index).bfill()

@FeatureRegistry.register("BB_Width")
def compute_bb_width(
    df: pd.DataFrame, period: int = 20, std_dev: float = 2.0
) -> pd.Series:
    """Compute Bollinger Band Width with Ta-Lib support"""
    close_prices = np.asarray(df["close"].values, dtype=float)
    upper, middle, lower = TaLibWrapper.bbands(close_prices, period, std_dev, std_dev)

    # Width = (upper - lower) / middle
    width = (upper - lower) / np.where(middle == 0, 1, middle)
    return pd.Series(width, index=df.index, dtype=float).fillna(0.0)

@FeatureRegistry.register("BB_Position")
def compute_bb_position(
    df: pd.DataFrame, period: int = 20, std_dev: float = 2.0
) -> pd.Series:
    """Compute Bollinger Band Position (%B) with Ta-Lib support"""
    close_prices = np.asarray(df["close"].values, dtype=float)
    upper, middle, lower = TaLibWrapper.bbands(close_prices, period, std_dev, std_dev)

    # Position = (close - lower) / (upper - lower)
    denominator = upper - lower
    position = (close_prices - lower) / np.where(denominator == 0, 1, denominator)
    return pd.Series(position, index=df.index, dtype=float).fillna(0.0)
