from typing import Any, Optional

import numpy as np
import pandas as pd

from ..base import BaseFeature
from ..registry import FeatureRegistry
from ..timeframe import Timeframe


@FeatureRegistry.register("HeikinAshi_Color")
def compute_heikin_ashi_color(
    df: pd.DataFrame, timeframe: Optional[Timeframe] = None
) -> pd.Series:
    """Heikin-Ashi Color: 1 (bullish), -1 (bearish), 0 (doji)"""
    feature = HeikinAshi()
    result_df = feature.compute(df, timeframe=timeframe)
    ha_open = result_df["ha_open"]
    ha_close = result_df["ha_close"]

    # Calculate color: 1 for bullish (close > open), -1 for bearish (close < open), 0 for doji (close == open)
    color = np.where(ha_close > ha_open, 1, np.where(ha_close < ha_open, -1, 0))
    return pd.Series(color, index=df.index)


class HeikinAshi(BaseFeature):
    """
    平均足 (Heikin-Ashi) feature implementation.
    Generates smoothed OHLC values to capture trend strength.

    Output columns:
      - ha_open
      - ha_high
      - ha_low
      - ha_close
    """

    def __init__(self, **kwargs: Any):
        super().__init__(name="HeikinAshi", deps=["open", "high", "low", "close"])

    def compute(
        self, df: pd.DataFrame, timeframe: Optional[Timeframe] = None, **params: Any
    ) -> pd.DataFrame:
        """
        Args:
            df (pd.DataFrame): Input DataFrame. Must include columns:
                - 'open' (float): Open price
                - 'high' (float): High price
                - 'low' (float): Low price
                - 'close' (float): Close price

        Example:
            pd.DataFrame({
                'open': [100.0, 101.0, ...],
                'high': [102.0, 103.0, ...],
                'low': [99.0, 100.5, ...],
                'close': [101.5, 102.0, ...]
            })

        Returns:
            pd.DataFrame: DataFrame with Heikin-Ashi OHLC columns:
                - ha_open
                - ha_high
                - ha_low
                - ha_close
        """
        ha_df = pd.DataFrame(index=df.index)

        # Heikin-Ashi close
        ha_df["ha_close"] = (df["open"] + df["high"] + df["low"] + df["close"]) / 4.0

        # Heikin-Ashi open
        ha_open = np.empty(len(df))
        if len(df) > 0:
            ha_open[0] = (df["open"].iloc[0] + df["close"].iloc[0]) / 2.0
            ha_close = ha_df["ha_close"].to_numpy()
            for i in range(1, len(df)):
                ha_open[i] = (ha_open[i - 1] + ha_close[i - 1]) / 2.0
            ha_df["ha_open"] = ha_open
        else:
            ha_open = np.empty(0)
            ha_df["ha_open"] = ha_open
        ha_df["ha_high"] = np.maximum.reduce(
            [
                df["high"].to_numpy(),
                ha_df["ha_open"].to_numpy(),
                ha_df["ha_close"].to_numpy(),
            ]
        )
        ha_df["ha_low"] = np.minimum.reduce(
            [
                df["low"].to_numpy(),
                ha_df["ha_open"].to_numpy(),
                ha_df["ha_close"].to_numpy(),
            ]
        )

        return ha_df


# === Multi-Timeframe Heikin-Ashi Features ===


@FeatureRegistry.register("HeikinAshi_Color_M1")
def compute_heikin_ashi_color_m1(df: pd.DataFrame) -> pd.Series:
    """Heikin-Ashi Color for 1-minute timeframe"""
    return compute_heikin_ashi_color(df, timeframe=Timeframe.M1)


@FeatureRegistry.register("HeikinAshi_Color_M5")
def compute_heikin_ashi_color_m5(df: pd.DataFrame) -> pd.Series:
    """Heikin-Ashi Color for 5-minute timeframe"""
    return compute_heikin_ashi_color(df, timeframe=Timeframe.M5)


@FeatureRegistry.register("HeikinAshi_Color_M15")
def compute_heikin_ashi_color_m15(df: pd.DataFrame) -> pd.Series:
    """Heikin-Ashi Color for 15-minute timeframe"""
    return compute_heikin_ashi_color(df, timeframe=Timeframe.M15)


@FeatureRegistry.register("HeikinAshi_Color_H1")
def compute_heikin_ashi_color_h1(df: pd.DataFrame) -> pd.Series:
    """Heikin-Ashi Color for 1-hour timeframe"""
    return compute_heikin_ashi_color(df, timeframe=Timeframe.H1)


@FeatureRegistry.register("HeikinAshi_Color_H4")
def compute_heikin_ashi_color_h4(df: pd.DataFrame) -> pd.Series:
    """Heikin-Ashi Color for 4-hour timeframe"""
    return compute_heikin_ashi_color(df, timeframe=Timeframe.H4)


@FeatureRegistry.register("HeikinAshi_Color_D1")
def compute_heikin_ashi_color_d1(df: pd.DataFrame) -> pd.Series:
    """Heikin-Ashi Color for daily timeframe"""
    return compute_heikin_ashi_color(df, timeframe=Timeframe.D1)
