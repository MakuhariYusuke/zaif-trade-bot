"""
ATR (Average True Range) implementation.
平均真の範囲 - ボラティリティ指標
"""

import numpy as np
import pandas as pd

from ztb.features.registry import FeatureRegistry


@FeatureRegistry.register("ATR")
def compute_atr(df: pd.DataFrame, period: int = 14) -> "pd.Series":
    """Compute Average True Range (ATR)"""
    tr = np.maximum(
        df["high"] - df["low"],
        np.maximum(
            (df["high"] - df["close"].shift(1)).abs(),
            (df["low"] - df["close"].shift(1)).abs(),
        ),
    )
    atr = pd.Series(tr).rolling(period).mean()
    return atr  # 初期値はNaNのまま返す


@FeatureRegistry.register("ATR_simplified")
def compute_atr_simplified(df: pd.DataFrame, period: int = 10) -> "pd.Series":
    """Compute Simplified ATR (period=10)"""
    return compute_atr(df, period)
