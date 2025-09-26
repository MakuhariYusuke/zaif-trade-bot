"""
ATR (Average True Range) implementation.
平均真の範囲 - ボラティリティ指標
"""

import numpy as np
import pandas as pd
from ztb.features.registry import FeatureRegistry


@FeatureRegistry.register("ATR")
def compute_atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """Compute Average True Range (ATR)"""
    tr = np.maximum(
        df['high'] - df['low'],
        np.maximum(
            (df['high'] - df['close'].shift(1)).abs(),
            (df['low'] - df['close'].shift(1)).abs()
        )
    )
    atr = pd.Series(tr).rolling(period).mean()
    return atr.fillna(0)