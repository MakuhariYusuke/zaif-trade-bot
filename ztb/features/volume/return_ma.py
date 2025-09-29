"""
Moving Averages of Returns implementation.
リターンの移動平均を計算
"""

import pandas as pd

from ztb.features.registry import FeatureRegistry


@FeatureRegistry.register("ReturnMA_Short")
def compute_return_ma_short(df: pd.DataFrame, period: int = 5) -> pd.Series:
    """Compute short-term moving average of returns"""
    returns = df["close"].pct_change()
    return returns.rolling(window=period).mean()


@FeatureRegistry.register("ReturnMA_Medium")
def compute_return_ma_medium(df: pd.DataFrame, period: int = 20) -> pd.Series:
    """Compute medium-term moving average of returns"""
    returns = df["close"].pct_change()
    return returns.rolling(window=period).mean()
