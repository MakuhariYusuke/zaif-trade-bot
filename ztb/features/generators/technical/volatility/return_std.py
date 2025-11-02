"""
Return Standard Deviation implementation.
リターンの標準偏差 - ボラティリティ指標
"""

import pandas as pd

from ztb.features.registry import FeatureRegistry


@FeatureRegistry.register("ReturnStdDev")
def compute_return_stddev(df: pd.DataFrame, period: int = 20) -> pd.Series:
    """Compute standard deviation of returns"""
    returns = df["close"].pct_change()
    return returns.rolling(window=period).std()
