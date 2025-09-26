"""
CCI (Commodity Channel Index) implementation.
CCIの実装
"""

import pandas as pd
from ztb.features.registry import FeatureRegistry
from ztb.features.utils.rolling import rolling_mean


@FeatureRegistry.register("CCI")
def compute_cci(df: pd.DataFrame, period: int = 20) -> pd.Series:
    """Compute CCI (Commodity Channel Index)"""
    # Calculate rolling mean if not present
    if f'rolling_mean_{period}' not in df.columns:
        rolling_mean_val = rolling_mean(df['close'], period)
    else:
        rolling_mean_val = df[f'rolling_mean_{period}']

    # Mean deviation
    mean_dev = (df['close'] - rolling_mean_val).abs().rolling(period).mean()
    cci = (df['close'] - rolling_mean_val) / (0.015 * mean_dev)
    return cci.fillna(0)