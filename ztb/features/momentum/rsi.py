"""
RSI (Relative Strength Index) implementation.
RSIの実装
"""

import numpy as np
import pandas as pd

from ztb.features.feature_cache import feature_cache
from ztb.features.registry import FeatureRegistry


@FeatureRegistry.register("RSI")
def compute_rsi(df: pd.DataFrame, period: int = 14) -> pd.Series[float]:
    """Compute RSI (Relative Strength Index) - Optimized numpy version"""
    if not FeatureRegistry.is_cache_enabled():
        return _compute_rsi_numpy(df, period)

    cache_key = f"rsi_{feature_cache.generate_dataframe_hash(df, ['close'], {'period': period})}"

    def compute() -> pd.Series:
        return _compute_rsi_numpy(df, period)

    return feature_cache.get_or_compute(cache_key, compute)


def _compute_rsi_numpy(df: pd.DataFrame, period: int = 14) -> pd.Series[float]:
    """Optimized RSI computation using numpy + pandas EWM"""
    # Calculate returns if not present
    if "return" not in df.columns:
        returns = df["close"].pct_change().fillna(0)
    else:
        returns = df["return"].fillna(0)

    # Use numpy for gain/loss calculation
    gains = np.where(returns > 0, returns, 0)
    losses = np.where(returns < 0, -returns, 0)

    # Use pandas EWM for exponential smoothing (faster than rolling mean)
    # RSI traditionally uses SMA, but EWM is often used for efficiency
    gain_series = pd.Series(gains, index=df.index)
    loss_series = pd.Series(losses, index=df.index)

    # Use span for exponential smoothing (approximates SMA for efficiency)
    gain_avg = gain_series.ewm(span=period, adjust=False).mean()
    loss_avg = loss_series.ewm(span=period, adjust=False).mean()

    rs = gain_avg / loss_avg.replace(0, np.inf)  # Avoid division by zero
    rsi = 100 - (100 / (1 + rs))

    return rsi.fillna(50)
