"""
RSI (Relative Strength Index) implementation.
RSIの実装
"""

import pandas as pd
import numpy as np
from ztb.features.registry import FeatureRegistry
import hashlib


# Simple cache for DataFrame-based computations
_cache = {}


def _get_df_hash(df: pd.DataFrame, period: int) -> str:
    """Generate hash for DataFrame + parameters"""
    # Use close prices and period for hash
    close_values = df['close'].astype(float).values
    data_str = f"{period}_{close_values.tobytes()}"
    return hashlib.md5(data_str.encode()).hexdigest()


@FeatureRegistry.register("RSI")
def compute_rsi(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """Compute RSI (Relative Strength Index) - Optimized numpy version"""
    if not FeatureRegistry.is_cache_enabled():
        return _compute_rsi_numpy(df, period)
    
    cache_key = f"rsi_{_get_df_hash(df, period)}"
    
    if cache_key in _cache:
        return _cache[cache_key].copy()
    
    result = _compute_rsi_numpy(df, period)
    _cache[cache_key] = result.copy()
    return result


def _compute_rsi_numpy(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """Optimized RSI computation using numpy + pandas EWM"""
    # Calculate returns if not present
    if 'return' not in df.columns:
        returns = df['close'].pct_change().fillna(0)
    else:
        returns = df['return'].fillna(0)
    
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