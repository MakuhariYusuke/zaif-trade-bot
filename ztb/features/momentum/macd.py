"""
MACD (Moving Average Convergence Divergence) implementation.
MACDの実装
"""

import pandas as pd
from ztb.features.registry import FeatureRegistry


@FeatureRegistry.register("MACD")
def compute_macd(df: pd.DataFrame, fast_period: int = 12, slow_period: int = 26, signal_period: int = 9) -> pd.Series:
    """Compute MACD (Moving Average Convergence Divergence) - Optimized version"""
    # Calculate EMAs efficiently, sharing intermediate results
    close_series = df['close']
    
    # Calculate fast EMA
    if f'ema_{fast_period}' not in df.columns:
        ema_fast = close_series.ewm(span=fast_period, adjust=False).mean()
    else:
        ema_fast = df[f'ema_{fast_period}']
    
    # Calculate slow EMA
    if f'ema_{slow_period}' not in df.columns:
        ema_slow = close_series.ewm(span=slow_period, adjust=False).mean()
    else:
        ema_slow = df[f'ema_{slow_period}']
    
    # MACD line
    macd = ema_fast - ema_slow
    
    # Signal line (EMA of MACD)
    if f'ema_{signal_period}' not in df.columns:
        signal = macd.ewm(span=signal_period, adjust=False).mean()
    else:
        signal = df[f'ema_{signal_period}']
    
    # Return MACD histogram (MACD - Signal) as it's the most useful component
    macd_hist = macd - signal
    return macd_hist.fillna(0)