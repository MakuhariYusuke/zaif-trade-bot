"""
MACD (Moving Average Convergence Divergence) implementation.
MACDの実装
"""

import pandas as pd

from ztb.features.registry import FeatureRegistry


@FeatureRegistry.register("MACD")
def compute_macd(
    df: pd.DataFrame,
    fast_period: int = 12,
    slow_period: int = 26,
    signal_period: int = 9,
) -> pd.Series:
    """Compute MACD (Moving Average Convergence Divergence) - Optimized version"""
    # Use pre-computed EMAs if available, otherwise compute them
    fast_col = f"ema_{fast_period}"
    slow_col = f"ema_{slow_period}"

    if fast_col in df.columns:
        ema_fast = df[fast_col]
    else:
        ema_fast = df["close"].ewm(span=fast_period, adjust=False).mean()

    if slow_col in df.columns:
        ema_slow = df[slow_col]
    else:
        ema_slow = df["close"].ewm(span=slow_period, adjust=False).mean()

    # MACD line
    macd = ema_fast - ema_slow

    # Signal line (EMA of MACD)
    signal_col = f"ema_{signal_period}"
    if signal_col in df.columns:
        signal = df[signal_col]
    else:
        signal = macd.ewm(span=signal_period, adjust=False).mean()

    # Return MACD histogram (MACD - Signal) as it's the most useful component
    macd_hist = macd - signal
    return macd_hist.fillna(0)
