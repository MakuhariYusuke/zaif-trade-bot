"""
VWAP (Volume Weighted Average Price) implementation.
VWAPの実装
"""

import pandas as pd

from ztb.features.registry import FeatureRegistry


@FeatureRegistry.register("VWAP")
def compute_vwap(df: pd.DataFrame) -> pd.Series:
    """Compute VWAP (Volume Weighted Average Price)"""
    typical_price = (df["high"] + df["low"] + df["close"]) / 3
    cum_vol_price = (typical_price * df["volume"]).cumsum()
    cum_volume = df["volume"].cumsum()
    vwap = cum_vol_price / cum_volume.replace(0, 1e-8)  # Avoid division by zero
    return vwap.fillna(df["close"])  # Fill with close price if no volume
