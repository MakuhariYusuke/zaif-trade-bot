"""
Price-Volume Correlation implementation.
価格と出来高の相関係数を計算
"""

import pandas as pd
from ztb.features.registry import FeatureRegistry


@FeatureRegistry.register("PriceVolumeCorr")
def compute_price_volume_corr(df: pd.DataFrame, period: int = 20) -> pd.Series:
    """Compute rolling correlation between price and volume"""
    price = df['close']
    volume = df['volume']
    corr = price.rolling(window=period).corr(volume)
    return corr