"""
MFI (Money Flow Index) implementation.
MFIの実装
"""

import pandas as pd
from ztb.features.registry import FeatureRegistry


@FeatureRegistry.register("MFI")
def compute_mfi(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """Compute MFI (Money Flow Index)"""
    # Typical Price
    typical_price = (df['high'] + df['low'] + df['close']) / 3

    # Raw Money Flow
    money_flow = typical_price * df['volume']

    # Positive and Negative Money Flow
    price_diff: pd.Series = typical_price.diff()
    positive_flow = money_flow.where(price_diff > 0, 0.0)
    negative_flow = money_flow.where(price_diff < 0, 0.0)

    # Money Flow Ratio
    pos_mf_sum = positive_flow.rolling(period).sum()
    neg_mf_sum = negative_flow.rolling(period).sum()

    money_flow_ratio = pos_mf_sum / neg_mf_sum.replace(0, 1e-8)  # Avoid division by zero

    # MFI calculation
    mfi = 100 - (100 / (1 + money_flow_ratio))

    return mfi.fillna(50)