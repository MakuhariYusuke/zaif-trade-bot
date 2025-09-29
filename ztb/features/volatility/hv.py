"""
Historical Volatility (HV) implementation.
過去のボラティリティを測定
"""

import numpy as np
import pandas as pd
from typing import cast

# 年間取引日数（一般的に252日）
TRADING_DAYS_PER_YEAR = 252

from ztb.features.registry import FeatureRegistry


@FeatureRegistry.register("HV")
def compute_hv(df: pd.DataFrame, period: int = 14) -> "pd.Series[float]":
    """
    log_returns = np.log(df['close'] / df['close'].shift(1))
    hv = log_returns.rolling(window=period).std() * np.sqrt(252)  # Annualized volatility
    return cast("pd.Series[float]", hv)
        df: DataFrame containing at least a 'close' column.
        period: Rolling window size in days for volatility calculation.

    Returns:
        pd.Series[float]: Annualized historical volatility.
    """
    log_returns = np.log(df['close'] / df['close'].shift(1))
    hv = pd.Series(log_returns).rolling(window=period).std() * np.sqrt(TRADING_DAYS_PER_YEAR)  # 年換算ボラティリティ
    return cast("pd.Series[float]", hv)