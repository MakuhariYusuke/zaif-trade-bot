"""
Historical Volatility (HV) implementation.
過去のボラティリティを測定
"""

from typing import cast

import numpy as np
import pandas as pd

from ztb.features.registry import FeatureRegistry

# 年間取引日数（一般的に252日）


@FeatureRegistry.register("HV")
def compute_hv(df: pd.DataFrame, period: int = 14) -> "pd.Series":
    """
    log_returns = np.log(df['close'] / df['close'].shift(1))
    hv = log_returns.rolling(window=period).std() * np.sqrt(252)  # Annualized volatility
    return cast("pd.Series", hv)
        df: DataFrame containing at least a 'close' column.
        period: Rolling window size in days for volatility calculation.

    Returns:
        pd.Series: Annualized historical volatility.
    """
    log_returns = np.log(df["close"] / df["close"].shift(1))
    from ztb.metrics.technical import calculate_rolling_volatility

    hv = calculate_rolling_volatility(
        log_returns, window=period, annualize=True
    )  # 年換算ボラティリティ
    return cast("pd.Series", hv)
