"""
Historical Volatility (HV) implementation.
過去のボラティリティを測定
"""

import numpy as np
import pandas as pd
from typing import cast
from ztb.features.registry import FeatureRegistry


@FeatureRegistry.register("HV")
def compute_hv(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """Compute Historical Volatility"""
    log_returns = np.log(df['close'] / df['close'].shift(1))
    hv = pd.Series(log_returns).rolling(window=period).std() * np.sqrt(252)  # Annualized volatility
    return cast(pd.Series, hv)