"""
CCI (Commodity Channel Index) feature implementation.
CCI is an oscillator that measures the current price level relative to an average price level over a given period.

Output columns:
  - cci: CCI value (typically -100 to +100)
"""

from typing import Any

import numpy as np
import pandas as pd

from ..base import BaseFeature
from ..registry import FeatureRegistry

@FeatureRegistry.register("CCI")
def compute_cci(df: pd.DataFrame) -> pd.Series:
    """CCI (Commodity Channel Index) using Ta-Lib wrapper"""
    feature = CCI()
    result_df = feature.compute(df)
    return result_df["cci"]

class CCI(BaseFeature):
    """
    Commodity Channel Index (CCI).
    Measures the current price level relative to an average price level over a given period.
    """

    def __init__(self, period: int = 20, **kwargs: Any):
        super().__init__("CCI", deps=["high", "low", "close"])
        self.period = period

    def compute(self, df: pd.DataFrame, **params: Any) -> pd.DataFrame:
        """
        df columns must include: ['high', 'low', 'close'].
        Returns a DataFrame with CCI values.
        """
        period = params.get("period", self.period)

        # Calculate CCI manually
        # Typical Price = (High + Low + Close) / 3
        typical_price = (df["high"] + df["low"] + df["close"]) / 3

        # SMA of Typical Price
        sma_tp = typical_price.rolling(window=period).mean()

        # Mean Deviation = Average of absolute differences between TP and SMA(TP)
        mean_deviation = typical_price.rolling(window=period).apply(
            lambda x: np.mean(np.abs(x - x.mean())), raw=False
        )

        # CCI = (TP - SMA(TP)) / (0.015 * Mean Deviation)
        cci = (typical_price - sma_tp) / (0.015 * mean_deviation)

        return pd.DataFrame({"cci": cci}, index=df.index)
