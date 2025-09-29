"""
TEMA (Triple Exponential Moving Average) feature implementation.
Triple exponential smoothing for trend analysis.

Parameters:
  - period: EMA period for TEMA calculation (default=14)
Output columns:
  - tema_{period}
"""

import pandas as pd
from typing import Any

from ztb.features.base import BaseFeature


class TEMA(BaseFeature):
    """
    Triple Exponential Moving Average for trend analysis.
    """

    def __init__(self, period: int = 14, **kwargs: Any):
        super().__init__("TEMA", deps=["close"])
        self.period = period

    def compute(self, df: pd.DataFrame, **params: Any) -> pd.DataFrame:
        """
        df columns must include: ['close'].
        Returns a DataFrame with TEMA values.
        """
        # Check if 'close' column exists
        if "close" not in df.columns:
            raise ValueError(
                "Input DataFrame must contain a 'close' column for TEMA calculation."
            )

        # Calculate three EMAs
        ema1 = df["close"].ewm(span=self.period, adjust=False).mean()
        ema2 = ema1.ewm(span=self.period, adjust=False).mean()
        ema3 = ema2.ewm(span=self.period, adjust=False).mean()
        # TEMA = 3 * EMA1 - 3 * EMA2 + EMA3
        tema = 3 * ema1 - 3 * ema2 + ema3

        return pd.DataFrame({f"tema_{self.period}": tema}, index=df.index)
