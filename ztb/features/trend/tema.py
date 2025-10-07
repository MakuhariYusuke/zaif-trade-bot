"""
TEMA (Triple Exponential Moving Average) feature implementation.
Triple exponential smoothing for trend analysis.

Parameters:
  - period: EMA period for TEMA calculation (default=14)
Output columns:
  - tema_{period}
"""

import numpy as np
import pandas as pd

from ztb.features.registry import FeatureRegistry
from ztb.utils.talib_wrapper import TaLibWrapper

from typing import Any

from ztb.features.base import BaseFeature


@FeatureRegistry.register("TEMA")
def compute_tema(df: pd.DataFrame) -> pd.Series:
    """Triple Exponential Moving Average (TEMA)"""
    feature = TEMA()
    result_df = feature.compute(df)
    return result_df["tema_14"]


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

        # Use Ta-Lib wrapper for TEMA calculation
        result = TaLibWrapper.tema(df["close"].values.astype(np.float64), self.period)

        return pd.DataFrame({f"tema_{self.period}": result}, index=df.index)
