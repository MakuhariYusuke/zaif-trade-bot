"""
Chaikin AD Oscillator feature implementation.
Chaikin AD Oscillator is the difference between the 3-day EMA and 10-day EMA of the Chaikin AD line.

Output columns:
  - chaikin_ad_oscillator: Chaikin AD Oscillator values
"""

from typing import Any

import numpy as np
import pandas as pd

from ztb.utils.talib_wrapper import TaLibWrapper

from ..base import BaseFeature
from ..registry import FeatureRegistry

@FeatureRegistry.register("Chaikin_AD_Oscillator")
def compute_chaikin_ad_oscillator(df: pd.DataFrame) -> pd.Series:
    """Chaikin AD Oscillator using Ta-Lib wrapper"""
    feature = ChaikinADOscillator()
    result_df = feature.compute(df)
    return result_df["chaikin_ad_oscillator"]

class ChaikinADOscillator(BaseFeature):
    """
    Chaikin AD Oscillator indicator.
    The difference between the 3-day EMA and 10-day EMA of the Chaikin AD line.
    """

    def __init__(self, fast_period: int = 3, slow_period: int = 10, **kwargs: Any):
        super().__init__("ChaikinADOscillator", deps=["high", "low", "close", "volume"])
        self.fast_period = fast_period
        self.slow_period = slow_period

    def compute(self, df: pd.DataFrame, **params: Any) -> pd.DataFrame:
        """
        df columns must include: ['high', 'low', 'close', 'volume'].
        Returns a DataFrame with Chaikin AD Oscillator values.
        """
        fast_period = params.get("fast_period", self.fast_period)
        slow_period = params.get("slow_period", self.slow_period)

        # Calculate Chaikin AD first
        chaikin_ad_values = TaLibWrapper.ad(
            df["high"].values.astype(np.float64),
            df["low"].values.astype(np.float64),
            df["close"].values.astype(np.float64),
            df["volume"].values.astype(np.float64),
        )

        # Calculate fast and slow EMAs of Chaikin AD
        fast_ema = TaLibWrapper.ema(chaikin_ad_values, fast_period)
        slow_ema = TaLibWrapper.ema(chaikin_ad_values, slow_period)

        # Chaikin AD Oscillator is the difference
        chaikin_ad_oscillator = fast_ema - slow_ema
        # Normalize oscillator by average volume to keep values in a reasonable range
        chaikin_ad_oscillator = chaikin_ad_oscillator / df["volume"].mean()

        result_df = pd.DataFrame(
            {"chaikin_ad_oscillator": chaikin_ad_oscillator}, index=df.index
        )
        return result_df
