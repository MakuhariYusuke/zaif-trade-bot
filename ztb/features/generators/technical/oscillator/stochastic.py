"""
Stochastic Oscillator feature implementation.
Stochastic Oscillator is a momentum indicator that shows the location of the close relative to the high-low range over a set number of periods.

Output columns:
  - stoch_k: %K line (fast stochastic)
  - stoch_d: %D line (slow stochastic, SMA of %K)
"""

from typing import Any

import pandas as pd

from ..base import BaseFeature
from ..registry import FeatureRegistry


@FeatureRegistry.register("Stochastic")
def compute_stochastic(df: pd.DataFrame) -> pd.Series:
    """Stochastic Oscillator using Ta-Lib wrapper"""
    feature = Stochastic()
    result_df = feature.compute(df)
    return result_df["stoch_k"]  # Return %K as primary value


class Stochastic(BaseFeature):
    """
    Stochastic Oscillator.
    Consists of two lines: %K (fast) and %D (slow, SMA of %K).
    """

    def __init__(self, k_period: int = 14, d_period: int = 3, **kwargs: Any):
        super().__init__("Stochastic", deps=["high", "low", "close"])
        self.k_period = k_period
        self.d_period = d_period

    def compute(self, df: pd.DataFrame, **params: Any) -> pd.DataFrame:
        """
        df columns must include: ['high', 'low', 'close'].
        Returns a DataFrame with Stochastic %K and %D values.
        """
        k_period = params.get("k_period", self.k_period)
        d_period = params.get("d_period", self.d_period)

        # Calculate Stochastic manually
        # %K = (Close - Lowest Low) / (Highest High - Lowest Low) * 100
        highest_high = df["high"].rolling(window=k_period).max()
        lowest_low = df["low"].rolling(window=k_period).min()

        stoch_k = ((df["close"] - lowest_low) / (highest_high - lowest_low)) * 100

        # %D = SMA of %K
        stoch_d = stoch_k.rolling(window=d_period).mean()

        return pd.DataFrame({"stoch_k": stoch_k, "stoch_d": stoch_d}, index=df.index)