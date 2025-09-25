"""
ADX (Average Directional Index) feature implementation.
Measures trend strength using directional movement indicators.

Parameters:
  - period: ADX calculation period (default=14)
Output columns:
  - adx_{period}
  - plus_di_{period}
  - minus_di_{period}
"""

import numpy as np
import pandas as pd
from numba import jit
from ..base import BaseFeature


class ADX(BaseFeature):
    """
    ADX (Average Directional Index) feature implementation.
    Measures trend strength and direction.
    """

    def __init__(self, period: int = 14, **kwargs):
        super().__init__("ADX", deps=["high", "low", "close"])
        self.period = period

    def compute(self, df: pd.DataFrame, **params) -> pd.DataFrame:
        """
        df columns must include: ['high', 'low', 'close'].
        Returns a DataFrame with ADX, +DI, -DI values.
        """
        adx_df = pd.DataFrame(index=df.index)

        high = df['high']
        low = df['low']
        close = df['close']

        # True Range
        tr1 = high - low
        tr2 = (high - close.shift(1)).abs()
        tr3 = (low - close.shift(1)).abs()
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

        # Directional Movement
        dm_plus = np.where(
            (high - high.shift(1)) > (low.shift(1) - low),
            np.maximum(high - high.shift(1), 0),
            0
        )
        dm_minus = np.where(
            (low.shift(1) - low) > (high - high.shift(1)),
            np.maximum(low.shift(1) - low, 0),
            0
        )

        # Smooth with EMA
        tr_smooth = tr.ewm(span=self.period, adjust=False).mean()
        dm_plus_smooth = pd.Series(dm_plus, index=df.index).ewm(span=self.period, adjust=False).mean()
        dm_minus_smooth = pd.Series(dm_minus, index=df.index).ewm(span=self.period, adjust=False).mean()

        # Directional Indicators
        plus_di = 100 * dm_plus_smooth / tr_smooth
        minus_di = 100 * dm_minus_smooth / tr_smooth

        # DX and ADX
        dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di + 1e-8)
        adx = dx.ewm(span=self.period, adjust=False).mean()

        adx_df[f'adx_{self.period}'] = adx
        adx_df[f'plus_di_{self.period}'] = plus_di
        adx_df[f'minus_di_{self.period}'] = minus_di

        return adx_df