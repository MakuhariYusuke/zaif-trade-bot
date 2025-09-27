"""
KAMA (Kaufman's Adaptive Moving Average) feature implementation.
Adaptive moving average that adjusts to market volatility.

Output columns:
  - kama
"""

import numpy as np
import pandas as pd
from typing import Any
from numba import jit  # type: ignore
from numpy.typing import NDArray
from ztb.features.base import MovingAverageFeature, ComputableFeature


class KAMA(MovingAverageFeature, ComputableFeature):
    """
    Kaufman's Adaptive Moving Average.
    Adjusts smoothing based on market efficiency ratio.
    """

    def __init__(self, **kwargs: Any) -> None:
        super().__init__("KAMA", deps=["close"])

    def compute(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        df columns must include: ['close'].
        Returns a DataFrame with KAMA values.
        """
        close = df['close'].astype(float).to_numpy()
        kama = self._calculate_kama(close)

        return pd.DataFrame({'kama': kama})

    @staticmethod
    @jit(nopython=True)  # type: ignore
    def _calculate_kama(close: NDArray[np.float64]) -> NDArray[np.float64]:
        """
        Calculate KAMA using numba for performance.
        """
        n = len(close)
        kama = np.zeros(n)
        kama[0] = close[0]

        # Need at least 10 periods for initial calculation
        for i in range(10, n):
            # Efficiency Ratio: |close[i] - close[i-10]| / sum(|close[j] - close[j-1]| for j in i-9 to i)
            # Efficiency Ratio (ER) の計算:
            #   - change: 直近10期間の価格変化量 = abs(close[i] - close[i-10])
            #   - volatility: 直近10期間の価格変動合計 = sum(abs(close[j] - close[j-1]) for j in i-9 to i)
            #   - ER = change / volatility (volatilityが0の場合は0)
            change = abs(close[i] - close[i-10])
            volatility = 0.0
            for j in range(i-9, i+1):
                volatility += abs(close[j] - close[j-1])
            er = change / volatility if volatility != 0 else 0

            # Smoothing constant: ER * (fast SC - slow SC) + slow SC
            # fast SC = 2/(2+1) = 0.6667, slow SC = 2/(30+1) = 0.0645
            fast_sc = 2.0 / (2.0 + 1.0)
            slow_sc = 2.0 / (30.0 + 1.0)
            sc = er * (fast_sc - slow_sc) + slow_sc
            sc = sc ** 2  # Square for faster adaptation

            kama[i] = kama[i-1] + sc * (close[i] - kama[i-1])

        return kama