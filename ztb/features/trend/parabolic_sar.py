"""
Parabolic SAR feature implementation.
Parabolic SAR (Stop and Reverse) is a trend-following indicator.

Output columns:
  - psar: Parabolic SAR value
  - psar_trend: Trend direction (1 for uptrend, -1 for downtrend)
  - psar_acceleration: Current acceleration factor
"""

from typing import Any

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from ..base import BaseFeature
from ..registry import FeatureRegistry


@FeatureRegistry.register("PSAR")
def compute_psar(df: pd.DataFrame) -> pd.Series:
    """Parabolic SAR (Stop and Reverse)"""
    feature = ParabolicSAR()
    result_df = feature.compute(df)
    return result_df["psar"]


@FeatureRegistry.register("PSAR_Trend")
def compute_psar_trend(df: pd.DataFrame) -> pd.Series:
    """Parabolic SAR Trend Direction (1=uptrend, -1=downtrend)"""
    feature = ParabolicSAR()
    result_df = feature.compute(df)
    return result_df["psar_trend"]


class ParabolicSAR(BaseFeature):
    """
    Parabolic SAR (Stop and Reverse) indicator.
    """

    def __init__(
        self, acceleration: float = 0.02, max_acceleration: float = 0.2, **kwargs: Any
    ):
        super().__init__("ParabolicSAR", deps=["high", "low", "close"])
        self.acceleration = acceleration
        self.max_acceleration = max_acceleration

    def compute(self, df: pd.DataFrame, **params: Any) -> pd.DataFrame:
        """
        df columns must include: ['high', 'low', 'close'].
        Returns a DataFrame with Parabolic SAR values.
        """
        acceleration = params.get("acceleration", self.acceleration)
        max_acceleration = params.get("max_acceleration", self.max_acceleration)

        high = np.asarray(df["high"])
        low = np.asarray(df["low"])
        close = np.asarray(df["close"])

        psar, trend, acceleration_factor = self._compute_psar(
            high, low, close, acceleration, max_acceleration
        )

        return pd.DataFrame(
            {
                "psar": psar,
                "psar_trend": trend,
                "psar_acceleration": acceleration_factor,
            },
            index=df.index,
        )

    @staticmethod
    def _compute_psar(
        high: NDArray[np.floating[Any]],
        low: NDArray[np.floating[Any]],
        close: NDArray[np.floating[Any]],
        acceleration: float,
        max_acceleration: float
    ) -> tuple[NDArray[np.floating[Any]], NDArray[np.int32], NDArray[np.floating[Any]]]:
        """
        Calculate Parabolic SAR using pure numpy (no numba).
        """
        n = len(close)
        psar = np.zeros(n, dtype=np.float64)
        trend = np.zeros(n, dtype=np.int32)  # 1 for uptrend, -1 for downtrend
        acceleration_factor = np.full(n, acceleration, dtype=np.float64)

        # Initialize first values - use more robust initialization
        if n < 2:
            return psar, trend, acceleration_factor

        # Initialize based on first two candles
        if close[1] > close[0]:  # Price rising - assume uptrend
            psar[0] = low[0]
            trend[0] = 1
        else:  # Price falling - assume downtrend
            psar[0] = high[0]
            trend[0] = -1

        # Calculate Parabolic SAR
        for i in range(1, n):
            if trend[i - 1] == 1:  # Uptrend
                psar[i] = psar[i - 1] + acceleration_factor[i - 1] * (
                    high[i - 1] - psar[i - 1]
                )

                # Ensure SAR doesn't go above previous low
                psar[i] = min(psar[i], low[i - 1])

                # Check for trend reversal
                if low[i] < psar[i]:
                    trend[i] = -1
                    psar[i] = high[i - 1]  # Reset to previous high
                    acceleration_factor[i] = acceleration
                else:
                    trend[i] = 1
                    acceleration_factor[i] = min(
                        acceleration_factor[i - 1] + acceleration, max_acceleration
                    )

            else:  # Downtrend
                psar[i] = psar[i - 1] + acceleration_factor[i - 1] * (
                    low[i - 1] - psar[i - 1]
                )

                # Ensure SAR doesn't go below previous high
                psar[i] = max(psar[i], high[i - 1])

                # Check for trend reversal
                if high[i] > psar[i]:
                    trend[i] = 1
                    psar[i] = low[i - 1]  # Reset to previous low
                    acceleration_factor[i] = acceleration
                else:
                    trend[i] = -1
                    acceleration_factor[i] = min(
                        acceleration_factor[i - 1] + acceleration, max_acceleration
                    )

        return psar, trend, acceleration_factor
