from typing import Any

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from ztb.features.base import ParameterizedFeature
from ztb.features.registry import FeatureRegistry


@FeatureRegistry.register("Supertrend")
def compute_supertrend(df: pd.DataFrame) -> pd.Series:
    """Supertrend Indicator"""
    feature = Supertrend()
    result_df = feature.compute(df)
    return result_df["supertrend"]


@FeatureRegistry.register("Supertrend_Direction")
def compute_supertrend_direction(df: pd.DataFrame) -> pd.Series:
    """Supertrend Direction (1 for uptrend, -1 for downtrend)"""
    feature = Supertrend()
    result_df = feature.compute(df)
    return result_df["supertrend_direction"]


class Supertrend(ParameterizedFeature):
    """
    Supertrend feature implementation.
    Identifies trend direction and potential reversals.

    Parameters:
      - period: ATR calculation period (default=10)
      - multiplier: ATR multiplier for band calculation (default=3.0)

    Output columns:
      - supertrend
      - supertrend_direction (1 for uptrend, -1 for downtrend)

    Note:
      - The DataFrame must include an ATR column named 'atr_{period}' (e.g., 'atr_10').
      - You should compute and add the ATR column beforehand, for example using an ATR feature or function.
    """

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(
            "Supertrend",
            deps=["high", "low", "close"],
            default_params={"period": 10, "multiplier": 3.0},
        )

    def _compute_with_params(self, df: pd.DataFrame, **params: Any) -> pd.DataFrame:
        """
        df columns must include: ['high', 'low', 'close'] and ATR column.
        Returns a DataFrame with Supertrend values.
        """
        period = params.get("period", 10)
        multiplier = params.get("multiplier", 3.0)
        atr_col = f"atr_{period}"

        if atr_col not in df.columns:
            # Compute ATR if not present
            from ztb.features.volatility.atr import compute_atr_simplified

            atr_series = compute_atr_simplified(df, period)
            df = df.copy()
            df[atr_col] = atr_series

        supertrend_df = pd.DataFrame(index=df.index)
        high = np.asarray(df["high"])
        low = np.asarray(df["low"])
        close = np.asarray(df["close"])
        atr = np.asarray(df[atr_col])

        supertrend, direction = self._compute_supertrend(
            high, low, close, atr, multiplier
        )

        supertrend_df["supertrend"] = supertrend
        supertrend_df["supertrend_direction"] = direction

        return supertrend_df

    @staticmethod
    def _compute_supertrend(
        high: NDArray[np.floating[Any]],
        low: NDArray[np.floating[Any]],
        close: NDArray[np.floating[Any]],
        atr: NDArray[np.floating[Any]],
        multiplier: float
    ) -> tuple[NDArray[np.floating[Any]], NDArray[np.floating[Any]]]:
        """Compute Supertrend using pure numpy (no numba)"""
        n = len(close)
        supertrend = np.full(n, np.nan, dtype=np.float64)
        direction = np.zeros(n, dtype=np.float64)

        for i in range(n):
            if i == 0:
                # 初期化
                hl2 = (high[i] + low[i]) / 2
                upperband = hl2 + (multiplier * atr[i])
                lowerband = hl2 - (multiplier * atr[i])
                supertrend[i] = lowerband
                direction[i] = 1  # 初期はアップトレンド
                continue

            hl2 = (high[i] + low[i]) / 2
            upperband = hl2 + (multiplier * atr[i])
            lowerband = hl2 - (multiplier * atr[i])

            prev_supertrend = supertrend[i - 1]
            prev_direction = direction[i - 1]

            # トレンド方向の決定
            if np.isclose(prev_supertrend, upperband) or np.isclose(
                prev_supertrend, lowerband
            ):
                if close[i] > upperband:
                    direction[i] = 1
                    supertrend[i] = lowerband
                elif close[i] < lowerband:
                    direction[i] = -1
                    supertrend[i] = upperband
                else:
                    direction[i] = prev_direction
                    supertrend[i] = prev_supertrend
            else:
                if prev_direction == 1:
                    if close[i] > prev_supertrend:
                        direction[i] = 1
                        supertrend[i] = max(lowerband, prev_supertrend)
                    else:
                        direction[i] = -1
                        supertrend[i] = upperband
                else:  # prev_direction == -1
                    if close[i] < prev_supertrend:
                        direction[i] = -1
                        supertrend[i] = min(upperband, prev_supertrend)
                    else:
                        direction[i] = 1
                        supertrend[i] = lowerband

        return supertrend, direction
