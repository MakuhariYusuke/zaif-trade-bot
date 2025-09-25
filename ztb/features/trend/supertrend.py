import numpy as np
import pandas as pd
from numba import jit
from ..base import ParameterizedFeature

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
    """
    def __init__(self, **kwargs):
        super().__init__(
            "Supertrend",
            deps=["high", "low", "close"],
            default_params={"period": 10, "multiplier": 3.0}
        )

    def _compute_with_params(self, df: pd.DataFrame, **params) -> pd.DataFrame:
        """
        df columns must include: ['high', 'low', 'close'] and ATR column.
        Returns a DataFrame with Supertrend values.
        """
        period = params.get('period', 10)
        multiplier = params.get('multiplier', 3.0)
        atr_col = f'atr_{period}'

        if atr_col not in df.columns:
            raise ValueError(f"ATR column '{atr_col}' not found in DataFrame. Please compute ATR first.")

        supertrend_df = pd.DataFrame(index=df.index)
        high = df['high'].values
        low = df['low'].values
        close = df['close'].values
        atr = df[atr_col].values

        supertrend, direction = self._compute_supertrend(
            high, low, close, atr, multiplier
        )

        supertrend_df['supertrend'] = supertrend
        supertrend_df['supertrend_direction'] = direction

        return supertrend_df

    @staticmethod
    @jit(nopython=True)
    def _compute_supertrend(high, low, close, atr, multiplier):
        n = len(close)
        supertrend = np.full(n, np.nan)
        direction = np.zeros(n)

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

            prev_supertrend = supertrend[i-1]
            prev_direction = direction[i-1]

            # トレンド方向の決定
            if prev_supertrend == upperband or prev_supertrend == lowerband:
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