import numpy as np
import pandas as pd
from numba import jit
from ..base import BaseFeature

class Donchian(BaseFeature):
    """Donchian Channel with normalized position and relative width"""

    def __init__(self):
        super().__init__("Donchian", deps=["high", "low", "close", "ATR_simplified"])

    @staticmethod
    @jit(nopython=True)
    def _compute_donchian(high, low, close, atr, period=20):
        n = len(high)
        upper = np.zeros(n)
        lower = np.zeros(n)

        for i in range(period-1, n):
            upper[i] = np.max(high[i-period+1:i+1])
            lower[i] = np.min(low[i-period+1:i+1])

        middle = (upper + lower) / 2
        width = upper - lower
        position = np.zeros(n)
        width_rel = np.zeros(n)

        for i in range(n):
            if width[i] != 0:
                position[i] = (close[i] - middle[i]) / width[i]
            if atr[i] != 0:
                width_rel[i] = width[i] / atr[i]

        return position, width_rel

    def compute(self, df: pd.DataFrame, **params) -> pd.DataFrame:
        periods = params.get('periods', [20, 55])
        high = df['high'].values
        low = df['low'].values
        close = df['close'].values
        atr = df['ATR_simplified'].values

        df_copy = df.copy()
        for period in periods:
            position, width_rel = self._compute_donchian(high, low, close, atr, period)
            df_copy[f'donchian_pos_{period}'] = position
            # slope
            pos_series = pd.Series(position)
            slope = pos_series.diff().fillna(0).values
            df_copy[f'donchian_slope_{period}'] = slope

        # 出力列
        output_cols = []
        for period in periods:
            output_cols.extend([f'donchian_pos_{period}', f'donchian_slope_{period}'])

        return df_copy[output_cols]