import numpy as np
import pandas as pd
from numba import jit
from ..base import ChannelFeature, ComputableFeature

class Ichimoku(ChannelFeature, ComputableFeature):
    """Ichimoku Cloud with normalized diff and cross signal"""

    def __init__(self):
        super().__init__("Ichimoku", deps=["high", "low", "close", "ATR_simplified"])
        self._required_calculations = set()  # Internal calculation only

    @staticmethod
    @jit(nopython=True)
    def _compute_ichimoku(high, low, close):
        n = len(high)
        tenkan = np.zeros(n)
        kijun = np.zeros(n)
        senkou_a = np.zeros(n)
        senkou_b = np.zeros(n)
        chikou = np.zeros(n)

        for i in range(9, n):
            tenkan[i] = (np.max(high[i-9:i+1]) + np.min(low[i-9:i+1])) / 2

        for i in range(26, n):
            kijun[i] = (np.max(high[i-26:i+1]) + np.min(low[i-26:i+1])) / 2

        for i in range(26, n):
            if i >= 26:
                senkou_a[i] = (tenkan[i-26] + kijun[i-26]) / 2

        for i in range(52, n):
            if i >= 52:
                senkou_b[i] = (np.max(high[i-52:i-26+1]) + np.min(low[i-52:i-26+1])) / 2

        for i in range(26, n):
            chikou[i] = close[i-26]

        return tenkan, kijun, senkou_a, senkou_b, chikou

    def compute(self, df: pd.DataFrame) -> pd.DataFrame:
        high = df['high'].values
        low = df['low'].values
        close = df['close'].values

        tenkan, kijun, senkou_a, senkou_b, chikou = self._compute_ichimoku(high, low, close)

        # 差分正規化
        diff = tenkan - kijun
        atr = df['ATR_simplified'].to_numpy(dtype=float) if 'ATR_simplified' in df.columns else np.ones(len(close))
        normalized_diff = np.where(atr != 0, diff / atr, 0)
        # クロスシグナル
        cross = (tenkan > kijun).astype(float)

        df_copy = df.copy()
        df_copy['ichimoku_diff_norm'] = normalized_diff
        df_copy['ichimoku_cross'] = cross

        return df_copy[['ichimoku_diff_norm', 'ichimoku_cross']]