import numpy as np
import pandas as pd
from numba import jit
from ..base import BaseFeature

class Supertrend(BaseFeature):
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
    def __init__(self, period: int = 10, multiplier: float = 3.0, **kwargs):
        super().__init__(**kwargs)
        self.period = period
        self.multiplier = multiplier
        self.atr_col = f'atr_{self.period}'
        self.prev_supertrend = 0.0
        self.prev_direction = 0.0
        self.prev_final_upperband = 0.0
        self.prev_final_lowerband = 0.0
        self.initialized = False
        self.epsilon = 1e-8  # To prevent division by zero
    def compute(self, df: pd.DataFrame, **params) -> pd.DataFrame:
        """ 
        df columns must include: ['high', 'low', 'close'] and ATR column.
        Returns a DataFrame with Supertrend values.
        """
        if self.atr_col not in df.columns:
            raise ValueError(f"ATR column '{self.atr_col}' not found in DataFrame. Please compute ATR first.")

        supertrend_df = pd.DataFrame(index=df.index)
        high = df['high'].values
        low = df['low'].values
        close = df['close'].values
        atr = df[self.atr_col].values

        supertrend, direction = self._compute_supertrend(
            high, low, close, atr, self.multiplier,
            self.prev_supertrend, self.prev_direction,
            self.prev_final_upperband, self.prev_final_lowerband,
            self.initialized, self.epsilon
        )

        supertrend_df['supertrend'] = supertrend
        supertrend_df['supertrend_direction'] = direction

        # Update previous state for next computation
        if len(supertrend) > 0:
            self.prev_supertrend = supertrend[-1]
            self.prev_direction = direction[-1]
            # NumbaのJITコンパイルのために、Noneではなく具体的な値を保持する必要があります。
            # しかし、ロジック上は次の計算で使われないため、更新は不要かもしれません。
            # ここではロジックを維持しつつ、Noneの代わりにnp.nanを使用します。
            if direction[-1] == 1:
                self.prev_final_lowerband = supertrend[-1]
                self.prev_final_upperband = np.nan
            else:
                self.prev_final_upperband = supertrend[-1]
                self.prev_final_lowerband = np.nan
            self.initialized = True

        return supertrend_df

    @staticmethod
    @jit(nopython=True)
    def _compute_supertrend(
        high, low, close, atr, multiplier,
        prev_supertrend, prev_direction,
        prev_final_upperband, prev_final_lowerband,
        initialized, epsilon
    ):
        n = len(close)
        supertrend = np.full(n, np.nan)
        direction = np.zeros(n)

        for i in range(n):
            if i == 0:
                # 初期化
                final_upperband = high[i] + multiplier * atr[i]
                final_lowerband = low[i] - multiplier * atr[i]
                supertrend[i] = final_upperband
                direction[i] = -1  # 初期はダウントレンドと仮定
                continue

            # 基本バンドの計算
            basic_upperband = high[i] + multiplier * atr[i]
            basic_lowerband = low[i] - multiplier * atr[i]

            # 最終バンドの計算
            if initialized:
                final_upperband = min(basic_upperband, prev_final_upperband) if prev_direction == -1 else basic_upperband
                final_lowerband = max(basic_lowerband, prev_final_lowerband) if prev_direction == 1 else basic_lowerband
            else:
                final_upperband = basic_upperband
                final_lowerband = basic_lowerband

            # Supertrendの決定
            if close[i] > final_upperband:
                supertrend[i] = final_lowerband
                direction[i] = 1  # アップトレンド
            elif close[i] < final_lowerband:
                supertrend[i] = final_upperband
                direction[i] = -1  # ダウントレンド
            else:
                supertrend[i] = prev_supertrend
                direction[i] = prev_direction

            # 状態の更新
            prev_supertrend = supertrend[i]
            prev_direction = direction[i]
            prev_final_upperband = final_upperband
            prev_final_lowerband = final_lowerband
            initialized = True

        return supertrend, direction