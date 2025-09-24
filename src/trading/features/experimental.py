"""
experimental.py
未確定・次期候補の特徴量を試験的に実装するモジュール。

ここには Wave5 以降に導入を検討する特徴量の草案を置きます。
安定して評価できるようになったら trend/、volatility/、momentum/、volume/ に移動します。

現在の候補:
- MovingAverages (EMA/SMA multi-period)
- GradientSign
- その他開発中の新規特徴量
"""

import numpy as np
import pandas as pd
from numba import jit
from .base import BaseFeature

# Moved to trend/heikin_ashi.py

# Moved to trend/supertrend.py
    
class MovingAverages(BaseFeature):
    """
    Multi-period Moving Averages (EMA/SMA) feature implementation.
    Computes EMAs and SMAs for multiple periods to capture trends at different time scales.

    Parameters:
      - ema_periods: List of EMA periods (default=[5, 10, 20, 50])
      - sma_periods: List of SMA periods (default=[5, 10, 20, 50])
    Output columns:
      - ema_{period} for each EMA period
      - sma_{period} for each SMA period
    """
    def __init__(self, ema_periods: list = [5, 10, 20, 50], sma_periods: list = [5, 10, 20, 50], **kwargs):
        super().__init__("MovingAverages", deps=["close"])
        self.ema_periods = ema_periods
        self.sma_periods = sma_periods
    def compute(self, df: pd.DataFrame, **params) -> pd.DataFrame:
        """
        df columns must include: ['close'].
        Returns a DataFrame with EMA and SMA columns for each specified period.
        """
        ma_df = pd.DataFrame(index=df.index)
        # EMAs
        for period in self.ema_periods:
            ema_col = f'ema_{period}'
            ma_df[ema_col] = df['close'].ewm(span=period, adjust=False).mean()
        # SMAs
        for period in self.sma_periods:
            sma_col = f'sma_{period}'
            ma_df[sma_col] = df['close'].rolling(window=period).mean()
        return ma_df

class GradientSign(BaseFeature):
    """
    Price Gradient Sign feature implementation.
    Indicates the direction of price movement.

    Output columns:
      - price_gradient_sign (1 for up, -1 for down, 0 for no change)
    """
    def __init__(self, **kwargs):
        super().__init__("GradientSign", deps=["close"])
    def compute(self, df: pd.DataFrame, **params) -> pd.DataFrame:
        """
        df columns must include: ['close'].
        Returns a DataFrame with price gradient sign.
        """
        grad_df = pd.DataFrame(index=df.index)
        grad = np.sign(df['close'].diff().fillna(0))
        grad_df['price_gradient_sign'] = grad
        return grad_df
