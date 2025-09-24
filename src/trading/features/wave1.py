"""
Wave1 features implementation.
Wave1特徴量の実装
"""

import numpy as np
import pandas as pd
from typing import Optional, List
from .base import BaseFeature


class ROC(BaseFeature):
    """Rate of Change"""

    def __init__(self, period: int = 14):
        super().__init__("ROC", deps=["close"])
        self.period = period

    def compute(self, df: pd.DataFrame, **params) -> pd.DataFrame:
        period = params.get('period', self.period)
        roc = (df['close'] - df['close'].shift(period)) / df['close'].shift(period) * 100
        return pd.DataFrame({'ROC': roc.fillna(0)})


# Moved to momentum/rsi.py


class RollingMean(BaseFeature):
    """Rolling Mean"""

    def __init__(self, window: int):
        super().__init__(f"rolling_mean_{window}", deps=["close"])
        self.window = window

    def compute(self, df: pd.DataFrame, **params) -> pd.DataFrame:
        col_name = f"rolling_mean_{self.window}"
        return pd.DataFrame({col_name: df['close'].rolling(self.window).mean()})


class RollingStd(BaseFeature):
    """Rolling Standard Deviation"""

    def __init__(self, window: int):
        super().__init__(f"rolling_std_{window}", deps=["close"])
        self.window = window

    def compute(self, df: pd.DataFrame, **params) -> pd.DataFrame:
        col_name = f"rolling_std_{self.window}"
        return pd.DataFrame({col_name: df['close'].rolling(self.window).std()})


class ZScore(BaseFeature):
    """Z-Score of returns"""

    def __init__(self, window: int = 20):
        self.window = window
        super().__init__(f"zscore_{window}", deps=["return", f"rolling_mean_{window}", f"rolling_std_{window}"])

    def compute(self, df: pd.DataFrame, **params) -> pd.DataFrame:
        window = params.get('window', self.window)
        mean = df['return'].rolling(window).mean()
        std = df['return'].rolling(window).std()
        zscore = (df['return'] - mean) / std
        return pd.DataFrame({f'zscore_{window}': zscore.fillna(0)})


# Moved to volatility/atr.py


class Lags(BaseFeature):
    """Lag features"""

    def __init__(self):
        super().__init__("Lags", deps=["close"])

    def compute(self, df: pd.DataFrame, **params) -> pd.DataFrame:
        lags: Optional[List[int]] = params.get('lags')
        if lags is None:
            lags = [1, 2, 3, 4, 5]
        result = {}
        for lag in lags:
            result[f'r{lag}'] = df['close'].shift(lag).ffill()
        return pd.DataFrame(result)


class DOW(BaseFeature):
    """Day of Week"""

    def __init__(self):
        super().__init__("DOW", deps=["ts"])

    def compute(self, df: pd.DataFrame, **params) -> pd.DataFrame:
        # ts を datetime に変換（仮定）
        if 'ts' in df.columns:
            dt = pd.to_datetime(df['ts'], unit='s')
            dow = dt.dt.dayofweek
        else:
            dow = pd.Series([0] * len(df))  # デフォルト
        return pd.DataFrame({'DOW': dow})


class HourOfDay(BaseFeature):
    """Hour of Day"""

    def __init__(self):
        super().__init__("HourOfDay", deps=["ts"])

    def compute(self, df: pd.DataFrame, **params) -> pd.DataFrame:
        if 'ts' in df.columns:
            dt = pd.to_datetime(df['ts'], unit='s')
            hour = dt.dt.hour
        else:
            hour = pd.Series([12] * len(df))  # デフォルト正午
        return pd.DataFrame({'HourOfDay': hour})


class OBV(BaseFeature):
    """On Balance Volume"""

    def __init__(self):
        super().__init__("OBV", deps=["close", "volume"])

    def compute(self, df: pd.DataFrame, **params) -> pd.DataFrame:
        direction = np.sign(df['close'].diff().fillna(0))
        obv = (direction * df['volume']).cumsum()
        return pd.DataFrame({'OBV': obv})


class MFI(BaseFeature):
    """Money Flow Index"""

    def __init__(self, period: int = 14):
        super().__init__("MFI", deps=["high", "low", "close", "volume"])
        self.period = period

    def compute(self, df: pd.DataFrame, **params) -> pd.DataFrame:
        period = params.get('period', self.period)
        typical_price = (df['high'] + df['low'] + df['close']) / 3
        money_flow = typical_price * df['volume']

        # 差分を計算
        price_diff = typical_price.diff().astype(float)
        positive_flow = money_flow.where(price_diff > 0, 0.0)
        negative_flow = money_flow.where(price_diff < 0, 0.0)

        pos_mf = positive_flow.rolling(period).sum()
        neg_mf = negative_flow.rolling(period).sum()
        epsilon = 1e-6
        mfr = pos_mf / (neg_mf + epsilon)
        mfi = 100 - (100 / (1 + mfr))
        return pd.DataFrame({'MFI': mfi.fillna(50)})
    
class VWAP(BaseFeature):
    """Volume Weighted Average Price"""

    def __init__(self):
        super().__init__("VWAP", deps=["high", "low", "close", "volume"])

    def compute(self, df: pd.DataFrame, **params) -> pd.DataFrame:
        typical_price = (df['high'] + df['low'] + df['close']) / 3
        cum_vol = df['volume'].cumsum()
        cum_vol_price = (typical_price * df['volume']).cumsum()
        vwap = cum_vol_price / cum_vol
        return pd.DataFrame({'VWAP': vwap.fillna(0)})