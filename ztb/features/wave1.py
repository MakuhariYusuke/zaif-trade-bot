"""
Wave1 features implementation.
Wave1特徴量の実装
"""

import numpy as np
import pandas as pd
from typing import Optional, List
from .base import BaseFeature, StrictComputableFeature


class ROC(BaseFeature, StrictComputableFeature):
    """Rate of Change"""

    def __init__(self, period: int = 14):
        super().__init__("ROC", deps=["close"])
        self.period = period

    def compute(self, df: pd.DataFrame) -> pd.DataFrame:
        roc = (df['close'] - df['close'].shift(self.period)) / df['close'].shift(self.period) * 100
        return pd.DataFrame({'ROC': roc.fillna(0)})


# Moved to momentum/rsi.py


class RollingMean(BaseFeature, StrictComputableFeature):
    """Rolling Mean"""

    def __init__(self, window: int):
        super().__init__(f"rolling_mean_{window}", deps=["close"])
        self.window = window

    def compute(self, df: pd.DataFrame) -> pd.DataFrame:
        col_name = f"rolling_mean_{self.window}"
        return pd.DataFrame({col_name: df['close'].rolling(self.window).mean()})


class RollingStd(BaseFeature, StrictComputableFeature):
    """Rolling Standard Deviation"""

    def __init__(self, window: int):
        super().__init__(f"rolling_std_{window}", deps=["close"])
        self.window = window

    def compute(self, df: pd.DataFrame) -> pd.DataFrame:
        col_name = f"rolling_std_{self.window}"
        return pd.DataFrame({col_name: df['close'].rolling(self.window).std()})


class ZScore(BaseFeature, StrictComputableFeature):
    """Z-Score of returns"""

    def __init__(self, window: int = 20):
        self.window = window
        super().__init__(f"zscore_{window}", deps=["return", f"rolling_mean_{window}", f"rolling_std_{window}"])

    def compute(self, df: pd.DataFrame) -> pd.DataFrame:
        mean = df['return'].rolling(self.window).mean()
        std = df['return'].rolling(self.window).std()
        zscore = (df['return'] - mean) / std
        return pd.DataFrame({f'zscore_{self.window}': zscore.fillna(0)})


# Moved to volatility/atr.py


class Lags(BaseFeature, StrictComputableFeature):
    """Lag features"""

    def __init__(self, lags: Optional[List[int]] = None):
        super().__init__("Lags", deps=["close"])
        self.lags = lags or [1, 2, 3, 4, 5]

    def compute(self, df: pd.DataFrame) -> pd.DataFrame:
        result = {}
        for lag in self.lags:
            result[f'r{lag}'] = df['close'].shift(lag).ffill()
        return pd.DataFrame(result)


class DOW(BaseFeature, StrictComputableFeature):
    """Day of Week"""

    def __init__(self):
        super().__init__("DOW", deps=["ts"])

    def compute(self, df: pd.DataFrame) -> pd.DataFrame:
        # ts を datetime に変換（仮定）
        if 'ts' in df.columns:
            dt = pd.to_datetime(df['ts'], unit='s')
            dow = dt.dt.dayofweek
        else:
            dow = pd.Series([0] * len(df))  # デフォルト
        return pd.DataFrame({'DOW': dow})


class HourOfDay(BaseFeature, StrictComputableFeature):
    """Hour of Day"""

    def __init__(self):
        super().__init__("HourOfDay", deps=["ts"])

    def compute(self, df: pd.DataFrame) -> pd.DataFrame:
        if 'ts' in df.columns:
            dt = pd.to_datetime(df['ts'], unit='s')
            hour = dt.dt.hour
        else:
            hour = pd.Series([12] * len(df))  # デフォルト正午
        return pd.DataFrame({'HourOfDay': hour})


class OBV(BaseFeature, StrictComputableFeature):
    """On Balance Volume"""

    def __init__(self):
        super().__init__("OBV", deps=["close", "volume"])

    def compute(self, df: pd.DataFrame) -> pd.DataFrame:
        direction = np.sign(df['close'].diff().fillna(0))
        obv = (direction * df['volume']).cumsum()
        return pd.DataFrame({'OBV': obv})


class MFI(BaseFeature, StrictComputableFeature):
    """Money Flow Index"""

    def __init__(self, period: int = 14):
        super().__init__("MFI", deps=["high", "low", "close", "volume"])
        self.period = period

    def compute(self, df: pd.DataFrame) -> pd.DataFrame:
        typical_price = (df['high'] + df['low'] + df['close']) / 3
        money_flow = typical_price * df['volume']

        # 差分を計算
        price_diff = typical_price.diff().astype(float)
        positive_flow = money_flow.where(price_diff > 0, 0.0)
        negative_flow = money_flow.where(price_diff < 0, 0.0)

        pos_mf = positive_flow.rolling(self.period).sum()
        neg_mf = negative_flow.rolling(self.period).sum()
        epsilon = 1e-6
        mfr = pos_mf / (neg_mf + epsilon)
        mfi = 100 - (100 / (1 + mfr))
        return pd.DataFrame({'MFI': mfi.fillna(50)})
    
class VWAP(BaseFeature, StrictComputableFeature):
    """Volume Weighted Average Price"""

    def __init__(self):
        super().__init__("VWAP", deps=["high", "low", "close", "volume"])

    def compute(self, df: pd.DataFrame) -> pd.DataFrame:
        typical_price = (df['high'] + df['low'] + df['close']) / 3
        cum_vol = df['volume'].cumsum()
        cum_vol_price = (typical_price * df['volume']).cumsum()
        vwap = cum_vol_price / cum_vol
        return pd.DataFrame({'VWAP': vwap.fillna(0)})


class ZScore(BaseFeature, StrictComputableFeature):
    """Z-Score (Standard Score)"""

    def __init__(self, period: int = 20):
        super().__init__("ZScore", deps=["close"])
        self.period = period

    def compute(self, df: pd.DataFrame) -> pd.DataFrame:
        mean = df['close'].rolling(window=self.period).mean()
        std = df['close'].rolling(window=self.period).std()
        zscore = (df['close'] - mean) / std
        return pd.DataFrame({'ZScore': zscore.fillna(0)})