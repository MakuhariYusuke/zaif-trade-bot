"""
wave2.py

This module implements a collection of technical analysis features for trading strategies,
including medium-cost, medium-effectiveness indicators such as MACD, Bollinger Bands,
Stochastic Oscillator, CCI, ADX, EMA/SMA Cross, TEMA, and KAMA. Each feature is implemented
as a class inheriting from BaseFeature, providing a compute method to generate indicator values
from a pandas DataFrame. These features are intended to be used as part of a trading bot's
feature engineering pipeline.
"""

import numpy as np
import pandas as pd
from numba import jit
from .base import BaseFeature


class MACD(BaseFeature):
    """MACD (Moving Average Convergence Divergence)"""

    def __init__(self):
        super().__init__("MACD", deps=["ema_12", "ema_26", "ema_9"])

    def compute(self, df: pd.DataFrame, **params) -> pd.DataFrame:
        df_copy = df.copy()
        df_copy['macd'] = df_copy['ema_12'] - df_copy['ema_26']
        df_copy['macd_signal'] = df_copy['ema_9']
        df_copy['macd_hist'] = df_copy['macd'] - df_copy['macd_signal']
        return df_copy[['macd', 'macd_signal', 'macd_hist']]


class Bollinger(BaseFeature):
    """Bollinger Bands"""

    def __init__(self):
        super().__init__("Bollinger", deps=["rolling_mean_20", "rolling_std_20"])

    def compute(self, df: pd.DataFrame, **params) -> pd.DataFrame:
        df_copy = df.copy()
        df_copy['bb_upper'] = df_copy['rolling_mean_20'] + 2 * df_copy['rolling_std_20']
        df_copy['bb_lower'] = df_copy['rolling_mean_20'] - 2 * df_copy['rolling_std_20']
        df_copy['bb_middle'] = df_copy['rolling_mean_20']
        df_copy['bb_width'] = np.where(
            df_copy['bb_middle'] != 0,
            (df_copy['bb_upper'] - df_copy['bb_lower']) / df_copy['bb_middle'],
            0
        )
        denominator = df_copy['bb_upper'] - df_copy['bb_lower']
        df_copy['bb_position'] = np.where(
            denominator != 0,
            (df_copy['close'] - df_copy['bb_lower']) / denominator,
            0.5  # neutral position if width is zero
        )
        return df_copy[['bb_upper', 'bb_lower', 'bb_middle', 'bb_width', 'bb_position']]


class Stochastic(BaseFeature):
    """Stochastic Oscillator"""

    def __init__(self):
        super().__init__("Stochastic", deps=["rolling_max_14", "rolling_min_14"])

    def compute(self, df: pd.DataFrame, **params) -> pd.DataFrame:
        df_copy = df.copy()
        denominator = df_copy['rolling_max_14'] - df_copy['rolling_min_14']
        df_copy['stoch_k'] = np.where(
            denominator != 0,
            100 * (df_copy['close'] - df_copy['rolling_min_14']) / denominator,
            50  # neutral value when denominator is zero
        )
        df_copy['stoch_k'] = df_copy['stoch_k'].fillna(50)  # neutral
        df_copy['stoch_d'] = df_copy['stoch_k'].rolling(3).mean()
        return df_copy[['stoch_k', 'stoch_d']]


class CCI(BaseFeature):
    """Commodity Channel Index"""

    def __init__(self):
        super().__init__("CCI", deps=["rolling_mean_20"])

    def compute(self, df: pd.DataFrame, **params) -> pd.DataFrame:
        df_copy = df.copy()
        # Mean deviation
        mean_dev = (df_copy['close'] - df_copy['rolling_mean_20']).abs().rolling(20).mean()
        df_copy['cci'] = (df_copy['close'] - df_copy['rolling_mean_20']) / (0.015 * mean_dev)
        df_copy['cci'] = df_copy['cci'].fillna(0)
        return df_copy[['cci']]


class ADX(BaseFeature):
    """Average Directional Index"""

    def __init__(self):
        super().__init__("ADX")

    def compute(self, df: pd.DataFrame, **params) -> pd.DataFrame:
        df_copy = df.copy()
        adx, plus_di, minus_di = self._calculate_adx(
            np.asarray(df_copy['high']), np.asarray(df_copy['low']), np.asarray(df_copy['close'])
        )
        df_copy['adx'] = adx
        df_copy['plus_di'] = plus_di
        df_copy['minus_di'] = minus_di
        return df_copy[['adx', 'plus_di', 'minus_di']]

    @staticmethod
    @jit(nopython=True)
    def _calculate_adx(high: np.ndarray, low: np.ndarray, close: np.ndarray) -> tuple:
        n = len(high)
        adx = np.zeros(n)
        plus_di = np.zeros(n)
        minus_di = np.zeros(n)

        tr = np.zeros(n)
        plus_dm = np.zeros(n)
        minus_dm = np.zeros(n)

        for i in range(1, n):
            # True Range
            tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))

            # Directional Movement
            move_up = high[i] - high[i-1]
            move_down = low[i-1] - low[i]

            plus_dm[i] = move_up if move_up > move_down and move_up > 0 else 0
            minus_dm[i] = move_down if move_down > move_up and move_down > 0 else 0

        # Smooth TR and DM
        dx_arr = np.zeros(n)
        atr = np.zeros(n)
        plus_dm_smooth = np.zeros(n)
        minus_dm_smooth = np.zeros(n)
        for i in range(14, n):
            atr[i] = np.mean(tr[i-13:i+1])
            plus_dm_smooth[i] = np.mean(plus_dm[i-13:i+1])
            minus_dm_smooth[i] = np.mean(minus_dm[i-13:i+1])

            plus_di[i] = 100 * plus_dm_smooth[i] / atr[i] if atr[i] != 0 else 0
            minus_di[i] = 100 * minus_dm_smooth[i] / atr[i] if atr[i] != 0 else 0

            dx_arr[i] = 100 * abs(plus_di[i] - minus_di[i]) / (plus_di[i] + minus_di[i]) if (plus_di[i] + minus_di[i]) != 0 else 0

        # 初期ADX値を最初の14個のDXの平均で初期化
        if n >= 28:
            adx[27] = np.mean(dx_arr[14:28])
            for i in range(28, n):
                adx[i] = (adx[i-1] * 13 + dx_arr[i]) / 14  # EMA-like smoothing

        return adx, plus_di, minus_di


class EMACross(BaseFeature):
    """EMA/SMA Cross signals"""

    def __init__(self):
        super().__init__("EMACross", deps=["ema_5", "rolling_mean_20"])

    def compute(self, df: pd.DataFrame, **params) -> pd.DataFrame:
        df_copy = df.copy()
        df_copy['ema_sma_cross'] = (df_copy['ema_5'] - df_copy['rolling_mean_20']) / df_copy['rolling_mean_20']
        df_copy['ema_above_sma'] = (df_copy['ema_5'] > df_copy['rolling_mean_20']).astype(int)
        return df_copy[['ema_sma_cross', 'ema_above_sma']]


class TEMA(BaseFeature):
    """Triple Exponential Moving Average"""

    def __init__(self):
        super().__init__("TEMA", deps=["ema_14"])

    def compute(self, df: pd.DataFrame, **params) -> pd.DataFrame:
        df_copy = df.copy()
        ema1 = df_copy['ema_14']
        ema2 = ema1.ewm(span=14, adjust=False).mean()
        ema3 = ema2.ewm(span=14, adjust=False).mean()
        # TEMA calculation: TEMA = 3 * EMA1 - 3 * EMA2 + EMA3
        # Reference: https://www.investopedia.com/terms/t/triple-exponential-moving-average.asp
        df_copy['tema'] = 3 * ema1 - 3 * ema2 + ema3
        return df_copy[['tema']]


class KAMA(BaseFeature):
    """Kaufman's Adaptive Moving Average"""

    def __init__(self):
        super().__init__("KAMA")

    def compute(self, df: pd.DataFrame, **params) -> pd.DataFrame:
        df_copy = df.copy()
        close_np = df_copy['close'].to_numpy() if hasattr(df_copy['close'], 'to_numpy') else np.asarray(df_copy['close'])
        df_copy['kama'] = self._calculate_kama(close_np)
        return df_copy[['kama']]

    @staticmethod
    @jit(nopython=True)
    def _calculate_kama(close: np.ndarray) -> np.ndarray:
        n = len(close)
        kama = np.zeros(n)
        kama[0] = close[0]

        for i in range(10, n):
            # Efficiency Ratio
            change = abs(close[i] - close[i-10])
            volatility = np.sum(np.abs(np.diff(close[i-10:i+1])))
            er = change / volatility if volatility != 0 else 0

            # Smoothing constant
            sc = er * (2/(2+1) - 2/(30+1)) + 2/(30+1)
            sc = sc ** 2  # squared for faster adaptation

            kama[i] = kama[i-1] + sc * (close[i] - kama[i-1])

        return kama