"""
wave3.py

This module implements advanced technical analysis features for trading strategies,
including Ichimoku Cloud, Donchian Channel, Regime Clustering, and Kalman Filter.
Each feature is implemented as a class inheriting from BaseFeature, providing a compute
method to generate indicator values from a pandas DataFrame. These features are intended
to be used as part of a trading bot's feature engineering pipeline.
"""

import numpy as np
import pandas as pd
from numba import jit
from .base import BaseFeature


class Ichimoku(BaseFeature):
    """Ichimoku Cloud with normalized diff and cross signal"""

    def __init__(self):
        super().__init__("Ichimoku", deps=["high", "low", "close", "ATR_simplified"])

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

    def compute(self, df: pd.DataFrame, **params) -> pd.DataFrame:
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


class RegimeClustering(BaseFeature):
    """Regime Clustering using volatility levels"""

    def __init__(self):
        super().__init__("RegimeClustering", deps=["close"])

    def compute(self, df: pd.DataFrame, **params) -> pd.DataFrame:
        window = params.get('window', 20)
        vol_threshold = params.get('vol_threshold', 0.02)

        returns = df['close'].pct_change().fillna(0)
        volatility = returns.rolling(window).std().fillna(0)

        # Simple regime: 0=low vol, 1=high vol
        regime = (volatility > vol_threshold).astype(int)

        # One-hot
        df_copy = df.copy()
        df_copy['regime_cluster_0'] = (regime == 0).astype(float)
        df_copy['regime_cluster_1'] = (regime == 1).astype(float)

        return df_copy[['regime_cluster_0', 'regime_cluster_1']]


class KalmanFilter(BaseFeature):
    """Kalman Filter for price smoothing"""

    def __init__(self):
        super().__init__("KalmanFilter", deps=["close"])

    @staticmethod
    @jit(nopython=True)
    def _kalman_filter(prices, process_noise=1e-5, measurement_noise=1e-3):
        n = len(prices)
        xhat = np.zeros(n)  # state estimate
        P = np.zeros(n)     # error covariance
        xhat[0] = prices[0]
        P[0] = 1.0

        for i in range(1, n):
            # Predict
            xhat_minus = xhat[i-1]
            P_minus = P[i-1] + process_noise

            # Update
            K = P_minus / (P_minus + measurement_noise)
            xhat[i] = xhat_minus + K * (prices[i] - xhat_minus)
            P[i] = (1 - K) * P_minus

        return xhat

    def compute(self, df: pd.DataFrame, **params) -> pd.DataFrame:
        process_noise = params.get('process_noise', 1e-5)
        measurement_noise = params.get('measurement_noise', 1e-3)

        prices = df['close'].values
        filtered = self._kalman_filter(prices, process_noise, measurement_noise)

        residual = prices - filtered
        # Z-score of residual (rolling std, clip min)
        window = 20
        residual_series = pd.Series(residual)
        residual_std = residual_series.rolling(window).std().fillna(1e-6).clip(lower=1e-6).values
        zscore = residual / residual_std
        # Residual diff
        residual_diff = residual_series.diff().fillna(0).values

        df_copy = df.copy()
        df_copy['kalman_residual'] = residual
        df_copy['kalman_residual_diff'] = residual_diff
        df_copy['kalman_zscore'] = zscore

        return df_copy[['kalman_residual', 'kalman_residual_diff', 'kalman_zscore']]