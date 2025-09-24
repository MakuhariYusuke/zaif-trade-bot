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


# Moved to trend/ichimoku.py
# class Ichimoku(BaseFeature):
#     ...

# Moved to trend/donchian.py
# class Donchian(BaseFeature):
#     ...


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