"""
Kalman Filter features for price smoothing and prediction.

This module implements Kalman filter-based price smoothing and trend detection.
"""

from typing import Any, Optional, Tuple, Union

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from ztb.features.base import ComputableFeature
from ztb.features.registry import FeatureRegistry
from ztb.utils.errors import safe_operation


@FeatureRegistry.register("KalmanFilter")
def compute_kalman_filter(df: pd.DataFrame) -> pd.Series:
    """Kalman filter smoothed price"""
    feature = KalmanFilter()
    result_df = feature.compute(df)
    return result_df["kalman_price"]


@FeatureRegistry.register("KalmanVelocity")
def compute_kalman_velocity(df: pd.DataFrame) -> pd.Series:
    """Kalman filter velocity (trend strength)"""
    feature = KalmanFilter()
    result_df = feature.compute(df)
    return result_df["kalman_velocity"]


class KalmanFilter(ComputableFeature):
    """Kalman filter for price smoothing and trend detection"""

    def __init__(
        self,
        process_noise: float = 0.001,
        measurement_noise: float = 0.1,
        initial_velocity: float = 0.0
    ) -> None:
        super().__init__(
            "KalmanFilter",
            deps=["close"]
        )
        self.process_noise = process_noise
        self.measurement_noise = measurement_noise
        self.initial_velocity = initial_velocity

    def compute(self, df: pd.DataFrame) -> pd.DataFrame:
        """Compute Kalman filter features"""
        return safe_operation(
            logger=None,
            operation=lambda: self._compute_kalman_filter(df),
            context="kalman_filter_computation",
            default_result=pd.DataFrame(index=df.index, columns=["kalman_price", "kalman_velocity"])
        )

    def _compute_kalman_filter(self, df: pd.DataFrame) -> pd.DataFrame:
        """Implementation of Kalman filter computation"""
        prices = df["close"].values
        n = len(prices)

        if n == 0:
            return pd.DataFrame({
                "kalman_price": [],
                "kalman_velocity": []
            }, index=df.index)

        # Initialize Kalman filter state
        # State: [price, velocity]
        x = np.zeros((2, n))  # State estimate
        P = np.zeros((2, 2, n))  # State covariance

        # Initial state
        x[0, 0] = prices[0]  # Initial price estimate
        x[1, 0] = self.initial_velocity  # Initial velocity

        # Initial covariance (uncertainty)
        P[:, :, 0] = np.eye(2) * 1000

        # Process model: constant velocity
        # x_k = [1, 1; 0, 1] * x_{k-1} + noise
        F = np.array([[1, 1], [0, 1]])

        # Measurement model: we only measure price
        # z_k = [1, 0] * x_k + noise
        H = np.array([[1, 0]])

        # Process noise covariance
        Q = np.array([[self.process_noise, 0], [0, self.process_noise]])

        # Measurement noise covariance
        R = np.array([[self.measurement_noise]])

        # Run Kalman filter
        for k in range(1, n):
            # Prediction step
            x_pred = F @ x[:, k-1]
            P_pred = F @ P[:, :, k-1] @ F.T + Q

            # Update step
            y = prices[k] - H @ x_pred  # Measurement residual
            S = H @ P_pred @ H.T + R  # Residual covariance
            K = P_pred @ H.T @ np.linalg.inv(S)  # Kalman gain

            # Update state estimate
            x[:, k] = x_pred + K.flatten() * y
            # Update covariance
            P[:, :, k] = (np.eye(2) - K @ H) @ P_pred

        return pd.DataFrame({
            "kalman_price": x[0, :].astype(np.float32),
            "kalman_velocity": x[1, :].astype(np.float32)
        }, index=df.index)


@FeatureRegistry.register("KalmanTrend")
def compute_kalman_trend(df: pd.DataFrame) -> pd.Series:
    """Kalman filter based trend indicator"""
    feature = KalmanFilter()
    result_df = feature.compute(df)

    # Trend based on velocity sign and magnitude
    velocity = result_df["kalman_velocity"]
    trend = np.sign(velocity) * np.minimum(np.abs(velocity) * 100, 1.0)

    return trend.astype(np.float32)


@FeatureRegistry.register("KalmanResidual")
def compute_kalman_residual(df: pd.DataFrame) -> pd.Series:
    """Kalman filter residual (price - smoothed_price)"""
    feature = KalmanFilter()
    result_df = feature.compute(df)

    kalman_price = result_df["kalman_price"]
    actual_price = df["close"]

    residual = actual_price - kalman_price
    # Normalize by price for scale invariance
    residual_norm = residual / actual_price.replace(0, 1)

    return residual_norm.astype(np.float32)