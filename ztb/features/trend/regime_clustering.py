"""
Regime Clustering features for market regime detection.

This module implements unsupervised clustering-based market regime detection
using volatility and trend strength indicators.
"""

from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

from ztb.features.registry import FeatureRegistry
from ztb.utils.errors import safe_operation

import logging

logger = logging.getLogger(__name__)


@FeatureRegistry.register("RegimeClustering")
def compute_regime_clustering(df: pd.DataFrame) -> pd.Series:
    """Market regime clustering based on volatility and trend strength"""
    feature = RegimeClustering()
    result_df = feature.compute(df)
    return result_df["regime_cluster"]


class RegimeClustering:
    """Market regime detection using clustering of volatility and trend indicators"""

    def __init__(self) -> None:
        self.n_clusters = 3
        self.lookback_window = 20
        self._scaler: Optional[StandardScaler] = None
        self._kmeans: Optional[KMeans] = None
        self._is_fitted = False

    def compute(self, df: pd.DataFrame) -> pd.DataFrame:
        """Compute regime clustering features"""
        return safe_operation(
            logger=None,
            operation=lambda: self._compute_regime_clustering(df),
            context="regime_clustering_computation",
            default_result=pd.DataFrame(index=df.index, columns=["regime_cluster"])
        )

    def _compute_regime_clustering(self, df: pd.DataFrame) -> pd.DataFrame:
        """Implementation of regime clustering computation"""
        if len(df) < self.lookback_window:
            # Not enough data, return neutral regime (cluster 1)
            return pd.DataFrame({
                "regime_cluster": [1] * len(df)
            }, index=df.index)

        # Calculate regime indicators
        volatility = self._calculate_volatility_regime(df)
        trend_strength = self._calculate_trend_regime(df)
        volume_regime = self._calculate_volume_regime(df)

        # Combine features for clustering
        features = np.column_stack([
            volatility,
            trend_strength,
            volume_regime
        ])

        # Handle NaN values
        features = np.nan_to_num(features, nan=0.0)

        # Fit clustering model if not already fitted
        if not self._is_fitted:
            self._fit_clustering_model(features)
            self._is_fitted = True

        # Scale features
        if self._scaler is not None and self._is_fitted:
            features_scaled = self._scaler.transform(features)
        else:
            features_scaled = features

        # Predict clusters
        if self._kmeans is not None and self._is_fitted:
            clusters = self._kmeans.predict(features_scaled)
        else:
            clusters = np.ones(len(df), dtype=int)

        return pd.DataFrame({
            "regime_cluster": clusters.astype(np.int8)  # Use int8 for cluster labels (0-2)
        }, index=df.index)

    def _calculate_volatility_regime(self, df: pd.DataFrame) -> np.ndarray[Any, np.dtype[Any]]:
        """Calculate volatility-based regime indicator"""
        # Use ATR normalized by price
        atr = df["atr_14"].bfill().fillna(0)
        price = df["close"]
        volatility = atr / price

        # Smooth with rolling mean
        volatility_smooth = volatility.rolling(window=self.lookback_window, min_periods=1).mean()

        return np.asarray(volatility_smooth.values)

    def _calculate_trend_regime(self, df: pd.DataFrame) -> np.ndarray[Any, np.dtype[Any]]:
        """Calculate trend strength regime indicator"""
        # Combine RSI and MACD histogram for trend strength
        rsi = df["rsi_14"].fillna(50)
        macd_hist = df["macd_hist"].fillna(0)

        # Normalize RSI deviation from 50
        rsi_deviation = (rsi - 50).abs() / 50

        # Use absolute MACD histogram as momentum indicator
        momentum = macd_hist.abs()

        # Combine indicators
        trend_strength = (rsi_deviation + momentum) / 2

        # Smooth with rolling mean
        trend_smooth = trend_strength.rolling(window=self.lookback_window, min_periods=1).mean()

        return np.asarray(trend_smooth.values)

    def _calculate_volume_regime(self, df: pd.DataFrame) -> np.ndarray[Any, np.dtype[Any]]:
        """Calculate volume-based regime indicator"""
        volume = df["volume"].fillna(0)

        # Calculate volume relative to recent average
        volume_ma = volume.rolling(window=self.lookback_window, min_periods=1).mean()
        volume_ratio = volume / volume_ma.replace(0, 1)  # Avoid division by zero

        # Smooth the ratio
        volume_smooth = volume_ratio.rolling(window=self.lookback_window, min_periods=1).mean()

        return np.asarray(volume_smooth.values)

    def _fit_clustering_model(self, features: np.ndarray[Any, np.dtype[Any]]) -> None:
        """Fit the clustering model using historical data"""
        try:
            # Use only non-NaN data for fitting
            valid_mask = ~np.isnan(features).any(axis=1)
            valid_features = features[valid_mask]

            if len(valid_features) >= self.n_clusters:
                # Standardize features
                self._scaler = StandardScaler()
                features_scaled = self._scaler.fit_transform(valid_features)

                # Fit K-means clustering
                self._kmeans = KMeans(
                    n_clusters=self.n_clusters,
                    random_state=42,
                    n_init=10
                )
                self._kmeans.fit(features_scaled)
                self._is_fitted = True
            else:
                # Not enough data, use simple clustering
                self._scaler = None
                self._kmeans = None
                self._is_fitted = False

        except Exception:
            # Fallback to no clustering
            self._scaler = None
            self._kmeans = None
            self._is_fitted = False