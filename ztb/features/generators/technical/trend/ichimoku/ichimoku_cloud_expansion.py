"""
Ichimoku Cloud Expansion feature implementation.
Cloud Expansion analyzes the expansion and contraction of the Ichimoku cloud.

Output columns:
  - ichimoku_cloud_expansion: Cloud expansion/contraction measurement
"""

from typing import Any

import numpy as np
import pandas as pd

from ztb.features.core.base import BaseFeature
from ztb.features.core.registry import FeatureRegistry

from .ichimoku_ext import calculate_ichimoku_extended

@FeatureRegistry.register("Ichimoku_Cloud_Expansion")
def compute_ichimoku_cloud_expansion(df: pd.DataFrame) -> pd.Series:
    """Ichimoku Cloud Expansion - cloud expansion and contraction analysis"""
    feature = IchimokuCloudExpansion()
    result_df = feature.compute(df)
    return result_df["ichimoku_cloud_expansion"]

class IchimokuCloudExpansion(BaseFeature):
    """
    Ichimoku Cloud Expansion feature.
    Analyzes the expansion and contraction of the Ichimoku cloud.
    """

    def __init__(self, **kwargs: Any):
        super().__init__("IchimokuCloudExpansion", deps=["high", "low", "close"])

    def compute(self, df: pd.DataFrame, **params: Any) -> pd.DataFrame:
        """
        df columns must include: ['high', 'low', 'close'].
        Returns a DataFrame with Cloud Expansion analysis.
        """
        # Get Ichimoku components
        ichimoku_data = calculate_ichimoku_extended(df)
        senkou_a = ichimoku_data["ichimoku_senkou_a"]
        senkou_b = ichimoku_data["ichimoku_senkou_b"]
        cloud_thickness = ichimoku_data["ichimoku_cloud_thickness"]

        # 1. Cloud expansion rate (change in thickness)
        thickness_change = cloud_thickness.pct_change(5)  # 5-period percentage change
        thickness_change_rate = cloud_thickness.diff(5) / 5  # Absolute change rate

        # 2. Cloud expansion momentum
        expansion_momentum = thickness_change_rate.rolling(10).mean()

        # 3. Cloud stability (consistency of thickness)
        thickness_volatility = cloud_thickness.rolling(20).std() / (
            cloud_thickness.rolling(20).mean() + 0.001
        )
        cloud_stability = 1 - np.clip(thickness_volatility, 0, 1)

        # 4. Expansion/contraction classification
        expansion_threshold = (
            cloud_thickness.rolling(20).mean() * 0.02
        )  # 2% of average thickness

        expansion_signal = pd.Series(
            np.where(
                thickness_change_rate > expansion_threshold,
                1,  # Expanding
                np.where(
                    thickness_change_rate < -expansion_threshold, -1, 0
                ),  # Contracting
            ),
            index=df.index,
        )

        # 5. Cloud breathing pattern (cyclical expansion/contraction)
        # Detect if cloud is in expansion or contraction phase
        thickness_ma5 = cloud_thickness.rolling(5).mean()
        thickness_ma20 = cloud_thickness.rolling(20).mean()

        breathing_phase = np.where(
            thickness_ma5 > thickness_ma20,
            1,  # Expansion phase
            np.where(thickness_ma5 < thickness_ma20, -1, 0),  # Contraction phase
        )

        # 6. Cloud expansion strength
        avg_thickness = cloud_thickness.rolling(20).mean()
        expansion_strength = thickness_change_rate / (avg_thickness + 0.001)
        expansion_strength_norm = np.clip(
            expansion_strength * 10, -2, 2
        )  # Scale and clip

        # 7. Cloud expansion trend
        expansion_trend = expansion_signal.rolling(10).mean()  # Smoothed trend

        # 8. Composite cloud expansion score
        # Combine all expansion metrics
        expansion_score = (
            0.3 * expansion_signal
            + 0.2 * breathing_phase  # Current expansion/contraction
            + 0.2 * expansion_strength_norm  # Breathing phase
            + 0.2 * expansion_trend  # Strength of expansion
            + 0.1 * cloud_stability  # Trend direction  # Stability factor
        )

        # Normalize to -1 to 1 scale
        cloud_expansion_score = np.clip(expansion_score, -1, 1)

        result_df = pd.DataFrame(
            {"ichimoku_cloud_expansion": cloud_expansion_score}, index=df.index
        )
        return result_df
