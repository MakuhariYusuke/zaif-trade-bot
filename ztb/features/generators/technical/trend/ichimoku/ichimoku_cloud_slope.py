"""
Ichimoku Cloud Slope feature implementation.
Cloud Slope analyzes the angle and direction of the Ichimoku cloud movement.

Output columns:
  - ichimoku_cloud_slope: Cloud slope/angle measurement
"""

from typing import Any

import numpy as np
import pandas as pd

from ztb.features.core.base import BaseFeature
from ztb.features.core.registry import FeatureRegistry

from .ichimoku_ext import calculate_ichimoku_extended


@FeatureRegistry.register("Ichimoku_Cloud_Slope")
def compute_ichimoku_cloud_slope(df: pd.DataFrame) -> pd.Series:
    """Ichimoku Cloud Slope - cloud angle and direction analysis"""
    feature = IchimokuCloudSlope()
    result_df = feature.compute(df)
    return result_df["ichimoku_cloud_slope"]


class IchimokuCloudSlope(BaseFeature):
    """
    Ichimoku Cloud Slope feature.
    Analyzes the angle and direction of the Ichimoku cloud movement.
    """

    def __init__(self, **kwargs: Any):
        super().__init__("IchimokuCloudSlope", deps=["high", "low", "close"])

    def compute(self, df: pd.DataFrame, **params: Any) -> pd.DataFrame:
        """
        df columns must include: ['high', 'low', 'close'].
        Returns a DataFrame with Cloud Slope analysis.
        """
        # Get Ichimoku components
        ichimoku_data = calculate_ichimoku_extended(df)
        senkou_a = ichimoku_data["ichimoku_senkou_a"]
        senkou_b = ichimoku_data["ichimoku_senkou_b"]

        # Calculate cloud center and slope
        cloud_center = (senkou_a + senkou_b) / 2

        # Cloud slope using linear regression over different periods
        periods = [5, 10, 20]

        slope_scores = []
        for period in periods:
            if len(cloud_center) >= period:
                # Calculate slope using linear regression
                x = np.arange(period)
                y = cloud_center.iloc[-period:].values

                # Simple slope calculation (rise over run)
                slope = (y[-1] - y[0]) / period if period > 1 else 0

                # Normalize by average price to make it scale-invariant
                avg_price = df["close"].iloc[-period:].mean()
                norm_slope = (
                    slope / (avg_price + 0.001) * 1000
                )  # Scale up for visibility

                slope_scores.append(norm_slope)
            else:
                slope_scores.append(0)

        # Average slope across different periods
        avg_slope = np.mean(slope_scores)

        # Cloud angle (arctangent of slope, converted to degrees)
        cloud_angle = np.degrees(
            np.arctan(avg_slope / 100)
        )  # Scale down for reasonable angles

        # Slope direction and strength
        slope_direction = np.sign(avg_slope)
        slope_strength = np.clip(abs(avg_slope) / 10, 0, 1)  # 0-1 scale

        # Cloud rotation (how much the cloud is tilting)
        cloud_rotation = senkou_a - senkou_b
        rotation_slope = cloud_rotation.diff(5).rolling(10).mean()

        # Final cloud slope score
        # Combine slope, angle, and rotation
        cloud_slope_score = (
            0.5 * slope_direction * slope_strength
            + 0.3 * np.clip(cloud_angle / 45, -1, 1)  # Direction and strength
            + 0.2  # Angle component (-45 to 45 degrees)
            * np.sign(rotation_slope)
            * np.clip(abs(rotation_slope) / 10, 0, 1)
        )

        result_df = pd.DataFrame(
            {"ichimoku_cloud_slope": cloud_slope_score}, index=df.index
        )
        return result_df
