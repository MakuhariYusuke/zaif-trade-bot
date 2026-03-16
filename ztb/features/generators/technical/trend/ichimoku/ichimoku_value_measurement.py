"""
Ichimoku Value Measurement feature implementation.
Value Measurement analyzes price fluctuation measurements using Ichimoku components.

Output columns:
  - ichimoku_value_measurement: Price fluctuation measurement score
"""

from typing import Any

import numpy as np
import pandas as pd

from ztb.features.core.base import BaseFeature
from ztb.features.core.registry import FeatureRegistry

from .ichimoku_ext import calculate_ichimoku_extended

@FeatureRegistry.register("Ichimoku_Value_Measurement")
def compute_ichimoku_value_measurement(df: pd.DataFrame) -> pd.Series:
    """Ichimoku Value Measurement - price fluctuation measurement"""
    feature = IchimokuValueMeasurement()
    result_df = feature.compute(df)
    return result_df["ichimoku_value_measurement"]

class IchimokuValueMeasurement(BaseFeature):
    """
    Ichimoku Value Measurement feature.
    Analyzes price fluctuation measurements using Ichimoku components.
    """

    def __init__(self, **kwargs: Any):
        super().__init__("IchimokuValueMeasurement", deps=["high", "low", "close"])

    def compute(self, df: pd.DataFrame, **params: Any) -> pd.DataFrame:
        """
        df columns must include: ['high', 'low', 'close'].
        Returns a DataFrame with Value Measurement analysis.
        """
        # Get Ichimoku components
        ichimoku_data = calculate_ichimoku_extended(df)
        tenkan = ichimoku_data["ichimoku_tenkan"]
        kijun = ichimoku_data["ichimoku_kijun"]
        senkou_a = ichimoku_data["ichimoku_senkou_a"]
        senkou_b = ichimoku_data["ichimoku_senkou_b"]
        cloud_thickness = ichimoku_data["ichimoku_cloud_thickness"]

        # 1. Price range measurements
        price_range = df["high"] - df["low"]
        price_range_ma = price_range.rolling(20).mean()

        # 2. Ichimoku-based value measurements
        tenkan_range = tenkan.rolling(9).std() * 2  # 2 standard deviations
        kijun_range = kijun.rolling(26).std() * 2

        # 3. Cloud value zone
        cloud_center = (senkou_a + senkou_b) / 2
        cloud_value_range = cloud_thickness

        # 4. Price fluctuation ratios
        # How much price movement vs Ichimoku ranges
        price_tenkan_ratio = price_range / (
            tenkan_range + 0.001
        )  # Avoid division by zero
        price_kijun_ratio = price_range / (kijun_range + 0.001)
        price_cloud_ratio = price_range / (cloud_value_range + 0.001)

        # 5. Value measurement efficiency
        # How well Ichimoku captures price movements
        price_in_cloud = np.where(
            (df["close"] >= np.minimum(senkou_a, senkou_b))
            & (df["close"] <= np.maximum(senkou_a, senkou_b)),
            1,
            0,
        )

        # 6. Value zone utilization
        cloud_utilization = price_range_ma / (cloud_value_range + 0.001)
        cloud_utilization = np.clip(cloud_utilization, 0, 2)  # Cap at 2

        # 7. Measurement consistency
        # How consistent the value measurements are
        range_consistency = 1 - price_range.rolling(20).std() / (price_range_ma + 0.001)

        # 8. Composite value measurement score
        # Weight different aspects
        ratio_score = (price_tenkan_ratio + price_kijun_ratio + price_cloud_ratio) / 3
        ratio_norm = np.clip(ratio_score / 2, 0, 1)  # Normalize to 0-1

        utilization_score = np.where(
            cloud_utilization > 1, 1, cloud_utilization
        )  # 0-1 scale

        consistency_score = np.clip(range_consistency, 0, 1)

        # Final value measurement score
        value_measurement_score = (
            0.4 * ratio_norm
            + 0.3 * utilization_score  # Range ratios
            + 0.3 * consistency_score  # Cloud utilization  # Measurement consistency
        )

        # Adjust based on price position in cloud
        position_adjustment = np.where(price_in_cloud == 1, 0.1, -0.1)
        value_measurement_score += position_adjustment

        result_df = pd.DataFrame(
            {"ichimoku_value_measurement": value_measurement_score}, index=df.index
        )
        return result_df
