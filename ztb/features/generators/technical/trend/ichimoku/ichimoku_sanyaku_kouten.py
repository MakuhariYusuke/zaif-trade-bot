"""
Ichimoku Sanyaku Kouten feature implementation.
Sanyaku Kouten (Three Roles Reversal) analyzes the classic Ichimoku reversal pattern.

Output columns:
  - ichimoku_sanyaku_kouten: Three roles reversal signal
"""

from typing import Any

import numpy as np
import pandas as pd

from ztb.features.core.base import BaseFeature
from ztb.features.core.registry import FeatureRegistry

from .ichimoku_ext import calculate_ichimoku_extended


@FeatureRegistry.register("Ichimoku_Sanyaku_Kouten")
def compute_ichimoku_sanyaku_kouten(df: pd.DataFrame) -> pd.Series:
    """Ichimoku Sanyaku Kouten - three roles reversal analysis"""
    feature = IchimokuSanyakuKouten()
    result_df = feature.compute(df)
    return result_df["ichimoku_sanyaku_kouten"]


class IchimokuSanyakuKouten(BaseFeature):
    """
    Ichimoku Sanyaku Kouten (Three Roles Reversal) feature.
    Analyzes the classic Ichimoku reversal pattern combining price, cloud, and lines.
    """

    def __init__(self, **kwargs: Any):
        super().__init__("IchimokuSanyakuKouten", deps=["high", "low", "close"])

    def compute(self, df: pd.DataFrame, **params: Any) -> pd.DataFrame:
        """
        df columns must include: ['high', 'low', 'close'].
        Returns a DataFrame with Sanyaku Kouten analysis.
        """
        # Get Ichimoku components
        ichimoku_data = calculate_ichimoku_extended(df)
        tenkan = ichimoku_data["ichimoku_tenkan"]
        kijun = ichimoku_data["ichimoku_kijun"]
        senkou_a = ichimoku_data["ichimoku_senkou_a"]
        senkou_b = ichimoku_data["ichimoku_senkou_b"]
        chikou = ichimoku_data["ichimoku_chikou"]

        current_price = df["close"]

        # Three roles for bullish reversal (Sanyaku Kouten):
        # 1. Price above cloud
        price_above_cloud = current_price > np.maximum(senkou_a, senkou_b)

        # 2. Tenkan above Kijun
        tenkan_above_kijun = tenkan > kijun

        # 3. Chikou above price (26 periods ago)
        price_26_ago = current_price.shift(26)
        chikou_above_price = chikou > price_26_ago

        # Bullish Sanyaku Kouten (all three conditions met)
        bullish_kouten = price_above_cloud & tenkan_above_kijun & chikou_above_price

        # Three roles for bearish reversal (Sanyaku Gyaku):
        # 1. Price below cloud
        price_below_cloud = current_price < np.minimum(senkou_a, senkou_b)

        # 2. Tenkan below Kijun
        tenkan_below_kijun = tenkan < kijun

        # 3. Chikou below price (26 periods ago)
        chikou_below_price = chikou < price_26_ago

        # Bearish Sanyaku Gyaku (all three conditions met)
        bearish_gyaku = price_below_cloud & tenkan_below_kijun & chikou_below_price

        # Calculate strength of each role
        # Price-cloud distance normalized
        cloud_center = (senkou_a + senkou_b) / 2
        cloud_thickness = abs(senkou_a - senkou_b)
        price_cloud_distance = (current_price - cloud_center) / (
            cloud_thickness + 0.001
        )
        price_strength = np.clip(price_cloud_distance, -2, 2) / 2  # -1 to 1

        # Tenkan-Kijun distance normalized
        tk_distance = (tenkan - kijun) / (
            kijun * 0.01 + 0.001
        )  # 1% of Kijun as reference
        tk_strength = np.clip(tk_distance, -2, 2) / 2

        # Chikou momentum strength
        chikou_momentum = chikou.diff(5) / (
            current_price * 0.005 + 0.001
        )  # 0.5% of price
        chikou_strength = np.clip(chikou_momentum, -2, 2) / 2

        # Composite Sanyaku Kouten score
        # Base score from pattern completion
        base_score = np.where(bullish_kouten, 1, np.where(bearish_gyaku, -1, 0))

        # Strength multiplier
        strength_multiplier = (
            abs(price_strength) + abs(tk_strength) + abs(chikou_strength)
        ) / 3

        # Final Sanyaku Kouten score
        sanyaku_kouten_score = base_score * (0.5 + 0.5 * strength_multiplier)

        # Add trend confirmation (recent crosses strengthen the signal)
        recent_tk_cross = (
            (tenkan > kijun).rolling(5).mean()
        )  # % of time TK was bullish recently
        trend_confirmation = (recent_tk_cross - 0.5) * 2  # -1 to 1 scale

        sanyaku_kouten_score = sanyaku_kouten_score * (1 + 0.2 * trend_confirmation)

        result_df = pd.DataFrame(
            {"ichimoku_sanyaku_kouten": sanyaku_kouten_score}, index=df.index
        )
        return result_df
