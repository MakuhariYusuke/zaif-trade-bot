"""
Ichimoku Momentum Confirmation feature implementation.
Momentum Confirmation analyzes lagging span momentum using Chikou Span analysis.

Output columns:
  - ichimoku_momentum_confirmation: Momentum confirmation score
"""

from typing import Any

import numpy as np
import pandas as pd

from ...base import BaseFeature
from ...registry import FeatureRegistry
from .ichimoku_ext import calculate_ichimoku_extended


@FeatureRegistry.register("Ichimoku_Momentum_Confirmation")
def compute_ichimoku_momentum_confirmation(df: pd.DataFrame) -> pd.Series:
    """Ichimoku Momentum Confirmation - lagging span momentum analysis"""
    feature = IchimokuMomentumConfirmation()
    result_df = feature.compute(df)
    return result_df["ichimoku_momentum_confirmation"]


class IchimokuMomentumConfirmation(BaseFeature):
    """
    Ichimoku Momentum Confirmation feature.
    Analyzes lagging span momentum using Chikou Span analysis.
    """

    def __init__(self, **kwargs: Any):
        super().__init__("IchimokuMomentumConfirmation", deps=["high", "low", "close"])

    def compute(self, df: pd.DataFrame, **params: Any) -> pd.DataFrame:
        """
        df columns must include: ['high', 'low', 'close'].
        Returns a DataFrame with Momentum Confirmation analysis.
        """
        # Get Ichimoku components
        ichimoku_data = calculate_ichimoku_extended(df)
        chikou = ichimoku_data["ichimoku_chikou"]
        tenkan = ichimoku_data["ichimoku_tenkan"]
        kijun = ichimoku_data["ichimoku_kijun"]

        # 1. Chikou momentum vs current price
        current_price = df["close"]
        chikou_vs_current = chikou - current_price

        # 2. Chikou trend direction
        chikou_trend = np.where(chikou > chikou.shift(1), 1, -1)

        # 3. Chikou slope (momentum strength)
        chikou_slope = chikou.diff(5)  # 5-period slope
        chikou_slope_norm = chikou_slope / (current_price * 0.01)  # Normalize by 1% of price

        # 4. Chikou confirmation with Tenkan/Kijun
        chikou_tenkan_diff = chikou - tenkan
        chikou_kijun_diff = chikou - kijun

        # 5. Momentum divergence detection
        # Compare Chikou momentum with price momentum
        price_momentum = current_price.pct_change(5)
        chikou_momentum = chikou.pct_change(5)

        momentum_divergence = np.where(
            (price_momentum > 0) & (chikou_momentum < 0), -1,  # Bearish divergence
            np.where((price_momentum < 0) & (chikou_momentum > 0), 1, 0)  # Bullish divergence
        )

        # 6. Chikou strength relative to ATR
        from ...volatility.atr import compute_atr_simplified
        atr_series = compute_atr_simplified(df)
        chikou_strength = abs(chikou_slope) / (atr_series + 0.001)  # Avoid division by zero

        # 7. Multi-timeframe momentum confirmation
        # Short-term (5-period), medium-term (13-period), long-term (21-period)
        chikou_ma5 = chikou.rolling(5).mean()
        chikou_ma13 = chikou.rolling(13).mean()
        chikou_ma21 = chikou.rolling(21).mean()

        # Alignment score: all MAs in same direction
        short_alignment = np.where(chikou > chikou_ma5, 1, -1)
        medium_alignment = np.where(chikou_ma5 > chikou_ma13, 1, -1)
        long_alignment = np.where(chikou_ma13 > chikou_ma21, 1, -1)

        alignment_score = (short_alignment + medium_alignment + long_alignment) / 3

        # 8. Composite momentum confirmation score
        # Weight different components
        position_score = np.where(chikou_vs_current > 0, 1, -1)
        slope_score = np.clip(chikou_slope_norm, -1, 1)
        strength_score = np.clip(chikou_strength, 0, 2) / 2  # 0-1 scale

        # Final momentum confirmation score
        momentum_confirmation_score = (
            0.3 * position_score +      # Chikou position vs current price
            0.2 * slope_score +         # Chikou slope strength
            0.2 * alignment_score +     # Multi-timeframe alignment
            0.2 * strength_score +      # Strength relative to volatility
            0.1 * momentum_divergence   # Divergence detection
        )

        result_df = pd.DataFrame({"ichimoku_momentum_confirmation": momentum_confirmation_score}, index=df.index)
        return result_df