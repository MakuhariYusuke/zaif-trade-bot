"""
Ichimoku Wave Theory feature implementation.
Wave Theory analyzes cloud wave patterns and momentum using Senkou Span A and B.

Output columns:
  - ichimoku_wave_theory: Wave pattern analysis score
"""

from typing import Any

import numpy as np
import pandas as pd

from ztb.features.core.base import BaseFeature
from ztb.features.core.registry import FeatureRegistry

from .ichimoku_ext import calculate_ichimoku_extended


@FeatureRegistry.register("Ichimoku_Wave_Theory")
def compute_ichimoku_wave_theory(df: pd.DataFrame) -> pd.Series:
    """Ichimoku Wave Theory - cloud wave patterns and momentum"""
    feature = IchimokuWaveTheory()
    result_df = feature.compute(df)
    return result_df["ichimoku_wave_theory"]


class IchimokuWaveTheory(BaseFeature):
    """
    Ichimoku Wave Theory feature.
    Analyzes cloud wave patterns and momentum using Senkou Span A and B.
    """

    def __init__(self, **kwargs: Any):
        super().__init__("IchimokuWaveTheory", deps=["high", "low", "close"])

    def compute(self, df: pd.DataFrame, **params: Any) -> pd.DataFrame:
        """
        df columns must include: ['high', 'low', 'close'].
        Returns a DataFrame with Wave Theory analysis.
        """
        # Get Ichimoku components
        ichimoku_data = calculate_ichimoku_extended(df)
        # Ensure all data is aligned with df length
        ichimoku_data = ichimoku_data.iloc[: len(df)]
        senkou_a = ichimoku_data["ichimoku_senkou_a"].iloc[: len(df)]
        senkou_b = ichimoku_data["ichimoku_senkou_b"].iloc[: len(df)]
        cloud_thickness = ichimoku_data["ichimoku_cloud_thickness"].iloc[: len(df)]

        # 1. Cloud wave momentum (rate of change of cloud edges)
        senkou_a_momentum = senkou_a.pct_change(5)  # 5-period momentum
        senkou_b_momentum = senkou_b.pct_change(10)  # 10-period momentum (slower)

        # 2. Cloud wave direction (which span is leading)
        wave_direction = np.where(
            senkou_a > senkou_b,
            np.where(senkou_a_momentum > senkou_b_momentum, 1, 0.5),  # Strong bullish
            np.where(senkou_b_momentum > senkou_a_momentum, -1, -0.5),  # Strong bearish
        )

        # 3. Cloud wave amplitude (normalized thickness change)
        thickness_change = cloud_thickness.pct_change(5)
        wave_amplitude = np.clip(thickness_change * 10, -1, 1)  # Scale and clip

        # 5. Cloud wave frequency (how often the cloud changes direction)
        cloud_color = np.where(senkou_a > senkou_b, 1, -1)
        color_changes = np.abs(np.diff(cloud_color, prepend=cloud_color[0])) / 2
        wave_frequency = pd.Series(color_changes).rolling(20).sum()

        # 5. Wave convergence/divergence
        span_divergence = (senkou_a - senkou_b) / cloud_thickness.replace(0, np.nan)
        span_divergence = span_divergence.fillna(0)

        # 6. Composite wave theory score
        # Combine momentum, direction, amplitude, and frequency
        momentum_score = (senkou_a_momentum + senkou_b_momentum) / 2
        momentum_norm = np.clip(momentum_score * 5, -1, 1)  # Normalize

        frequency_norm = np.clip(wave_frequency / 5, 0, 1)  # 0-1 scale

        # Wave strength: combination of direction strength and momentum
        wave_strength = wave_direction * (1 + momentum_norm) * (1 + wave_amplitude)

        # Apply frequency damping (higher frequency = weaker signal)
        wave_damping = 1 / (1 + frequency_norm)
        wave_damping = pd.Series(
            wave_damping.values, index=df.index
        )  # Ensure same index as df

        # Final wave theory score
        wave_theory_score = wave_strength * wave_damping

        # Add convergence/divergence component
        convergence_factor = np.where(
            span_divergence > 0.5, 0.2, np.where(span_divergence < -0.5, -0.2, 0)
        )
        # Ensure convergence_factor has the same length as df
        convergence_factor = convergence_factor[: len(df)]
        wave_theory_score += convergence_factor

        result_df = pd.DataFrame(
            {"ichimoku_wave_theory": wave_theory_score}, index=df.index
        )
        return result_df
