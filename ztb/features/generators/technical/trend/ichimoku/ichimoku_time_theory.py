"""
Ichimoku Time Theory feature implementation.
Time Theory analyzes temporal relationships between Tenkan-sen and Kijun-sen,
measuring the duration and frequency of cross signals.

Output columns:
  - ichimoku_time_theory: Time-based analysis score
"""

from typing import Any

import numpy as np
import pandas as pd

from ztb.features.core.base import BaseFeature
from ztb.features.core.registry import FeatureRegistry

from .ichimoku_ext import calculate_ichimoku_extended


@FeatureRegistry.register("Ichimoku_Time_Theory")
def compute_ichimoku_time_theory(df: pd.DataFrame) -> pd.Series:
    """Ichimoku Time Theory - temporal relationships between Tenkan and Kijun"""
    feature = IchimokuTimeTheory()
    result_df = feature.compute(df)
    return result_df["ichimoku_time_theory"]


class IchimokuTimeTheory(BaseFeature):
    """
    Ichimoku Time Theory feature.
    Analyzes temporal relationships between Tenkan-sen and Kijun-sen.
    """

    def __init__(self, **kwargs: Any) -> None:
        super().__init__("IchimokuTimeTheory", deps=["high", "low", "close"])

    def compute(self, df: pd.DataFrame, **params: Any) -> pd.DataFrame:
        """
        df columns must include: ['high', 'low', 'close'].
        Returns a DataFrame with Time Theory analysis.
        """
        # Get basic Ichimoku components
        ichimoku_data = calculate_ichimoku_extended(df)
        tenkan = ichimoku_data["ichimoku_tenkan"]
        kijun = ichimoku_data["ichimoku_kijun"]

        # Calculate cross signals
        tk_cross = np.where(tenkan > kijun, 1, -1)

        # 1. Duration of current trend (how long Tenkan has been above/below Kijun)
        trend_duration = pd.Series(index=df.index, dtype=float)
        current_trend = 0
        duration_count = 0

        for i in range(len(tk_cross)):
            if tk_cross[i] != current_trend:
                current_trend = tk_cross[i]
                duration_count = 1
            else:
                duration_count += 1
            trend_duration.iloc[i] = duration_count

        # 2. Cross frequency (rolling count of crosses in last N periods)
        cross_changes = (
            np.abs(np.diff(tk_cross, prepend=tk_cross[0])) / 2
        )  # 1 when cross occurs, 0 otherwise
        cross_frequency = pd.Series(cross_changes, index=df.index).rolling(50).sum()

        # 3. Time since last cross
        last_cross_idx = -1
        time_since_cross = pd.Series(index=df.index, dtype=float)

        for i in range(len(cross_changes)):
            if cross_changes[i] > 0:
                last_cross_idx = i
            if last_cross_idx >= 0:
                time_since_cross.iloc[i] = i - last_cross_idx

        # 4. Trend stability score (consistency of direction)
        trend_stability = trend_duration / (
            time_since_cross + 1
        )  # Avoid division by zero

        # 5. Composite time theory score
        # Normalize components to 0-1 scale
        duration_norm = np.clip(trend_duration / 50, 0, 1)  # Max 50 periods
        frequency_norm = np.clip(
            cross_frequency / 10, 0, 1
        )  # Max 10 crosses in 50 periods
        stability_norm = np.clip(trend_stability, 0, 1)

        # Weight the components (duration 40%, stability 40%, frequency 20%)
        time_theory_score = (
            0.4 * duration_norm
            + 0.4 * stability_norm
            + 0.2 * (1 - frequency_norm)  # Lower frequency = more stable
        )

        # Adjust based on current trend direction
        time_theory_score = time_theory_score * tk_cross

        result_df = pd.DataFrame(
            {"ichimoku_time_theory": time_theory_score}, index=df.index
        )
        return result_df
