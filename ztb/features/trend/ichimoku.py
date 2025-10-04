from typing import Any, Tuple, Union

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from ztb.features.base import ChannelFeature, ComputableFeature
from ztb.features.registry import FeatureRegistry


@FeatureRegistry.register("Ichimoku_Diff_Norm")
def compute_ichimoku_diff_norm(df: pd.DataFrame) -> pd.Series:
    """Ichimoku Cloud Normalized Difference (Tenkan - Kijun)"""
    feature = Ichimoku()
    result_df = feature.compute(df)
    return result_df["ichimoku_diff_norm"]


@FeatureRegistry.register("Ichimoku_Cross")
def compute_ichimoku_cross(df: pd.DataFrame) -> pd.Series:
    """Ichimoku Cloud Cross Signal (1 if Tenkan > Kijun, 0 otherwise)"""
    feature = Ichimoku()
    result_df = feature.compute(df)
    return result_df["ichimoku_cross"]


class Ichimoku(ChannelFeature, ComputableFeature):
    """Ichimoku Cloud with normalized diff and cross signal"""

    def __init__(self) -> None:
        super().__init__("Ichimoku", deps=["high", "low", "close", "ATR_simplified"])
        self._required_calculations: set[str] = set()  # Internal calculation only

    @staticmethod
    def _compute_ichimoku(
        high: Any,
        low: Any,
        close: Any,
    ) -> Tuple[Any, Any, Any, Any, Any]:
        """Compute Ichimoku components using pure numpy (no numba)"""
        if len(high) != len(low) or len(high) != len(close):
            raise ValueError(
                "Input arrays 'high', 'low', and 'close' must have the same length."
            )
        n = len(high)
        tenkan = np.zeros(n)
        kijun = np.zeros(n)
        senkou_a = np.zeros(n)
        senkou_b = np.zeros(n)
        chikou = np.zeros(n)

        # Calculate Tenkan-sen (Conversion Line): (9-period high + 9-period low) / 2
        for i in range(9, n):
            tenkan[i] = (np.max(high[i - 9 : i + 1]) + np.min(low[i - 9 : i + 1])) / 2

        # Calculate Kijun-sen (Base Line): (26-period high + 26-period low) / 2
        for i in range(26, n):
            kijun[i] = (np.max(high[i - 26 : i + 1]) + np.min(low[i - 26 : i + 1])) / 2

        # Calculate Senkou Span A (Leading Span A): (Tenkan + Kijun) / 2, plotted 26 periods ahead
        for i in range(26, n):
            senkou_a[i] = (tenkan[i - 26] + kijun[i - 26]) / 2

        # Calculate Senkou Span B (Leading Span B): (52-period high + 52-period low) / 2, plotted 26 periods ahead
        for i in range(52, n):
            senkou_b[i] = (
                np.max(high[i - 52 : i - 26 + 1]) + np.min(low[i - 52 : i - 26 + 1])
            ) / 2

        # Calculate Chikou Span (Lagging Span): Current close plotted 26 periods back
        for i in range(26, n):
            chikou[i] = close[i - 26]

        return tenkan, kijun, senkou_a, senkou_b, chikou

    def compute(self, df: pd.DataFrame) -> pd.DataFrame:
        high = df["high"].values
        low = df["low"].values
        close = df["close"].values

        tenkan, kijun, _, _, _ = self._compute_ichimoku(high, low, close)

        # 差分正規化
        diff = tenkan - kijun
        if "ATR_simplified" not in df.columns:
            from ztb.features.volatility.atr import compute_atr_simplified

            atr_series = compute_atr_simplified(df)
            df = df.copy()
            df["ATR_simplified"] = atr_series
        atr = df["ATR_simplified"].to_numpy(dtype=float)
        normalized_diff = np.where(atr != 0, diff / atr, 0)
        # クロスシグナル
        cross = (tenkan > kijun).astype(float)

        df_copy = df.copy()
        df_copy["ichimoku_diff_norm"] = normalized_diff
        df_copy["ichimoku_cross"] = cross

        return df_copy[["ichimoku_diff_norm", "ichimoku_cross"]]
