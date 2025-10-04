from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from ..base import BaseFeature
from ..registry import FeatureRegistry


@FeatureRegistry.register("Donchian_Pos_20")
def compute_donchian_pos_20(df: pd.DataFrame) -> pd.Series:
    """Donchian Channel Position (20-period)"""
    feature = Donchian()
    result_df = feature.compute(df, periods=[20])
    return result_df["donchian_pos_20"]


@FeatureRegistry.register("Donchian_Width_Rel_20")
def compute_donchian_width_rel_20(df: pd.DataFrame) -> pd.Series:
    """Donchian Channel Relative Width (20-period)"""
    feature = Donchian()
    result_df = feature.compute(df, periods=[20])
    return result_df["donchian_width_rel_20"]


@FeatureRegistry.register("Donchian_Slope_20")
def compute_donchian_slope_20(df: pd.DataFrame) -> pd.Series:
    """Donchian Channel Slope (20-period)"""
    feature = Donchian()
    result_df = feature.compute(df, periods=[20])
    return result_df["donchian_slope_20"]


class Donchian(BaseFeature):
    """Donchian Channel with normalized position and relative width"""

    def __init__(self) -> None:
        super().__init__("Donchian", deps=["high", "low", "close", "ATR_simplified"])

    @staticmethod
    def _compute_donchian(
        high: NDArray[np.floating[Any]],
        low: NDArray[np.floating[Any]],
        close: NDArray[np.floating[Any]],
        atr: NDArray[np.floating[Any]],
        period: int = 20,
    ) -> Tuple[NDArray[np.floating[Any]], NDArray[np.floating[Any]]]:
        """Compute Donchian Channel using pure numpy (no numba)"""
        n = len(high)
        upper = np.zeros(n)
        lower = np.zeros(n)

        for i in range(period - 1, n):
            upper[i] = np.max(high[i - period + 1 : i + 1])
            lower[i] = np.min(low[i - period + 1 : i + 1])

        middle = (upper + lower) / 2
        width = upper - lower
        position = np.zeros(n)
        width_rel = np.zeros(n)

        for i in range(n):
            if width[i] != 0:
                position[i] = (close[i] - middle[i]) / width[i]
            if atr[i] != 0:
                width_rel[i] = width[i] / atr[i]

        return position, width_rel

    def compute(
        self,
        df: pd.DataFrame,
        periods: Optional[List[int]] = None,
        **params: Dict[str, Any],
    ) -> pd.DataFrame:
        if periods is None:
            periods_raw = params.get("periods", [20, 55])
            periods = periods_raw if isinstance(periods_raw, list) else [20, 55]
        # Ensure ATR_simplified is available
        if "ATR_simplified" not in df.columns:
            from ztb.features.volatility.atr import compute_atr_simplified

            atr_series = compute_atr_simplified(df)
            df = df.copy()
            df["ATR_simplified"] = atr_series
        # Ensure inputs are numpy arrays
        high = np.asarray(df["high"].values, dtype=np.float64)
        low = np.asarray(df["low"].values, dtype=np.float64)
        close = np.asarray(df["close"].values, dtype=np.float64)
        atr = np.asarray(df["ATR_simplified"].values, dtype=np.float64)

        df_copy = df.copy()
        for period in periods:
            position, width_rel = self._compute_donchian(
                high, low, close, atr, int(period)
            )
            df_copy[f"donchian_pos_{period}"] = position
            df_copy[f"donchian_width_rel_{period}"] = width_rel
            # Slope: first-order difference of Donchian position (rate of change)
            pos_series = pd.Series(position)
            slope = pos_series.diff().fillna(0).values
            df_copy[f"donchian_slope_{period}"] = slope
        # 出力列
        output_cols = []
        for period in periods:
            output_cols.extend(
                [
                    f"donchian_pos_{period}",
                    f"donchian_slope_{period}",
                    f"donchian_width_rel_{period}",
                ]
            )

        return df_copy[output_cols]
