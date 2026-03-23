"""571# ロバスト統計ユーティリティ (eDRC 入力保護用).

eDRC への入力となる σ や OFI をノイズから保護するための
高速・頑健な計算モジュール。ステートレス設計でメモリリーク回避。

参照: 570# (ロバスト入力設計), 568# (eDRC 数理仕様)
"""

from __future__ import annotations

import numpy as np


class RobustStats:
    """頑健な統計計算ユーティリティ (メモリ効率・低遅延重視)."""

    __slots__ = ()  # インスタンス化不要のクラス

    @staticmethod
    def clip_outliers_mad(data: np.ndarray, threshold: float = 3.0) -> np.ndarray:
        """MAD (Median Absolute Deviation) ベースの外れ値クリッピング.

        MAD = median(|x_i - median(x)|) を基準とし、
        [median - threshold * MAD, median + threshold * MAD] にクリップ。
        全データ同一値 (MAD=0) の場合はそのまま返す。
        """
        if len(data) == 0:
            return data
        median = np.median(data)
        mad = np.median(np.abs(data - median))
        if mad == 0:
            return data

        lower = median - threshold * mad
        upper = median + threshold * mad
        return np.clip(data, lower, upper)

    @staticmethod
    def robust_ema(
        current_val: float,
        prev_ema: float,
        alpha: float,
        sigma_limit: float | None = None,
    ) -> float:
        """入力クリッピング付き EMA.

        急激なスパイクによる指数移動平均の跳ね上がりを抑制する。
        sigma_limit 指定時、前回 EMA からの乖離が大きい入力を制限値でクリップ。
        """
        if sigma_limit is not None and abs(current_val - prev_ema) > sigma_limit:
            sign = 1.0 if current_val > prev_ema else -1.0
            clipped_val = prev_ema + sign * sigma_limit
            return alpha * clipped_val + (1.0 - alpha) * prev_ema

        return alpha * current_val + (1.0 - alpha) * prev_ema

    @staticmethod
    def asymmetric_ema(
        current_val: float,
        prev_ema: float,
        alpha_up: float,
        alpha_down: float,
    ) -> float:
        """非対称 EMA. 反転（価格逆行）に対する感度を高める設計.

        current_val > prev_ema → alpha_up (上昇追従)
        current_val <= prev_ema → alpha_down (下落追従、通常 alpha_down > alpha_up)
        """
        alpha = alpha_up if current_val > prev_ema else alpha_down
        return alpha * current_val + (1.0 - alpha) * prev_ema

    @staticmethod
    def median_filter_fast(buffer: np.ndarray) -> float:
        """過去 N 件の中央値を高速に算出 (OFI 等のノイズ除去用)."""
        return float(np.median(buffer))
