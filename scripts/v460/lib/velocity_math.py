"""200# L: Velocity 計算ユーティリティ — SSOT (Single Source of Truth).

velocity 関連の数学的計算を一元化し、maker_price (VG) と skip_gate (SG) の
重複計算を排除する。

使い分けガイド:
  - VG velocity: 瞬間的 mid 変化 (cycle 間差分). maker_price._last_vg_velocity_bps
    → 短期 price action への即応に使用
  - SG velocity: 60s EMA (price_velocity_60s from gate_features)
    → skip/offset 判定の中期シグナルに使用

両者は **同一の符号規約** (正=上昇, 負=下降) を使用する。
"""

from __future__ import annotations


def compute_velocity_offset_multiplier(
    *,
    observed_velocity_bps: float,
    threshold_bps: float,
    base_multiplier: float,
    max_multiplier: float,
    proportional: bool,
) -> tuple[float, bool]:
    """velocity soft mode の offset 乗数を安全に解決する.

    SkipGateEvaluator._compute_velocity_offset_multiplier() から抽出。
    skip_gate_evaluator と maker_price の両方から使用可能。

    - 0 除算回避: threshold=0 の場合は proportional を使わず固定倍率へフォールバック
    - 保守性維持: 1.0 未満の倍率は無効化し、少なくとも現状維持 (1.0) に丸める
    - 上限暴走防止: max_multiplier で頭打ち

    Args:
        observed_velocity_bps: 観測された velocity (bps)
        threshold_bps: 閾値 (bps) — この値を超えた部分で boost が発動
        base_multiplier: 固定モード時の乗数 / 比例モードの基準
        max_multiplier: 乗数上限
        proportional: True=閾値超過量に比例, False=固定値

    Returns:
        (multiplier, was_proportional)
    """
    capped_max = max(1.0, float(max_multiplier))
    bounded_base = min(max(1.0, float(base_multiplier)), capped_max)
    if not proportional:
        return bounded_base, False

    threshold_abs = abs(float(threshold_bps))
    if threshold_abs <= 0.0:
        return bounded_base, False

    excess_ratio = abs(float(observed_velocity_bps)) / threshold_abs
    boost = 1.0 + (bounded_base - 1.0) * excess_ratio
    return min(boost, capped_max), True
