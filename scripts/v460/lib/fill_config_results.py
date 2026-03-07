"""
329# fill_config_results — Result dataclasses + EV offset utility.

328# God Object 分割 Step 1: fill_config.py から Result classes を分離.
SkipGateResult, FillMonitorResult, PnlMeasurement, compute_ev_offset_multiplier()
を独立モジュール化し、fill_config.py の肥大化を緩和する。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ztb.metrics.fill_quality import FillRecord


# ======================================================================
# 113# R1: run_single_cycle 分割用 内部データクラス
# ======================================================================

@dataclass
class SkipGateResult:
    """SkipGate ML 判定結果 (run_single_cycle 内部)."""

    skipped: bool | None = None
    score: float | None = None
    reason: str | None = None
    model_used: str | None = None
    as_prob: float | None = None
    threshold_used: float | None = None
    # 158# P1-6: 時間帯別閾値調整のオフセット
    hour_offset: float = 0.0
    # 165# AS-R1: velocity logging
    price_velocity_bps: float | None = None
    early_return_record: FillRecord | None = None
    # 193#: ev_weighted score (offset 修飾子用)
    ev_score: float | None = None
    # 195#: velocity_skip ソフトモード — offset boost 倍率
    velocity_offset_mult: float | None = None


@dataclass
class FillMonitorResult:
    """約定監視結果 (run_single_cycle 内部)."""

    filled: bool = False
    fill_price: float | None = None
    t_fill: float | None = None
    cancel_reason: str | None = None
    queue_wait: float = 0.0
    reprice_count: int = 0
    # 158# P1-3: reprice 累積 drift (bps)
    reprice_drift_bps: float = 0.0
    final_order_price: float = 0.0
    # 145# §9-#2: regime 調整済みの実効タイムアウト (cancel_reason 判定で使用)
    effective_timeout: float = 0.0
    # 166# C.7: cancel 失敗後に約定を検出した場合のフラグ (Bug11)
    cancel_failed_likely_filled: bool = False
    # 237# phantom position guard: status_unknown 時の注文 ID (遅延照合用)
    order_id_for_reconciliation: str | None = None


@dataclass
class PnlMeasurement:
    """PnL 計測結果 (run_single_cycle 内部)."""

    mid_at_fill: float | None = None
    mid_30s_after: float | None = None
    mid_60s_after: float | None = None
    mid_120s_after: float | None = None
    post_fill_pnl: float | None = None
    post_fill_60s_pnl: float | None = None
    post_fill_120s_pnl: float | None = None
    adverse_selected: bool | None = None
    adverse_selected_raw: bool | None = None
    actual_measurement_sec: float | None = None
    # 120# PnlMeasurer: early_exit_triggered を戻り値に含める
    early_exit_triggered: bool = False
    # 120# A4-2: EE 発動時の中断時点 PnL (post_fill_pnl は常に固定30s)
    pnl_at_exit_bps: float | None = None
    # 305# Execution Quality 分解 (Kissell & Glantz 2003):
    #   PnL = spread_capture + adverse_selection_cost
    #   spread_capture: fill_price vs mid_at_fill (MM の付加価値)
    #   adverse_selection_cost: mid_at_fill vs mid_after (情報コスト)
    spread_capture_bps: float | None = None
    adverse_selection_cost_bps: float | None = None


# ======================================================================
# 200# M: ev_offset 計算ユーティリティ (DRY — executor と evaluator で共通使用)
# ======================================================================
def compute_ev_offset_multiplier(
    *,
    ev_score: float,
    sensitivity: float,
    min_mult: float,
    max_mult: float,
    warning_threshold: float = -4.0,
    warning_factor: float = 1.0,
) -> float:
    """ev_score → offset 乗数の共通計算.

    Args:
        ev_score: EV スコア (正=有利, 負=不利)
        sensitivity: ev_score → mult 感度 (mult = 1.0 + sensitivity × ev_score)
        min_mult, max_mult: クランプ範囲
        warning_threshold: この値未満で warning zone (追加保守化)
        warning_factor: warning zone での追加乗数 (< 1.0 で保守的)

    Returns:
        クランプ済み offset 乗数
    """
    raw = 1.0 + sensitivity * ev_score
    mult = max(min_mult, min(max_mult, raw))
    # 200# M: warning zone — emergency ではないが低 EV → 追加保守化
    if warning_factor != 1.0 and ev_score < warning_threshold:
        mult *= warning_factor
        mult = max(min_mult, min(max_mult, mult))
    return mult
