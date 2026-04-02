"""
G1.1-exec Fill Quality Metrics — 009# §4.3 準拠.

maker 注文の約定品質を評価する指標の算出と Gate 判定を提供する。

E1: fill_rate P90  — 日別 fill rate の 90th percentile
E2: cancel_ratio   — 未約定キャンセル率
E3: queue_wait_median_sec — 発注→約定 の待ち時間中央値
E4: post_fill_30s_pnl    — 約定後 30 秒の mid 価格変動平均 + 片側 t 検定
E5: adverse_selection_ratio — 約定後 30 秒で逆行した注文の割合
"""

from __future__ import annotations

import logging
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Final, Iterable, Iterator, Mapping, Sequence, cast

import numpy as np
from ztb.io.jsonl import iter_jsonl_objects
from ztb.metrics.fill_metrics_core import (
    format_utc_day,
    mean_and_one_sided_pvalue,
    scan_fill_metric_inputs,
)
from ztb.metrics.fill_judgment_core import (
    build_exec_gate_checks,
    build_full_gate_pnl_checks,
    build_full_gate_structural_checks,
    build_gate_payload,
    build_quick_gate_checks,
    build_quick_watch_detail,
    resolve_exec_judgment_type,
    resolve_gate_result,
)
from ztb.metrics.fill_exec_monitoring import build_exec_monitoring_checks
from ztb.metrics.fill_group_metrics import (
    GroupedMetricsBase,
    HourlyMetrics,
    RegimeMetrics,
    compute_hourly_metrics,
    compute_regime_metrics,
)
from ztb.metrics.fill_record_integrity import (
    detect_split_brain,
    filter_clean_records,
    partition_clean_records,
    quarantine_reason as _quarantine_reason,
)
from ztb.metrics.pnl_accumulators import PnlAccumulator, PnlWinAccumulator
from ztb.metrics.fill_record_io import (
    apply_fill_record_filters,
    fill_records_to_dataframe,
    iter_fill_record_dicts,
    iter_fill_record_objects_from_files,
    iter_fill_record_objects_glob,
    iter_fill_records,
    iter_fill_records_glob,
    list_fill_record_files,
    load_fill_record_objects_glob,
    load_fill_records,
    load_fill_records_glob,
    save_fill_records,
)
from ztb.metrics.fill_round_trip_metrics import (
    RoundTripMetrics,
    RoundTripRecord,
    compute_round_trip_metrics,
)
from ztb.utils.dataclass_utils import get_dataclass_field_names, shallow_asdict

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    import pandas as pd

# ======================================================================
# Data classes
# ======================================================================

@dataclass
class FillRecord:
    """1 回の maker 注文サイクルの結果.

    009# §4.2 FillRecord スキーマ準拠.
    """

    cycle_id: str
    timestamp: float  # t_submit (epoch)
    side: str  # 'buy' or 'sell'
    order_price: float  # 発注価格
    order_quantity: float  # 発注数量
    fill_price: float | None = None  # 約定価格 (未約定は None)
    filled: bool = False
    cancelled: bool = False
    queue_wait_sec: float = 0.0  # 発注→約定 (or cancel) の秒数
    mid_at_fill: float | None = None  # 約定時の mid price
    mid_30s_after: float | None = None  # 約定 30 秒後の mid price
    post_fill_30s_pnl: float | None = None  # 30 秒後 PnL (bps)
    adverse_selected: bool | None = None  # 30 秒後に逆行したか (CM-3 deadzone 適用後)
    adverse_selected_raw: bool | None = None  # 020# O5: 生の逆行判定 (deadzone 非適用)
    cancel_reason: str | None = None  # CM-2: キャンセル理由 (api_error / timeout / post_only_reject)
    run_id: str | None = None  # 020# O4: 実験ラン識別子
    git_sha: str | None = None  # 020# O4: コミットハッシュ
    # 031# 追加フィールド
    spread_at_order: float | None = None  # 発注時スプレッド (JPY)
    error_message: str | None = None  # エラー詳細メッセージ
    spread_offset_ratio: float | None = None  # 使用した spread_offset_ratio
    # 047# E3: multi-timeframe PnL 計測 (exit timing 最適化のデータ基盤)
    mid_60s_after: float | None = None   # 約定 60 秒後の mid price
    mid_120s_after: float | None = None  # 約定 120 秒後の mid price
    post_fill_60s_pnl: float | None = None   # 60 秒後 PnL (bps)
    post_fill_120s_pnl: float | None = None  # 120 秒後 PnL (bps)
    # 577# P1: Kissell & Glantz の執行品質指標未保存バグ修正
    spread_capture_bps: float | None = None  # MM の付加価値
    adverse_selection_cost_bps: float | None = None  # 逆行損失
    # 037# レジーム情報 (035# §7 Week 1)
    regime: str | None = None  # FillTestRegime.value (trending/ranging/high_vol/unknown)
    regime_confidence: float | None = None  # 0.0–1.0
    regime_stability: int | None = None  # 連続同一レジーム数
    # 156# §18: データシンク解消 — 下流分析で方向強度/ボラ比を活用
    regime_trend_pct: float | None = None    # トレンド強度 (%)
    regime_volatility_ratio: float | None = None  # ボラティリティ比
    # 054# AS 予測データ基盤 — orderbook imbalance + spread + mid trend
    orderbook_imbalance: float | None = None   # 板不均衡 [-1, +1] (+1=bid圧倒)
    bid_depth_total: float | None = None       # bid 側合計数量 (BTC)
    ask_depth_total: float | None = None       # ask 側合計数量 (BTC)
    mid_price_trend_5s: float | None = None    # 直前 5s の mid 変化率 (bps)
    trend_5s_guard_triggered: bool | None = None  # 684# trend_5s guard が反応したか
    trend_5s_guard_action: str | None = None      # "boost" / "veto" / "none"
    trend_5s_at_order: float | None = None        # 判定時に使った trend_5s 値 (bps)
    as_trailing_gate_action: str | None = None    # 694# "boost" / "veto" / "none"
    as_trailing_gate_rate: float | None = None    # 694# trailing AS rate
    as_trailing_gate_offset_mult: float | None = None  # 694# offset multiplier
    spread_bps: float | None = None            # 発注時スプレッド (bps)
    effective_offset_used: float | None = None # 実際に適用された offset 比率
    # 062# SkipGate ML 判定情報
    skip_gate_skipped: bool | None = None      # SkipGate によるスキップ判定
    skip_gate_bypassed: bool | None = None     # 686# bypass mode で本来 skip されるはずだった判定
    skip_gate_score: float | None = None       # SkipGate 予測スコア (AS確率 or PnL予測値)
    skip_gate_reason: str | None = None        # SkipGate 判定理由
    # 068# OB 品質 + SkipGate モデル使用ログ
    ob_quality_ok: bool | None = None          # OB 特徴量が品質基準を満たしたか
    ob_age_ms: float | None = None             # OB 取得からの経過ミリ秒
    skip_gate_model_used: str | None = None    # "primary" or "fallback"
    # 084# SkipGate 可観測性改善: P(AS) と使用閾値を直接記録
    skip_gate_as_prob: float | None = None     # AS 確率 (0.0-1.0), mode="as" 時のみ
    skip_gate_threshold_used: float | None = None  # 実際に適用された閾値 (side別解決後)
    # 158# P1-6: 時間帯別 skip_gate 閾値調整のオフセット値 (bps)
    skip_gate_hour_offset: float | None = None    # 適用された hour-based offset (0.0=調整なし)
    # 094# stale order cancel-replace 追跡
    reprice_count: int = 0                        # 1 サイクル内で再発注した回数
    # 158# P1-3: reprice 累積 drift (bps) — 全 reprice の合計 drift を記録
    reprice_drift_bps: float | None = None     # 全 reprice 発動時の累積ドリフト (bps)
    # 100# P1-4: 実際の PnL 計測経過秒数 (early_exit で 30s 未満になる場合の記録)
    actual_measurement_sec: float | None = None  # mid_30s_after の実計測秒数
    # 120# A4: Early Exit 明示フラグ (推定ではなく PnlMeasurer の判定値を直接保存)
    early_exit_triggered: bool | None = None     # EE 発火したか (True/False/None=計測前)
    # 120# A4-2: EE 中断時点 PnL (post_fill_30s_pnl は固定30s、計測バイアス分離用)
    pnl_at_exit_bps: float | None = None          # EE 発動時の中断時点 PnL (bps)
    # 120# P2-1: 寄与分解基盤 — FFD/VG イベントフラグ
    ffd_boost_active: bool | None = None          # FastFillDefense boost 中だったか
    vg_triggered: bool | None = None              # VolatilityGuard 発動したか
    # 158# P2-6: VG 詳細ログ (ヒンドサイト分析用)
    vg_velocity_bps: float | None = None          # VG 評価時の mid_price_trend (bps)
    vg_vpin: float | None = None                  # VG 評価時の VPIN 値
    vg_boost_factor: float | None = None          # 実際に適用された boost 倍率 (1.0=未発動)
    # 510# VG boost 理由の粒度向上
    vg_reason: str | None = None                   # "velocity" / "vpin" / "velocity+vpin" / None
    # 510# 在庫偏重状態の追跡
    inv_skew_factor: float | None = None           # inventory skew factor (負=sell offset 緩和)
    # 165# AS-R1: SkipGate 特徴量ログ (閾値キャリブレーション用)
    price_velocity_bps: float | None = None        # 直近60sの価格速度 (bps)
    # 129# D.2: 残高制約による side 強制切替フラグ (評価/学習での交絡分離用)
    balance_forced_switch: bool | None = None     # 残高不足で side が強制切替されたか
    # 155# §9.4 #2: balance_forced_skip 連続回数追跡
    balance_forced_consecutive: int | None = None  # スキップ時点の連続 forced skip 数
    # 151# P3-03: confidence lot 情報 (§10 #7 可観測性)
    confidence_lot_factor: float | None = None    # 適用された倍率 [0, 1]
    order_lot_regime: float | None = None         # regime_adjusted_lot (confidence 未適用)
    order_lot_effective: float | None = None      # 最終発注ロット (= order_quantity)
    confidence_lot_mode: str | None = None        # "as" / "pnl" / None (無効時)
    # 158# P1-5: A/B テスト variant 識別子 (実験分析用)
    ab_test_variant: str | None = None             # 例: "sell_offset_015", None=実験なし
    # 166# C.7: cancel 失敗後に約定確認されたフラグ (Bug11 KPI 分離)
    cancel_failed_likely_filled: bool | None = None  # True=cancel失敗→約定検出
    # 237# phantom position guard: status_unknown 後の再照合待ちフラグ
    pending_reconciliation: bool | None = None  # True=status_unknown発生→次サイクルで再照合
    # 181# EV_weighted: 30s/120s 加重平均 PnL (178# §1.3 設計)
    ev_weighted_pnl: float | None = None  # 0.4*pnl30 + 0.6*pnl120 (bps)
    # 292# P0: ev_weighted 可観測性強化 (290#/291# review)
    ev_score_pretrade: float | None = None        # ランタイム ev_score (ex-ante 予測値)
    ev_offset_mult_applied: float | None = None   # 実適用 offset 乗数 (1.0=変更なし)
    decision_path: str | None = None              # "primary_only" / "ev_offset" / "ev_no_change" / "ev_emergency_skip" / "ev_normal_skip"
    decision_trace_id: str | None = None          # 688# 同一サイクル判断トレース ID
    timeout_applied_sec: float | None = None      # 688# 実効 timeout
    timeout_reason: str | None = None             # 688# timeout 採用理由
    entry_gate_ev: float | None = None            # 690# entry gate EV
    entry_gate_blocked: bool | None = None        # 690# enabled 時に EV<=0 だったか
    entry_gate_guard_suppressed: bool | None = None  # 690# safety guard 抑制
    entry_gate_regime: str | None = None          # 690# entry gate 評価 regime
    # 187# B-2: guard_trace — gated_regime + effective_cycle_interval 記録
    gated_regime: str | None = None              # ヒステリシス適用後の実効 regime
    effective_cycle_interval: float | None = None  # 使用されたサイクル間隔 (秒)
    # 189# D: MacroRegime 統合
    macro_trend: str | None = None               # macro_strong_up / neutral / ... / macro_insufficient
    macro_slope_5m: float | None = None          # 5分 slope (bps/min)
    macro_slope_15m: float | None = None         # 15分 slope (bps/min)
    macro_aligned: bool | None = None            # micro/macro 一致フラグ
    # 458# F-lite: macro offset boost 適用フラグ
    macro_boost_applied: bool | None = None      # macro→offset boost が発火したか
    # 285# 283# P0-1: Split-Brain 検知用 — プロセス ID 記録
    # 同一時刻帯に複数 run_id/pid が存在すれば多重起動を検出可能
    pid: int | None = None                       # os.getpid() at record creation
    # ---- 306# O1: Queue Position Estimation ----
    queue_depth_ahead: float | None = None       # 発注時の same_side_depth_ahead (BTC)
    queue_fill_prob_est: float | None = None      # 推定 fill probability [0,1]
    # ---- 306# S1: Offset Stage 寄与記録 (301# F6, 300# T0-1) ----
    # JSON 形式で各ステージの寄与を記録: {"as_shift": 0.02, "vg": 0.05, ...}
    offset_stages: str | None = None              # JSON-encoded stage contributions
    # ---- 306# L2: Microprice Side Selection ----
    microprice_bias_bps: float | None = None      # microprice vs mid の偏向 (bps)
    # ---- 318# F5-3: None regime 可観測性 (307# F5) ----
    # regime_at_order: 発注時 (pricing 時) のレジーム値。post-cycle の regime とは
    # 異なる場合がある (cycle 中に detector が更新されるため)
    regime_at_order: str | None = None            # 発注時の regime value
    # regime_observation_count: detector の蓄積観測数。
    # < window (通常20) なら warmup 中、>= window なら成熟 unknown (低信頼度)
    regime_observation_count: int | None = None   # detector observation count at order time
    # ---- 319# S-3: mid_at_order (316# S-3: spread capture 精度向上) ----
    # 注文発行時の mid price。mid_at_fill (post-fill 測定時の mid) との差で
    # 検出レイテンシバイアスを分離し、spread capture 計算の精度を向上。
    mid_at_order: float | None = None             # mid price at order submission
    # ---- 372# F1: SAC Sidecar offset 記録 ----
    sidecar_offset_bps: float | None = None       # 適用された sidecar offset (bps, 正=攻撃的)
    sidecar_bias: float | None = None             # SAC directional_bias [-1,+1]
    # ---- 487# P0: SAC Sidecar attribution 可観測性 ----
    sidecar_confidence: float | None = None       # SAC confidence [0,1]
    sidecar_model_version: str | None = None      # モデル識別子 (e.g. 'sac_sidecar_v460_...')
    sidecar_signal_status: str | None = None      # "fresh"/"stale"/"missing"/"error"
    ppo_sidecar_action: str | None = None         # "buy"/"sell"/"skip"
    ppo_sidecar_confidence: float | None = None   # PPO top action confidence [0,1]
    ppo_sidecar_action_margin: float | None = None  # top1 - top2 probability
    ppo_sidecar_model_version: str | None = None  # PPO モデル識別子
    ppo_sidecar_signal_status: str | None = None  # "fresh"/"stale"/"missing"/"error"
    ppo_sidecar_override_active: bool | None = None  # threshold 通過で veto 判定が有効化されたか
    # ---- 439# P1: Cross-Venue Lead-Lag 可観測性 ----
    cross_venue_reference_exchange: str | None = None
    cross_venue_lead_lag_direction: str | None = None
    cross_venue_lead_lag_adverse_side: str | None = None
    cross_venue_lead_lag_spread_bps: float | None = None
    cross_venue_lead_lag_velocity_bps: float | None = None
    cross_venue_lead_lag_age_sec: float | None = None
    cross_venue_lead_lag_applied: bool | None = None
    cross_venue_lead_lag_vetoed: bool | None = None
    # 533# veto deadlock 防止: 連続 veto 回数の記録
    cross_venue_lead_lag_veto_consecutive: int | None = None
    # ---- 442# Cross-Venue microprice + depth imbalance ----
    cross_venue_microprice_spread_bps: float | None = None
    cross_venue_depth_imbalance: float | None = None
    # ---- 445# Cross-Venue confidence scoring ----
    cross_venue_confidence: float | None = None
    # ---- 508# Cross-Venue basis correction 可観測性 ----
    cross_venue_basis_bps: float | None = None
    cross_venue_adjusted_spread_bps: float | None = None
    # ---- 448# Cross-Venue EMA/点spread分離 + No-Op可視化 ----
    cross_venue_lead_lag_point_spread_bps: float | None = None
    cross_venue_lead_lag_pre_offset: float | None = None
    cross_venue_lead_lag_post_offset: float | None = None
    cross_venue_lead_lag_cap_hit: bool | None = None
    cross_venue_buy_offset_mult: float | None = None
    # ---- 421# P0: Execution Final Clamp 記録 ----
    # executor multiplier chain 適用後・Final Clamp 適用前の offset ratio。
    # None=clamp 未発火 or 無効。値あり=clamp が発火し、ceiling に切り詰められた。
    execution_pre_clamp_offset: float | None = None
    # ---- 420# P1: Executor Offset Stages (6 multiplier 寄与記録) ----
    # JSON: {"ev": 1.05, "velocity": null, "trending": null, "toxicity": 1.15,
    #        "vg_supp": null, "alert": 1.0}
    # null = 未適用 / 1.0 = 適用されたが変更なし
    executor_offset_stages: str | None = None
    # ---- 420# P1: start_git_sha (run 開始時 SHA 固定) ----
    # hot_reload で git_sha が変わっても start_git_sha は不変。
    # コード attribution 分析で run 開始版を特定するために使用。
    start_git_sha: str | None = None
    # ---- 467# config_hash: 設定識別子 (462# 残課題) ----
    # compute_config_hash(config) で生成。同一 run 中の config drift 検出に使用。
    config_hash: str | None = None
    # ---- 420# P1: Side 切替可観測性 (416# §4.2) ----
    # SideSelector が最初に返した side (balance/veto 切替前)
    requested_side: str | None = None
    # 切替理由: 522# で balance_switch/recovery_skew 撤廃 → 現在は常に None
    resolved_side_reason: str | None = None
    # 687# state separation: 実行 side と試行 side を分離して記録
    last_executed_side: str | None = None
    last_attempted_side: str | None = None
    # ---- 452# Micro-timeout (TIF Emulation) ----
    requote_attempts: int | None = None  # サブサイクル re-quote 回数 (0=初回で約定, None=micro_timeout 無効)
    micro_timeout_partial_filled_qty: float | None = None  # re-quote ループ中の部分約定合計
    # ---- 533# log_cycle_no: ログ⇔JSONL join key ----
    # "=== Cycle NNN" ログの NNN と一致させ、ログファイルと fill_records の突合を容易にする
    log_cycle_no: int | None = None
    # ---- 573# eDRC / additive pipeline telemetry ----
    execution_sigma: float | None = None
    execution_adverse_ofi: float | None = None
    execution_additive_enabled: bool | None = None
    # ---- 642# 可観測性改善: skip_rate / hard_skip / CV / balance ----
    skip_gate_forced_pass: bool | None = None       # rate_limit が skip を override したか
    skip_gate_side_skip_rate: float | None = None   # 判定時の side 別 skip 率
    skip_gate_budget_regime: str | None = None      # 690# bucket 判定に使った regime
    skip_gate_budget_remaining: int | None = None   # 690# 判定後の残り budget
    skip_gate_budget_exhausted: bool | None = None  # 690# budget 枯渇で PASS 強制
    execution_hard_skip_mult_used: float | None = None  # hard skip 時に使用した mult 値
    cv_offset_action: str | None = None             # "widen"/"tighten"/None (CV 適用方向)
    balance_jpy_at_order: float | None = None       # 発注時 JPY 残高
    balance_btc_at_order: float | None = None       # 発注時 BTC 残高
    # ---- 671# NFQ (no_feasible_quote) 構造化ログ ----
    # error_message パース不要で NFQ 原因分析を可能にする
    nfq_actual_spread: float | None = None         # 実際の市場スプレッド (JPY)
    nfq_min_spread_effective: float | None = None  # 適用された最小スプレッド (JPY)
    nfq_min_spread_abs: float | None = None        # config.min_spread_jpy (JPY)
    nfq_min_spread_atr: float | None = None        # ATR ベース最小スプレッド (JPY)
    nfq_sigma: float | None = None                 # ATR 計算時の σ 値

    def to_dict(self) -> dict:
        """JSON serializable dict."""
        return shallow_asdict(self)

    @classmethod
    def from_dict(cls, d: Mapping[str, object]) -> FillRecord:
        """Reconstruct from dict."""
        return cls(**_sanitize_fill_record_fields(d, context="FillRecord.from_dict"))

_FILL_RECORD_FIELD_NAMES: Final[frozenset[str]] = get_dataclass_field_names(FillRecord)
_SKIP_RECORD_PROTECTED_FIELDS: Final[frozenset[str]] = frozenset({
    "cycle_id",
    "timestamp",
    "side",
    "order_price",
    "order_quantity",
    "filled",
    "cancelled",
    "cancel_reason",
    "run_id",
    "git_sha",
    "spread_at_order",
    "spread_offset_ratio",
    "regime",
    "balance_forced_switch",
    "ab_test_variant",
})
FillRecordPayload = dict[str, object]

def _sanitize_fill_record_fields(
    values: Mapping[str, object],
    *,
    context: str,
    protected_keys: frozenset[str] = frozenset(),
) -> FillRecordPayload:
    """FillRecord に存在するキーだけ通し、不要キーは無視する."""
    # 216# §6: 旧フィールド名 → 新フィールド名 のマイグレーション
    _FIELD_ALIASES: dict[str, str] = {
        "price_velocity_60s": "price_velocity_bps",
    }
    filtered: FillRecordPayload = {}
    unknown_keys: list[str] = []
    protected_hits: list[str] = []

    for key, value in values.items():
        # エイリアス解決 (新名が既に存在しない場合のみ)
        resolved = _FIELD_ALIASES.get(key, key)
        if resolved != key and resolved in values:
            # 新名が既に存在 → 旧名は無視
            unknown_keys.append(key)
            continue
        key = resolved

        if key in protected_keys:
            protected_hits.append(key)
            continue
        if key not in _FILL_RECORD_FIELD_NAMES:
            unknown_keys.append(key)
            continue
        filtered[key] = value

    if protected_hits:
        logger.debug(
            "%s: protected keys ignored: %s",
            context,
            sorted(protected_hits),
        )
    if unknown_keys:
        logger.debug(
            "%s: unknown fields ignored: %s",
            context,
            sorted(unknown_keys),
        )
    return filtered

def build_fill_record(**data: object) -> FillRecord:
    """FillRecord の generic builder.

    known field のみを通す。必須フィールド不足は FillRecord 側の TypeError に委ねる。
    """
    return FillRecord(**_sanitize_fill_record_fields(data, context="build_fill_record"))


def _build_skip_fill_record_payload(
    *,
    cycle_id: str,
    timestamp: float,
    side: str,
    order_price: float,
    order_quantity: float,
    cancel_reason: str,
    run_id: str | None,
    git_sha: str | None,
    spread_at_order: float | None = None,
    spread_offset_ratio: float | None = None,
    regime: str | None = None,
    balance_forced_switch: bool = False,
    ab_test_variant: str | None = None,
    extra: Mapping[str, object],
) -> FillRecordPayload:
    """skip/監査系 FillRecord の payload shaping を集約する."""
    payload: FillRecordPayload = {
        "cycle_id": cycle_id,
        "timestamp": timestamp,
        "side": side,
        "order_price": order_price,
        "order_quantity": order_quantity,
        "filled": False,
        "cancelled": True,
        "cancel_reason": cancel_reason,
        "run_id": run_id,
        "git_sha": git_sha,
        "spread_at_order": spread_at_order,
        "spread_offset_ratio": spread_offset_ratio,
        "regime": regime,
        "balance_forced_switch": balance_forced_switch,
        "ab_test_variant": ab_test_variant,
    }
    payload.update(
        _sanitize_fill_record_fields(
            extra,
            context="build_skip_fill_record",
            protected_keys=_SKIP_RECORD_PROTECTED_FIELDS,
        )
    )
    return payload

def build_skip_fill_record(
    *,
    cycle_id: str,
    timestamp: float,
    side: str,
    order_price: float,
    order_quantity: float,
    cancel_reason: str,
    run_id: str | None,
    git_sha: str | None,
    spread_at_order: float | None = None,
    spread_offset_ratio: float | None = None,
    regime: str | None = None,
    balance_forced_switch: bool = False,
    ab_test_variant: str | None = None,
    **extra: object,
) -> FillRecord:
    """skip/監査系 FillRecord の共通 builder.

    追加フィールドは FillRecord に存在するものだけを反映し、それ以外は無視する。
    """
    payload = _build_skip_fill_record_payload(
        cycle_id=cycle_id,
        timestamp=timestamp,
        side=side,
        order_price=order_price,
        order_quantity=order_quantity,
        cancel_reason=cancel_reason,
        run_id=run_id,
        git_sha=git_sha,
        spread_at_order=spread_at_order,
        spread_offset_ratio=spread_offset_ratio,
        regime=regime,
        balance_forced_switch=balance_forced_switch,
        ab_test_variant=ab_test_variant,
        extra=extra,
    )
    return build_fill_record(**payload)

def compute_record_pnl_jpy(record: FillRecord) -> float | None:
    """FillRecord の 30s PnL を JPY 概算へ変換.

    filled でない、または必要値が不足/非有限な場合は None を返す。
    """
    if not record.filled:
        return None
    if record.post_fill_30s_pnl is None or record.fill_price is None:
        return None
    pnl_bps = float(record.post_fill_30s_pnl)
    fill_price = float(record.fill_price)
    order_qty = float(record.order_quantity)
    if not (np.isfinite(pnl_bps) and np.isfinite(fill_price) and np.isfinite(order_qty)):
        return None
    return pnl_bps * 1e-4 * fill_price * order_qty

@dataclass
class FillMetrics:
    """G1.1 Gate 指標の算出結果.

    009# §4.3 FillMetrics 準拠.
    """

    total_orders: int = 0
    filled_orders: int = 0
    cancelled_orders: int = 0
    fill_rate_p90: float = 0.0  # E1
    cancel_ratio: float = 0.0  # E2
    queue_wait_median_sec: float = 0.0  # E3
    post_fill_30s_pnl_mean: float = 0.0  # E4
    post_fill_30s_pnl_pvalue: float = 1.0  # E4 片側 t 検定
    adverse_selection_ratio: float = 0.0  # E5 (deadzone 適用後)
    adverse_selection_ratio_raw: float = 0.0  # E5-raw (020# O5: 並行監視用)

    # 補助情報
    daily_fill_rates: list[float] = field(default_factory=list)
    measurement_days: int = 0
    sample_sufficient: bool = False  # 020# O1: n>=200 & 3暦日 達成フラグ
    # 047# Finding4: AS coverage (分母透明化)
    as_coverage: int = 0       # adverse_selected != None の件数
    as_raw_coverage: int = 0   # adverse_selected_raw != None の件数

    # 116# attempted ベース指標 (skip_gate 除外)
    attempted_orders: int = 0          # skip_gate 除外後のサイクル数
    skip_gate_count: int = 0           # skip_gate でスキップされた件数
    skip_gate_ratio: float = 0.0       # skip_gate_count / total_orders
    attempted_fill_rate: float = 0.0   # filled / attempted_orders
    attempted_cancel_ratio: float = 0.0  # (attempted - filled) / attempted
    overall_fill_rate: float = 0.0     # filled / total_orders (全体ベース)
    # 116# PnL CI (Kill Gate 複合条件用)
    post_fill_30s_pnl_ci_upper: float = 0.0  # 95% CI 上限

    # 122# B3: multi-timeframe PnL 統計 (047# データ基盤活用)
    post_fill_60s_pnl_mean: float = 0.0
    post_fill_60s_pnl_pvalue: float = 1.0
    post_fill_120s_pnl_mean: float = 0.0
    post_fill_120s_pnl_pvalue: float = 1.0

    # 117# cancel reason 内訳 (115# Q10.6: cancel 理由の内訳管理)
    cancel_reason_breakdown: dict[str, int] = field(default_factory=dict)

    def to_dict(self) -> dict:
        """JSON serializable dict."""
        return shallow_asdict(self)

# ======================================================================
# Metrics computation
# ======================================================================

def compute_fill_metrics(records: list[FillRecord]) -> FillMetrics:
    """FillRecord のリストから G1.1 Gate 指標を算出.

    009# §2.1 E1-E5 準拠.

    Returns:
        FillMetrics with all E1-E5 indicators computed.
    """
    if not records:
        return FillMetrics()

    scan = scan_fill_metric_inputs(records)

    fill_rate_p90 = (
        float(np.percentile(scan.daily_fill_rates, 10))
        if scan.daily_fill_rates
        else 0.0
    )
    # NOTE: P90 = "90% of days have fill rate >= this value" = 10th percentile
    # (lower bound of the distribution)

    # --- E2: cancel_ratio ---
    cancel_ratio = scan.cancelled_count / scan.total if scan.total > 0 else 0.0

    # --- E3: queue_wait_median_sec (filled orders only) ---
    queue_wait_median = float(np.median(scan.wait_times)) if scan.wait_times else 0.0

    # --- E4: post_fill_30s_pnl ---
    pnl_mean, pnl_pvalue = mean_and_one_sided_pvalue(scan.pnl_values)

    # --- E5: adverse_selection_ratio ---
    adverse_ratio = scan.n_adverse / scan.as_coverage if scan.as_coverage else 0.0

    # --- E5-raw: 020# O5 — deadzone 非適用の生データ並行監視 ---
    adverse_ratio_raw = (
        scan.n_adverse_raw / scan.as_raw_coverage if scan.as_raw_coverage else adverse_ratio
    )

    # --- 020# O1: サンプル充足判定 ---
    # 047# Finding3: 3日ではなく 7日を要求 (000# §3.3 準拠)
    sample_sufficient = scan.total >= 200 and len(scan.daily_fill_rates) >= 7

    # --- 116# attempted ベース指標 ---
    attempted_orders = scan.total - scan.skip_gate_count
    skip_gate_ratio = scan.skip_gate_count / scan.total if scan.total > 0 else 0.0
    attempted_fill_rate = (
        scan.filled_count / attempted_orders if attempted_orders > 0 else 0.0
    )
    attempted_cancel_ratio = (
        (attempted_orders - scan.filled_count) / attempted_orders
        if attempted_orders > 0
        else 0.0
    )
    overall_fill_rate = scan.filled_count / scan.total if scan.total > 0 else 0.0

    # --- 116# PnL CI 上限 (Kill Gate 複合条件用) ---
    pnl_ci_upper = 0.0
    if scan.pnl_values and len(scan.pnl_values) >= 2:
        se = float(np.std(scan.pnl_values, ddof=1)) / np.sqrt(len(scan.pnl_values))
        from scipy import stats

        t_crit = float(stats.t.ppf(0.975, df=len(scan.pnl_values) - 1))
        pnl_ci_upper = pnl_mean + t_crit * se

    # --- 122# B3: multi-timeframe PnL (047# データ基盤活用) ---
    pnl60_mean, pnl60_pvalue = mean_and_one_sided_pvalue(scan.pnl60_values)
    pnl120_mean, pnl120_pvalue = mean_and_one_sided_pvalue(scan.pnl120_values)

    return FillMetrics(
        total_orders=scan.total,
        filled_orders=scan.filled_count,
        cancelled_orders=scan.cancelled_count,
        fill_rate_p90=fill_rate_p90,
        cancel_ratio=cancel_ratio,
        queue_wait_median_sec=queue_wait_median,
        post_fill_30s_pnl_mean=pnl_mean,
        post_fill_30s_pnl_pvalue=pnl_pvalue,
        adverse_selection_ratio=adverse_ratio,
        adverse_selection_ratio_raw=adverse_ratio_raw,
        daily_fill_rates=scan.daily_fill_rates,
        measurement_days=len(scan.daily_fill_rates),
        sample_sufficient=sample_sufficient,
        # 047# Finding4: AS coverage
        as_coverage=scan.as_coverage,
        as_raw_coverage=scan.as_raw_coverage,
        # 116# attempted
        attempted_orders=attempted_orders,
        skip_gate_count=scan.skip_gate_count,
        skip_gate_ratio=skip_gate_ratio,
        attempted_fill_rate=attempted_fill_rate,
        attempted_cancel_ratio=attempted_cancel_ratio,
        overall_fill_rate=overall_fill_rate,
        post_fill_30s_pnl_ci_upper=pnl_ci_upper,
        # 122# B3: multi-timeframe PnL
        post_fill_60s_pnl_mean=pnl60_mean,
        post_fill_60s_pnl_pvalue=pnl60_pvalue,
        post_fill_120s_pnl_mean=pnl120_mean,
        post_fill_120s_pnl_pvalue=pnl120_pvalue,
        cancel_reason_breakdown=scan.cancel_reason_breakdown,
    )

# ======================================================================
# G1.1 Gate Judgment
# ======================================================================

def g1_1_judgment(
    metrics: FillMetrics,
    thresholds: dict,
    records: list[FillRecord] | None = None,
) -> dict:
    """G1.1 Gate 合否判定.

    009# §2.1 / 000# §3.3 準拠.
    092# 追加: E6 (round-trip mean PnL), E7 (net inventory drift).

    Args:
        metrics: compute_fill_metrics() の出力.
        thresholds: gate_thresholds.yaml の ``g1_1_exec`` セクション.
        records: round-trip KPI 算出用の FillRecord リスト (省略時は E6/E7 スキップ).

    Returns:
        dict with gate_result, per-check details.
    """
    checks = cast(dict[str, dict[str, object]], build_exec_gate_checks(metrics, thresholds))

    if records is not None:
        checks.update(build_exec_monitoring_checks(records, thresholds))

    # Gate 判定には informational=True のチェックは含めない
    gate_checks = {k: v for k, v in checks.items() if not v.get("informational")}
    all_pass = all(c["pass"] for c in gate_checks.values())

    judgment_type = resolve_exec_judgment_type(metrics)

    return build_gate_payload(
        gate="G1.1-exec",
        gate_result="PASS" if all_pass else "FAIL",
        checks=checks,
        metrics=metrics,
        extras={
            "judgment_type": judgment_type,  # 020# O1 / 047# Finding3
            "sample_sufficient": metrics.sample_sufficient,
        },
    )

# ======================================================================
# 116# Two-Stage Gate Judgment (115# review)
# ======================================================================

def g1_1_quick_judgment(
    metrics: FillMetrics,
    thresholds: dict,
    cumulative_loss_jpy: float = 0.0,
) -> dict:
    """G1.1-quick (72h Kill Gate) 判定.

    116# 実装 / 115# レビュー反映.
    明らかに不成立な戦略を早期棄却する。

    Args:
        metrics: compute_fill_metrics() の出力.
        thresholds: gate_thresholds.yaml の ``g1_1_quick_exec`` セクション.
        cumulative_loss_jpy: fill_test の累積実損 (JPY, 正値=損失).

    Returns:
        dict with gate_result (PASS/FAIL/WATCH), per-check details.
    """
    checks = cast(
        dict[str, dict[str, object]],
        build_quick_gate_checks(
            metrics,
            thresholds,
            cumulative_loss_jpy=cumulative_loss_jpy,
        ),
    )

    all_pass = all(c["pass"] for c in checks.values())

    # 115# Q10.4: Watch 層 — Kill には至らないが黄信号
    pnl_watch_p = thresholds.get("pnl_watch_p_threshold", 0.05)
    pnl_watch_mean = thresholds.get("pnl_watch_mean_threshold", -0.3)
    is_watch = (
        all_pass
        and metrics.post_fill_30s_pnl_pvalue < pnl_watch_p
        and metrics.post_fill_30s_pnl_mean < pnl_watch_mean
    )

    if not all_pass:
        gate_result = "FAIL"
    elif is_watch:
        gate_result = "WATCH"
    else:
        gate_result = "PASS"

    return build_gate_payload(
        gate="G1.1-quick",
        gate_result=gate_result,
        checks=checks,
        metrics=metrics,
        watch=is_watch,
        watch_detail=(
            build_quick_watch_detail(
                metrics,
                pnl_watch_p=pnl_watch_p,
                pnl_watch_mean=pnl_watch_mean,
            )
            if is_watch
            else None
        ),
    )

def g1_2_full_judgment(
    metrics: FillMetrics,
    thresholds: dict,
) -> dict:
    """G1.2-full (168h Qualification Gate) 判定.

    116# 実装 / 115# レビュー反映.
    完全な品質適格性の確認。

    Args:
        metrics: compute_fill_metrics() の出力.
        thresholds: gate_thresholds.yaml の ``g1_2_full_exec`` セクション.

    Returns:
        dict with gate_result (PASS/FAIL), per-check details.
    """
    checks = cast(
        dict[str, dict[str, object]],
        build_full_gate_structural_checks(metrics, thresholds),
    )
    checks.update(build_full_gate_pnl_checks(metrics, thresholds))

    gate_result, is_watch = resolve_gate_result(checks)

    return build_gate_payload(
        gate="G1.2-full",
        gate_result=gate_result,
        checks=checks,
        metrics=metrics,
        watch=is_watch,
    )

# ======================================================================
# 051# P2-2: Round-trip 評価 (buy→sell ペアリング)
# ======================================================================
