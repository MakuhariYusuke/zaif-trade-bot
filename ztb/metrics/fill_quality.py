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

import json
import logging
import math
from collections import deque
from dataclasses import asdict, dataclass, field
from datetime import datetime, timedelta, timezone
from functools import lru_cache
from pathlib import Path
from typing import TYPE_CHECKING, Final, Iterable, Iterator, Mapping, Sequence

import numpy as np
from scipy import stats
from ztb.io.jsonl import iter_jsonl_objects
from ztb.utils.dataclass_utils import get_dataclass_field_names

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    import pandas as pd

_SECONDS_PER_DAY: Final[float] = 86_400.0

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
    spread_bps: float | None = None            # 発注時スプレッド (bps)
    effective_offset_used: float | None = None # 実際に適用された offset 比率
    # 062# SkipGate ML 判定情報
    skip_gate_skipped: bool | None = None      # SkipGate によるスキップ判定
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
    # 187# B-2: guard_trace — gated_regime + effective_cycle_interval 記録
    gated_regime: str | None = None              # ヒステリシス適用後の実効 regime
    effective_cycle_interval: float | None = None  # 使用されたサイクル間隔 (秒)
    # 189# D: MacroRegime 統合
    macro_trend: str | None = None               # macro_strong_up / neutral / ... / macro_insufficient
    macro_slope_5m: float | None = None          # 5分 slope (bps/min)
    macro_slope_15m: float | None = None         # 15分 slope (bps/min)
    macro_aligned: bool | None = None            # micro/macro 一致フラグ
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

    def to_dict(self) -> dict:
        """JSON serializable dict."""
        return asdict(self)

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

def _sanitize_fill_record_fields(
    values: Mapping[str, object],
    *,
    context: str,
    protected_keys: frozenset[str] = frozenset(),
) -> dict[str, object]:
    """FillRecord に存在するキーだけ通し、不要キーは無視する."""
    # 216# §6: 旧フィールド名 → 新フィールド名 のマイグレーション
    _FIELD_ALIASES: dict[str, str] = {
        "price_velocity_60s": "price_velocity_bps",
    }
    filtered: dict[str, object] = {}
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
    payload: dict[str, object] = {
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
        return asdict(self)

@dataclass
class PnlAccumulator:
    """Finite PnL 値だけを集計するストリーム集計器."""

    count: int = 0
    total_bps: float = 0.0

    def add(self, value: float | None) -> None:
        if value is None:
            return
        numeric = float(value)
        if not np.isfinite(numeric):
            return
        self.count += 1
        self.total_bps += numeric

    @property
    def mean_bps(self) -> float:
        return self.total_bps / self.count if self.count else 0.0

@dataclass
class PnlWinAccumulator:
    """PnL の平均と勝率を同時に集計するストリーム集計器."""

    pnl: PnlAccumulator = field(default_factory=PnlAccumulator)
    positive_count: int = 0

    def add(self, value: float | None) -> None:
        if value is None:
            return
        numeric = float(value)
        if not np.isfinite(numeric):
            return
        self.pnl.count += 1
        self.pnl.total_bps += numeric
        if numeric > 0:
            self.positive_count += 1

    @property
    def count(self) -> int:
        return self.pnl.count

    @property
    def total_bps(self) -> float:
        return self.pnl.total_bps

    @property
    def mean_bps(self) -> float:
        return self.pnl.mean_bps

    @property
    def win_rate(self) -> float:
        return self.positive_count / self.pnl.count if self.pnl.count else 0.0

@dataclass
class _DailyFillCount:
    """日次 fill rate 用の軽量カウンタ."""

    total: int = 0
    filled: int = 0

    def add(self, *, filled: bool) -> None:
        self.total += 1
        if filled:
            self.filled += 1

# ======================================================================
# Metrics computation
# ======================================================================

def format_utc_day(timestamp: object, *, compact: bool = True) -> str | None:
    """epoch 秒を UTC 日付文字列へ変換する."""
    try:
        ts = float(timestamp)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(ts):
        return None
    try:
        pattern = "%Y%m%d" if compact else "%Y-%m-%d"
        return datetime.fromtimestamp(ts, tz=timezone.utc).strftime(pattern)
    except (OverflowError, OSError, ValueError):
        return None

def _utc_day_bucket(timestamp: object) -> int | None:
    """epoch 秒を UTC 日単位バケットに変換する."""
    try:
        ts = float(timestamp)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(ts):
        return None
    try:
        return int(ts // _SECONDS_PER_DAY)
    except (OverflowError, ValueError):
        return None

def _mean_and_one_sided_pvalue(values: list[float]) -> tuple[float, float]:
    """平均値と片側 t 検定 p 値を返す."""
    if not values:
        return 0.0, 1.0
    mean_val = float(np.mean(values))
    if len(values) < 2:
        return mean_val, 1.0

    t_stat, two_sided_p = stats.ttest_1samp(values, 0.0)
    if t_stat < 0:
        return mean_val, float(two_sided_p / 2)
    return mean_val, 1.0 - float(two_sided_p / 2)

def compute_fill_metrics(records: list[FillRecord]) -> FillMetrics:
    """FillRecord のリストから G1.1 Gate 指標を算出.

    009# §2.1 E1-E5 準拠.

    Returns:
        FillMetrics with all E1-E5 indicators computed.
    """
    if not records:
        return FillMetrics()

    total = len(records)
    filled_count = 0
    cancelled_count = 0
    cancel_reason_breakdown: dict[str, int] = {}
    skip_gate_count = 0
    daily_groups: dict[int, _DailyFillCount] = {}
    wait_times: list[float] = []
    pnl_values: list[float] = []
    pnl60_values: list[float] = []
    pnl120_values: list[float] = []
    as_coverage = 0
    as_raw_coverage = 0
    n_adverse = 0
    n_adverse_raw = 0

    # --- E1-E5 / cancel reason / attempted 用の前処理 ---
    for record in records:
        if record.filled:
            filled_count += 1
            if record.queue_wait_sec > 0:
                wait_times.append(record.queue_wait_sec)

            if record.post_fill_30s_pnl is not None:
                pnl_values.append(record.post_fill_30s_pnl)
            if record.post_fill_60s_pnl is not None:
                pnl60_values.append(record.post_fill_60s_pnl)
            if record.post_fill_120s_pnl is not None:
                pnl120_values.append(record.post_fill_120s_pnl)

            if record.adverse_selected is not None:
                as_coverage += 1
                if record.adverse_selected:
                    n_adverse += 1

            if record.adverse_selected_raw is not None:
                as_raw_coverage += 1
                if record.adverse_selected_raw:
                    n_adverse_raw += 1

        if record.cancelled:
            cancelled_count += 1
            reason = record.cancel_reason or "unknown"
            cancel_reason_breakdown[reason] = cancel_reason_breakdown.get(reason, 0) + 1
        if record.skip_gate_skipped is True or record.cancel_reason == "skip_gate":
            skip_gate_count += 1

        day_key = _utc_day_bucket(record.timestamp)
        if day_key is None:
            continue
        day_stats = daily_groups.get(day_key)
        if day_stats is None:
            day_stats = _DailyFillCount()
            daily_groups[day_key] = day_stats
        day_stats.add(filled=record.filled)

    daily_fill_rates: list[float] = []
    for day_key in sorted(daily_groups):
        day_stats = daily_groups[day_key]
        daily_fill_rates.append(
            day_stats.filled / day_stats.total if day_stats.total > 0 else 0.0
        )

    fill_rate_p90 = float(np.percentile(daily_fill_rates, 10)) if daily_fill_rates else 0.0
    # NOTE: P90 = "90% of days have fill rate >= this value" = 10th percentile
    # (lower bound of the distribution)

    # --- E2: cancel_ratio ---
    cancel_ratio = cancelled_count / total if total > 0 else 0.0

    # --- E3: queue_wait_median_sec (filled orders only) ---
    queue_wait_median = float(np.median(wait_times)) if wait_times else 0.0

    # --- E4: post_fill_30s_pnl ---
    pnl_mean, pnl_pvalue = _mean_and_one_sided_pvalue(pnl_values)

    # --- E5: adverse_selection_ratio ---
    adverse_ratio = n_adverse / as_coverage if as_coverage else 0.0

    # --- E5-raw: 020# O5 — deadzone 非適用の生データ並行監視 ---
    adverse_ratio_raw = (
        n_adverse_raw / as_raw_coverage if as_raw_coverage else adverse_ratio
    )

    # --- 020# O1: サンプル充足判定 ---
    # 047# Finding3: 3日ではなく 7日を要求 (000# §3.3 準拠)
    sample_sufficient = total >= 200 and len(daily_fill_rates) >= 7

    # --- 116# attempted ベース指標 ---
    attempted_orders = total - skip_gate_count
    skip_gate_ratio = skip_gate_count / total if total > 0 else 0.0
    attempted_fill_rate = filled_count / attempted_orders if attempted_orders > 0 else 0.0
    attempted_cancel_ratio = (
        (attempted_orders - filled_count) / attempted_orders
        if attempted_orders > 0
        else 0.0
    )
    overall_fill_rate = filled_count / total if total > 0 else 0.0

    # --- 116# PnL CI 上限 (Kill Gate 複合条件用) ---
    pnl_ci_upper = 0.0
    if pnl_values and len(pnl_values) >= 2:
        se = float(np.std(pnl_values, ddof=1)) / np.sqrt(len(pnl_values))
        t_crit = float(stats.t.ppf(0.975, df=len(pnl_values) - 1))
        pnl_ci_upper = pnl_mean + t_crit * se

    # --- 122# B3: multi-timeframe PnL (047# データ基盤活用) ---
    pnl60_mean, pnl60_pvalue = _mean_and_one_sided_pvalue(pnl60_values)
    pnl120_mean, pnl120_pvalue = _mean_and_one_sided_pvalue(pnl120_values)

    return FillMetrics(
        total_orders=total,
        filled_orders=filled_count,
        cancelled_orders=cancelled_count,
        fill_rate_p90=fill_rate_p90,
        cancel_ratio=cancel_ratio,
        queue_wait_median_sec=queue_wait_median,
        post_fill_30s_pnl_mean=pnl_mean,
        post_fill_30s_pnl_pvalue=pnl_pvalue,
        adverse_selection_ratio=adverse_ratio,
        adverse_selection_ratio_raw=adverse_ratio_raw,
        daily_fill_rates=daily_fill_rates,
        measurement_days=len(daily_fill_rates),
        sample_sufficient=sample_sufficient,
        # 047# Finding4: AS coverage
        as_coverage=as_coverage,
        as_raw_coverage=as_raw_coverage,
        # 116# attempted
        attempted_orders=attempted_orders,
        skip_gate_count=skip_gate_count,
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
        cancel_reason_breakdown=cancel_reason_breakdown,
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
    checks: dict[str, dict] = {}

    # E1: fill_rate P90
    min_fill = thresholds.get("min_fill_rate_p90", 0.90)
    checks["E1_fill_rate_p90"] = {
        "value": metrics.fill_rate_p90,
        "threshold": min_fill,
        "pass": metrics.fill_rate_p90 >= min_fill,
    }

    # E2: cancel_ratio
    max_cancel = thresholds.get("max_cancel_ratio", 0.30)
    checks["E2_cancel_ratio"] = {
        "value": metrics.cancel_ratio,
        "threshold": max_cancel,
        "pass": metrics.cancel_ratio <= max_cancel,
    }

    # E3: queue_wait_median_sec
    max_wait = thresholds.get("max_queue_wait_median_sec", 60)
    checks["E3_queue_wait_median"] = {
        "value": metrics.queue_wait_median_sec,
        "threshold": max_wait,
        "pass": metrics.queue_wait_median_sec <= max_wait,
    }

    # E4: post_fill_30s_pnl (§2.4 統計的補足)
    min_pnl = thresholds.get("min_post_fill_30s_pnl", 0.0)
    pnl_pass: bool
    if metrics.post_fill_30s_pnl_mean >= min_pnl:
        pnl_pass = True
    elif metrics.post_fill_30s_pnl_pvalue >= 0.05:
        pnl_pass = True  # 負だが統計的に有意でない → PASS
    else:
        pnl_pass = False  # 負かつ有意 → systemic adverse selection
    checks["E4_post_fill_pnl"] = {
        "value": metrics.post_fill_30s_pnl_mean,
        "threshold": min_pnl,
        "pvalue": metrics.post_fill_30s_pnl_pvalue,
        "pass": pnl_pass,
    }

    # E5: adverse_selection_ratio
    max_adverse = thresholds.get("max_adverse_selection_ratio", 0.20)
    checks["E5_adverse_selection"] = {
        "value": metrics.adverse_selection_ratio,
        "threshold": max_adverse,
        "pass": metrics.adverse_selection_ratio <= max_adverse,
    }

    # E5-raw: 020# O5 — deadzone 非適用の並行監視
    checks["E5_adverse_selection_raw"] = {
        "value": metrics.adverse_selection_ratio_raw,
        "threshold": max_adverse,
        "pass": metrics.adverse_selection_ratio_raw <= max_adverse,
        "informational": True,  # Gate 判定には影響しない (監視用)
    }

    # 092# E6: round-trip mean PnL (087# P1-1 / 083# §4.2-3)
    # mean が負でもテール損失管理の監視として重要
    if records is not None:
        filled_recs = [r for r in records if r.filled]
        if len(filled_recs) >= 2:
            rt_metrics, _ = compute_round_trip_metrics(records)
            if rt_metrics.total_pairs > 0:
                min_rt_pnl = thresholds.get("min_round_trip_pnl_mean", -2.0)
                checks["E6_round_trip_pnl"] = {
                    "value": rt_metrics.pnl_mean_bps,
                    "threshold": min_rt_pnl,
                    "pass": rt_metrics.pnl_mean_bps >= min_rt_pnl,
                    "pairs": rt_metrics.total_pairs,
                    "median": rt_metrics.pnl_median_bps,
                    "total_jpy": rt_metrics.pnl_total_jpy,
                    "informational": True,  # 当面は監視用、安定したら Gate 昇格
                }

                # E7: net inventory drift — 在庫偏り警告
                max_inventory = thresholds.get("max_net_inventory", 5)
                checks["E7_net_inventory"] = {
                    "value": abs(rt_metrics.net_inventory),
                    "threshold": max_inventory,
                    "pass": abs(rt_metrics.net_inventory) <= max_inventory,
                    "net_inventory": rt_metrics.net_inventory,
                    "unpaired_buys": rt_metrics.unpaired_buys,
                    "unpaired_sells": rt_metrics.unpaired_sells,
                    "informational": True,  # 監視用
                }

    # Gate 判定には informational=True のチェックは含めない
    gate_checks = {k: v for k, v in checks.items() if not v.get("informational")}
    all_pass = all(c["pass"] for c in gate_checks.values())

    # 047# Finding3: PROVISIONAL/INTERIM/FINAL の 3 段階判定
    # - PROVISIONAL: n<200 or days<3
    # - INTERIM: n>=200 & 3<=days<7 (暗定判定)
    # - FINAL: n>=200 & days>=7 (000# §3.3 準拠)
    if metrics.sample_sufficient:
        judgment_type = "FINAL"  # n>=200 & days>=7
    elif metrics.total_orders >= 200 and metrics.measurement_days >= 3:
        judgment_type = "INTERIM"  # 十分なサンプルだが 7 日未満
    else:
        judgment_type = "PROVISIONAL"

    return {
        "gate": "G1.1-exec",
        "gate_result": "PASS" if all_pass else "FAIL",
        "judgment_type": judgment_type,  # 020# O1 / 047# Finding3
        "sample_sufficient": metrics.sample_sufficient,
        "checks": checks,
        "metrics_summary": metrics.to_dict(),
    }

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
    checks: dict[str, dict] = {}

    # K1: attempted_fill_rate
    min_att_fill = thresholds.get("min_attempted_fill_rate", 0.60)
    checks["K1_attempted_fill_rate"] = {
        "value": metrics.attempted_fill_rate,
        "threshold": min_att_fill,
        "pass": metrics.attempted_fill_rate >= min_att_fill,
    }

    # K2: attempted_cancel_ratio
    max_att_cancel = thresholds.get("max_attempted_cancel_ratio", 0.40)
    checks["K2_attempted_cancel_ratio"] = {
        "value": metrics.attempted_cancel_ratio,
        "threshold": max_att_cancel,
        "pass": metrics.attempted_cancel_ratio <= max_att_cancel,
    }

    # K3: queue_wait_median
    max_wait = thresholds.get("max_queue_wait_median_sec", 120)
    checks["K3_queue_wait_median"] = {
        "value": metrics.queue_wait_median_sec,
        "threshold": max_wait,
        "pass": metrics.queue_wait_median_sec <= max_wait,
    }

    # K4: PnL 複合条件 — p < threshold かつ mean <= threshold で FAIL
    # 115# Q10.2(C): 単独 p 値判定は不十分。効果量条件を併設。
    pnl_kill_p = thresholds.get("pnl_kill_p_threshold", 0.02)
    pnl_kill_mean = thresholds.get("pnl_kill_mean_threshold", -0.8)
    pnl_is_significant = metrics.post_fill_30s_pnl_pvalue < pnl_kill_p
    pnl_is_large_loss = metrics.post_fill_30s_pnl_mean <= pnl_kill_mean
    # FAIL = 両条件同時成立
    k4_pass = not (pnl_is_significant and pnl_is_large_loss)
    checks["K4_pnl_kill"] = {
        "value": metrics.post_fill_30s_pnl_mean,
        "pvalue": metrics.post_fill_30s_pnl_pvalue,
        "threshold_p": pnl_kill_p,
        "threshold_mean": pnl_kill_mean,
        "significant": pnl_is_significant,
        "large_loss": pnl_is_large_loss,
        "pass": k4_pass,
    }

    # K5: 累積実損
    max_loss = thresholds.get("max_cumulative_loss_jpy", 10000)
    checks["K5_cumulative_loss"] = {
        "value": cumulative_loss_jpy,
        "threshold": max_loss,
        "pass": cumulative_loss_jpy < max_loss,
    }

    # K6: skip_gate_ratio
    max_skip = thresholds.get("max_skip_gate_ratio", 0.25)
    checks["K6_skip_gate_ratio"] = {
        "value": metrics.skip_gate_ratio,
        "threshold": max_skip,
        "pass": metrics.skip_gate_ratio <= max_skip,
    }

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

    return {
        "gate": "G1.1-quick",
        "gate_result": gate_result,
        "checks": checks,
        "watch": is_watch,
        "watch_detail": {
            "pnl_mean": metrics.post_fill_30s_pnl_mean,
            "pnl_pvalue": metrics.post_fill_30s_pnl_pvalue,
            "watch_thresholds": {"p": pnl_watch_p, "mean": pnl_watch_mean},
        } if is_watch else None,
        "metrics_summary": metrics.to_dict(),
    }

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
    checks: dict[str, dict] = {}

    # F1: attempted_fill_rate
    min_att_fill = thresholds.get("min_attempted_fill_rate", 0.70)
    checks["F1_attempted_fill_rate"] = {
        "value": metrics.attempted_fill_rate,
        "threshold": min_att_fill,
        "pass": metrics.attempted_fill_rate >= min_att_fill,
    }

    # F1b: overall_fill_rate (115# Q10.2(A): SkipGate 過剰回避)
    min_overall_fill = thresholds.get("min_overall_fill_rate", 0.62)
    checks["F1b_overall_fill_rate"] = {
        "value": metrics.overall_fill_rate,
        "threshold": min_overall_fill,
        "pass": metrics.overall_fill_rate >= min_overall_fill,
    }

    # F2: attempted_cancel_ratio
    max_att_cancel = thresholds.get("max_attempted_cancel_ratio", 0.30)
    checks["F2_attempted_cancel_ratio"] = {
        "value": metrics.attempted_cancel_ratio,
        "threshold": max_att_cancel,
        "pass": metrics.attempted_cancel_ratio <= max_att_cancel,
    }

    # F3: queue_wait_median
    max_wait = thresholds.get("max_queue_wait_median_sec", 60)
    checks["F3_queue_wait_median"] = {
        "value": metrics.queue_wait_median_sec,
        "threshold": max_wait,
        "pass": metrics.queue_wait_median_sec <= max_wait,
    }

    # F4: PnL — 有意に負でないこと (原初 E4 維持)
    pnl_alpha = thresholds.get("pnl_alpha", 0.05)

    # 122# B2: multi-timeframe PnL + Holm-Bonferroni 補正
    # 3 タイムフレーム (30s, 60s, 120s) の p値を収集し Holm 補正
    pnl_tests = [
        ("F4_pnl_30s", metrics.post_fill_30s_pnl_mean, metrics.post_fill_30s_pnl_pvalue,
         metrics.post_fill_30s_pnl_ci_upper),
        ("F4b_pnl_60s", metrics.post_fill_60s_pnl_mean, metrics.post_fill_60s_pnl_pvalue,
         None),
        ("F4c_pnl_120s", metrics.post_fill_120s_pnl_mean, metrics.post_fill_120s_pnl_pvalue,
         None),
    ]
    # Holm-Bonferroni: p値を昇順ソートし α/(m-rank) で比較
    # 「有意に負でない」ことを確認 → 有意に負 = FAIL
    raw_pvals = [(name, p) for name, _, p, _ in pnl_tests]
    sorted_pvals = sorted(raw_pvals, key=lambda x: x[1])
    m = len(sorted_pvals)
    holm_adjusted: dict[str, float] = {}
    for rank, (name, p_raw) in enumerate(sorted_pvals):
        # Holm 補正済み p値 = min(p_raw * (m - rank), 1.0)
        holm_adjusted[name] = min(p_raw * (m - rank), 1.0)

    for name, mean_val, p_raw, ci_upper in pnl_tests:
        p_holm = holm_adjusted[name]
        if mean_val >= 0:
            f_pass = True
        elif p_holm >= pnl_alpha:
            f_pass = True  # 負だが Holm 補正後も有意でない
        else:
            f_pass = False
        check_data: dict = {
            "value": mean_val,
            "pvalue_raw": p_raw,
            "pvalue_holm": round(p_holm, 6),
            "alpha": pnl_alpha,
            "pass": f_pass,
        }
        if ci_upper is not None:
            check_data["ci_upper"] = ci_upper
        checks[name] = check_data

    # F4d: PnL mean floor — 期待値がマイナスならリスク警告 (123# Gemini review Critical 1)
    # 「有意に負でない」だけでなく、平均自体が許容範囲内であることを要求
    pnl_mean_floor = thresholds.get("pnl_mean_floor_bps", -0.10)  # default: -0.10 bps
    pnl_mean_hard_floor = thresholds.get("pnl_mean_hard_floor_bps", -0.50)  # hard FAIL
    pnl_30s_mean = metrics.post_fill_30s_pnl_mean
    if pnl_30s_mean >= pnl_mean_floor:
        f4d_pass = True
        f4d_watch = False
    elif pnl_30s_mean >= pnl_mean_hard_floor:
        f4d_pass = True  # soft WATCH — 統計的に有意でなくても注意
        f4d_watch = True
    else:
        f4d_pass = False  # hard FAIL — 許容損失を超過
        f4d_watch = False
    checks["F4d_pnl_mean_floor"] = {
        "value": pnl_30s_mean,
        "floor": pnl_mean_floor,
        "hard_floor": pnl_mean_hard_floor,
        "pass": f4d_pass,
        "watch": f4d_watch,
    }

    # 後方互換: F4_pnl キーも維持 (旧テスト参照用)
    checks["F4_pnl"] = {
        "value": metrics.post_fill_30s_pnl_mean,
        "pvalue": metrics.post_fill_30s_pnl_pvalue,
        "ci_upper": metrics.post_fill_30s_pnl_ci_upper,
        "alpha": pnl_alpha,
        "pass": checks["F4_pnl_30s"]["pass"],
    }

    # F5: AS_ratio (115# Q10.2(B): 35→30)
    max_as = thresholds.get("max_adverse_selection_ratio", 0.30)
    checks["F5_adverse_selection"] = {
        "value": metrics.adverse_selection_ratio,
        "threshold": max_as,
        "pass": metrics.adverse_selection_ratio <= max_as,
    }

    # F6: skip_gate_ratio
    max_skip = thresholds.get("max_skip_gate_ratio", 0.20)
    checks["F6_skip_gate_ratio"] = {
        "value": metrics.skip_gate_ratio,
        "threshold": max_skip,
        "pass": metrics.skip_gate_ratio <= max_skip,
    }

    # F7: calendar_days
    min_days = thresholds.get("min_calendar_days", 7)
    checks["F7_calendar_days"] = {
        "value": metrics.measurement_days,
        "threshold": min_days,
        "pass": metrics.measurement_days >= min_days,
    }

    # F8: n_attempted
    min_n = thresholds.get("min_attempted_samples", 500)
    checks["F8_n_attempted"] = {
        "value": metrics.attempted_orders,
        "threshold": min_n,
        "pass": metrics.attempted_orders >= min_n,
    }

    all_pass = all(c["pass"] for c in checks.values())
    is_watch = any(c.get("watch", False) for c in checks.values())

    if not all_pass:
        gate_result = "FAIL"
    elif is_watch:
        gate_result = "WATCH"
    else:
        gate_result = "PASS"

    return {
        "gate": "G1.2-full",
        "gate_result": gate_result,
        "checks": checks,
        "watch": is_watch,
        "metrics_summary": metrics.to_dict(),
    }

# ======================================================================
# I/O utilities
# ======================================================================

def save_fill_records(records: Iterable[FillRecord], path: str | Path) -> None:
    """JSONL 形式で FillRecord を保存.

    032#17: バッチ全体を tempfile に書き出してから append する。
    SIGINT / ディスクフル時の不完全行混入を防止。
    """
    import os
    import shutil
    import tempfile

    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    payload_parts: list[str] = []
    count = 0
    for record in records:
        payload_parts.append(json.dumps(record.to_dict(), ensure_ascii=False))
        payload_parts.append("\n")
        count += 1
    payload = "".join(payload_parts)
    # Atomic batch: write to temp, fsync, then append to target
    tmp_fd, tmp_path = tempfile.mkstemp(dir=str(p.parent), suffix=".tmp")
    try:
        with os.fdopen(tmp_fd, "w", encoding="utf-8") as tmp_f:
            tmp_f.write(payload)
            tmp_f.flush()
            os.fsync(tmp_f.fileno())
        with open(p, "a", encoding="utf-8") as f:
            with open(tmp_path, "r", encoding="utf-8") as tmp_r:
                shutil.copyfileobj(tmp_r, f, length=1024 * 1024)
            f.flush()
            os.fsync(f.fileno())
    finally:
        try:
            os.remove(tmp_path)
        except OSError:
            pass
    logger.info(f"Saved {count} fill records to {p}")

def iter_fill_records(path: str | Path) -> Iterator[FillRecord]:
    """JSONL ファイルから FillRecord を逐次読み込み.

    032# #19: 破損行はスキップしてログ出力。
    101# §5: cycle_id による重複排除 (SIGINT 中断時の partial+emergency 重複対策)。
    349# P2: iter_jsonl_objects に委譲してパースロジックを一元化。
    """
    p = Path(path)
    if not p.exists():
        return
    seen_ids: set[str] = set()
    duplicates = 0
    loaded = 0
    for obj in iter_jsonl_objects(p, warn_malformed=True):
        try:
            rec = FillRecord.from_dict(obj)
        except (TypeError, KeyError) as e:
            logger.warning(f"Skipped invalid record in {p.name}: {e}")
            continue
        if rec.cycle_id in seen_ids:
            duplicates += 1
            continue
        seen_ids.add(rec.cycle_id)
        loaded += 1
        yield rec
    if duplicates:
        logger.info(f"Deduplicated {duplicates} records in {p.name}")
    logger.info(f"Loaded {loaded} fill records from {p}")

def load_fill_records(path: str | Path) -> list[FillRecord]:
    """JSONL ファイルから FillRecord を読み込み."""
    return list(iter_fill_records(path))


def detect_split_brain(
    records: Sequence[FillRecord],
    *,
    overlap_window_sec: float = 300.0,
) -> list[dict[str, object]]:
    """286# 283# P0-1: Split-Brain (多重起動) を事後検出する.

    同一時刻帯に複数の run_id または pid が記録を書き込んでいるケースを検出。
    FillRecord.pid (285# 追加) と run_id を使用。

    検出ロジック:
    - 隣接レコード間で run_id が異なり、かつ timestamp 差が overlap_window_sec 以内
      → Split-Brain イベントとして報告
    - pid が異なる場合も同様

    Args:
        records: 時系列順の FillRecord リスト
        overlap_window_sec: 時間重複を判定する窓幅 (秒)

    Returns:
        検出された Split-Brain イベントのリスト。各要素は:
        {"timestamp": float, "run_id_a": str, "run_id_b": str,
         "pid_a": int|None, "pid_b": int|None, "gap_sec": float}
    """
    if len(records) < 2:
        return []

    events: list[dict[str, object]] = []
    for i in range(1, len(records)):
        prev, curr = records[i - 1], records[i]
        # run_id チェック
        if prev.run_id and curr.run_id and prev.run_id != curr.run_id:
            gap = abs(curr.timestamp - prev.timestamp)
            if gap <= overlap_window_sec:
                events.append({
                    "timestamp": curr.timestamp,
                    "run_id_a": prev.run_id,
                    "run_id_b": curr.run_id,
                    "pid_a": prev.pid,
                    "pid_b": curr.pid,
                    "gap_sec": gap,
                })
        # 同一 run_id でも pid が異なるケース (プロセス入替の検出)
        elif (
            prev.pid is not None
            and curr.pid is not None
            and prev.pid != curr.pid
            and prev.run_id == curr.run_id
        ):
            gap = abs(curr.timestamp - prev.timestamp)
            if gap <= overlap_window_sec:
                events.append({
                    "timestamp": curr.timestamp,
                    "run_id_a": prev.run_id,
                    "run_id_b": curr.run_id,
                    "pid_a": prev.pid,
                    "pid_b": curr.pid,
                    "gap_sec": gap,
                })

    if events:
        logger.critical(
            f"[286# SPLIT-BRAIN] {len(events)} overlapping process events "
            f"detected! Multiple processes wrote to the same JSONL. "
            f"First event: run_ids={events[0].get('run_id_a')}/{events[0].get('run_id_b')}, "
            f"pids={events[0].get('pid_a')}/{events[0].get('pid_b')}"
        )
    return events


def fill_records_to_dataframe(records: Iterable[FillRecord]) -> "pd.DataFrame":
    """FillRecord iterable を DataFrame に変換する."""
    import pandas as pd

    return pd.DataFrame.from_records(record.to_dict() for record in records)

def _normalize_fill_record_date(value: str | None) -> str | None:
    if value is None:
        return None
    normalized = value.replace("-", "")
    return normalized if len(normalized) == 8 and normalized.isdigit() else None

def _extract_fill_record_file_date(path: Path) -> str | None:
    date_part = path.stem.split("_")[-1]
    return date_part if len(date_part) == 8 and date_part.isdigit() else None


def _directory_signature(path: Path) -> tuple[int, int]:
    """Directory signature for cache invalidation."""
    try:
        st = path.stat()
    except OSError:
        return -1, -1
    return st.st_mtime_ns, st.st_ctime_ns


def _scan_jsonl_by_prefix(directory: Path, prefix: str) -> list[Path]:
    if not directory.is_dir():
        return []
    try:
        files = [
            path
            for path in directory.iterdir()
            if path.is_file()
            and path.name.startswith(prefix)
            and path.suffix == ".jsonl"
        ]
    except OSError:
        return []
    return sorted(files)


@lru_cache(maxsize=256)
def _list_fill_record_files_cached(
    directory: str,
    include_emergency: bool,
    root_sig: tuple[int, int],
    emergency_sig: tuple[int, int],
) -> tuple[Path, ...]:
    del root_sig, emergency_sig  # signature-only cache keys
    root = Path(directory)
    files = _scan_jsonl_by_prefix(root, "fill_records_")
    if include_emergency:
        files.extend(_scan_jsonl_by_prefix(root / "emergency", "emergency_"))
    return tuple(files)


def _resolve_fill_record_files_by_date_range(
    directory: Path,
    *,
    include_emergency: bool,
    start_date: str,
    end_date: str,
) -> list[Path] | None:
    """Resolve date-bounded file list directly without directory scan."""
    try:
        start = datetime.strptime(start_date, "%Y%m%d")
        end = datetime.strptime(end_date, "%Y%m%d")
    except ValueError:
        return None
    if start > end:
        return []
    if start_date == end_date:
        files: list[Path] = []
        fill_path = directory / f"fill_records_{start_date}.jsonl"
        if fill_path.is_file():
            files.append(fill_path)
        if include_emergency:
            emergency_path = directory / "emergency" / f"emergency_{start_date}.jsonl"
            if emergency_path.is_file():
                files.append(emergency_path)
        return files
    day_count = (end - start).days
    if day_count > 3650:
        return None

    files: list[Path] = []
    for i in range(day_count + 1):
        day_str = (start + timedelta(days=i)).strftime("%Y%m%d")
        fill_path = directory / f"fill_records_{day_str}.jsonl"
        if fill_path.is_file():
            files.append(fill_path)
    if include_emergency:
        emergency_dir = directory / "emergency"
        for i in range(day_count + 1):
            day_str = (start + timedelta(days=i)).strftime("%Y%m%d")
            emergency_path = emergency_dir / f"emergency_{day_str}.jsonl"
            if emergency_path.is_file():
                files.append(emergency_path)
    return files

def list_fill_record_files(
    directory: str | Path,
    *,
    include_emergency: bool = True,
    start_date: str | None = None,
    end_date: str | None = None,
) -> list[Path]:
    """fill record 系 JSONL ファイルを順序付きで列挙する."""
    d = Path(directory)
    norm_start = _normalize_fill_record_date(start_date)
    norm_end = _normalize_fill_record_date(end_date)

    if norm_start is not None and norm_end is not None:
        direct_files = _resolve_fill_record_files_by_date_range(
            d,
            include_emergency=include_emergency,
            start_date=norm_start,
            end_date=norm_end,
        )
        if direct_files is not None:
            return direct_files

    def _within_date(path: Path) -> bool:
        if norm_start is None and norm_end is None:
            return True
        file_date = _extract_fill_record_file_date(path)
        if file_date is None:
            return False
        if norm_start is not None and file_date < norm_start:
            return False
        if norm_end is not None and file_date > norm_end:
            return False
        return True

    root_sig = _directory_signature(d)
    emergency_sig = _directory_signature(d / "emergency") if include_emergency else (-1, -1)
    all_files = _list_fill_record_files_cached(
        str(d),
        include_emergency,
        root_sig,
        emergency_sig,
    )
    return [path for path in all_files if _within_date(path)]

def iter_fill_record_objects_from_files(files: Iterable[Path]) -> Iterator[dict[str, object]]:
    """指定ファイル群から raw object を逐次読み込み、cycle_id を跨いで重複排除する."""
    seen_ids: set[str] = set()
    for path in files:
        for record in iter_jsonl_objects(path, warn_malformed=True):
            cycle_id = record.get("cycle_id")
            if isinstance(cycle_id, str):
                if cycle_id in seen_ids:
                    continue
                seen_ids.add(cycle_id)
            yield record

def iter_fill_record_objects_glob(
    directory: str | Path,
    *,
    include_emergency: bool = True,
    start_date: str | None = None,
    end_date: str | None = None,
) -> Iterator[dict[str, object]]:
    """ディレクトリ内の全 JSONL から raw object を逐次読み込み.

    FillRecord dataclass を経由せず、analysis 系で使う dict payload をそのまま返す。
    cycle_id が文字列のものは cross-file で重複排除する。
    """
    yield from iter_fill_record_objects_from_files(
        list_fill_record_files(
            directory,
            include_emergency=include_emergency,
            start_date=start_date,
            end_date=end_date,
        )
    )

def load_fill_record_objects_glob(
    directory: str | Path,
    *,
    include_emergency: bool = True,
    start_date: str | None = None,
    end_date: str | None = None,
) -> list[dict[str, object]]:
    """ディレクトリ内の全 JSONL から raw object を読み込み."""
    return list(
        iter_fill_record_objects_glob(
            directory,
            include_emergency=include_emergency,
            start_date=start_date,
            end_date=end_date,
        )
    )

def _coerce_filter_timestamp(value: object) -> float | None:
    """filter 用 timestamp を有限 float に正規化する."""
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        ts = float(value)
        if math.isfinite(ts):
            return ts
    return None

def apply_fill_record_filters(
    records: Iterable[Mapping[str, object]],
    *,
    run_id: str | None = None,
    git_sha: str | None = None,
    date_from: str | None = None,
    date_to: str | None = None,
) -> tuple[list[dict[str, object]], dict[str, str | None]]:
    """fill record dict に run_id/git_sha/date フィルタを適用する."""
    ts_from: float | None = None
    ts_to: float | None = None
    if date_from:
        ts_from = datetime.strptime(date_from, "%Y-%m-%d").replace(
            tzinfo=timezone.utc
        ).timestamp()
    if date_to:
        ts_to = (
            datetime.strptime(date_to, "%Y-%m-%d").replace(
                tzinfo=timezone.utc
            ).timestamp()
            + 86400
        )

    filtered: list[dict[str, object]] = []
    for record in records:
        if run_id and record.get("run_id") != run_id:
            continue
        if git_sha and not str(record.get("git_sha", "")).startswith(git_sha):
            continue
        timestamp = _coerce_filter_timestamp(record.get("timestamp")) or 0.0
        if ts_from is not None and timestamp < ts_from:
            continue
        if ts_to is not None and timestamp >= ts_to:
            continue
        filtered.append(record if isinstance(record, dict) else dict(record))

    filters = {
        "run_id": run_id,
        "git_sha": git_sha,
        "date_from": date_from,
        "date_to": date_to,
    }
    return filtered, filters

def _iter_fill_record_files(
    directory: Path,
    *,
    include_emergency: bool = True,
) -> Iterator[Path]:
    """内部互換: fill record 系 JSONL の対象ファイルを順序付きで列挙."""
    yield from list_fill_record_files(directory, include_emergency=include_emergency)

def iter_fill_records_glob(
    directory: str | Path,
    *,
    include_emergency: bool = True,
) -> Iterator[FillRecord]:
    """ディレクトリ内の全 JSONL ファイルから FillRecord を逐次読み込み.

    101# §5: cross-file 重複排除 (emergency dump との重複対策)."""
    d = Path(directory)
    files = list_fill_record_files(d, include_emergency=include_emergency)
    if not files:
        return
    if len(files) == 1:
        yield from iter_fill_records(files[0])
        return
    seen_ids: set[str] = set()
    for path in files:
        for record in iter_fill_records(path):
            if record.cycle_id in seen_ids:
                continue
            seen_ids.add(record.cycle_id)
            yield record

def load_fill_records_glob(
    directory: str | Path,
    *,
    include_emergency: bool = True,
) -> list[FillRecord]:
    """ディレクトリ内の全 JSONL ファイルから FillRecord を読み込み."""
    return list(iter_fill_records_glob(directory, include_emergency=include_emergency))

def partition_clean_records(
    records: Iterable[FillRecord],
    *,
    require_git_sha: bool = True,
) -> tuple[list[FillRecord], list[FillRecord]]:
    """046# clean/quarantine 分離 + 047# A5 拡張基準 (iterable 対応).

    以下のいずれかに該当するレコードを quarantine へ分類:
    - git_sha が blank/None (ゾンビプロセス由来)
    - run_id が blank/None (020# O4 以前の旧形式レコード)
    - 必須フィールド (side, order_price, order_quantity) が不正

    Args:
        records: 全 FillRecord iterable.
        require_git_sha: True の場合 git_sha が blank/None のレコードを quarantine.
            False の場合は全チェックをバイパスして全件 clean を返す.

    Returns:
        (clean, quarantine) のタプル.
    """
    if not require_git_sha:
        if isinstance(records, list):
            return records, []  # 既存互換: list 入力時は同一参照を返す
        return list(records), []  # テスト用: 全件 clean (本番は常に True)

    clean: list[FillRecord] = []
    quarantine: list[FillRecord] = []
    total = 0
    for r in records:
        total += 1
        # 047# A5: 複合チェック — git_sha + run_id + 必須フィールド
        reason = _quarantine_reason(r)
        if reason:
            quarantine.append(r)
        else:
            clean.append(r)

    if quarantine:
        logger.info(
            f"[quarantine] {len(quarantine)}/{total} records quarantined. "
            f"clean={len(clean)}"
        )
    return clean, quarantine

def filter_clean_records(
    records: list[FillRecord],
    *,
    require_git_sha: bool = True,
) -> tuple[list[FillRecord], list[FillRecord]]:
    """046# clean/quarantine 分離 + 047# A5 拡張基準.

    list 入力の既存 API。新規コードでは `partition_clean_records()` を優先。
    """
    return partition_clean_records(records, require_git_sha=require_git_sha)

def _quarantine_reason(r: FillRecord) -> str | None:
    """047# A5: レコードの quarantine 理由を返す (None=clean)."""
    if not (r.git_sha and r.git_sha.strip()):
        return "blank_git_sha"
    if not (r.run_id and r.run_id.strip()):
        return "blank_run_id"
    # 144# #3: cancel_reason バイパスは「監査系 reason + side=none」に限定
    # 145# §9-#6: 定数化 → cancel_reasons モジュールに集約
    from scripts.v460.lib.cancel_reasons import AUDIT_CANCEL_REASONS
    _cancel = getattr(r, "cancel_reason", None)
    _is_audit = _cancel in AUDIT_CANCEL_REASONS and r.side in ("none", "buy", "sell")
    if r.side not in ("buy", "sell"):
        if _is_audit:
            return None  # 監査レコードは clean 扱い
        return f"invalid_side={r.side}"
    if not r.order_price or r.order_price <= 0:
        if not _is_audit:
            return "invalid_order_price"
    if not r.order_quantity or r.order_quantity <= 0:
        if not _is_audit:
            return "invalid_order_quantity"
    return None

# ======================================================================
# 051# P2-2: Round-trip 評価 (buy→sell ペアリング)
# ======================================================================

@dataclass
class RoundTripRecord:
    """往復取引記録 (buy→sell / sell→buy 双方向対応).

    055# Fix: sell先行ペアも対称的に評価.
    """

    entry_record: FillRecord
    exit_record: FillRecord
    pnl_bps: float  # エントリー基準の損益 (bps)
    pnl_jpy: float  # 実損益 (JPY)
    hold_sec: float  # 保持時間 (秒)
    direction: str  # "buy_first" or "sell_first"

    # 後方互換: buy_record/sell_record プロパティ
    @property
    def buy_record(self) -> FillRecord:
        return self.entry_record if self.direction == "buy_first" else self.exit_record

    @property
    def sell_record(self) -> FillRecord:
        return self.exit_record if self.direction == "buy_first" else self.entry_record

@dataclass
class RoundTripMetrics:
    """Round-trip 集計指標.

    055# Fix: unpaired_sells / net_inventory 追加.
    """

    total_pairs: int = 0
    pnl_mean_bps: float = 0.0
    pnl_median_bps: float = 0.0
    pnl_std_bps: float = 0.0
    pnl_total_jpy: float = 0.0
    win_rate: float = 0.0  # pnl > 0 の割合
    hold_sec_median: float = 0.0
    unpaired_buys: int = 0  # ペアリング未完了の buy 件数
    unpaired_sells: int = 0  # 055# ペアリング未完了の sell 件数
    net_inventory: int = 0  # 055# 純在庫 (unpaired_buys - unpaired_sells)

def compute_round_trip_metrics(
    records: list[FillRecord],
) -> tuple[RoundTripMetrics, list[RoundTripRecord]]:
    """051# P2-2 → 055# Fix: 双方向 FIFO ペアリングで往復損益を算出.

    inventory-aware 方式:
    - net inventory を追跡し、buy/sell どちらが先でもペアリング.
    - buy 先行時: sell が来たら close. PnL = (sell - buy) / buy
    - sell 先行時: buy が来たら close. PnL = (sell - buy) / buy (空売り利益)

    Args:
        records: FillRecord リスト (時系列ソート済み想定).

    Returns:
        (RoundTripMetrics, list[RoundTripRecord]) タプル.
    """
    filled = [r for r in records if r.filled and r.fill_price is not None]
    filled.sort(key=lambda r: r.timestamp)

    pending_buys: deque[FillRecord] = deque()
    pending_sells: deque[FillRecord] = deque()
    trips: list[RoundTripRecord] = []

    for r in filled:
        if r.side == "buy":
            if pending_sells:
                # sell 先行 → buy で close
                sell_entry = pending_sells.popleft()  # FIFO
                sell_price = sell_entry.fill_price
                buy_price = r.fill_price
                if sell_price is None or buy_price is None:
                    continue
                pnl_bps = (sell_price - buy_price) / buy_price * 10_000
                qty = min(r.order_quantity, sell_entry.order_quantity)
                pnl_jpy = (sell_price - buy_price) * qty
                hold_sec = r.timestamp - sell_entry.timestamp
                trips.append(RoundTripRecord(
                    entry_record=sell_entry,
                    exit_record=r,
                    pnl_bps=pnl_bps,
                    pnl_jpy=pnl_jpy,
                    hold_sec=hold_sec,
                    direction="sell_first",
                ))
            else:
                pending_buys.append(r)
        elif r.side == "sell":
            if pending_buys:
                # buy 先行 → sell で close
                buy_entry = pending_buys.popleft()  # FIFO
                sell_price = r.fill_price
                buy_price = buy_entry.fill_price
                if sell_price is None or buy_price is None:
                    continue
                pnl_bps = (sell_price - buy_price) / buy_price * 10_000
                qty = min(r.order_quantity, buy_entry.order_quantity)
                pnl_jpy = (sell_price - buy_price) * qty
                hold_sec = r.timestamp - buy_entry.timestamp
                trips.append(RoundTripRecord(
                    entry_record=buy_entry,
                    exit_record=r,
                    pnl_bps=pnl_bps,
                    pnl_jpy=pnl_jpy,
                    hold_sec=hold_sec,
                    direction="buy_first",
                ))
            else:
                pending_sells.append(r)

    if not trips:
        return RoundTripMetrics(
            unpaired_buys=len(pending_buys),
            unpaired_sells=len(pending_sells),
            net_inventory=len(pending_buys) - len(pending_sells),
        ), []

    pnl_arr = [t.pnl_bps for t in trips]
    hold_arr = [t.hold_sec for t in trips]

    return RoundTripMetrics(
        total_pairs=len(trips),
        pnl_mean_bps=float(np.mean(pnl_arr)),
        pnl_median_bps=float(np.median(pnl_arr)),
        pnl_std_bps=float(np.std(pnl_arr)),
        pnl_total_jpy=sum(t.pnl_jpy for t in trips),
        win_rate=sum(1 for p in pnl_arr if p > 0) / len(pnl_arr),
        hold_sec_median=float(np.median(hold_arr)),
        unpaired_buys=len(pending_buys),
        unpaired_sells=len(pending_sells),
        net_inventory=len(pending_buys) - len(pending_sells),
    ), trips

# ======================================================================
# 051# P2-4: レジーム別メトリクス
# ======================================================================

@dataclass
class GroupedMetricsBase:
    """グループ集計で共通となる損益・AS 指標."""

    count: int = 0
    filled: int = 0
    pnl_mean_bps: float = 0.0
    as_ratio: float = 0.0

@dataclass
class RegimeMetrics(GroupedMetricsBase):
    """レジーム別の集計指標."""

    regime: str = "unknown"
    fill_rate: float = 0.0
    queue_wait_median_sec: float = 0.0

def _summarize_filled_records(
    records: list[FillRecord],
    *,
    include_queue_wait: bool = False,
) -> tuple[int, float, float, float]:
    """filled レコードの共通集計を単一パスで算出."""
    filled_count = 0
    pnl_acc = PnlAccumulator()
    as_total = 0
    as_positive = 0
    queue_waits: list[float] = []

    for rec in records:
        if not rec.filled:
            continue
        filled_count += 1
        pnl_acc.add(rec.post_fill_30s_pnl)
        if rec.adverse_selected is not None:
            as_total += 1
            if rec.adverse_selected:
                as_positive += 1
        if include_queue_wait and rec.queue_wait_sec > 0:
            queue_waits.append(rec.queue_wait_sec)

    as_ratio = (as_positive / as_total) if as_total else 0.0
    wait_median = float(np.median(queue_waits)) if queue_waits else 0.0
    return filled_count, pnl_acc.mean_bps, as_ratio, wait_median

def compute_regime_metrics(records: list[FillRecord]) -> list[RegimeMetrics]:
    """051# P2-4: レジーム別にメトリクスを算出.

    Args:
        records: FillRecord リスト.

    Returns:
        RegimeMetrics のリスト (レジーム名でソート).
    """
    from collections import defaultdict

    groups: dict[str, list[FillRecord]] = defaultdict(list)
    for r in records:
        regime = r.regime or "unknown"
        groups[regime].append(r)

    result: list[RegimeMetrics] = []
    for regime_name in sorted(groups.keys()):
        recs = groups[regime_name]
        filled_count, pnl_mean, as_ratio, wait_median = _summarize_filled_records(
            recs,
            include_queue_wait=True,
        )

        result.append(RegimeMetrics(
            regime=regime_name,
            count=len(recs),
            filled=filled_count,
            fill_rate=filled_count / len(recs) if recs else 0.0,
            pnl_mean_bps=pnl_mean,
            as_ratio=as_ratio,
            queue_wait_median_sec=wait_median,
        ))
    return result

# ======================================================================
# 051# UTC 時間帯別分析
# ======================================================================

@dataclass
class HourlyMetrics(GroupedMetricsBase):
    """UTC 時間帯別の集計指標."""

    utc_hour: int = 0

def compute_hourly_metrics(records: list[FillRecord]) -> list[HourlyMetrics]:
    """051# UTC 時間帯別にメトリクスを算出.

    time_filter 検証用: 各 UTC hour の AS/PnL を可視化.

    Args:
        records: FillRecord リスト.

    Returns:
        HourlyMetrics のリスト (utc_hour 昇順).
    """
    from collections import defaultdict

    groups: dict[int, list[FillRecord]] = defaultdict(list)
    for r in records:
        utc_hour = datetime.fromtimestamp(r.timestamp, tz=timezone.utc).hour
        groups[utc_hour].append(r)

    result: list[HourlyMetrics] = []
    for hour in range(24):
        if hour not in groups:
            continue
        recs = groups[hour]
        filled_count, pnl_mean, as_ratio, _wait_median_unused = _summarize_filled_records(
            recs,
            include_queue_wait=False,
        )

        result.append(HourlyMetrics(
            utc_hour=hour,
            count=len(recs),
            filled=filled_count,
            pnl_mean_bps=pnl_mean,
            as_ratio=as_ratio,
        ))
    return result
