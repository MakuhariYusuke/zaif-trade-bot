"""155# 後知恵フィルター分析 — 「こうしたら儲かったのに」を定量化.

fill_records の全レコードを時系列で再構成し、
スキップ/タイムアウト/キャンセルされたサイクルが
実行されていたら得られたであろう PnL を推定する。

分析カテゴリ:
  H1: skip_gate で見逃した利益機会
  H2: timeout で逃した注文
  H3: side 選択ミス (buy/sell の逆が良かった)
  H4: 時間帯別の機会損失
  H5: balance_forced_skip (P0-08) による機会損失

Usage:
    python -m scripts.v460.analysis.hindsight_filter
    python -m scripts.v460.analysis.hindsight_filter --start 2026-02-17 --end 2026-02-23
"""

from __future__ import annotations

import argparse
import bisect
import json
import logging

from ztb.utils.safety import safe_to_finite
import sys
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import TypeAlias, TypedDict

from scripts.v460.analysis.reproduce_152_metrics import _load_records
from scripts.v460.lib import cancel_reasons as CR
from ztb.metrics.fill_quality import PnlAccumulator

logger = logging.getLogger(__name__)


RawRecord: TypeAlias = Mapping[str, object]


class AggregatePnlSummary(TypedDict):
    count: int
    avg_pnl_30s: float
    profitable_pct: float
    total_pnl: float


class WaitBandSummary(AggregatePnlSummary):
    """queue wait band stats."""


class RegimeSideSummary(AggregatePnlSummary):
    """regime × side stats."""


class SideReversalSummary(TypedDict):
    total_filled: int
    reverse_better_count: int
    reverse_better_pct: float
    avg_actual_pnl: float
    avg_reverse_pnl: float


class HourlySummary(TypedDict):
    total: int
    filled: int
    skipped: int
    avg_hindsight_30s: float
    filled_avg: float
    skipped_avg: float
    profitable_skipped: int


class SkipGateBinSummary(TypedDict):
    count: int
    avg_pnl_30s: float
    profitable_pct: float
    total_profit_bps: float
    total_loss_bps: float


class SkipGateThresholdSummary(TypedDict):
    would_execute: int
    would_skip: int
    avg_exec_pnl: float
    total_exec_pnl: float


class SkipGateCalibrationSummary(TypedDict):
    by_as_prob_bin: dict[str, SkipGateBinSummary]
    threshold_simulation: dict[str, SkipGateThresholdSummary]


class SkipGateCalibrationNote(TypedDict):
    note: str


SkipGateCalibrationReport: TypeAlias = SkipGateCalibrationSummary | SkipGateCalibrationNote


class InterpolatedSplitStats(TypedDict):
    count: int
    with_pnl: int
    avg_hindsight_30s: float | None


class InterpolatedStats(TypedDict):
    interpolated: InterpolatedSplitStats
    original_price: InterpolatedSplitStats


# ---------------------------------------------------------------------------
# 172# EV_per_cycle — Codex R3 + Gemini 9.4 推奨指標
# EV = fill_prob × avg_pnl_if_filled (bps/cycle)
# ---------------------------------------------------------------------------

class EvPerCycleSummary(TypedDict):
    """1 グループの EV_per_cycle 集計."""
    total_cycles: int
    filled_cycles: int
    fill_prob: float
    avg_pnl_if_filled: float  # bps (actual 30s)
    ev_per_cycle: float       # bps
    # guard 評価用: ブロックされたサイクルの後知恵 PnL
    blocked_cycles: int
    avg_hindsight_blocked: float | None  # bps (hindsight 30s)
    guard_value: float | None  # ev_per_cycle - avg_hindsight_blocked


class EvPerCycleReport(TypedDict):
    """EV_per_cycle 分析全体."""
    overall: EvPerCycleSummary
    by_regime_side: dict[str, EvPerCycleSummary]
    by_guard: dict[str, EvPerCycleSummary]


def _to_str(value: object | None, *, default: str = "") -> str:
    if isinstance(value, str):
        return value
    if value is None:
        return default
    return str(value)


def _to_optional_str(value: object | None) -> str | None:
    text = _to_str(value, default="")
    return text if text else None


def _pct(part: int, total: int, *, digits: int = 1) -> float:
    return round(part / total * 100, digits) if total else 0.0


def _mean(values: Sequence[float]) -> float | None:
    return (sum(values) / len(values)) if values else None


def _round_acc_mean(accumulator: PnlAccumulator, *, digits: int = 4) -> float:
    return round(accumulator.mean_bps, digits) if accumulator.count else 0.0


# ---------------------------------------------------------------------------
# Price timeline reconstruction
# ---------------------------------------------------------------------------

@dataclass
class PricePoint:
    """タイムライン上の価格ポイント."""
    timestamp: float
    price: float


def _build_price_timeline(records: Sequence[RawRecord]) -> list[PricePoint]:
    """全レコードの order_price + filled records の mid_at_fill/mid_Xs_after から
    価格タイムラインを構築."""
    points: list[PricePoint] = []

    for r in records:
        ts_f = safe_to_finite(r.get("timestamp"))
        if ts_f is None:
            continue

        # order_price は注文時の mid_price の近似
        op_f = safe_to_finite(r.get("order_price"))
        if op_f is not None and op_f > 0:
            points.append(PricePoint(ts_f, op_f))

        # filled records には mid_at_fill, mid_30s_after 等がある
        if bool(r.get("filled")):
            for field_name, offset in [
                ("mid_at_fill", 0),
                ("mid_30s_after", 30),
                ("mid_60s_after", 60),
                ("mid_120s_after", 120),
            ]:
                mid_price = safe_to_finite(r.get(field_name))
                if mid_price is not None:
                    points.append(PricePoint(ts_f + offset, mid_price))

    # 時系列ソート & 同一 timestamp の重複除去 (末尾で上書き)
    points.sort(key=lambda p: p.timestamp)
    if not points:
        return []

    deduped: list[PricePoint] = [points[0]]
    for point in points[1:]:
        if point.timestamp == deduped[-1].timestamp:
            deduped[-1] = point
            continue
        deduped.append(point)
    return deduped


def _interpolate_price(
    timeline: Sequence[PricePoint],
    ts: float,
    *,
    timestamps: Sequence[float] | None = None,
) -> float | None:
    """タイムライン上の指定時刻の価格を線形補間."""
    if not timeline:
        return None

    ts_index = timestamps if timestamps is not None else [point.timestamp for point in timeline]
    idx = bisect.bisect_left(ts_index, ts)

    if idx == 0:
        return timeline[0].price if abs(timeline[0].timestamp - ts) < 300 else None
    if idx >= len(timeline):
        return timeline[-1].price if abs(timeline[-1].timestamp - ts) < 300 else None

    p0, p1 = timeline[idx - 1], timeline[idx]
    # 5分以上離れていたら補間しない
    interval = p1.timestamp - p0.timestamp
    if interval <= 0 or interval > 300:
        return None

    ratio = (ts - p0.timestamp) / interval
    return p0.price + ratio * (p1.price - p0.price)


# ---------------------------------------------------------------------------
# Hindsight PnL calculation
# ---------------------------------------------------------------------------

@dataclass
class HindsightResult:
    """1 レコードの後知恵分析結果."""
    cycle_id: str
    timestamp: float
    side: str
    order_price: float
    cancel_reason: str
    filled: bool
    actual_pnl_30s: float | None
    # 後知恵PnL (order_price と Xs 後の mid_price の差)
    hindsight_pnl_30s: float | None
    hindsight_pnl_60s: float | None
    hindsight_pnl_120s: float | None
    # 逆サイドだったらの PnL
    reverse_pnl_30s: float | None
    skip_gate_score: float | None
    skip_gate_as_prob: float | None
    regime: str | None
    # §9.4 #1: order_price=0 時に補間で得た疑似参照価格
    interpolated_ref: bool = False
    # §9.2 #3: queue_wait_sec (fill 待機時間)
    queue_wait_sec: float | None = None


def _compute_hindsight_pnl(
    side: str,
    order_price: float,
    future_price: float | None,
) -> float | None:
    """side と order_price, future_price から後知恵 PnL (bps) を計算."""
    if future_price is None or order_price <= 0:
        return None
    diff_bps = (future_price - order_price) / order_price * 10000
    # buy: price goes up → profit, sell: price goes down → profit
    return diff_bps if side == "buy" else -diff_bps


def _analyze_records(
    records: Sequence[RawRecord],
    timeline: Sequence[PricePoint],
) -> list[HindsightResult]:
    """全レコードの後知恵PnLを計算.

    §9.4 #1: order_price=0 のレコードも timestamp 基準の補間価格を
    疑似参照価格として使い、H5/H6 の後知恵 PnL を出す。
    """
    results: list[HindsightResult] = []
    timeline_ts = [point.timestamp for point in timeline]
    _skipped_invalid_side = 0  # 156# §16: 除外カウント

    for r in records:
        ts_f = safe_to_finite(r.get("timestamp"))
        op_f = safe_to_finite(r.get("order_price"))
        if ts_f is None or op_f is None:
            continue

        side = _to_str(r.get("side"), default="unknown")

        # 156# §10 #2: buy/sell 以外の side を除外 (符号歪み防止)
        if side not in ("buy", "sell"):
            _skipped_invalid_side += 1
            continue

        # §9.4 #1: price=0 → 補間で疑似参照価格を取得
        interpolated = False
        if op_f <= 0:
            interp = _interpolate_price(timeline, ts_f, timestamps=timeline_ts)
            if interp is None:
                continue  # タイムラインからも取れない → 分析不能
            op_f = interp
            interpolated = True

        # 未来の価格を補間
        p30 = _interpolate_price(timeline, ts_f + 30, timestamps=timeline_ts)
        p60 = _interpolate_price(timeline, ts_f + 60, timestamps=timeline_ts)
        p120 = _interpolate_price(timeline, ts_f + 120, timestamps=timeline_ts)

        h30 = _compute_hindsight_pnl(side, op_f, p30)
        h60 = _compute_hindsight_pnl(side, op_f, p60)
        h120 = _compute_hindsight_pnl(side, op_f, p120)

        # 逆サイド
        rev_side = "sell" if side == "buy" else "buy"
        rev30 = _compute_hindsight_pnl(rev_side, op_f, p30)

        # §9.2 #3: queue_wait_sec
        qw_f = safe_to_finite(r.get("queue_wait_sec"))

        results.append(HindsightResult(
            cycle_id=_to_str(r.get("cycle_id"), default=""),
            timestamp=ts_f,
            side=side,
            order_price=op_f,
            cancel_reason=_to_str(r.get("cancel_reason"), default=""),
            filled=bool(r.get("filled")),
            actual_pnl_30s=safe_to_finite(r.get("post_fill_30s_pnl")),
            hindsight_pnl_30s=h30,
            hindsight_pnl_60s=h60,
            hindsight_pnl_120s=h120,
            reverse_pnl_30s=rev30,
            skip_gate_score=safe_to_finite(r.get("skip_gate_score")),
            skip_gate_as_prob=safe_to_finite(r.get("skip_gate_as_prob")),
            regime=_to_optional_str(r.get("regime")),
            interpolated_ref=interpolated,
            queue_wait_sec=qw_f,
        ))

    if _skipped_invalid_side > 0:
        logger.info(
            f"[hindsight] Excluded {_skipped_invalid_side} records "
            f"with invalid side (not buy/sell)"
        )

    return results


# ---------------------------------------------------------------------------
# Analysis reports
# ---------------------------------------------------------------------------

@dataclass
class CategoryAnalysis:
    """カテゴリ別分析結果."""
    category: str
    count: int
    avg_hindsight_30s: float | None
    avg_hindsight_60s: float | None
    avg_hindsight_120s: float | None
    profitable_30s_count: int
    profitable_30s_pct: float
    total_missed_profit_30s: float  # sum of positive hindsight PnL
    total_missed_profit_120s: float
    best_case: HindsightResult | None
    worst_case: HindsightResult | None


_DIRECT_CATEGORY_BY_REASON: dict[str, str] = {
    CR.SKIP_GATE: "H1_skip_gate",
    CR.TIMEOUT: "H2_timeout",
    # 348# balance_forced 撤廃: CR.BALANCE_FORCED_SKIP を削除 (旧 H5)
    "balance_forced_skip": "H5_balance_forced",  # 後方互換: 既存レコード用
    CR.DAILY_DRAWDOWN_HALT: "H9_daily_drawdown",  # 173#
    CR.HARD_SKIP_UTC_HOUR: "H10_hard_skip_hour",   # 205# §9.4
    CR.TOXIC_FILL_SIDE_VETO: "H11_toxic_veto",     # 205# §9.2
    CR.PER_SIDE_DD_HALT: "H12_per_side_dd",         # 205# §9.5
}
# 156# §10 #3/#4: cancel_reasons 定数と同期し技術要因を一括分類
_TECHNICAL_REASONS = frozenset({
    CR.POST_ONLY_REJECT,
    "postonly_reject",       # レガシー互換 (order_monitor 旧出力)
    CR.ORDERBOOK_ERROR,
    CR.ORDERBOOK_TIMEOUT,    # 130# 細分化
    CR.ORDERBOOK_RATE_LIMIT, # 130# 細分化
    CR.ORDERBOOK_EMPTY,      # 130# 細分化
    CR.SELL_GUARD_REJECT,    # 088# sell ガード
    CR.API_ERROR,
    CR.STALE_SKIP_GATE_BLOCKED,
    CR.STALE_REPRICE_FAILED,
})
_REGIME_GUARD_REASONS = frozenset({
    CR.UNKNOWN_REGIME_BUY_SKIP,
    CR.UNKNOWN_REGIME_SELL_SKIP,  # 173# 追加
    CR.SELL_DYNAMIC_KILL,
    CR.BUY_DYNAMIC_KILL,   # 157# §19
    CR.TRENDING_SELL_SKIP,
    CR.RANGING_LOW_VOL_SKIP,  # 173# 169# B1' 追加
    CR.SKIP_GATE_RULE_VELOCITY_SELL,  # 173# 165# AS-R1 追加
    CR.SKIP_GATE_RULE_VELOCITY_BUY,   # 173# 165# AS-R1 追加
})


@dataclass
class _PnlAggregateBase:
    """count + mean/profitable%/total を返す共通集計器."""

    sample_count: int = 0
    positive_count: int = 0
    pnl: PnlAccumulator = field(default_factory=PnlAccumulator)

    def add(self, value: float | None) -> None:
        self.sample_count += 1
        numeric = safe_to_finite(value)
        self.pnl.add(numeric)
        if numeric is not None and numeric > 0:
            self.positive_count += 1

    def to_summary(self) -> AggregatePnlSummary:
        return {
            "count": self.sample_count,
            "avg_pnl_30s": _round_acc_mean(self.pnl),
            "profitable_pct": _pct(self.positive_count, self.pnl.count),
            "total_pnl": round(self.pnl.total_bps, 2),
        }


@dataclass
class _SignedPnlAggregate(_PnlAggregateBase):
    """skip_gate 確率帯の損益内訳付き集計器."""

    positive_total_bps: float = 0.0
    negative_total_bps: float = 0.0

    def add(self, value: float | None) -> None:
        super().add(value)
        numeric = safe_to_finite(value)
        if numeric is None:
            return
        if numeric > 0:
            self.positive_total_bps += numeric
        elif numeric < 0:
            self.negative_total_bps += numeric

    def to_bin_summary(self) -> SkipGateBinSummary:
        return {
            "count": self.sample_count,
            "avg_pnl_30s": _round_acc_mean(self.pnl),
            "profitable_pct": _pct(self.positive_count, self.pnl.count),
            "total_profit_bps": round(self.positive_total_bps, 2),
            "total_loss_bps": round(self.negative_total_bps, 2),
        }


@dataclass
class _SideReversalAggregate:
    total_filled: int = 0
    reverse_better_count: int = 0
    actual_pnl: PnlAccumulator = field(default_factory=PnlAccumulator)
    reverse_pnl: PnlAccumulator = field(default_factory=PnlAccumulator)

    def add(self, result: HindsightResult) -> None:
        if not result.filled:
            return

        self.total_filled += 1
        self.actual_pnl.add(result.actual_pnl_30s)
        self.reverse_pnl.add(result.reverse_pnl_30s)
        if (
            result.actual_pnl_30s is not None
            and result.reverse_pnl_30s is not None
            and result.reverse_pnl_30s > result.actual_pnl_30s
        ):
            self.reverse_better_count += 1

    def to_summary(self) -> SideReversalSummary:
        return {
            "total_filled": self.total_filled,
            "reverse_better_count": self.reverse_better_count,
            "reverse_better_pct": _pct(self.reverse_better_count, self.total_filled),
            "avg_actual_pnl": _round_acc_mean(self.actual_pnl),
            "avg_reverse_pnl": _round_acc_mean(self.reverse_pnl),
        }


@dataclass
class _HourlyAggregate:
    total: int = 0
    filled: int = 0
    skipped: int = 0
    profitable_skipped: int = 0
    hindsight_pnl: PnlAccumulator = field(default_factory=PnlAccumulator)
    filled_pnl: PnlAccumulator = field(default_factory=PnlAccumulator)
    skipped_pnl: PnlAccumulator = field(default_factory=PnlAccumulator)

    def add(self, result: HindsightResult) -> None:
        pnl_30s = result.hindsight_pnl_30s
        if pnl_30s is None:
            return

        self.total += 1
        self.hindsight_pnl.add(pnl_30s)
        if result.filled:
            self.filled += 1
            self.filled_pnl.add(pnl_30s)
            return

        self.skipped += 1
        self.skipped_pnl.add(pnl_30s)
        if pnl_30s > 0:
            self.profitable_skipped += 1

    def to_summary(self) -> HourlySummary:
        return {
            "total": self.total,
            "filled": self.filled,
            "skipped": self.skipped,
            "avg_hindsight_30s": _round_acc_mean(self.hindsight_pnl),
            "filled_avg": _round_acc_mean(self.filled_pnl),
            "skipped_avg": _round_acc_mean(self.skipped_pnl),
            "profitable_skipped": self.profitable_skipped,
        }


def _category_from_result(result: HindsightResult) -> str:
    if result.filled:
        return "filled"
    if result.cancel_reason in _TECHNICAL_REASONS:
        return "H6_technical"
    if result.cancel_reason in _REGIME_GUARD_REASONS:
        return "H8_regime_guard"
    return _DIRECT_CATEGORY_BY_REASON.get(result.cancel_reason, "H7_other")


def _categorize(results: list[HindsightResult]) -> dict[str, list[HindsightResult]]:
    """cancel_reason でカテゴリ分け."""
    cats: dict[str, list[HindsightResult]] = defaultdict(list)
    for r in results:
        cats[_category_from_result(r)].append(r)
    return dict(cats)


def _analyze_category(name: str, records: list[HindsightResult]) -> CategoryAnalysis:
    """カテゴリの集計."""
    h30 = [r.hindsight_pnl_30s for r in records if r.hindsight_pnl_30s is not None]
    h60 = [r.hindsight_pnl_60s for r in records if r.hindsight_pnl_60s is not None]
    h120 = [r.hindsight_pnl_120s for r in records if r.hindsight_pnl_120s is not None]

    profitable_30s = [v for v in h30 if v > 0]

    best = max(
        (r for r in records if r.hindsight_pnl_30s is not None),
        key=lambda r: r.hindsight_pnl_30s or 0,
        default=None,
    )
    worst = min(
        (r for r in records if r.hindsight_pnl_30s is not None),
        key=lambda r: r.hindsight_pnl_30s or 0,
        default=None,
    )

    missed_120 = [v for v in h120 if v > 0]

    return CategoryAnalysis(
        category=name,
        count=len(records),
        avg_hindsight_30s=_mean(h30),
        avg_hindsight_60s=_mean(h60),
        avg_hindsight_120s=_mean(h120),
        profitable_30s_count=len(profitable_30s),
        profitable_30s_pct=(len(profitable_30s) / len(h30) * 100) if h30 else 0.0,
        total_missed_profit_30s=sum(profitable_30s),
        total_missed_profit_120s=sum(missed_120),
        best_case=best,
        worst_case=worst,
    )


def _analyze_side_reversal(results: list[HindsightResult]) -> dict[str, SideReversalSummary]:
    """H3: side 逆転分析 — 逆サイドの方が良かったケース."""
    side_aggs = {
        "buy": _SideReversalAggregate(),
        "sell": _SideReversalAggregate(),
    }
    for result in results:
        agg = side_aggs.get(result.side)
        if agg is not None:
            agg.add(result)

    return {
        side_name: agg.to_summary()
        for side_name, agg in side_aggs.items()
        if agg.total_filled > 0
    }


def _analyze_hourly(results: list[HindsightResult]) -> dict[str, HourlySummary]:
    """H4: 時間帯別の機会損失分析."""
    hourly: dict[int, _HourlyAggregate] = defaultdict(_HourlyAggregate)
    for r in results:
        if r.hindsight_pnl_30s is not None:
            # JST = UTC + 9
            h = datetime.fromtimestamp(r.timestamp, tz=timezone.utc)
            jst_hour = (h.hour + 9) % 24
            hourly[jst_hour].add(r)

    return {
        f"JST{hour:02d}": hourly[hour].to_summary()
        for hour in sorted(hourly.keys())
    }


_AS_BINS: tuple[tuple[float, float], ...] = (
    (0.50, 0.55),
    (0.55, 0.60),
    (0.60, 0.65),
    (0.65, 0.70),
    (0.70, 1.0),
)
_SKIP_GATE_THRESHOLDS: tuple[float, ...] = (0.50, 0.55, 0.60, 0.65, 0.70)


def _analyze_skip_gate_calibration(results: list[HindsightResult]) -> SkipGateCalibrationReport:
    """skip_gate の閾値キャリブレーション — 閾値を変えたら利益はどう変わるか."""
    calibration_aggs: dict[str, _SignedPnlAggregate] = {
        f"AS[{lo:.2f}-{hi:.2f})": _SignedPnlAggregate()
        for lo, hi in _AS_BINS
    }
    threshold_exec_aggs: dict[float, _PnlAggregateBase] = {
        threshold: _PnlAggregateBase()
        for threshold in _SKIP_GATE_THRESHOLDS
    }
    threshold_skip_counts: dict[float, int] = {threshold: 0 for threshold in _SKIP_GATE_THRESHOLDS}
    has_skip_gate_recs = False

    for result in results:
        prob = result.skip_gate_as_prob
        pnl_30s = result.hindsight_pnl_30s
        if prob is None or pnl_30s is None:
            continue

        for threshold in _SKIP_GATE_THRESHOLDS:
            if prob >= threshold:
                threshold_skip_counts[threshold] += 1
            else:
                threshold_exec_aggs[threshold].add(pnl_30s)

        if result.cancel_reason != "skip_gate":
            continue

        has_skip_gate_recs = True
        for lo, hi in _AS_BINS:
            if lo <= prob < hi:
                calibration_aggs[f"AS[{lo:.2f}-{hi:.2f})"].add(pnl_30s)
                break

    if not has_skip_gate_recs:
        return {"note": "No skip_gate records with AS prob and hindsight PnL"}

    calibration: dict[str, SkipGateBinSummary] = {
        label: agg.to_bin_summary()
        for label, agg in calibration_aggs.items()
    }
    threshold_impact: dict[str, SkipGateThresholdSummary] = {}
    for threshold in _SKIP_GATE_THRESHOLDS:
        exec_agg = threshold_exec_aggs[threshold]
        threshold_impact[f"threshold={threshold:.2f}"] = {
            "would_execute": exec_agg.sample_count,
            "would_skip": threshold_skip_counts[threshold],
            "avg_exec_pnl": _round_acc_mean(exec_agg.pnl),
            "total_exec_pnl": round(exec_agg.pnl.total_bps, 2),
        }

    return {
        "by_as_prob_bin": calibration,
        "threshold_simulation": threshold_impact,
    }


_WAIT_BANDS: tuple[tuple[str, float, float], ...] = (
    ("0-5s", 0.0, 5.0),
    ("5-15s", 5.0, 15.0),
    ("15-30s", 15.0, 30.0),
    ("30-60s", 30.0, 60.0),
    ("60s+", 60.0, float("inf")),
)


def _analyze_wait_bands(results: list[HindsightResult]) -> dict[str, WaitBandSummary]:
    """§9.2 #3: fill 待機時間帯別の PnL 分析.

    queue_wait_sec が 15-30s で最悪 (-0.563 bps) という指摘に対応。
    """
    band_aggs: dict[str, _PnlAggregateBase] = {
        label: _PnlAggregateBase()
        for label, _, _ in _WAIT_BANDS
    }

    for result in results:
        if not result.filled or result.queue_wait_sec is None:
            continue
        for label, lo, hi in _WAIT_BANDS:
            if lo <= result.queue_wait_sec < hi:
                band_aggs[label].add(result.actual_pnl_30s)
                break

    return {label: band_aggs[label].to_summary() for label, _, _ in _WAIT_BANDS}


def _analyze_regime_side(results: list[HindsightResult]) -> dict[str, RegimeSideSummary]:
    """§9.2 #4: レジーム×side クロス分析.

    trending sell -0.687 bps という指摘に対応し、全組み合わせを出す。
    """
    combos: dict[str, _PnlAggregateBase] = defaultdict(_PnlAggregateBase)
    for r in results:
        if not r.filled:
            continue
        key = f"{r.regime or 'none'}_{r.side}"
        combos[key].add(r.actual_pnl_30s)

    return {key: combos[key].to_summary() for key in sorted(combos.keys())}


def _analyze_interpolated_stats(results: list[HindsightResult]) -> InterpolatedStats:
    """§9.4 #1: 補間参照価格で分析したレコード (旧 price=0) の統計."""
    interp = [r for r in results if r.interpolated_ref]
    non_interp = [r for r in results if not r.interpolated_ref]

    def _stats(recs: Sequence[HindsightResult]) -> InterpolatedSplitStats:
        h30 = [r.hindsight_pnl_30s for r in recs if r.hindsight_pnl_30s is not None]
        return {
            "count": len(recs),
            "with_pnl": len(h30),
            "avg_hindsight_30s": round(mean_h30, 4) if (mean_h30 := _mean(h30)) is not None else None,
        }

    return {
        "interpolated": _stats(interp),
        "original_price": _stats(non_interp),
    }


def _compute_ev_summary(
    filled_pnls: Sequence[float],
    blocked_hindsight: Sequence[float],
    total_cycles: int,
) -> EvPerCycleSummary:
    """EV_per_cycle 集計の共通ロジック.

    172# EV_per_cycle = fill_prob × avg_pnl_if_filled.
    guard_value = EV(executed) − avg_hindsight(blocked) で
    ガードの「損失回避効果 − 機会損失」を評価する。

    guard_value の解釈:
      > 0: EV(実行) がブロック分の後知恵 PnL を上回る → ガードが有害な注文を正しく除外
      < 0: ブロック分の後知恵 PnL が EV(実行) を上回る → ガードが良い機会を逃している
    """
    filled_n = len(filled_pnls)
    fill_prob = filled_n / total_cycles if total_cycles > 0 else 0.0
    avg_pnl_raw = _mean(list(filled_pnls))
    avg_pnl = avg_pnl_raw if avg_pnl_raw is not None else 0.0
    ev = fill_prob * avg_pnl

    blocked_n = len(blocked_hindsight)
    avg_blocked = _mean(list(blocked_hindsight))
    guard_val: float | None = None
    if avg_blocked is not None:
        guard_val = round(ev - avg_blocked, 4)

    return {
        "total_cycles": total_cycles,
        "filled_cycles": filled_n,
        "fill_prob": round(fill_prob, 4),
        "avg_pnl_if_filled": round(avg_pnl, 4),
        "ev_per_cycle": round(ev, 4),
        "blocked_cycles": blocked_n,
        "avg_hindsight_blocked": round(avg_blocked, 4) if avg_blocked is not None else None,
        "guard_value": guard_val,
    }


def _analyze_ev_per_cycle(results: list[HindsightResult]) -> EvPerCycleReport:
    """172# EV_per_cycle 分析 — Codex R3 + Gemini 9.4 推奨.

    EV_per_cycle = fill_prob × avg_pnl_if_filled (bps/cycle)
    各 regime×side / guard カテゴリ別に算出。
    """
    # -- Overall --
    all_filled_pnl = [
        r.actual_pnl_30s for r in results
        if r.filled and r.actual_pnl_30s is not None
    ]
    all_blocked_h = [
        r.hindsight_pnl_30s for r in results
        if not r.filled and r.hindsight_pnl_30s is not None
    ]
    overall = _compute_ev_summary(all_filled_pnl, all_blocked_h, len(results))

    # -- By regime × side --
    regime_side_groups: dict[str, dict[str, list[float]]] = defaultdict(
        lambda: {"filled_pnl": [], "blocked_h": [], "total": []}
    )
    for r in results:
        key = f"{r.regime or 'none'}_{r.side}"
        regime_side_groups[key]["total"].append(0.0)  # count only
        if r.filled and r.actual_pnl_30s is not None:
            regime_side_groups[key]["filled_pnl"].append(r.actual_pnl_30s)
        elif not r.filled and r.hindsight_pnl_30s is not None:
            regime_side_groups[key]["blocked_h"].append(r.hindsight_pnl_30s)

    by_regime_side: dict[str, EvPerCycleSummary] = {}
    for key in sorted(regime_side_groups.keys()):
        g = regime_side_groups[key]
        by_regime_side[key] = _compute_ev_summary(
            g["filled_pnl"], g["blocked_h"], len(g["total"]),
        )

    # -- By guard (cancel_reason カテゴリ) --
    guard_groups: dict[str, dict[str, list[float]]] = defaultdict(
        lambda: {"filled_pnl": [], "blocked_h": [], "total": []}
    )
    for r in results:
        cat = _category_from_result(r)
        guard_groups[cat]["total"].append(0.0)
        if r.filled and r.actual_pnl_30s is not None:
            guard_groups[cat]["filled_pnl"].append(r.actual_pnl_30s)
        elif not r.filled and r.hindsight_pnl_30s is not None:
            guard_groups[cat]["blocked_h"].append(r.hindsight_pnl_30s)

    by_guard: dict[str, EvPerCycleSummary] = {}
    for cat in sorted(guard_groups.keys()):
        g = guard_groups[cat]
        by_guard[cat] = _compute_ev_summary(
            g["filled_pnl"], g["blocked_h"], len(g["total"]),
        )

    return {
        "overall": overall,
        "by_regime_side": by_regime_side,
        "by_guard": by_guard,
    }


# ---------------------------------------------------------------------------
# Display
# ---------------------------------------------------------------------------

def _print_report(
    categories: dict[str, CategoryAnalysis],
    side_reversal: dict[str, SideReversalSummary],
    hourly: dict[str, HourlySummary],
    skip_gate_cal: SkipGateCalibrationReport,
    total_records: int,
    *,
    wait_bands: dict[str, WaitBandSummary] | None = None,
    regime_side: dict[str, RegimeSideSummary] | None = None,
    interpolated_stats: InterpolatedStats | None = None,
    ev_per_cycle: EvPerCycleReport | None = None,
) -> None:
    """Print hindsight analysis report."""
    print("=" * 70)
    print("155# 後知恵フィルター分析レポート")
    print("=" * 70)
    print(f"Total records analyzed: {total_records}")

    # H1-H7 category analysis
    print("\n--- カテゴリ別後知恵PnL分析 ---")
    print(f"  {'Category':<22} {'N':>5} {'avg30s':>8} {'avg120s':>8} "
          f"{'profit%':>7} {'missed_30s':>10} {'missed_120s':>10}")
    for name in sorted(categories.keys()):
        c = categories[name]
        print(
            f"  {c.category:<22} {c.count:>5} "
            f"{c.avg_hindsight_30s or 0:>8.3f} "
            f"{c.avg_hindsight_120s or 0:>8.3f} "
            f"{c.profitable_30s_pct:>6.1f}% "
            f"{c.total_missed_profit_30s:>10.2f} "
            f"{c.total_missed_profit_120s:>10.2f}"
        )

    # H3: Side reversal
    print("\n--- H3: Side 逆転分析 (逆サイドが良かったケース) ---")
    for side_name, data in side_reversal.items():
        print(
            f"  [{side_name}] filled={data['total_filled']}, "
            f"reverse_better={data['reverse_better_count']} "
            f"({data['reverse_better_pct']}%), "
            f"avg_actual={data['avg_actual_pnl']:.4f} bps, "
            f"avg_reverse={data['avg_reverse_pnl']:.4f} bps"
        )

    # H4: Hourly
    print("\n--- H4: 時間帯別 (JST, skipped 機会損失) ---")
    print(f"  {'Hour':<8} {'skipped':>7} {'skip_avg':>9} {'profit_skip':>10}")
    for hour, data in hourly.items():
        if data["skipped"] > 0:
            print(
                f"  {hour:<8} {data['skipped']:>7} "
                f"{data['skipped_avg']:>9.4f} "
                f"{data['profitable_skipped']:>10}"
            )

    # Skip gate calibration
    print("\n--- skip_gate 閾値シミュレーション ---")
    if "threshold_simulation" in skip_gate_cal:
        print(f"  {'Threshold':<18} {'execute':>8} {'skip':>6} {'avg_pnl':>9} {'total_pnl':>10}")
        for thresh, data in skip_gate_cal["threshold_simulation"].items():
            print(
                f"  {thresh:<18} {data['would_execute']:>8} "
                f"{data['would_skip']:>6} "
                f"{data['avg_exec_pnl']:>9.4f} "
                f"{data['total_exec_pnl']:>10.2f}"
            )

    if "by_as_prob_bin" in skip_gate_cal:
        print("\n--- skip_gate AS確率帯別 (skipされたもの) ---")
        print(f"  {'AS Band':<20} {'N':>5} {'avg_pnl':>9} {'profit%':>8} {'profit':>8} {'loss':>8}")
        for band, data in skip_gate_cal["by_as_prob_bin"].items():
            print(
                f"  {band:<20} {data['count']:>5} "
                f"{data['avg_pnl_30s']:>9.4f} "
                f"{data['profitable_pct']:>7.1f}% "
                f"{data['total_profit_bps']:>8.2f} "
                f"{data['total_loss_bps']:>8.2f}"
            )

    # §9.2 #3: Wait time bands
    if wait_bands:
        print("\n--- 待機時間帯別 PnL (filled, actual 30s) ---")
        print(f"  {'Band':<10} {'N':>5} {'avg_pnl':>9} {'profit%':>8} {'total':>8}")
        for band, data in wait_bands.items():
            print(
                f"  {band:<10} {data['count']:>5} "
                f"{data['avg_pnl_30s']:>9.4f} "
                f"{data['profitable_pct']:>7.1f}% "
                f"{data['total_pnl']:>8.2f}"
            )

    # §9.2 #4: Regime × Side
    if regime_side:
        print("\n--- レジーム×Side 別 PnL (filled, actual 30s) ---")
        print(f"  {'Regime_Side':<20} {'N':>5} {'avg_pnl':>9} {'profit%':>8} {'total':>8}")
        for key, data in regime_side.items():
            print(
                f"  {key:<20} {data['count']:>5} "
                f"{data['avg_pnl_30s']:>9.4f} "
                f"{data['profitable_pct']:>7.1f}% "
                f"{data['total_pnl']:>8.2f}"
            )

    # §9.4 #1: Interpolated stats
    if interpolated_stats:
        print("\n--- 補間参照価格の統計 (§9.4 #1) ---")
        for label, data in interpolated_stats.items():
            avg = data.get("avg_hindsight_30s")
            avg_s = f"{avg:.4f}" if avg is not None else "N/A"
            print(f"  {label}: count={data['count']}, with_pnl={data['with_pnl']}, avg_30s={avg_s}")

    # 172# EV_per_cycle
    if ev_per_cycle:
        print("\n--- 172# EV_per_cycle (fill_prob × avg_pnl_if_filled) ---")
        ov = ev_per_cycle["overall"]
        print(f"  Overall: fill_prob={ov['fill_prob']:.3f}, "
              f"avg_pnl={ov['avg_pnl_if_filled']:.4f}, "
              f"EV={ov['ev_per_cycle']:.4f} bps/cycle")
        if ov["guard_value"] is not None:
            print(f"  guard_value={ov['guard_value']:.4f} "
                  f"(>0: guards help, <0: guards harmful)")

        print(f"\n  {'Regime_Side':<22} {'N':>5} {'fill%':>6} {'avg_pnl':>8} "
              f"{'EV':>8} {'blocked':>7} {'blk_h30':>8} {'guard_v':>8}")
        for key, s in ev_per_cycle["by_regime_side"].items():
            gv_s = f"{s['guard_value']:.4f}" if s["guard_value"] is not None else "N/A"
            bh_s = f"{s['avg_hindsight_blocked']:.4f}" if s["avg_hindsight_blocked"] is not None else "N/A"
            print(f"  {key:<22} {s['total_cycles']:>5} "
                  f"{s['fill_prob']:>5.1%} "
                  f"{s['avg_pnl_if_filled']:>8.4f} "
                  f"{s['ev_per_cycle']:>8.4f} "
                  f"{s['blocked_cycles']:>7} "
                  f"{bh_s:>8} {gv_s:>8}")

        print(f"\n  {'Guard_Cat':<22} {'N':>5} {'fill%':>6} {'avg_pnl':>8} "
              f"{'EV':>8} {'blocked':>7} {'blk_h30':>8} {'guard_v':>8}")
        for cat, s in ev_per_cycle["by_guard"].items():
            gv_s = f"{s['guard_value']:.4f}" if s["guard_value"] is not None else "N/A"
            bh_s = f"{s['avg_hindsight_blocked']:.4f}" if s["avg_hindsight_blocked"] is not None else "N/A"
            print(f"  {cat:<22} {s['total_cycles']:>5} "
                  f"{s['fill_prob']:>5.1%} "
                  f"{s['avg_pnl_if_filled']:>8.4f} "
                  f"{s['ev_per_cycle']:>8.4f} "
                  f"{s['blocked_cycles']:>7} "
                  f"{bh_s:>8} {gv_s:>8}")

    print(f"\n{'='*70}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main(argv: Sequence[str] | None = None) -> dict[str, object]:
    """Entry point."""
    parser = argparse.ArgumentParser(
        description="155# 後知恵フィルター分析 — missed profit opportunities",
    )
    parser.add_argument("--start", default=None, help="Start date (YYYY-MM-DD)")
    parser.add_argument("--end", default=None, help="End date (YYYY-MM-DD)")
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--data-dir", default="results/v460/fill_test")
    parser.add_argument("--output", default=None, help="Output JSON path")
    args = parser.parse_args(argv)

    records = _load_records(
        args.data_dir,
        start_date=args.start,
        end_date=args.end,
        run_id=args.run_id,
    )

    if not records:
        print("ERROR: No records", file=sys.stderr)
        sys.exit(1)

    print(f"Loaded {len(records)} records")

    # Build price timeline
    timeline = _build_price_timeline(records)
    if not timeline:
        print("ERROR: Price timeline is empty", file=sys.stderr)
        sys.exit(1)
    print(f"Price timeline: {len(timeline)} points "
          f"({timeline[0].timestamp:.0f} - {timeline[-1].timestamp:.0f})")

    # Compute hindsight PnL
    results = _analyze_records(records, timeline)
    print(f"Analyzed: {len(results)} records")

    # Categorize
    cats = _categorize(results)
    cat_analyses = {name: _analyze_category(name, recs) for name, recs in cats.items()}

    # Side reversal
    side_reversal = _analyze_side_reversal(results)

    # Hourly
    hourly = _analyze_hourly(results)

    # Skip gate calibration
    skip_gate_cal = _analyze_skip_gate_calibration(results)

    # §9.2 #3: Wait time band analysis
    wait_bands = _analyze_wait_bands(results)

    # §9.2 #4: Regime × Side cross analysis
    regime_side = _analyze_regime_side(results)

    # §9.4 #1: Interpolated stats
    interpolated_stats = _analyze_interpolated_stats(results)

    # 172# EV_per_cycle
    ev_per_cycle = _analyze_ev_per_cycle(results)

    # Print report
    _print_report(
        cat_analyses, side_reversal, hourly, skip_gate_cal, len(records),
        wait_bands=wait_bands,
        regime_side=regime_side,
        interpolated_stats=interpolated_stats,
        ev_per_cycle=ev_per_cycle,
    )

    # Build output
    output = {
        "total_records": len(records),
        "timeline_points": len(timeline),
        "categories": {
            name: {
                "count": a.count,
                "avg_hindsight_30s": a.avg_hindsight_30s,
                "avg_hindsight_60s": a.avg_hindsight_60s,
                "avg_hindsight_120s": a.avg_hindsight_120s,
                "profitable_30s_count": a.profitable_30s_count,
                "profitable_30s_pct": a.profitable_30s_pct,
                "total_missed_profit_30s": a.total_missed_profit_30s,
                "total_missed_profit_120s": a.total_missed_profit_120s,
            }
            for name, a in cat_analyses.items()
        },
        "side_reversal": side_reversal,
        "hourly_summary": hourly,
        "skip_gate_calibration": skip_gate_cal,
        "wait_bands": wait_bands,
        "regime_side": regime_side,
        "interpolated_stats": interpolated_stats,
        "ev_per_cycle": ev_per_cycle,
    }

    # Top missed opportunities
    skip_missed = [
        r for r in results
        if r.cancel_reason == "skip_gate"
        and r.hindsight_pnl_30s is not None
    ]
    skip_missed.sort(key=lambda r: r.hindsight_pnl_30s or 0, reverse=True)
    output["top_missed_skip_gate"] = [
        {
            "cycle_id": r.cycle_id,
            "timestamp": r.timestamp,
            "dt": datetime.fromtimestamp(r.timestamp, tz=timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC"),
            "side": r.side,
            "order_price": r.order_price,
            "hindsight_pnl_30s": r.hindsight_pnl_30s,
            "hindsight_pnl_120s": r.hindsight_pnl_120s,
            "skip_gate_as_prob": r.skip_gate_as_prob,
        }
        for r in skip_missed[:10]
    ]

    # Top timeout misses
    timeout_missed = [
        r for r in results
        if r.cancel_reason == "timeout"
        and r.hindsight_pnl_30s is not None
    ]
    timeout_missed.sort(key=lambda r: r.hindsight_pnl_30s or 0, reverse=True)
    output["top_missed_timeout"] = [
        {
            "cycle_id": r.cycle_id,
            "timestamp": r.timestamp,
            "dt": datetime.fromtimestamp(r.timestamp, tz=timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC"),
            "side": r.side,
            "order_price": r.order_price,
            "hindsight_pnl_30s": r.hindsight_pnl_30s,
            "hindsight_pnl_120s": r.hindsight_pnl_120s,
        }
        for r in timeout_missed[:10]
    ]

    if args.output:
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(
            json.dumps(output, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        print(f"\nSaved to {out_path}")

    return output


if __name__ == "__main__":
    main()
