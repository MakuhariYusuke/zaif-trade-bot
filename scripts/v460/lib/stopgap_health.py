"""165# P1: 日次 per-regime 3指標 + Stopgap 退出判定.

162# §7.3 P1: 「160# 判定ロジックを stopgap 検証に統合」
受入基準: 「3指標 + per-regime を日次で出力」

既存資産活用:
- scripts.v460.lib.metrics_utils.MetricsAccumulator (3指標 + AS/reprice/VG 拡張)
- scripts.v460.lib.ab_judgment.evaluate_per_regime (regime 別 A/B 判定)
- 163# Stopgap 退出基準表 (§7 Table) の自動評価
"""

from __future__ import annotations

import logging
import math
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import NotRequired, TypedDict, cast

from scripts.v460.lib.metrics_utils import MetricsAccumulator
from ztb.io.json_io import JSONObject
from ztb.metrics.fill_quality import (
    PnlAccumulator,
    apply_fill_record_filters,
    format_utc_day,
    load_fill_record_objects_glob,
)
from ztb.utils.dataclass_utils import shallow_asdict
from ztb.utils.safety import safe_to_finite

logger = logging.getLogger(__name__)

FillRecord = JSONObject
MetricScalar = bool | int | float | None
CriteriaScalar = int | float
StopgapMetrics = dict[str, MetricScalar]
StopgapCriteria = dict[str, CriteriaScalar]


class DailyMetricRow(TypedDict, total=False):
    day: str
    regime: str
    side: str
    n_total: int
    n_filled: int
    fill_rate: float
    avg_pnl30_bps: NotRequired[float]
    downside_p10_bps: NotRequired[float]
    as_rate: float
    dynamic_kill_count: NotRequired[int]
    unknown_regime_count: NotRequired[int]
    velocity_skip_count: NotRequired[int]


class ModelUsedMetricsRow(TypedDict):
    model_used: str
    n_filled: int
    as_count: int
    as_rate: float
    avg_pnl30_bps: float | None
    avg_as_loss_bps: float


class StopgapCheckRow(TypedDict):
    stopgap_id: str
    name: str
    verdict: str
    metrics: StopgapMetrics
    criteria: StopgapCriteria
    detail: str


class AlertRow(TypedDict):
    severity: str
    stopgap_id: str
    message: str


# ======================================================================
# Data Structures
# ======================================================================


class ExitVerdict(str, Enum):
    """退出判定結果."""

    CAN_EXIT = "can_exit"       # OFF 判定基準を満たす
    KEEP = "keep"               # まだ ON 維持が必要
    INSUFFICIENT = "insufficient"  # データ不足


@dataclass
class DailyMetrics:
    """日次 3指標 + AS 率."""

    day: str  # YYYYMMDD
    regime: str  # "all" | "ranging" | "trending" | ...
    side: str  # "all" | "buy" | "sell"

    # 3指標
    n_total: int = 0
    n_filled: int = 0
    fill_rate: float = 0.0
    avg_pnl30_bps: float = float("nan")
    downside_p10_bps: float = float("nan")

    # AS + 拡張
    as_rate: float = 0.0
    avg_as_loss_bps: float = 0.0

    # Stopgap 固有
    dynamic_kill_count: int = 0
    unknown_regime_count: int = 0
    velocity_skip_count: int = 0


@dataclass
class StopgapExitCheck:
    """163# 退出基準表の単一 stopgap 評価結果."""

    stopgap_id: str     # "2-A", "2-C", "6-A" etc.
    name: str
    verdict: ExitVerdict
    metrics: StopgapMetrics = field(default_factory=dict)
    criteria: StopgapCriteria = field(default_factory=dict)
    detail: str = ""


@dataclass
class ModelUsedMetrics:
    """model_used 経路別 AS/PnL (165# 7.3 対応)."""

    model_used: str
    n_filled: int = 0
    as_count: int = 0
    as_rate: float = 0.0
    avg_pnl30_bps: float = float("nan")
    avg_as_loss_bps: float = 0.0


@dataclass
class _ModelUsedAggregate:
    """model_used ごとの stream 集計."""

    n_filled: int = 0
    as_count: int = 0
    pnl_acc: PnlAccumulator = field(default_factory=PnlAccumulator)
    as_pnl_acc: PnlAccumulator = field(default_factory=PnlAccumulator)

    def add(self, record: FillRecord) -> None:
        self.n_filled += 1
        pnl_value = safe_to_finite(record.get("post_fill_30s_pnl"))
        self.pnl_acc.add(pnl_value)
        if record.get("adverse_selected"):
            self.as_count += 1
            self.as_pnl_acc.add(pnl_value)


@dataclass
class _DailyAggregate:
    """日次 stopgap 集計の内部状態."""

    metrics: MetricsAccumulator = field(default_factory=MetricsAccumulator)
    dynamic_kill_count: int = 0
    unknown_regime_count: int = 0
    velocity_skip_count: int = 0

    def add(self, record: FillRecord) -> None:
        self.metrics.add(record)
        if record.get("cancel_reason") == "sell_dynamic_kill":
            self.dynamic_kill_count += 1
        if str(record.get("regime") or "") in ("unknown", "none", ""):
            self.unknown_regime_count += 1
        if str(record.get("cancel_reason") or "").startswith("skip_gate_rule_velocity"):
            self.velocity_skip_count += 1


def _build_daily_metrics_row(
    day: str,
    regime: str,
    side: str,
    agg: _DailyAggregate,
) -> DailyMetrics:
    """集計器から DailyMetrics を構築する."""
    base = agg.metrics.to_extended_metrics()
    return DailyMetrics(
        day=day,
        regime=regime,
        side=side,
        n_total=base["n_total"],
        n_filled=base["n_filled"],
        fill_rate=base["fill_rate"],
        avg_pnl30_bps=base["avg_pnl30_bps"],
        downside_p10_bps=base["downside_p10_bps"],
        as_rate=base["as_rate"],
        avg_as_loss_bps=base["avg_as_loss_bps"],
        dynamic_kill_count=agg.dynamic_kill_count,
        unknown_regime_count=agg.unknown_regime_count,
        velocity_skip_count=agg.velocity_skip_count,
    )


@dataclass
class AlertItem:
    """退出基準の閾値逸脱アラート (165# 7.5 P0 対応)."""

    severity: str  # "critical" | "warning" | "info"
    stopgap_id: str
    message: str


@dataclass
class DailyHealthReport:
    """日次ヘルスレポート全体."""

    generated_at: str
    window_hours: int
    total_records: int
    total_filled: int
    filters_applied: dict[str, str | None] = field(default_factory=dict)
    daily_metrics: list[DailyMetricRow] = field(default_factory=list)
    model_used_breakdown: list[ModelUsedMetricsRow] = field(default_factory=list)
    stopgap_checks: list[StopgapCheckRow] = field(default_factory=list)
    alerts: list[AlertRow] = field(default_factory=list)


# ======================================================================
# Record Loading
# ======================================================================


def load_fill_records(results_dir: Path) -> list[FillRecord]:
    """全 fill_records_*.jsonl をロード."""
    return cast(
        list[FillRecord],
        load_fill_record_objects_glob(results_dir, include_emergency=False),
    )


def apply_filters(
    records: list[FillRecord],
    *,
    run_id: str | None = None,
    git_sha: str | None = None,
    date_from: str | None = None,
    date_to: str | None = None,
) -> tuple[list[FillRecord], dict[str, str | None]]:
    """162# P0 再現性固定: run_id/git_sha/date でフィルタ.

    共有実装は `ztb.metrics.fill_quality.apply_fill_record_filters()` に統一。
    Returns: (filtered_records, applied_filters_dict)
    """
    filtered, filters = apply_fill_record_filters(
        records,
        run_id=run_id,
        git_sha=git_sha,
        date_from=date_from,
        date_to=date_to,
    )
    return cast(list[FillRecord], filtered), filters


def _filter_window(
    records: list[FillRecord],
    window_hours: int,
) -> list[FillRecord]:
    """最新 window_hours のレコードのみ抽出."""
    if window_hours <= 0:
        return records
    now = datetime.now(tz=timezone.utc).timestamp()
    cutoff = now - window_hours * 3600
    return [r for r in records if (safe_to_finite(r.get("timestamp")) or 0) >= cutoff]


def _get_day(r: FillRecord) -> str:
    """YYYYMMDD を抽出."""
    return format_utc_day(safe_to_finite(r.get("timestamp"))) or "unknown"


# ======================================================================
# Daily Metrics Computation
# ======================================================================


def compute_daily_metrics(
    records: list[FillRecord],
) -> list[DailyMetrics]:
    """日次 x regime x side の 3指標を算出.

    162# P1 受入基準のコア実装。
    """
    if not records:
        return []

    if len(records) == 1:
        record = records[0]
        day = _get_day(record)
        regime = str(record.get("regime") or "unknown")
        side = str(record.get("side") or "unknown")
        agg = _DailyAggregate()
        agg.add(record)
        return [
            _build_daily_metrics_row(day, "all", "all", agg),
            _build_daily_metrics_row(day, regime, "all", agg),
            _build_daily_metrics_row(day, "all", side, agg),
            _build_daily_metrics_row(day, regime, side, agg),
        ]

    # Group by (day, regime, side)
    groups: dict[tuple[str, str, str], _DailyAggregate] = defaultdict(_DailyAggregate)
    day_cache: dict[int | None, str] = {}

    for r in records:
        ts_value = safe_to_finite(r.get("timestamp"))
        day_bucket = int(ts_value // 86400) if ts_value is not None else None
        cached_day = day_cache.get(day_bucket)
        if cached_day is None:
            cached_day = _get_day(r)
            day_cache[day_bucket] = cached_day
        day = cached_day
        regime = str(r.get("regime") or "unknown")
        side = str(r.get("side") or "unknown")

        groups[(day, "all", "all")].add(r)
        groups[(day, regime, "all")].add(r)
        groups[(day, "all", side)].add(r)
        groups[(day, regime, side)].add(r)

    results: list[DailyMetrics] = []
    for (day, regime, side), agg in sorted(groups.items()):
        results.append(_build_daily_metrics_row(day, regime, side, agg))
    return results


# ======================================================================
# Stopgap Exit Evaluation (163# Table)
# ======================================================================


def _check_2a_trending_sell_skip(
    records: list[FillRecord],
) -> StopgapExitCheck:
    """2-A: trending_sell_skip 退出判定.

    OFF判定基準: sell AS_rate < 35% (trending regime), total PnL > 0
    ロールバック: AS_rate > 50% で即時 ON
    """
    trending_sell_filled = 0
    as_count = 0
    total_pnl_acc = PnlAccumulator()

    for record in records:
        if not record.get("filled"):
            continue
        total_pnl_acc.add(safe_to_finite(record.get("post_fill_30s_pnl")))
        if not str(record.get("regime") or "").startswith("trending"):
            continue
        if record.get("side") != "sell":
            continue
        trending_sell_filled += 1
        if record.get("adverse_selected"):
            as_count += 1

    if trending_sell_filled < 10:
        return StopgapExitCheck(
            stopgap_id="2-A",
            name="trending_sell_skip",
            verdict=ExitVerdict.INSUFFICIENT,
            metrics={"n_trending_sell_filled": trending_sell_filled},
            criteria={"min_sample": 10},
            detail=f"Insufficient: {trending_sell_filled} trending sell fills < 10",
        )

    as_rate = as_count / trending_sell_filled if trending_sell_filled else 0.0
    total_pnl = total_pnl_acc.total_bps

    can_exit = as_rate < 0.35 and total_pnl > 0
    rollback = as_rate > 0.50

    verdict = ExitVerdict.CAN_EXIT if can_exit else ExitVerdict.KEEP
    detail_parts = [
        f"trending_sell AS_rate={as_rate:.1%} ({'<35%' if as_rate < 0.35 else '>=35%'})",
        f"total_pnl={total_pnl:+.2f}bps ({'> 0' if total_pnl > 0 else '<= 0'})",
    ]
    if rollback:
        detail_parts.append("ROLLBACK: AS_rate > 50%")

    return StopgapExitCheck(
        stopgap_id="2-A",
        name="trending_sell_skip",
        verdict=verdict,
        metrics={
            "trending_sell_filled": trending_sell_filled,
            "as_count": as_count,
            "as_rate": round(as_rate, 4),
            "total_pnl_bps": round(total_pnl, 4),
            "rollback_triggered": rollback,
        },
        criteria={"as_rate_max": 0.35, "total_pnl_min": 0.0, "rollback_as_rate": 0.50},
        detail=" | ".join(detail_parts),
    )


def _check_2c_sell_dynamic_kill(
    records: list[FillRecord],
    window_days: float = 7.0,
) -> StopgapExitCheck:
    """2-C: sell_dynamic_kill 退出判定.

    OFF判定: kill 発動 < 1回/day (7日平均)
    ロールバック: kill 発動 > 3回/day
    """
    kill_count = sum(
        1 for r in records
        if r.get("cancel_reason") == "sell_dynamic_kill"
    )
    kills_per_day = kill_count / window_days if window_days > 0 else 0.0

    can_exit = kills_per_day < 1.0
    rollback = kills_per_day > 3.0

    detail_parts = [
        f"kill_count={kill_count} in {window_days:.0f}d",
        f"rate={kills_per_day:.2f}/day ({'<1' if can_exit else '>=1'})",
    ]
    if rollback:
        detail_parts.append("ROLLBACK: > 3/day")

    return StopgapExitCheck(
        stopgap_id="2-C",
        name="sell_dynamic_kill",
        verdict=ExitVerdict.CAN_EXIT if can_exit else ExitVerdict.KEEP,
        metrics={
            "kill_count": kill_count,
            "window_days": window_days,
            "kills_per_day": round(kills_per_day, 4),
            "rollback_triggered": rollback,
        },
        criteria={"max_kills_per_day": 1.0, "rollback_kills_per_day": 3.0},
        detail=" | ".join(detail_parts),
    )


def _check_6a_unknown_regime_skip(
    records: list[FillRecord],
    window_days: float = 7.0,
) -> StopgapExitCheck:
    """6-A: unknown_regime skip 退出判定.

    OFF判定: unknown < 5% (7日)
    ロールバック: unknown > 15%
    """
    n_total = len(records)
    unk_count = sum(
        1 for r in records
        if str(r.get("regime") or "") in ("unknown", "none", "")
    )
    unk_rate = unk_count / n_total if n_total > 0 else 0.0

    if n_total < 50:
        return StopgapExitCheck(
            stopgap_id="6-A",
            name="unknown_regime_skip",
            verdict=ExitVerdict.INSUFFICIENT,
            metrics={"n_total": n_total, "unknown_count": unk_count},
            criteria={"min_sample": 50},
            detail=f"Insufficient: {n_total} records < 50",
        )

    can_exit = unk_rate < 0.05
    rollback = unk_rate > 0.15

    detail_parts = [
        f"unknown_rate={unk_rate:.1%} ({'<5%' if can_exit else '>=5%'})",
        f"n={unk_count}/{n_total}",
    ]
    if rollback:
        detail_parts.append("ROLLBACK: > 15%")

    return StopgapExitCheck(
        stopgap_id="6-A",
        name="unknown_regime_skip",
        verdict=ExitVerdict.CAN_EXIT if can_exit else ExitVerdict.KEEP,
        metrics={
            "unknown_count": unk_count,
            "n_total": n_total,
            "unknown_rate": round(unk_rate, 4),
            "rollback_triggered": rollback,
        },
        criteria={"max_unknown_rate": 0.05, "rollback_rate": 0.15},
        detail=" | ".join(detail_parts),
    )


def _check_2d_sell_guard(
    records: list[FillRecord],
) -> StopgapExitCheck:
    """2-D: sell_guard 退出判定.

    OFF判定: sell cancel 率 < 10%, sell PnL > 0
    ロールバック: cancel 率 > 20%
    """
    sell_total = 0
    sell_cancelled = 0
    sell_pnl_acc = PnlAccumulator()
    for record in records:
        if record.get("side") != "sell":
            continue
        sell_total += 1
        if record.get("cancelled"):
            sell_cancelled += 1
        if record.get("filled"):
            sell_pnl_acc.add(safe_to_finite(record.get("post_fill_30s_pnl")))

    if sell_total < 20:
        return StopgapExitCheck(
            stopgap_id="2-D",
            name="sell_guard",
            verdict=ExitVerdict.INSUFFICIENT,
            metrics={"n_sell": sell_total},
            criteria={"min_sample": 20},
            detail=f"Insufficient: {sell_total} sell records < 20",
        )

    cancel_rate = sell_cancelled / sell_total if sell_total else 0.0
    avg_pnl = sell_pnl_acc.mean_bps if sell_pnl_acc.count else float("nan")

    can_exit = cancel_rate < 0.10 and (not math.isnan(avg_pnl) and avg_pnl > 0)
    rollback = cancel_rate > 0.20

    detail_parts = [
        f"sell_cancel_rate={cancel_rate:.1%} ({'<10%' if cancel_rate < 0.10 else '>=10%'})",
        f"sell_avg_pnl={avg_pnl:+.2f}bps",
    ]
    if rollback:
        detail_parts.append("ROLLBACK: cancel > 20%")

    return StopgapExitCheck(
        stopgap_id="2-D",
        name="sell_guard",
        verdict=ExitVerdict.CAN_EXIT if can_exit else ExitVerdict.KEEP,
        metrics={
            "sell_total": sell_total,
            "sell_cancelled": sell_cancelled,
            "cancel_rate": round(cancel_rate, 4),
            "sell_avg_pnl_bps": round(avg_pnl, 4) if not math.isnan(avg_pnl) else None,
            "rollback_triggered": rollback,
        },
        criteria={"max_cancel_rate": 0.10, "min_pnl": 0.0, "rollback_cancel_rate": 0.20},
        detail=" | ".join(detail_parts),
    )


def evaluate_stopgap_exit(
    records: list[FillRecord],
    window_hours: int = 168,
) -> list[StopgapExitCheck]:
    """163# Stopgap 退出基準表の自動評価.

    現時点で jsonl データから評価可能な stopgap:
    - 2-A: trending_sell_skip
    - 2-C: sell_dynamic_kill
    - 2-D: sell_guard
    - 6-A: unknown_regime_skip

    3-A/B/C は forced_skip / deadlock / rescue フィールドの実装後に追加。
    1-A/1-C は time_filter regime-adaptive 実装後に追加。
    """
    windowed = _filter_window(records, window_hours)
    window_days = window_hours / 24.0

    checks = [
        _check_2a_trending_sell_skip(windowed),
        _check_2c_sell_dynamic_kill(windowed, window_days=window_days),
        _check_2d_sell_guard(windowed),
        _check_6a_unknown_regime_skip(windowed, window_days=window_days),
    ]
    return checks


# ======================================================================
# Report Generation
# ======================================================================


def compute_model_used_metrics(
    records: list[FillRecord],
) -> list[ModelUsedMetrics]:
    """model_used 経路別の AS率PnL を算出 (165# 7.3)."""
    groups: dict[str, _ModelUsedAggregate] = defaultdict(_ModelUsedAggregate)
    for r in records:
        if not r.get("filled"):
            continue
        model = str(r.get("skip_gate_model_used") or "none")
        groups[model].add(r)

    results: list[ModelUsedMetrics] = []
    for model, agg in sorted(groups.items()):
        as_rate = agg.as_count / agg.n_filled if agg.n_filled else 0.0
        avg_pnl = agg.pnl_acc.mean_bps if agg.pnl_acc.count else float("nan")
        avg_as = agg.as_pnl_acc.mean_bps if agg.as_pnl_acc.count else 0.0

        results.append(ModelUsedMetrics(
            model_used=model,
            n_filled=agg.n_filled,
            as_count=agg.as_count,
            as_rate=round(as_rate, 4),
            avg_pnl30_bps=round(avg_pnl, 4) if not math.isnan(avg_pnl) else avg_pnl,
            avg_as_loss_bps=round(avg_as, 4),
        ))
    return results


def generate_alerts(
    checks: list[StopgapExitCheck],
) -> list[AlertItem]:
    """退出基準の閾値逸脱を自動アラート化 (165# 7.5 P0).

    KEEP 判定のうち、ロールバック条件に抵触するものを CRITICAL、
    KEEP で閾値近接をWARNING としてアラート生成。
    """
    alerts: list[AlertItem] = []
    for c in checks:
        if c.metrics.get("rollback_triggered"):
            alerts.append(AlertItem(
                severity="critical",
                stopgap_id=c.stopgap_id,
                message=f"ROLLBACK condition met for {c.name}: {c.detail}",
            ))
        elif c.verdict == ExitVerdict.KEEP:
            alerts.append(AlertItem(
                severity="warning",
                stopgap_id=c.stopgap_id,
                message=f"Still KEEP: {c.name}: {c.detail}",
            ))
        elif c.verdict == ExitVerdict.CAN_EXIT:
            alerts.append(AlertItem(
                severity="info",
                stopgap_id=c.stopgap_id,
                message=f"Ready to exit: {c.name}: {c.detail}",
            ))
    return alerts


def _serialize_daily_metric(m: DailyMetrics) -> DailyMetricRow:
    """DailyMetrics を JSON 出力向けに整形."""
    row: DailyMetricRow = {
        "day": m.day,
        "regime": m.regime,
        "side": m.side,
        "n_total": m.n_total,
        "n_filled": m.n_filled,
        "fill_rate": round(m.fill_rate, 4),
        "as_rate": round(m.as_rate, 4),
    }
    if not math.isnan(m.avg_pnl30_bps):
        row["avg_pnl30_bps"] = round(m.avg_pnl30_bps, 4)
    if not math.isnan(m.downside_p10_bps):
        row["downside_p10_bps"] = round(m.downside_p10_bps, 4)
    if m.dynamic_kill_count:
        row["dynamic_kill_count"] = m.dynamic_kill_count
    if m.unknown_regime_count:
        row["unknown_regime_count"] = m.unknown_regime_count
    if m.velocity_skip_count:
        row["velocity_skip_count"] = m.velocity_skip_count
    return row


def _serialize_stopgap_check(c: StopgapExitCheck) -> StopgapCheckRow:
    """StopgapExitCheck を JSON 出力向けに整形."""
    return {
        "stopgap_id": c.stopgap_id,
        "name": c.name,
        "verdict": c.verdict.value,
        "metrics": c.metrics,
        "criteria": c.criteria,
        "detail": c.detail,
    }


def _serialize_model_used_metric(m: ModelUsedMetrics) -> ModelUsedMetricsRow:
    """ModelUsedMetrics を JSON 出力向けに整形."""
    return {
        "model_used": m.model_used,
        "n_filled": m.n_filled,
        "as_count": m.as_count,
        "as_rate": m.as_rate,
        "avg_pnl30_bps": (
            None if math.isnan(m.avg_pnl30_bps) else m.avg_pnl30_bps
        ),
        "avg_as_loss_bps": m.avg_as_loss_bps,
    }


def _serialize_alert(a: AlertItem) -> AlertRow:
    """AlertItem を JSON 出力向けに整形."""
    return {
        "severity": a.severity,
        "stopgap_id": a.stopgap_id,
        "message": a.message,
    }


def generate_health_report(
    records: list[FillRecord],
    *,
    window_hours: int = 168,
    daily_limit: int = 7,
    filters_applied: dict[str, str | None] | None = None,
) -> DailyHealthReport:
    """日次ヘルスレポートを生成.

    Args:
        records: 全 fill records.
        window_hours: 評価対象ウィンドウ (default 168h = 7d).
        daily_limit: 日次出力の最大日数.
    """
    windowed = _filter_window(records, window_hours)
    daily = compute_daily_metrics(windowed)
    checks = evaluate_stopgap_exit(records, window_hours=window_hours)

    daily_dicts = [_serialize_daily_metric(m) for m in daily]

    # 日数制限: 最新 daily_limit 日分のみ
    days_seen = sorted({m.day for m in daily if m.day != "unknown"})
    if len(days_seen) > daily_limit:
        keep_days = set(days_seen[-daily_limit:])
        daily_dicts = [d for d in daily_dicts if d["day"] in keep_days]

    n_total = len(windowed)
    n_filled = sum(1 for r in windowed if r.get("filled"))

    check_dicts = [_serialize_stopgap_check(c) for c in checks]

    # model_used breakdown
    model_metrics = compute_model_used_metrics(windowed)
    model_dicts = [_serialize_model_used_metric(m) for m in model_metrics]

    # Alerts
    alert_items = generate_alerts(checks)
    alert_dicts = [_serialize_alert(a) for a in alert_items]

    return DailyHealthReport(
        generated_at=datetime.now(tz=timezone.utc).isoformat(),
        window_hours=window_hours,
        total_records=n_total,
        total_filled=n_filled,
        filters_applied=filters_applied or {},
        daily_metrics=daily_dicts,
        model_used_breakdown=model_dicts,
        stopgap_checks=check_dicts,
        alerts=alert_dicts,
    )


def serialize_health_report(report: DailyHealthReport) -> dict[str, object]:
    """DailyHealthReport を JSON 出力向け dict に変換する."""
    return shallow_asdict(report)


def print_health_summary(report: DailyHealthReport) -> None:
    """人間可読サマリを出力."""
    print("=" * 72)
    print(f"  165# Daily Health Report  ({report.generated_at})")
    print(f"  Window: {report.window_hours}h | Records: {report.total_records}"
          f" | Filled: {report.total_filled}")
    print("=" * 72)

    # Daily summary (all x all only)
    daily_all = [
        d for d in report.daily_metrics
        if d["regime"] == "all" and d["side"] == "all"
    ]
    if daily_all:
        print("\n  --- Daily 3-Indicator (all) ---")
        print(f"  {'Day':>10} {'N':>5} {'Fill%':>7} {'PnL30':>8} {'P10':>8} {'AS%':>6}")
        for d in daily_all:
            pnl = d.get("avg_pnl30_bps")
            p10 = d.get("downside_p10_bps")
            pnl_s = f"{pnl:>+8.2f}" if pnl is not None else f"{'N/A':>8}"
            p10_s = f"{p10:>+8.2f}" if p10 is not None else f"{'N/A':>8}"
            print(
                f"  {d['day']:>10} {d['n_filled']:>5} "
                f"{d['fill_rate']:>6.1%} "
                f"{pnl_s} {p10_s} "
                f"{d['as_rate']:>5.1%}"
            )

    # Per-regime sell summary (latest day)
    if daily_all:
        latest_day = daily_all[-1]["day"]
        regime_sell = [
            d for d in report.daily_metrics
            if d["day"] == latest_day
            and d["side"] == "sell"
            and d["regime"] != "all"
        ]
        if regime_sell:
            print(f"\n  --- Per-Regime Sell ({latest_day}) ---")
            print(f"  {'Regime':>15} {'N':>5} {'Fill%':>7} {'PnL30':>8} {'AS%':>6}")
            for d in regime_sell:
                pnl = d.get("avg_pnl30_bps")
                pnl_str = f"{pnl:>+8.2f}" if pnl is not None else f"{'N/A':>8}"
                print(
                    f"  {d['regime']:>15} {d['n_filled']:>5} "
                    f"{d['fill_rate']:>6.1%} "
                    f"{pnl_str} "
                    f"{d['as_rate']:>5.1%}"
                )

    # Model Used breakdown
    if report.model_used_breakdown:
        print("\n  --- Model Used Pathway (165# 7.3) ---")
        print(f"  {'Model':>25} {'N':>5} {'AS#':>4} {'AS%':>6} {'PnL30':>8} {'AS_Loss':>8}")
        for m in report.model_used_breakdown:
            pnl = m.get("avg_pnl30_bps")
            pnl_s = f"{pnl:>+8.2f}" if pnl is not None else f"{'N/A':>8}"
            print(
                f"  {m['model_used']:>25} {m['n_filled']:>5} "
                f"{m['as_count']:>4} {m['as_rate']:>5.1%} "
                f"{pnl_s} {m['avg_as_loss_bps']:>+8.2f}"
            )

    # Stopgap exit checks
    if report.stopgap_checks:
        print("\n  --- Stopgap Exit Evaluation (163# Table) ---")
        for c in report.stopgap_checks:
            icon = {
                "can_exit": "OK",
                "keep": "NG",
                "insufficient": "??",
            }.get(c["verdict"], "??")
            print(f"  [{icon}] {c['stopgap_id']} {c['name']}: {c['detail']}")

    # Alerts
    if report.alerts:
        print("\n  --- Alerts ---")
        for a in report.alerts:
            sev_icon = {"critical": "!!!", "warning": " ! ", "info": " i "}.get(a["severity"], " ? ")
            print(f"  [{sev_icon}] {a['stopgap_id']}: {a['message']}")

    # Filters
    if report.filters_applied and any(v for v in report.filters_applied.values()):
        print("\n  Filters:", " ".join(f"{k}={v}" for k, v in report.filters_applied.items() if v))

    print("\n" + "=" * 72)
