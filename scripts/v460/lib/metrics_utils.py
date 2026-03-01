"""161# 共通メトリクス計算ユーティリティ.

ab_judgment._compute_metrics と side_regime_dashboard._compute_side_metrics の
共通ロジックを DRY 統合。
"""
from __future__ import annotations

from datetime import datetime, timezone
from typing import TypedDict

import numpy as np

from ztb.io.json_io import JSONObject
from ztb.metrics.fill_quality import PnlAccumulator, PnlWinAccumulator
from ztb.utils.safety import safe_to_finite

MetricRecord = JSONObject


class BaseMetrics(TypedDict):
    n_total: int
    n_filled: int
    fill_rate: float
    avg_pnl30_bps: float
    std_pnl30_bps: float
    downside_p10_bps: float
    downside_p05_bps: float
    profitable_rate: float
    calendar_days: int
    pnl30_array: np.ndarray


class ExtendedMetrics(BaseMetrics):
    as_rate: float
    avg_as_loss_bps: float
    reprice_rate: float
    avg_reprice_drift_bps: float
    vg_trigger_rate: float


__all__ = [
    "MetricRecord",
    "BaseMetrics",
    "ExtendedMetrics",
    "compute_base_metrics",
    "compute_extended_metrics",
]
def compute_base_metrics(records: list[MetricRecord]) -> BaseMetrics:
    """fill レコード群から基本メトリクスを算出.

    ab_judgment / side_regime_dashboard 双方で使える共通ベース。

    Returns:
        dict with keys:
            n_total, n_filled, fill_rate,
            avg_pnl30_bps, std_pnl30_bps, downside_p10_bps, downside_p05_bps,
            profitable_rate, calendar_days,
            pnl30_array (np.ndarray — 下流で stat test 等に使用可)
    """
    n_total = len(records)
    n_filled = 0
    fill_rate = 0.0
    pnl_values: list[float] = []
    pnl_summary = PnlWinAccumulator()

    # カレンダー日数
    days: set[str] = set()
    for record in records:
        if not record.get("filled"):
            continue
        n_filled += 1
        pnl_value = safe_to_finite(record.get("post_fill_30s_pnl"))
        if pnl_value is not None:
            pnl_values.append(pnl_value)
            pnl_summary.add(pnl_value)
        ts = safe_to_finite(record.get("timestamp"))
        if ts is None:
            continue
        try:
            days.add(datetime.fromtimestamp(ts, tz=timezone.utc).strftime("%Y%m%d"))
        except (ValueError, OSError):
            continue

    if n_total > 0:
        fill_rate = n_filled / n_total

    if pnl_values:
        arr = np.array(pnl_values, dtype=float)
        avg_pnl30 = pnl_summary.mean_bps
        std_pnl30 = float(np.std(arr))
        p10 = float(np.percentile(arr, 10))
        p05 = float(np.percentile(arr, 5))
        profitable = pnl_summary.win_rate
    else:
        # 160# bugfix: 0.0 だと閾値を上回り誤PASS判定されるため NaN
        arr = np.array([], dtype=float)
        avg_pnl30 = float("nan")
        std_pnl30 = 0.0
        p10 = float("nan")
        p05 = float("nan")
        profitable = 0.0

    return {
        "n_total": n_total,
        "n_filled": n_filled,
        "fill_rate": fill_rate,
        "avg_pnl30_bps": avg_pnl30,
        "std_pnl30_bps": std_pnl30,
        "downside_p10_bps": p10,
        "downside_p05_bps": p05,
        "profitable_rate": profitable,
        "calendar_days": len(days),
        "pnl30_array": arr,
    }


def compute_extended_metrics(records: list[MetricRecord]) -> ExtendedMetrics:
    """基本メトリクス + AS / reprice / VG 拡張.

    side_regime_dashboard 向け。
    """
    base = compute_base_metrics(records)
    n_filled = base["n_filled"]

    # AS 率
    as_count = 0
    as_pnl_acc = PnlAccumulator()

    # reprice 集計 (159# P1-A)
    repriced_count = 0
    drift_acc = PnlAccumulator()

    # VG trigger 集計 (159# P1-C)
    vg_triggered_count = 0

    for record in records:
        if not record.get("filled"):
            continue
        if record.get("adverse_selected"):
            as_count += 1
            as_pnl_acc.add(safe_to_finite(record.get("post_fill_30s_pnl")))
        if (record.get("reprice_count") or 0) > 0:
            repriced_count += 1
            drift_acc.add(safe_to_finite(record.get("reprice_drift_bps")))
        if record.get("vg_boost_factor") is not None:
            vg_triggered_count += 1

    as_rate = as_count / n_filled if n_filled > 0 else 0.0
    avg_as_loss = as_pnl_acc.mean_bps if as_pnl_acc.count else 0.0
    reprice_rate = repriced_count / n_filled if n_filled > 0 else 0.0
    avg_reprice_drift = drift_acc.mean_bps if drift_acc.count else 0.0
    vg_trigger_rate = vg_triggered_count / n_filled if n_filled > 0 else 0.0

    base.update({
        "as_rate": as_rate,
        "avg_as_loss_bps": avg_as_loss,
        "reprice_rate": reprice_rate,
        "avg_reprice_drift_bps": avg_reprice_drift,
        "vg_trigger_rate": vg_trigger_rate,
    })
    return base
