"""161# 共通メトリクス計算ユーティリティ.

ab_judgment._compute_metrics と side_regime_dashboard._compute_side_metrics の
共通ロジックを DRY 統合。
"""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

import numpy as np

from ztb.utils.safety import safe_to_finite


def compute_base_metrics(records: list[dict[str, Any]]) -> dict[str, Any]:
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
    filled = [r for r in records if r.get("filled")]
    n_filled = len(filled)
    fill_rate = n_filled / n_total if n_total > 0 else 0.0

    pnl_vals = [safe_to_finite(r.get("post_fill_30s_pnl")) for r in filled]
    pnl_clean: list[float] = [v for v in pnl_vals if v is not None]

    if pnl_clean:
        arr = np.array(pnl_clean, dtype=float)
        avg_pnl30 = float(np.mean(arr))
        std_pnl30 = float(np.std(arr))
        p10 = float(np.percentile(arr, 10))
        p05 = float(np.percentile(arr, 5))
        profitable = float(np.sum(arr > 0) / len(arr))
    else:
        # 160# bugfix: 0.0 だと閾値を上回り誤PASS判定されるため NaN
        arr = np.array([], dtype=float)
        avg_pnl30 = float("nan")
        std_pnl30 = 0.0
        p10 = float("nan")
        p05 = float("nan")
        profitable = 0.0

    # カレンダー日数
    days: set[str] = set()
    for r in filled:
        ts = safe_to_finite(r.get("timestamp"))
        if ts is not None:
            try:
                days.add(datetime.fromtimestamp(ts, tz=timezone.utc).strftime("%Y%m%d"))
            except (ValueError, OSError):
                continue

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


def compute_extended_metrics(records: list[dict[str, Any]]) -> dict[str, Any]:
    """基本メトリクス + AS / reprice / VG 拡張.

    side_regime_dashboard 向け。
    """
    base = compute_base_metrics(records)
    filled = [r for r in records if r.get("filled")]
    n_filled = base["n_filled"]

    # AS 率
    as_records = [r for r in filled if r.get("adverse_selected")]
    as_rate = len(as_records) / n_filled if n_filled > 0 else 0.0
    as_pnl = [safe_to_finite(r.get("post_fill_30s_pnl")) for r in as_records]
    as_clean = [v for v in as_pnl if v is not None]
    avg_as_loss = float(np.mean(as_clean)) if as_clean else 0.0

    # reprice 集計 (159# P1-A)
    repriced = [r for r in filled if (r.get("reprice_count") or 0) > 0]
    reprice_rate = len(repriced) / n_filled if n_filled > 0 else 0.0
    drift_vals = [safe_to_finite(r.get("reprice_drift_bps")) for r in repriced]
    drift_clean = [v for v in drift_vals if v is not None]
    avg_reprice_drift = float(np.mean(drift_clean)) if drift_clean else 0.0

    # VG trigger 集計 (159# P1-C)
    vg_triggered = [r for r in filled if r.get("vg_boost_factor") is not None]
    vg_trigger_rate = len(vg_triggered) / n_filled if n_filled > 0 else 0.0

    base.update({
        "as_rate": as_rate,
        "avg_as_loss_bps": avg_as_loss,
        "reprice_rate": reprice_rate,
        "avg_reprice_drift_bps": avg_reprice_drift,
        "vg_trigger_rate": vg_trigger_rate,
    })
    return base
