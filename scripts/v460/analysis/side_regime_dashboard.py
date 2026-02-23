"""159# P0-B/C: side × regime 3指標ダッシュボード + trending 日次テンプレート.

sell offset A/B 評価で fill_rate 単独最適化は危険 (159# §3.1)。
最低限 fill_rate / avg_pnl30 / downside_tail (p10) の3指標同時管理が必要。

Usage:
    .venv\\Scripts\\python.exe scripts/v460/analysis/side_regime_dashboard.py
    .venv\\Scripts\\python.exe scripts/v460/analysis/side_regime_dashboard.py --results-dir results/v460/fill_test
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import TypedDict

import numpy as np

_SCRIPT_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _SCRIPT_DIR.parent.parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))

from ztb.io.jsonl import read_jsonl_objects


class SideMetrics(TypedDict, total=False):
    """Side 別メトリクス (159# §3.1 準拠)."""

    n_total: int
    n_filled: int
    fill_rate: float
    avg_pnl30_bps: float
    std_pnl30_bps: float
    downside_p10_bps: float  # p10 = worst decile
    downside_p05_bps: float  # p5
    profitable_rate: float
    as_rate: float
    avg_as_loss_bps: float


class RegimeSideMetrics(TypedDict, total=False):
    """Regime × Side メトリクス."""

    regime: str
    side: str
    metrics: SideMetrics


class DashboardResult(TypedDict, total=False):
    """ダッシュボード結果."""

    timestamp: str
    results_dir: str
    total_records: int
    total_filled: int
    overall_fill_rate: float
    side_summary: dict[str, SideMetrics]
    regime_side_detail: list[RegimeSideMetrics]
    trending_daily: list[dict[str, object]]


def _to_finite(value: object) -> float | None:
    """有限浮動小数点数への安全な変換."""
    if value is None:
        return None
    try:
        v = float(value)  # type: ignore[arg-type]
    except (ValueError, TypeError):
        return None
    return v if math.isfinite(v) else None


def _compute_side_metrics(records: list[dict[str, object]]) -> SideMetrics:
    """レコード群から SideMetrics を計算."""
    n_total = len(records)
    filled = [r for r in records if r.get("filled")]
    n_filled = len(filled)
    fill_rate = n_filled / n_total if n_total > 0 else 0.0

    pnl30_values = [
        _to_finite(r.get("post_fill_30s_pnl"))
        for r in filled
    ]
    pnl30_clean = [v for v in pnl30_values if v is not None]

    if pnl30_clean:
        arr = np.array(pnl30_clean)
        avg_pnl30 = float(np.mean(arr))
        std_pnl30 = float(np.std(arr))
        p10 = float(np.percentile(arr, 10))
        p05 = float(np.percentile(arr, 5))
        profitable = float(np.sum(arr > 0) / len(arr))
    else:
        avg_pnl30 = 0.0
        std_pnl30 = 0.0
        p10 = 0.0
        p05 = 0.0
        profitable = 0.0

    # AS 率
    as_records = [r for r in filled if r.get("adverse_selected")]
    as_rate = len(as_records) / n_filled if n_filled > 0 else 0.0
    as_pnl = [_to_finite(r.get("post_fill_30s_pnl")) for r in as_records]
    as_clean = [v for v in as_pnl if v is not None]
    avg_as_loss = float(np.mean(as_clean)) if as_clean else 0.0

    return {
        "n_total": n_total,
        "n_filled": n_filled,
        "fill_rate": round(fill_rate, 4),
        "avg_pnl30_bps": round(avg_pnl30, 4),
        "std_pnl30_bps": round(std_pnl30, 4),
        "downside_p10_bps": round(p10, 4),
        "downside_p05_bps": round(p05, 4),
        "profitable_rate": round(profitable, 4),
        "as_rate": round(as_rate, 4),
        "avg_as_loss_bps": round(avg_as_loss, 4),
    }


def _load_all_records(results_dir: Path) -> list[dict[str, object]]:
    """fill_records JSONL を全読み込み."""
    all_records: list[dict[str, object]] = []
    for path in sorted(results_dir.glob("fill_records_*.jsonl")):
        try:
            records = read_jsonl_objects(path)
            all_records.extend(records)
        except Exception:
            # BOM fallback
            with open(path, encoding="utf-8-sig") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        try:
                            all_records.append(json.loads(line))
                        except json.JSONDecodeError:
                            continue
    return all_records


def run_dashboard(
    results_dir: str = "results/v460/fill_test",
) -> DashboardResult:
    """3指標ダッシュボードを生成.

    Returns:
        DashboardResult: fill_rate / avg_pnl30 / downside_p10 を side×regime で算出。
    """
    results_path = Path(results_dir)
    records = _load_all_records(results_path)

    result: DashboardResult = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "results_dir": str(results_path),
        "total_records": len(records),
    }

    filled = [r for r in records if r.get("filled")]
    result["total_filled"] = len(filled)
    result["overall_fill_rate"] = round(len(filled) / len(records), 4) if records else 0.0

    # === Side 別サマリー (159# §3.1: 3指標) ===
    side_groups: dict[str, list[dict[str, object]]] = defaultdict(list)
    for r in records:
        side = str(r.get("side", "unknown"))
        side_groups[side].append(r)

    side_summary: dict[str, SideMetrics] = {}
    for side in ["buy", "sell"]:
        if side in side_groups:
            side_summary[side] = _compute_side_metrics(side_groups[side])
    result["side_summary"] = side_summary

    # === Regime × Side 詳細 ===
    regime_side_groups: dict[str, list[dict[str, object]]] = defaultdict(list)
    for r in records:
        regime = str(r.get("regime") or "none")
        side = str(r.get("side", "unknown"))
        regime_side_groups[f"{regime}:{side}"].append(r)

    detail: list[RegimeSideMetrics] = []
    for key in sorted(regime_side_groups.keys()):
        regime, side = key.split(":", 1)
        group = regime_side_groups[key]
        detail.append({
            "regime": regime,
            "side": side,
            "metrics": _compute_side_metrics(group),
        })
    result["regime_side_detail"] = detail

    # === P0-C: trending 日次テンプレート ===
    # trending_down × sell の日別集計
    trending_daily: list[dict[str, object]] = []
    td_by_day: dict[str, list[dict[str, object]]] = defaultdict(list)
    for r in filled:
        if r.get("regime") == "trending_down" and r.get("side") == "sell":
            ts = r.get("timestamp")
            if ts:
                try:
                    day = datetime.fromtimestamp(float(ts), tz=timezone.utc).strftime("%Y%m%d")  # type: ignore[arg-type]
                except (ValueError, TypeError, OSError):
                    continue
                td_by_day[day].append(r)

    for day in sorted(td_by_day.keys()):
        recs = td_by_day[day]
        pnls = [_to_finite(r.get("post_fill_30s_pnl")) for r in recs]
        clean = [v for v in pnls if v is not None]
        trending_daily.append({
            "day": day,
            "n_filled": len(recs),
            "avg_pnl30_bps": round(float(np.mean(clean)), 4) if clean else None,
            "p10_bps": round(float(np.percentile(clean, 10)), 4) if len(clean) >= 3 else None,
        })
    result["trending_daily"] = trending_daily

    return result


def _print_dashboard(result: DashboardResult) -> None:
    """ダッシュボードを人間可読形式で出力."""
    print("\n" + "=" * 74)
    print("  159# Side × Regime 3指標ダッシュボード")
    print("=" * 74)
    print(f"  Total: {result['total_records']} records, "
          f"{result['total_filled']} filled ({result['overall_fill_rate']:.1%})")

    for side in ["buy", "sell"]:
        sm = result.get("side_summary", {}).get(side)
        if not sm:
            continue
        print(f"\n  --- {side.upper()} ---")
        print(f"    fill_rate:     {sm['fill_rate']:.1%} ({sm['n_filled']}/{sm['n_total']})")
        print(f"    avg_pnl30:     {sm['avg_pnl30_bps']:+.4f} bps")
        print(f"    downside_p10:  {sm['downside_p10_bps']:+.4f} bps")
        print(f"    downside_p05:  {sm['downside_p05_bps']:+.4f} bps")
        print(f"    profitable:    {sm['profitable_rate']:.1%}")
        print(f"    AS rate:       {sm['as_rate']:.1%}, avg AS loss: {sm['avg_as_loss_bps']:+.4f} bps")

    print(f"\n  --- Regime × Side Detail ---")
    for item in result.get("regime_side_detail", []):
        m = item["metrics"]
        filled_n = m["n_filled"]
        if filled_n == 0:
            continue
        print(f"    {item['regime']:15s} {item['side']:4s}  "
              f"fill={m['fill_rate']:.1%}  "
              f"pnl30={m['avg_pnl30_bps']:+.4f}  "
              f"p10={m['downside_p10_bps']:+.4f}  "
              f"AS={m['as_rate']:.1%}  "
              f"n={filled_n}")

    td = result.get("trending_daily", [])
    if td:
        print(f"\n  --- Trending Down Sell (Daily) ---")
        for entry in td:
            avg = entry.get("avg_pnl30_bps")
            avg_str = f"{avg:+.4f}" if avg is not None else "N/A"
            print(f"    {entry['day']}: n={entry['n_filled']}, avg_pnl30={avg_str} bps")

    print("=" * 74)


def main() -> None:
    """CLI エントリポイント."""
    parser = argparse.ArgumentParser(
        description="159# P0-B/C: Side × Regime 3指標ダッシュボード",
    )
    parser.add_argument(
        "--results-dir", type=str, default="results/v460/fill_test",
        help="fill_records ディレクトリ",
    )
    parser.add_argument("--json", action="store_true", help="JSON 出力")
    args = parser.parse_args()

    result = run_dashboard(results_dir=args.results_dir)
    _print_dashboard(result)

    if args.json:
        print(f"\n{json.dumps(result, indent=2, default=str)}")


if __name__ == "__main__":
    main()
