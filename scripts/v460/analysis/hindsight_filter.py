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
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

from scripts.v460.analysis.reproduce_152_metrics import _load_records


# ---------------------------------------------------------------------------
# Price timeline reconstruction
# ---------------------------------------------------------------------------

@dataclass
class PricePoint:
    """タイムライン上の価格ポイント."""
    timestamp: float
    price: float


def _build_price_timeline(records: list[dict[str, Any]]) -> list[PricePoint]:
    """全レコードの order_price + filled records の mid_at_fill/mid_Xs_after から
    価格タイムラインを構築."""
    points: list[PricePoint] = []

    for r in records:
        ts = r.get("timestamp")
        if ts is None:
            continue
        ts_f = float(ts)

        # order_price は注文時の mid_price の近似
        op = r.get("order_price")
        if op is not None and float(op) > 0:
            points.append(PricePoint(ts_f, float(op)))

        # filled records には mid_at_fill, mid_30s_after 等がある
        if r.get("filled"):
            for field_name, offset in [
                ("mid_at_fill", 0),
                ("mid_30s_after", 30),
                ("mid_60s_after", 60),
                ("mid_120s_after", 120),
            ]:
                val = r.get(field_name)
                if val is not None:
                    points.append(PricePoint(ts_f + offset, float(val)))

    # 時系列ソート & 重複除去
    points.sort(key=lambda p: p.timestamp)
    return points


def _interpolate_price(timeline: list[PricePoint], ts: float) -> float | None:
    """タイムライン上の指定時刻の価格を線形補間."""
    if not timeline:
        return None

    timestamps = [p.timestamp for p in timeline]
    idx = bisect.bisect_left(timestamps, ts)

    if idx == 0:
        return timeline[0].price if abs(timeline[0].timestamp - ts) < 300 else None
    if idx >= len(timeline):
        return timeline[-1].price if abs(timeline[-1].timestamp - ts) < 300 else None

    p0, p1 = timeline[idx - 1], timeline[idx]
    # 5分以上離れていたら補間しない
    if p1.timestamp - p0.timestamp > 300:
        return None

    ratio = (ts - p0.timestamp) / (p1.timestamp - p0.timestamp)
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
    records: list[dict[str, Any]],
    timeline: list[PricePoint],
) -> list[HindsightResult]:
    """全レコードの後知恵PnLを計算."""
    results: list[HindsightResult] = []

    for r in records:
        ts = r.get("timestamp")
        op = r.get("order_price")
        side = r.get("side", "unknown")
        if ts is None or op is None:
            continue

        ts_f = float(ts)
        op_f = float(op)
        if op_f <= 0:
            continue

        # 未来の価格を補間
        p30 = _interpolate_price(timeline, ts_f + 30)
        p60 = _interpolate_price(timeline, ts_f + 60)
        p120 = _interpolate_price(timeline, ts_f + 120)

        h30 = _compute_hindsight_pnl(side, op_f, p30)
        h60 = _compute_hindsight_pnl(side, op_f, p60)
        h120 = _compute_hindsight_pnl(side, op_f, p120)

        # 逆サイド
        rev_side = "sell" if side == "buy" else "buy"
        rev30 = _compute_hindsight_pnl(rev_side, op_f, p30)

        results.append(HindsightResult(
            cycle_id=str(r.get("cycle_id", "")),
            timestamp=ts_f,
            side=side,
            order_price=op_f,
            cancel_reason=str(r.get("cancel_reason") or ""),
            filled=bool(r.get("filled")),
            actual_pnl_30s=r.get("post_fill_30s_pnl"),
            hindsight_pnl_30s=h30,
            hindsight_pnl_60s=h60,
            hindsight_pnl_120s=h120,
            reverse_pnl_30s=rev30,
            skip_gate_score=r.get("skip_gate_score"),
            skip_gate_as_prob=r.get("skip_gate_as_prob"),
            regime=r.get("regime"),
        ))

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


def _categorize(results: list[HindsightResult]) -> dict[str, list[HindsightResult]]:
    """cancel_reason でカテゴリ分け."""
    cats: dict[str, list[HindsightResult]] = defaultdict(list)
    for r in results:
        if r.filled:
            cats["filled"].append(r)
        elif r.cancel_reason == "skip_gate":
            cats["H1_skip_gate"].append(r)
        elif r.cancel_reason == "timeout":
            cats["H2_timeout"].append(r)
        elif r.cancel_reason == "balance_forced_skip":
            cats["H5_balance_forced"].append(r)
        elif r.cancel_reason in ("postonly_reject", "orderbook_error",
                                  "api_error", "stale_skip_gate_blocked",
                                  "stale_reprice_failed"):
            cats["H6_technical"].append(r)
        else:
            cats["H7_other"].append(r)
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
        avg_hindsight_30s=sum(h30) / len(h30) if h30 else None,
        avg_hindsight_60s=sum(h60) / len(h60) if h60 else None,
        avg_hindsight_120s=sum(h120) / len(h120) if h120 else None,
        profitable_30s_count=len(profitable_30s),
        profitable_30s_pct=len(profitable_30s) / len(h30) * 100 if h30 else 0,
        total_missed_profit_30s=sum(profitable_30s),
        total_missed_profit_120s=sum(missed_120),
        best_case=best,
        worst_case=worst,
    )


def _analyze_side_reversal(results: list[HindsightResult]) -> dict[str, Any]:
    """H3: side 逆転分析 — 逆サイドの方が良かったケース."""
    reversals: dict[str, dict[str, Any]] = {}

    for side_name in ["buy", "sell"]:
        side_recs = [r for r in results if r.side == side_name and r.filled]
        better_reverse = [
            r for r in side_recs
            if r.actual_pnl_30s is not None
            and r.reverse_pnl_30s is not None
            and r.reverse_pnl_30s > r.actual_pnl_30s
        ]
        if side_recs:
            reversals[side_name] = {
                "total_filled": len(side_recs),
                "reverse_better_count": len(better_reverse),
                "reverse_better_pct": round(len(better_reverse) / len(side_recs) * 100, 1),
                "avg_actual_pnl": round(
                    sum(r.actual_pnl_30s for r in side_recs if r.actual_pnl_30s is not None)
                    / max(sum(1 for r in side_recs if r.actual_pnl_30s is not None), 1),
                    4,
                ),
                "avg_reverse_pnl": round(
                    sum(r.reverse_pnl_30s for r in side_recs if r.reverse_pnl_30s is not None)
                    / max(sum(1 for r in side_recs if r.reverse_pnl_30s is not None), 1),
                    4,
                ),
            }

    return reversals


def _analyze_hourly(results: list[HindsightResult]) -> dict[str, dict[str, Any]]:
    """H4: 時間帯別の機会損失分析."""
    hourly: dict[int, list[HindsightResult]] = defaultdict(list)
    for r in results:
        if r.hindsight_pnl_30s is not None:
            # JST = UTC + 9
            h = datetime.fromtimestamp(r.timestamp, tz=timezone.utc)
            jst_hour = (h.hour + 9) % 24
            hourly[jst_hour].append(r)

    summary: dict[str, dict[str, Any]] = {}
    for hour in sorted(hourly.keys()):
        recs = hourly[hour]
        h30 = [r.hindsight_pnl_30s for r in recs if r.hindsight_pnl_30s is not None]
        filled = [r for r in recs if r.filled]
        skipped = [r for r in recs if not r.filled]
        filled_h30 = [r.hindsight_pnl_30s for r in filled if r.hindsight_pnl_30s is not None]
        skipped_h30 = [r.hindsight_pnl_30s for r in skipped if r.hindsight_pnl_30s is not None]

        summary[f"JST{hour:02d}"] = {
            "total": len(recs),
            "filled": len(filled),
            "skipped": len(skipped),
            "avg_hindsight_30s": round(sum(h30) / len(h30), 4) if h30 else 0,
            "filled_avg": round(sum(filled_h30) / len(filled_h30), 4) if filled_h30 else 0,
            "skipped_avg": round(sum(skipped_h30) / len(skipped_h30), 4) if skipped_h30 else 0,
            "profitable_skipped": sum(1 for v in skipped_h30 if v > 0),
        }

    return summary


def _analyze_skip_gate_calibration(results: list[HindsightResult]) -> dict[str, Any]:
    """skip_gate の閾値キャリブレーション — 閾値を変えたら利益はどう変わるか."""
    skip_gate_recs = [
        r for r in results
        if r.cancel_reason == "skip_gate"
        and r.skip_gate_as_prob is not None
        and r.hindsight_pnl_30s is not None
    ]

    if not skip_gate_recs:
        return {"note": "No skip_gate records with AS prob and hindsight PnL"}

    # AS probability bins and their hindsight PnL
    bins = [(0.50, 0.55), (0.55, 0.60), (0.60, 0.65), (0.65, 0.70), (0.70, 1.0)]
    calibration: dict[str, dict[str, Any]] = {}
    for lo, hi in bins:
        in_bin = [
            r for r in skip_gate_recs
            if lo <= (r.skip_gate_as_prob or 0) < hi
        ]
        h30 = [r.hindsight_pnl_30s for r in in_bin if r.hindsight_pnl_30s is not None]
        profitable = sum(1 for v in h30 if v > 0)
        label = f"AS[{lo:.2f}-{hi:.2f})"
        calibration[label] = {
            "count": len(in_bin),
            "avg_pnl_30s": round(sum(h30) / len(h30), 4) if h30 else 0,
            "profitable_pct": round(profitable / len(h30) * 100, 1) if h30 else 0,
            "total_profit_bps": round(sum(v for v in h30 if v > 0), 2),
            "total_loss_bps": round(sum(v for v in h30 if v < 0), 2),
        }

    # What if threshold was higher (skip fewer)?
    current_threshold = 0.55  # approximate default

    thresholds = [0.50, 0.55, 0.60, 0.65, 0.70]
    threshold_impact: dict[str, dict[str, Any]] = {}
    all_hindsight = [
        r for r in results
        if r.skip_gate_as_prob is not None
        and r.hindsight_pnl_30s is not None
    ]
    for thresh in thresholds:
        would_skip = [r for r in all_hindsight if (r.skip_gate_as_prob or 0) >= thresh]
        would_execute = [r for r in all_hindsight if (r.skip_gate_as_prob or 0) < thresh]
        exec_pnl = [r.hindsight_pnl_30s for r in would_execute if r.hindsight_pnl_30s is not None]
        threshold_impact[f"threshold={thresh:.2f}"] = {
            "would_execute": len(would_execute),
            "would_skip": len(would_skip),
            "avg_exec_pnl": round(sum(exec_pnl) / len(exec_pnl), 4) if exec_pnl else 0,
            "total_exec_pnl": round(sum(exec_pnl), 2),
        }

    return {
        "by_as_prob_bin": calibration,
        "threshold_simulation": threshold_impact,
    }


# ---------------------------------------------------------------------------
# Display
# ---------------------------------------------------------------------------

def _print_report(
    categories: dict[str, CategoryAnalysis],
    side_reversal: dict[str, Any],
    hourly: dict[str, dict[str, Any]],
    skip_gate_cal: dict[str, Any],
    total_records: int,
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

    # Best missed opportunities
    print("\n--- 最大の見逃し利益 TOP5 (skip_gate) ---")
    skip_recs = []
    for name, c in categories.items():
        if "skip_gate" in name and c.best_case:
            skip_recs.extend(
                r for r in [c.best_case]
                if r.hindsight_pnl_30s is not None
            )
    # Get actual records for top 5
    all_skip = []
    for name, recs in categories.items():
        pass
    if "H1_skip_gate" in categories:
        h1_recs = [r for r in [] if r.hindsight_pnl_30s is not None]  # placeholder

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

    print(f"\n{'='*70}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main(argv: Sequence[str] | None = None) -> dict[str, Any]:
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

    # Print report
    _print_report(cat_analyses, side_reversal, hourly, skip_gate_cal, len(records))

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
