"""152# §9 P0-1: 集計再現スクリプト — §1 の数値を再現可能化.

Usage:
    python -m scripts.v460.analysis.reproduce_152_metrics
    python -m scripts.v460.analysis.reproduce_152_metrics --start 2026-02-13 --end 2026-02-22
    python -m scripts.v460.analysis.reproduce_152_metrics --run-id abc123
    python -m scripts.v460.analysis.reproduce_152_metrics --output results/v460/reproduce_152.json
"""

from __future__ import annotations

import argparse
import datetime
import glob
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Sequence

from ztb.io.jsonl import read_jsonl_objects

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
DEFAULT_DATA_DIR = "results/v460/fill_test"
DEFAULT_PATTERN = "fill_records_*.jsonl"


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def _load_records(
    data_dir: str,
    *,
    start_date: str | None = None,
    end_date: str | None = None,
    run_id: str | None = None,
) -> list[dict[str, Any]]:
    """Load fill records with optional date/run_id filtering."""
    files = sorted(glob.glob(str(Path(data_dir) / DEFAULT_PATTERN)))
    if not files:
        print(f"ERROR: No fill record files found in {data_dir}", file=sys.stderr)
        sys.exit(1)

    # Date filtering: filename format is fill_records_YYYYMMDD.jsonl
    if start_date or end_date:
        # Normalize input dates (accept YYYY-MM-DD or YYYYMMDD)
        norm_start = start_date.replace("-", "") if start_date else None
        norm_end = end_date.replace("-", "") if end_date else None
        filtered: list[str] = []
        for f in files:
            stem = Path(f).stem  # fill_records_20260213
            date_part = stem.split("_")[-1]  # 20260213
            if norm_start and date_part < norm_start:
                continue
            if norm_end and date_part > norm_end:
                continue
            filtered.append(f)
        files = filtered

    records: list[dict[str, Any]] = []
    for f in files:
        records.extend(read_jsonl_objects(Path(f)))

    # run_id filtering
    if run_id:
        records = [r for r in records if r.get("run_id") == run_id]

    return records


# ---------------------------------------------------------------------------
# Metrics computation
# ---------------------------------------------------------------------------

def _compute_metrics(
    records: list[dict[str, Any]],
    *,
    include_zero_qty: bool = False,
) -> dict[str, Any]:
    """Compute all §1-equivalent metrics."""
    total = len(records)
    # §12 #4: デフォルトは order_quantity > 0 (0.0000 除外)
    if include_zero_qty:
        with_qty = [r for r in records if r.get("order_quantity") is not None]
    else:
        with_qty = [
            r for r in records
            if r.get("order_quantity") is not None and float(r["order_quantity"]) > 0
        ]
    filled = [r for r in records if r.get("filled")]

    # Regime distribution (over all records with regime)
    regime_all: Counter[str] = Counter()
    for r in records:
        regime = r.get("regime")
        if regime is not None:
            regime_all[regime] += 1
    regime_tagged_total = sum(regime_all.values())

    # Regime × PnL (filled only, post_fill_30s_pnl)
    regime_pnl: dict[str, list[float]] = defaultdict(list)
    for r in filled:
        pnl = r.get("post_fill_30s_pnl")
        if pnl is not None:
            regime_pnl[r.get("regime", "n/a")].append(pnl)

    regime_pnl_summary: dict[str, dict[str, float]] = {}
    for regime, vals in sorted(regime_pnl.items(), key=lambda x: -len(x[1])):
        regime_pnl_summary[regime] = {
            "fills": len(vals),
            "avg_pnl_bps": round(sum(vals) / len(vals), 4) if vals else 0.0,
            "sum_pnl_bps": round(sum(vals), 2),
        }

    # Lot distribution
    lot_counter: Counter[str] = Counter()
    for r in with_qty:
        qty = r.get("order_quantity")
        if qty is not None:
            lot_counter[f"{float(qty):.4f}"] += 1

    # AS probability distribution (skip_gate_as_prob)
    as_probs = [
        float(r["skip_gate_as_prob"])
        for r in records
        if r.get("skip_gate_as_prob") is not None
    ]
    as_dist: dict[str, float] = {}
    if as_probs:
        import math
        bins = [(0.0, 0.1), (0.1, 0.2), (0.2, 0.3), (0.3, 0.4),
                (0.4, 0.5), (0.5, 0.6), (0.6, 1.01)]
        for lo, hi in bins:
            cnt = sum(1 for p in as_probs if lo <= p < hi)
            label = f"[{lo:.1f},{hi:.1f})" if hi <= 1.0 else f"[{lo:.1f}+)"
            as_dist[label] = round(cnt / len(as_probs) * 100, 1)
        as_dist["mean"] = round(sum(as_probs) / len(as_probs), 3)
        as_dist["median"] = round(sorted(as_probs)[len(as_probs) // 2], 3)

    # Side × regime × PnL cross-tabulation (P0-3 寄与分解統合)
    side_regime_pnl: dict[str, dict[str, dict[str, float]]] = {}
    for side_name in ["buy", "sell"]:
        side_data: dict[str, dict[str, float]] = {}
        for regime in sorted(regime_pnl.keys()):
            side_vals = [
                r["post_fill_30s_pnl"]
                for r in filled
                if r.get("side") == side_name
                and r.get("regime") == regime
                and r.get("post_fill_30s_pnl") is not None
            ]
            if side_vals:
                side_data[regime] = {
                    "fills": len(side_vals),
                    "avg_pnl_bps": round(sum(side_vals) / len(side_vals), 4),
                    "sum_pnl_bps": round(sum(side_vals), 2),
                }
        side_regime_pnl[side_name] = side_data

    # Hour × PnL (UTC)
    hour_pnl: dict[int, list[float]] = defaultdict(list)
    for r in filled:
        ts = r.get("timestamp")
        pnl = r.get("post_fill_30s_pnl")
        if ts is not None and pnl is not None:
            h = datetime.datetime.fromtimestamp(
                float(ts), tz=datetime.timezone.utc
            ).hour
            hour_pnl[h].append(pnl)

    hour_summary: dict[str, dict[str, float]] = {}
    for h in sorted(hour_pnl.keys()):
        vals = hour_pnl[h]
        hour_summary[f"UTC{h:02d}_JST{(h+9)%24:02d}"] = {
            "fills": len(vals),
            "avg_pnl_bps": round(sum(vals) / len(vals), 4) if vals else 0.0,
        }

    # run_id breakdown
    run_ids: dict[str, int] = Counter()
    for r in records:
        run_ids[str(r.get("run_id", "none"))] += 1

    return {
        "total_records": total,
        "records_with_order_quantity": len(with_qty),
        "filled": len(filled),
        "fill_rate_pct": round(len(filled) / total * 100, 1) if total else 0,
        "regime_tagged": regime_tagged_total,
        "regime_distribution": dict(regime_all.most_common()),
        "regime_pnl_30s": regime_pnl_summary,
        "lot_distribution": dict(lot_counter.most_common()),
        "as_probability_distribution": as_dist,
        "side_regime_pnl": side_regime_pnl,
        "hour_pnl": hour_summary,
        "run_ids": dict(run_ids.most_common()),
    }


# ---------------------------------------------------------------------------
# Display
# ---------------------------------------------------------------------------

def _print_report(metrics: dict[str, Any], params: dict[str, Any]) -> None:
    """Print human-readable report matching §1 format."""
    print("=" * 60)
    print("152# 集計再現レポート")
    print("=" * 60)
    print(f"Parameters: {json.dumps(params, ensure_ascii=False)}")
    print()

    print(f"Total records: {metrics['total_records']:,}")
    print(f"Records with order_quantity: {metrics['records_with_order_quantity']:,}")
    print(f"Filled: {metrics['filled']:,}")
    print(f"Fill rate: {metrics['fill_rate_pct']}%")
    print(f"Regime-tagged records: {metrics['regime_tagged']:,}")

    print("\n--- Regime Distribution ---")
    for regime, count in metrics["regime_distribution"].items():
        pct = count / metrics["regime_tagged"] * 100 if metrics["regime_tagged"] else 0
        print(f"  {regime}: {count} ({pct:.1f}%)")

    print("\n--- Regime × PnL (30s) ---")
    print(f"  {'Regime':<12} {'fills':>6} {'avg PnL':>10} {'sum PnL':>10}")
    for regime, data in metrics["regime_pnl_30s"].items():
        print(
            f"  {regime:<12} {data['fills']:>6} "
            f"{data['avg_pnl_bps']:>10.4f} {data['sum_pnl_bps']:>10.2f}"
        )

    print("\n--- Lot Distribution ---")
    for lot, count in metrics["lot_distribution"].items():
        pct = count / metrics["records_with_order_quantity"] * 100 if metrics["records_with_order_quantity"] else 0
        print(f"  {lot} BTC: {count} ({pct:.1f}%)")

    print("\n--- Side × Regime × PnL (30s) ---")
    for side, regimes in metrics["side_regime_pnl"].items():
        print(f"  [{side}]")
        for regime, data in regimes.items():
            print(
                f"    {regime:<12} fills={data['fills']:>4}, "
                f"avg={data['avg_pnl_bps']:>8.4f} bps, sum={data['sum_pnl_bps']:>8.2f} bps"
            )

    print("\n--- Hour × PnL (worst 6) ---")
    sorted_hours = sorted(
        metrics["hour_pnl"].items(), key=lambda x: x[1]["avg_pnl_bps"]
    )
    for hour_label, data in sorted_hours[:6]:
        print(f"  {hour_label}: n={data['fills']}, avg={data['avg_pnl_bps']:.4f} bps")

    print(f"\n--- Run IDs ({len(metrics['run_ids'])}) ---")
    for rid, count in metrics["run_ids"].items():
        print(f"  {rid}: {count}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main(argv: Sequence[str] | None = None) -> dict[str, Any]:
    """Entry point. Returns metrics dict for programmatic use."""
    parser = argparse.ArgumentParser(
        description="152# 集計再現スクリプト — §1 データの再現",
    )
    parser.add_argument(
        "--start", default=None,
        help="Start date (YYYY-MM-DD), inclusive",
    )
    parser.add_argument(
        "--end", default=None,
        help="End date (YYYY-MM-DD), inclusive",
    )
    parser.add_argument(
        "--run-id", default=None,
        help="Filter by specific run_id",
    )
    parser.add_argument(
        "--data-dir", default=DEFAULT_DATA_DIR,
        help=f"Data directory (default: {DEFAULT_DATA_DIR})",
    )
    parser.add_argument(
        "--output", default=None,
        help="Output JSON file path (optional)",
    )
    parser.add_argument(
        "--quiet", action="store_true",
        help="Suppress human-readable output",
    )
    parser.add_argument(
        "--include-zero-qty", action="store_true",
        help="order_quantity=0 のレコードも集計に含める (デフォルト: 除外)",
    )
    args = parser.parse_args(argv)

    records = _load_records(
        args.data_dir,
        start_date=args.start,
        end_date=args.end,
        run_id=args.run_id,
    )

    if not records:
        print("ERROR: No records after filtering", file=sys.stderr)
        sys.exit(1)

    params = {
        "data_dir": args.data_dir,
        "start_date": args.start,
        "end_date": args.end,
        "run_id": args.run_id,
        "record_count": len(records),
    }

    metrics = _compute_metrics(records, include_zero_qty=args.include_zero_qty)

    if not args.quiet:
        _print_report(metrics, params)

    if args.output:
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(
            json.dumps({"params": params, "metrics": metrics}, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        print(f"\nSaved to {out_path}")

    return metrics


if __name__ == "__main__":
    main()
