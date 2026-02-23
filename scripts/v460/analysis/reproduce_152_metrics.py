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
import math
import sys
from collections import Counter, defaultdict
from collections.abc import Sequence
from pathlib import Path

from ztb.io.jsonl import read_jsonl_objects
from ztb.io.json_io import write_json
from ztb.utils.safety import ensure_dict, safe_to_float, safe_to_int

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
DEFAULT_DATA_DIR = "results/v460/fill_test"
DEFAULT_PATTERN = "fill_records_*.jsonl"
FillRecord = dict[str, object]
MetricsMap = dict[str, object]


def _to_float(value: object) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    parsed = safe_to_float(value, float("nan"))
    return parsed if math.isfinite(parsed) else None


def _to_str(value: object) -> str | None:
    return value if isinstance(value, str) and value else None


def _to_dict(value: object) -> dict[str, object]:
    return ensure_dict(value)


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def _load_records(
    data_dir: str,
    *,
    start_date: str | None = None,
    end_date: str | None = None,
    run_id: str | None = None,
) -> list[FillRecord]:
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

    records: list[FillRecord] = []
    for f in files:
        raw_records = read_jsonl_objects(Path(f))
        records.extend(_to_dict(record) for record in raw_records if isinstance(record, dict))

    # run_id filtering
    if run_id:
        records = [r for r in records if r.get("run_id") == run_id]

    return records


# ---------------------------------------------------------------------------
# Metrics computation
# ---------------------------------------------------------------------------

def _compute_metrics(
    records: list[FillRecord],
    *,
    include_zero_qty: bool = False,
) -> MetricsMap:
    """Compute all §1-equivalent metrics."""
    total = len(records)
    records_with_qty = 0
    filled_count = 0

    regime_all: Counter[str] = Counter()
    lot_counter: Counter[str] = Counter()
    run_ids: Counter[str] = Counter()
    regime_pnl_values: dict[str, list[float]] = defaultdict(list)
    side_regime_values: dict[str, dict[str, list[float]]] = {
        "buy": defaultdict(list),
        "sell": defaultdict(list),
    }
    hour_pnl_values: dict[int, list[float]] = defaultdict(list)
    as_probs: list[float] = []

    for record in records:
        run_ids[str(record.get("run_id", "none"))] += 1

        regime = _to_str(record.get("regime"))
        if regime is not None:
            regime_all[regime] += 1

        qty = _to_float(record.get("order_quantity"))
        if qty is not None and (include_zero_qty or qty > 0):
            records_with_qty += 1
            lot_counter[f"{qty:.4f}"] += 1

        as_prob = _to_float(record.get("skip_gate_as_prob"))
        if as_prob is not None:
            as_probs.append(as_prob)

        if not bool(record.get("filled")):
            continue
        filled_count += 1

        pnl = _to_float(record.get("post_fill_30s_pnl"))
        if pnl is None:
            continue

        regime_key = regime if regime is not None else "n/a"
        regime_pnl_values[regime_key].append(pnl)

        side = _to_str(record.get("side"))
        if side in ("buy", "sell"):
            side_regime_values[side][regime_key].append(pnl)

        ts = _to_float(record.get("timestamp"))
        if ts is not None:
            hour = datetime.datetime.fromtimestamp(ts, tz=datetime.timezone.utc).hour
            hour_pnl_values[hour].append(pnl)

    regime_tagged_total = sum(regime_all.values())

    regime_pnl_summary: dict[str, dict[str, float]] = {}
    for regime, values in sorted(regime_pnl_values.items(), key=lambda item: -len(item[1])):
        regime_pnl_summary[regime] = {
            "fills": len(values),
            "avg_pnl_bps": round(sum(values) / len(values), 4) if values else 0.0,
            "sum_pnl_bps": round(sum(values), 2),
        }

    as_dist: dict[str, float] = {}
    if as_probs:
        bins = [
            (0.0, 0.1),
            (0.1, 0.2),
            (0.2, 0.3),
            (0.3, 0.4),
            (0.4, 0.5),
            (0.5, 0.6),
            (0.6, 1.01),
        ]
        total_probs = len(as_probs)
        for lo, hi in bins:
            count = sum(1 for prob in as_probs if lo <= prob < hi)
            label = f"[{lo:.1f},{hi:.1f})" if hi <= 1.0 else f"[{lo:.1f}+)"
            as_dist[label] = round(count / total_probs * 100, 1)
        sorted_probs = sorted(as_probs)
        mid = len(sorted_probs) // 2
        as_dist["mean"] = round(sum(sorted_probs) / len(sorted_probs), 3)
        as_dist["median"] = round(sorted_probs[mid], 3)

    side_regime_pnl: dict[str, dict[str, dict[str, float]]] = {"buy": {}, "sell": {}}
    for side_name in ("buy", "sell"):
        side_data: dict[str, dict[str, float]] = {}
        for regime, values in sorted(
            side_regime_values[side_name].items(), key=lambda item: -len(item[1])
        ):
            if not values:
                continue
            side_data[regime] = {
                "fills": len(values),
                "avg_pnl_bps": round(sum(values) / len(values), 4),
                "sum_pnl_bps": round(sum(values), 2),
            }
        side_regime_pnl[side_name] = side_data

    hour_summary: dict[str, dict[str, float]] = {}
    for hour in sorted(hour_pnl_values):
        values = hour_pnl_values[hour]
        hour_summary[f"UTC{hour:02d}_JST{(hour + 9) % 24:02d}"] = {
            "fills": len(values),
            "avg_pnl_bps": round(sum(values) / len(values), 4) if values else 0.0,
        }

    return {
        "total_records": total,
        "records_with_order_quantity": records_with_qty,
        "filled": filled_count,
        "fill_rate_pct": round(filled_count / total * 100, 1) if total else 0,
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

def _as_int(value: object) -> int:
    return safe_to_int(value, 0)


def _as_float_or_zero(value: object) -> float:
    return safe_to_float(value, 0.0)


def _print_report(metrics: MetricsMap, params: dict[str, object]) -> None:
    """Print human-readable report matching §1 format."""
    regime_distribution = _to_dict(metrics.get("regime_distribution"))
    regime_pnl_30s = _to_dict(metrics.get("regime_pnl_30s"))
    lot_distribution = _to_dict(metrics.get("lot_distribution"))
    side_regime_pnl = _to_dict(metrics.get("side_regime_pnl"))
    hour_pnl = _to_dict(metrics.get("hour_pnl"))
    run_ids = _to_dict(metrics.get("run_ids"))
    regime_tagged = _as_int(metrics.get("regime_tagged"))
    records_with_qty = _as_int(metrics.get("records_with_order_quantity"))

    print("=" * 60)
    print("152# 集計再現レポート")
    print("=" * 60)
    print(f"Parameters: {json.dumps(params, ensure_ascii=False)}")
    print()

    print(f"Total records: {_as_int(metrics.get('total_records')):,}")
    print(f"Records with order_quantity: {records_with_qty:,}")
    print(f"Filled: {_as_int(metrics.get('filled')):,}")
    print(f"Fill rate: {_as_float_or_zero(metrics.get('fill_rate_pct'))}%")
    print(f"Regime-tagged records: {regime_tagged:,}")

    print("\n--- Regime Distribution ---")
    for regime, count in regime_distribution.items():
        count_int = _as_int(count)
        pct = count_int / regime_tagged * 100 if regime_tagged else 0
        print(f"  {regime}: {count} ({pct:.1f}%)")

    print("\n--- Regime × PnL (30s) ---")
    print(f"  {'Regime':<12} {'fills':>6} {'avg PnL':>10} {'sum PnL':>10}")
    for regime, raw_data in regime_pnl_30s.items():
        data = _to_dict(raw_data)
        print(
            f"  {regime:<12} {_as_int(data.get('fills')):>6} "
            f"{_as_float_or_zero(data.get('avg_pnl_bps')):>10.4f} "
            f"{_as_float_or_zero(data.get('sum_pnl_bps')):>10.2f}"
        )

    print("\n--- Lot Distribution ---")
    for lot, count in lot_distribution.items():
        count_int = _as_int(count)
        pct = count_int / records_with_qty * 100 if records_with_qty else 0
        print(f"  {lot} BTC: {count} ({pct:.1f}%)")

    print("\n--- Side × Regime × PnL (30s) ---")
    for side, raw_regimes in side_regime_pnl.items():
        regimes = _to_dict(raw_regimes)
        print(f"  [{side}]")
        for regime, raw_data in regimes.items():
            data = _to_dict(raw_data)
            print(
                f"    {regime:<12} fills={_as_int(data.get('fills')):>4}, "
                f"avg={_as_float_or_zero(data.get('avg_pnl_bps')):>8.4f} bps, "
                f"sum={_as_float_or_zero(data.get('sum_pnl_bps')):>8.2f} bps"
            )

    print("\n--- Hour × PnL (worst 6) ---")
    sorted_hours = sorted(
        hour_pnl.items(),
        key=lambda x: _as_float_or_zero(_to_dict(x[1]).get("avg_pnl_bps")),
    )
    for hour_label, data in sorted_hours[:6]:
        hour_data = _to_dict(data)
        print(
            f"  {hour_label}: n={_as_int(hour_data.get('fills'))}, "
            f"avg={_as_float_or_zero(hour_data.get('avg_pnl_bps')):.4f} bps"
        )

    print(f"\n--- Run IDs ({len(run_ids)}) ---")
    for rid, count in run_ids.items():
        print(f"  {rid}: {count}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main(argv: Sequence[str] | None = None) -> MetricsMap:
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

    params: dict[str, object] = {
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
        write_json(
            out_path,
            {"params": params, "metrics": metrics},
            indent=2,
            ensure_ascii=False,
        )
        print(f"\nSaved to {out_path}")

    return metrics


if __name__ == "__main__":
    main()
