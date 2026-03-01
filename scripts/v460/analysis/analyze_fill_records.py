"""132# 用 fill_records 分析スクリプト."""

from __future__ import annotations

import datetime
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path

from ztb.io.jsonl import iter_jsonl_objects
from ztb.metrics.fill_quality import (
    PnlWinAccumulator,
    iter_fill_record_objects_from_files,
    list_fill_record_files,
)
from ztb.utils.safety import safe_to_finite


@dataclass
class _RunStats:
    total: int = 0
    filled: int = 0
    pnl_30s: PnlWinAccumulator = field(default_factory=PnlWinAccumulator)




def _pct(numerator: int, denominator: int) -> float:
    return numerator / denominator * 100.0 if denominator > 0 else 0.0


def main() -> None:
    results_dir = Path("results/v460/fill_test")
    files = list_fill_record_files(results_dir, include_emergency=False)
    if not files:
        print("No fill_records files found: results/v460/fill_test/fill_records_*.jsonl")
        return

    first_date = files[0].stem.split("_")[-1]
    last_date = files[-1].stem.split("_")[-1]
    pnl_by_tf: dict[str, PnlWinAccumulator] = {
        "30s": PnlWinAccumulator(),
        "60s": PnlWinAccumulator(),
        "120s": PnlWinAccumulator(),
    }
    side_totals: Counter[str] = Counter()
    side_filled: Counter[str] = Counter()
    side_as: Counter[str] = Counter()
    side_pnl_30s: dict[str, PnlWinAccumulator] = defaultdict(PnlWinAccumulator)
    skip_reasons: Counter[str] = Counter()
    regime_counts: Counter[str] = Counter()
    regime_pnl_30s: dict[str, PnlWinAccumulator] = defaultdict(PnlWinAccumulator)
    hour_pnl_30s: dict[int, PnlWinAccumulator] = defaultdict(PnlWinAccumulator)
    hour_as: Counter[int] = Counter()
    hour_total: Counter[int] = Counter()
    run_stats: dict[str, _RunStats] = defaultdict(_RunStats)
    queue_waits: list[float] = []

    total_records = 0
    filled_count = 0
    skipped_count = 0
    as_total_count = 0
    bfs_count = 0
    latest_run: str | None = None

    for raw_record in iter_fill_record_objects_from_files(files):
        if not isinstance(raw_record, dict):
            continue

        total_records += 1
        record = raw_record
        side = str(record.get("side", ""))
        regime = str(record.get("regime", "n/a"))
        run_id = str(record.get("run_id", ""))
        latest_run = run_id

        regime_counts[regime] += 1
        if side in ("buy", "sell"):
            side_totals[side] += 1
        if bool(record.get("balance_forced_switch")):
            bfs_count += 1

        run_stat = run_stats[run_id]
        run_stat.total += 1

        filled = bool(record.get("filled"))
        if not filled:
            skipped_count += 1
            skip_reasons[str(record.get("cancel_reason", "unknown"))] += 1
            continue

        filled_count += 1
        run_stat.filled += 1

        if side in ("buy", "sell"):
            side_filled[side] += 1
        if bool(record.get("adverse_selected")):
            as_total_count += 1
            if side in ("buy", "sell"):
                side_as[side] += 1

        queue_wait_sec = safe_to_finite(record.get("queue_wait_sec"))
        if queue_wait_sec is not None:
            queue_waits.append(queue_wait_sec)

        timestamp = safe_to_finite(record.get("timestamp"))
        utc_hour: int | None = None
        if timestamp is not None:
            utc_hour = datetime.datetime.fromtimestamp(
                timestamp,
                tz=datetime.timezone.utc,
            ).hour
            hour_total[utc_hour] += 1
            if bool(record.get("adverse_selected")):
                hour_as[utc_hour] += 1

        for tf in ("30s", "60s", "120s"):
            key = f"post_fill_{tf}_pnl"
            pnl_value = safe_to_finite(record.get(key))
            if pnl_value is None:
                continue
            pnl_by_tf[tf].add(pnl_value)
            if tf == "30s":
                if side in ("buy", "sell"):
                    side_pnl_30s[side].add(pnl_value)
                regime_pnl_30s[regime].add(pnl_value)
                run_stat.pnl_30s.add(pnl_value)
                if utc_hour is not None:
                    hour_pnl_30s[utc_hour].add(pnl_value)

    print(f"Total records: {total_records}")
    print(f"Date range: {first_date} - {last_date}")

    # Fill/skip stats
    print(f"Filled: {filled_count}, Skipped: {skipped_count}")
    print(f"Fill rate: {_pct(filled_count, total_records):.1f}%")

    # PnL stats
    for tf in ("30s", "60s", "120s"):
        stats = pnl_by_tf[tf]
        if stats.count:
            print(
                f"PnL {tf}: n={stats.count}, mean={stats.mean_bps:.4f} bps, "
                f"win_rate={stats.win_rate * 100.0:.1f}%"
            )

    # Side stats
    for side_name in ("buy", "sell"):
        side_stats = side_pnl_30s.get(side_name)
        if side_stats is None or side_stats.count == 0:
            continue
        as_count = side_as.get(side_name, 0)
        filled_side = side_filled.get(side_name, 0)
        side_total = side_totals.get(side_name, 0)
        print(
            f"  {side_name}: n={side_stats.count}, mean_pnl={side_stats.mean_bps:.4f} bps, "
            f"AS={as_count}/{filled_side} ({_pct(as_count, filled_side):.1f}%), "
            f"fill_rate={_pct(filled_side, side_total):.1f}%"
        )

    # Skip reasons
    print("Skip reasons (top 10):")
    for reason, count in skip_reasons.most_common(10):
        print(f"  {reason}: {count} ({_pct(count, skipped_count):.1f}%)")

    # AS stats
    print(
        f"AS total: {as_total_count}/{filled_count} = "
        f"{_pct(as_total_count, filled_count):.1f}%"
    )

    # Regime stats
    print("Regimes:")
    for regime, count in regime_counts.most_common():
        print(f"  {regime}: {count} ({_pct(count, total_records):.1f}%)")

    # Regime × PnL breakdown
    print("Regime × PnL 30s:")
    for regime, stats in sorted(
        regime_pnl_30s.items(),
        key=lambda item: -item[1].count,
    ):
        print(f"  {regime}: n={stats.count}, mean={stats.mean_bps:.4f} bps")

    # Hour × PnL (UTC)
    print("Hour (UTC) × PnL 30s (worst 6):")
    worst_hours = sorted(
        hour_pnl_30s.items(),
        key=lambda item: item[1].mean_bps,
    )[:6]
    for hour, stats in worst_hours:
        as_rate = _pct(hour_as.get(hour, 0), hour_total.get(hour, 0))
        print(
            f"  UTC{hour:02d} (JST{(hour + 9) % 24:02d}): n={stats.count}, "
            f"mean={stats.mean_bps:.4f} bps, AS={as_rate:.1f}%"
        )

    # balance_forced_switch 割合
    print(
        f"balance_forced_switch: {bfs_count}/{total_records} "
        f"({_pct(bfs_count, total_records):.1f}%)"
    )

    # Latest run analysis
    if latest_run is not None:
        latest = run_stats[latest_run]
        print(f"\nLatest run ({latest_run}):")
        print(
            f"  records={latest.total}, filled={latest.filled}, "
            f"fill_rate={_pct(latest.filled, latest.total):.1f}%"
        )
        if latest.pnl_30s.count:
            print(f"  PnL 30s: mean={latest.pnl_30s.mean_bps:.4f} bps")
    else:
        print("\nLatest run: N/A (no valid records)")

    # Queue wait time analysis
    if queue_waits:
        queue_waits.sort()
        median_qw = queue_waits[len(queue_waits) // 2]
        fast_fill = sum(1 for wait in queue_waits if wait <= 5)
        print(
            f"Queue wait: median={median_qw:.1f}s, fast_fill(<=5s)="
            f"{fast_fill}/{len(queue_waits)} ({_pct(fast_fill, len(queue_waits)):.1f}%)"
        )

    # Retrain status (from retrain_history.jsonl)
    retrain_path = Path("logs/retrain_history.jsonl")
    if retrain_path.exists():
        statuses: Counter[str] = Counter()
        last_deployed: object | None = None
        for entry in iter_jsonl_objects(retrain_path, warn_malformed=True):
            status = str(entry.get("status", "unknown"))
            statuses[status] += 1
            if status == "deployed":
                last_deployed = entry.get("timestamp")
        print("\nRetrain history:")
        for status, count in statuses.most_common():
            print(f"  {status}: {count}")
        if last_deployed:
            print(f"  Last deployed: {last_deployed}")
        else:
            print("  Last deployed: NEVER (no deployed entries)")


if __name__ == "__main__":
    main()
