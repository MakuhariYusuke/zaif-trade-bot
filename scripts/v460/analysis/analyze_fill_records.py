"""132# 用 fill_records 分析スクリプト."""
import json
import glob
from collections import defaultdict
from pathlib import Path


def main() -> None:
    files = sorted(glob.glob("results/v460/fill_test/fill_records_*.jsonl"))
    all_records: list[dict] = []
    for f in files:
        with open(f) as fh:
            for line in fh:
                try:
                    all_records.append(json.loads(line.strip()))
                except Exception:
                    pass

    print(f"Total records: {len(all_records)}")
    first_date = Path(files[0]).stem.split("_")[-1]
    last_date = Path(files[-1]).stem.split("_")[-1]
    print(f"Date range: {first_date} - {last_date}")

    # Fill/skip stats
    filled = [r for r in all_records if r.get("filled")]
    skipped = [r for r in all_records if not r.get("filled")]
    print(f"Filled: {len(filled)}, Skipped: {len(skipped)}")
    print(f"Fill rate: {len(filled)/len(all_records)*100:.1f}%")

    # PnL stats
    for tf in ["30s", "60s", "120s"]:
        key = f"post_fill_{tf}_pnl"
        vals = [r[key] for r in filled if r.get(key) is not None]
        if vals:
            mean_v = sum(vals) / len(vals)
            pos = sum(1 for v in vals if v > 0)
            print(f"PnL {tf}: n={len(vals)}, mean={mean_v:.4f} bps, win_rate={pos/len(vals)*100:.1f}%")

    # Side stats
    for side_name in ["buy", "sell"]:
        side_filled = [r for r in filled if r.get("side") == side_name]
        side_pnl = [r["post_fill_30s_pnl"] for r in side_filled if r.get("post_fill_30s_pnl") is not None]
        if side_pnl:
            as_cnt = sum(1 for r in side_filled if r.get("adverse_selected"))
            fill_side_total = [r for r in all_records if r.get("side") == side_name]
            fr = len(side_filled) / len(fill_side_total) * 100 if fill_side_total else 0
            print(
                f"  {side_name}: n={len(side_pnl)}, mean_pnl={sum(side_pnl)/len(side_pnl):.4f} bps, "
                f"AS={as_cnt}/{len(side_filled)} ({as_cnt/len(side_filled)*100:.1f}%), "
                f"fill_rate={fr:.1f}%"
            )

    # Skip reasons
    reasons: dict[str, int] = defaultdict(int)
    for r in skipped:
        reasons[r.get("cancel_reason", "unknown")] += 1
    print("Skip reasons (top 10):")
    for reason, count in sorted(reasons.items(), key=lambda x: -x[1])[:10]:
        print(f"  {reason}: {count} ({count/len(skipped)*100:.1f}%)")

    # AS stats
    as_records = [r for r in filled if r.get("adverse_selected")]
    print(f"AS total: {len(as_records)}/{len(filled)} = {len(as_records)/len(filled)*100:.1f}%")

    # Regime stats
    regimes: dict[str, int] = defaultdict(int)
    for r in all_records:
        regimes[r.get("regime", "n/a")] += 1
    print("Regimes:")
    for regime, count in sorted(regimes.items(), key=lambda x: -x[1]):
        print(f"  {regime}: {count} ({count/len(all_records)*100:.1f}%)")

    # Regime × PnL breakdown
    regime_pnl: dict[str, list[float]] = defaultdict(list)
    for r in filled:
        pnl = r.get("post_fill_30s_pnl")
        if pnl is not None:
            regime_pnl[r.get("regime", "n/a")].append(pnl)
    print("Regime × PnL 30s:")
    for regime, vals in sorted(regime_pnl.items(), key=lambda x: -len(x[1])):
        mean_v = sum(vals) / len(vals) if vals else 0
        print(f"  {regime}: n={len(vals)}, mean={mean_v:.4f} bps")

    # Hour × PnL (UTC)
    hour_pnl: dict[int, list[float]] = defaultdict(list)
    hour_as: dict[int, int] = defaultdict(int)
    hour_total: dict[int, int] = defaultdict(int)
    for r in filled:
        ts = r.get("timestamp")
        if ts is None:
            continue
        import datetime
        dt = datetime.datetime.fromtimestamp(ts, tz=datetime.timezone.utc)
        h = dt.hour
        pnl = r.get("post_fill_30s_pnl")
        if pnl is not None:
            hour_pnl[h].append(pnl)
        if r.get("adverse_selected"):
            hour_as[h] += 1
        hour_total[h] += 1

    print("Hour (UTC) × PnL 30s (worst 6):")
    hour_means = {h: sum(v)/len(v) for h, v in hour_pnl.items() if v}
    for h, mean_v in sorted(hour_means.items(), key=lambda x: x[1])[:6]:
        n = len(hour_pnl[h])
        as_rate = hour_as.get(h, 0) / hour_total.get(h, 1) * 100
        print(f"  UTC{h:02d} (JST{(h+9)%24:02d}): n={n}, mean={mean_v:.4f} bps, AS={as_rate:.1f}%")

    # balance_forced_switch 割合
    bfs = sum(1 for r in all_records if r.get("balance_forced_switch"))
    print(f"balance_forced_switch: {bfs}/{len(all_records)} ({bfs/len(all_records)*100:.1f}%)")

    # Latest run analysis
    latest_run = all_records[-1].get("run_id", "")
    run_recs = [r for r in all_records if r.get("run_id") == latest_run]
    run_filled = [r for r in run_recs if r.get("filled")]
    run_pnl = [r["post_fill_30s_pnl"] for r in run_filled if r.get("post_fill_30s_pnl") is not None]
    print(f"\nLatest run ({latest_run}):")
    print(f"  records={len(run_recs)}, filled={len(run_filled)}, fill_rate={len(run_filled)/max(len(run_recs),1)*100:.1f}%")
    if run_pnl:
        print(f"  PnL 30s: mean={sum(run_pnl)/len(run_pnl):.4f} bps")

    # Queue wait time analysis
    queue_waits = [r.get("queue_wait_sec", 0) for r in filled if r.get("queue_wait_sec") is not None]
    if queue_waits:
        queue_waits.sort()
        median_qw = queue_waits[len(queue_waits)//2]
        fast_fill = sum(1 for q in queue_waits if q <= 5)
        print(f"Queue wait: median={median_qw:.1f}s, fast_fill(<=5s)={fast_fill}/{len(queue_waits)} ({fast_fill/len(queue_waits)*100:.1f}%)")

    # Retrain status (from retrain_history.jsonl)
    retrain_path = Path("logs/retrain_history.jsonl")
    if retrain_path.exists():
        statuses: dict[str, int] = defaultdict(int)
        last_deployed = None
        with open(retrain_path) as fh:
            for line in fh:
                try:
                    entry = json.loads(line.strip())
                    statuses[entry.get("status", "unknown")] += 1
                    if entry.get("status") == "deployed":
                        last_deployed = entry.get("timestamp")
                except Exception:
                    pass
        print(f"\nRetrain history:")
        for status, count in sorted(statuses.items(), key=lambda x: -x[1]):
            print(f"  {status}: {count}")
        if last_deployed:
            print(f"  Last deployed: {last_deployed}")
        else:
            print("  Last deployed: NEVER (no deployed entries)")


if __name__ == "__main__":
    main()
