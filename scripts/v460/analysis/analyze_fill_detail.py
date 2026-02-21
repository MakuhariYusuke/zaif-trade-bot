"""132# 追加分析: daily, run_id, reprice 等."""
import json
import glob
import datetime
from collections import defaultdict
from pathlib import Path


def main() -> None:
    files = sorted(glob.glob("results/v460/fill_test/fill_records_*.jsonl"))
    all_recs: list[dict] = []
    for f in files:
        with open(f) as fh:
            for line in fh:
                try:
                    all_recs.append(json.loads(line.strip()))
                except Exception:
                    pass

    filled = [r for r in all_recs if r.get("filled")]

    # Stale/reprice
    repriced = [r for r in filled if r.get("reprice_count", 0) > 0]
    print(f"Repriced: {len(repriced)}/{len(filled)} ({len(repriced)/len(filled)*100:.1f}%)")

    # Consecutive same side
    last_side = None
    max_consec = 0
    cur_consec = 0
    for r in all_recs:
        s = r.get("side")
        if s == last_side:
            cur_consec += 1
            max_consec = max(max_consec, cur_consec)
        else:
            cur_consec = 1
            last_side = s
    print(f"Max consecutive same side: {max_consec}")

    # Day breakdown
    day_stats: dict[str, dict] = defaultdict(lambda: {"total": 0, "filled": 0, "pnl": []})
    for r in all_recs:
        ts = r.get("timestamp")
        if ts:
            d = datetime.datetime.fromtimestamp(ts, tz=datetime.timezone.utc).strftime("%Y-%m-%d")
            day_stats[d]["total"] += 1
            if r.get("filled"):
                day_stats[d]["filled"] += 1
                pnl = r.get("post_fill_30s_pnl")
                if pnl is not None:
                    day_stats[d]["pnl"].append(pnl)

    print("Daily breakdown:")
    for d in sorted(day_stats.keys()):
        s = day_stats[d]
        fr = s["filled"] / s["total"] * 100 if s["total"] else 0
        pnl_mean = sum(s["pnl"]) / len(s["pnl"]) if s["pnl"] else 0
        n = len(s["pnl"])
        print(f"  {d}: total={s['total']}, filled={s['filled']}, FR={fr:.0f}%, PnL30s={pnl_mean:.3f}bps (n={n})")

    # run_id breakdown
    runs: dict[str, dict] = defaultdict(lambda: {"total": 0, "filled": 0, "pnl": []})
    for r in all_recs:
        rid = r.get("run_id", "none")
        runs[rid]["total"] += 1
        if r.get("filled"):
            runs[rid]["filled"] += 1
            pnl = r.get("post_fill_30s_pnl")
            if pnl is not None:
                runs[rid]["pnl"].append(pnl)

    print(f"\nRun IDs: {len(runs)}")
    for rid, s in sorted(runs.items(), key=lambda x: -x[1]["total"]):
        fr = s["filled"] / s["total"] * 100 if s["total"] else 0
        pnl_mean = sum(s["pnl"]) / len(s["pnl"]) if s["pnl"] else 0
        n = len(s["pnl"])
        print(f"  {rid}: n={s['total']}, FR={fr:.0f}%, PnL30s={pnl_mean:.3f}bps (filled={n})")

    # Offset ratio distribution
    offsets = [r.get("effective_offset_used", 0) for r in filled if r.get("effective_offset_used") is not None]
    if offsets:
        offsets.sort()
        print(f"\nOffset ratio: min={offsets[0]:.4f}, median={offsets[len(offsets)//2]:.4f}, max={offsets[-1]:.4f}")

    # Spread distribution (bps)
    spreads = [r.get("spread_bps", 0) for r in filled if r.get("spread_bps") is not None]
    if spreads:
        spreads.sort()
        print(f"Spread bps: min={spreads[0]:.1f}, P25={spreads[len(spreads)//4]:.1f}, median={spreads[len(spreads)//2]:.1f}, P75={spreads[3*len(spreads)//4]:.1f}, max={spreads[-1]:.1f}")

    # SkipGate skip analysis
    # 133# P0-06: cancel_reason=None 時の startswith クラッシュを修正
    sg_skipped = [r for r in all_recs if (r.get("cancel_reason") or "").startswith("skip_gate")]
    sg_regime_skip = [r for r in all_recs if (r.get("cancel_reason") or "").startswith("skip_sell_unknown")]
    print(f"\nSkipGate skips: {len(sg_skipped)}")
    print(f"Regime sell skip (unknown): {len(sg_regime_skip)}")

    # Time filter skips
    tf_skipped = [r for r in all_recs if "time_filter" in (r.get("cancel_reason") or "")]
    print(f"Time filter skips: {len(tf_skipped)}")

    # VG / fast_fill defense
    vg_skips = [r for r in all_recs if "volatility" in (r.get("cancel_reason") or "").lower() or r.get("vg_triggered")]
    ff_boost = [r for r in filled if r.get("ffd_boost_active")]
    print(f"Volatility guard skips: {len(vg_skips)}")
    print(f"Fast fill defense active: {len(ff_boost)}")


if __name__ == "__main__":
    main()
