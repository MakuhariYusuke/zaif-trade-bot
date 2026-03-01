"""132# 追加分析: daily, run_id, reprice 等."""
import datetime
from collections import defaultdict
from dataclasses import dataclass, field

from ztb.metrics.fill_quality import PnlAccumulator, load_fill_record_objects_glob
from ztb.utils.safety import safe_to_finite


@dataclass
class _CountPnlSummary:
    total: int = 0
    filled: int = 0
    pnl_30s: PnlAccumulator = field(default_factory=PnlAccumulator)

    def add(self, *, filled: bool, pnl_30s: float | None) -> None:
        self.total += 1
        if filled:
            self.filled += 1
            self.pnl_30s.add(pnl_30s)


def _pct(numerator: int, denominator: int) -> float:
    return numerator / denominator * 100.0 if denominator > 0 else 0.0


def main() -> None:
    all_recs = load_fill_record_objects_glob(
        "results/v460/fill_test",
        include_emergency=False,
    )

    filled = [r for r in all_recs if r.get("filled")]

    # Stale/reprice
    repriced = [r for r in filled if r.get("reprice_count", 0) > 0]
    print(f"Repriced: {len(repriced)}/{len(filled)} ({_pct(len(repriced), len(filled)):.1f}%)")

    # Consecutive same side
    last_side = None
    max_consec = 0
    cur_consec = 0
    day_stats: dict[str, _CountPnlSummary] = defaultdict(_CountPnlSummary)
    runs: dict[str, _CountPnlSummary] = defaultdict(_CountPnlSummary)
    for r in all_recs:
        s = r.get("side")
        if s == last_side:
            cur_consec += 1
            max_consec = max(max_consec, cur_consec)
        else:
            cur_consec = 1
            last_side = s

        filled_flag = bool(r.get("filled"))
        pnl_30s = safe_to_finite(r.get("post_fill_30s_pnl"))
        rid = str(r.get("run_id", "none"))
        runs[rid].add(filled=filled_flag, pnl_30s=pnl_30s)

        ts = safe_to_finite(r.get("timestamp"))
        if ts is not None:
            day = datetime.datetime.fromtimestamp(
                ts,
                tz=datetime.timezone.utc,
            ).strftime("%Y-%m-%d")
            day_stats[day].add(filled=filled_flag, pnl_30s=pnl_30s)

    print(f"Max consecutive same side: {max_consec}")

    print("Daily breakdown:")
    for d in sorted(day_stats.keys()):
        s = day_stats[d]
        print(
            f"  {d}: total={s.total}, filled={s.filled}, FR={_pct(s.filled, s.total):.0f}%, "
            f"PnL30s={s.pnl_30s.mean_bps:.3f}bps (n={s.pnl_30s.count})"
        )

    print(f"\nRun IDs: {len(runs)}")
    for rid, s in sorted(runs.items(), key=lambda x: -x[1].total):
        print(
            f"  {rid}: n={s.total}, FR={_pct(s.filled, s.total):.0f}%, "
            f"PnL30s={s.pnl_30s.mean_bps:.3f}bps (filled={s.pnl_30s.count})"
        )

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
