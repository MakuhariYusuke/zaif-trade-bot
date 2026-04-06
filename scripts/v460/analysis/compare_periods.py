"""705# Post-deployment multi-perspective period comparison script.

Usage:
    python -m scripts.v460.analysis.compare_periods --pre 2026-04-01:2026-04-03 --post 2026-04-04:2026-04-06
    python -m scripts.v460.analysis.compare_periods --pre 2026-04-01:2026-04-03 --post 2026-04-04:2026-04-06 --output analysis_results/705_comparison.txt

Purpose: 二つの期間を多角的に比較し、デプロイ前後の効果を自動分析。
    - Side×Regime別 PnL、offset pipeline分解、guard統計、skip_gate統計
    - 自動的に劣化/改善を検出し、root cause候補を提示
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from collections import defaultdict
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

from scripts.v460.analysis.analysis_common import (
    DEFAULT_RESULTS_DIR,
    Record,
)


def _parse_date_range(spec: str) -> tuple[str, str]:
    """'YYYY-MM-DD:YYYY-MM-DD' → (start, end) as YYYYMMDD."""
    parts = spec.split(":")
    if len(parts) != 2:
        raise ValueError(f"Date range must be START:END, got {spec!r}")
    start = parts[0].replace("-", "")
    end = parts[1].replace("-", "")
    return start, end


def _load_period(
    data_dir: Path, start: str, end: str
) -> list[dict[str, Any]]:
    """Load fill records for all dates in [start, end]."""
    records: list[dict[str, Any]] = []
    d = datetime.strptime(start, "%Y%m%d")
    end_d = datetime.strptime(end, "%Y%m%d")
    while d <= end_d:
        fname = data_dir / f"fill_records_{d.strftime('%Y%m%d')}.jsonl"
        if fname.exists():
            with open(fname) as f:
                for line in f:
                    if line.strip():
                        try:
                            records.append(json.loads(line))
                        except json.JSONDecodeError:
                            pass
        d += timedelta(days=1)
    return records


def _get_side(r: dict[str, Any]) -> str:
    return str(
        r.get("side") or r.get("last_attempted_side") or r.get("requested_side") or "unknown"
    )


def _safe_mean(vals: list[float]) -> float:
    return statistics.mean(vals) if vals else 0.0


def _safe_median(vals: list[float]) -> float:
    return statistics.median(vals) if vals else 0.0


def _pct(n: int, total: int) -> str:
    if total == 0:
        return "N/A"
    return f"{n / total * 100:.1f}%"


def _analyze_period(
    records: list[dict[str, Any]], label: str, out: list[str]
) -> dict[str, Any]:
    """Analyze a single period and collect metrics."""
    fills = [r for r in records if r.get("filled")]
    buys = [r for r in fills if _get_side(r) == "buy"]
    sells = [r for r in fills if _get_side(r) == "sell"]

    buy_pnl = [r["post_fill_30s_pnl"] for r in buys if r.get("post_fill_30s_pnl") is not None]
    sell_pnl = [r["post_fill_30s_pnl"] for r in sells if r.get("post_fill_30s_pnl") is not None]
    all_pnl = buy_pnl + sell_pnl

    buy_120 = [r["post_fill_120s_pnl"] for r in buys if r.get("post_fill_120s_pnl") is not None]
    sell_120 = [r["post_fill_120s_pnl"] for r in sells if r.get("post_fill_120s_pnl") is not None]

    buy_off = [r["effective_offset_used"] for r in buys if r.get("effective_offset_used") is not None]
    sell_off = [r["effective_offset_used"] for r in sells if r.get("effective_offset_used") is not None]

    buy_sc = [r["spread_capture_bps"] for r in buys if r.get("spread_capture_bps") is not None]
    sell_sc = [r["spread_capture_bps"] for r in sells if r.get("spread_capture_bps") is not None]

    spreads = [r["spread_bps"] for r in fills if r.get("spread_bps") is not None]

    sag_triggered = len([r for r in fills if r.get("spread_as_guard_triggered")])
    eg_blocked = len([r for r in records if r.get("entry_gate_blocked")])
    eg_suppressed = len([r for r in records if r.get("entry_gate_guard_suppressed")])

    buy_as = len([r for r in buys if r.get("adverse_selected") or r.get("adverse_selected_raw")])
    sell_as = len([r for r in sells if r.get("adverse_selected") or r.get("adverse_selected_raw")])

    buy_skips = len([r for r in records if r.get("skip_gate_skipped") and _get_side(r) == "buy"])

    buy_sg_scores = [
        r["skip_gate_score"]
        for r in records
        if r.get("skip_gate_score") is not None and _get_side(r) == "buy"
    ]

    # Side × Regime breakdown
    sr_metrics: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    for r in fills:
        side = _get_side(r)
        regime = str(r.get("regime") or r.get("regime_at_order") or "unknown")
        pnl = r.get("post_fill_30s_pnl")
        if pnl is not None:
            sr_metrics[f"{side}×{regime}"]["pnl"].append(pnl)

    # Hour breakdown
    hour_pnl: dict[str, dict[int, list[float]]] = {"buy": defaultdict(list), "sell": defaultdict(list)}
    for r in fills:
        side = _get_side(r)
        ts = r.get("timestamp")
        pnl = r.get("post_fill_30s_pnl")
        if ts and pnl is not None and side in ("buy", "sell"):
            hour = datetime.utcfromtimestamp(ts).hour
            hour_pnl[side][hour].append(pnl)

    # Offset stages (buy only)
    offset_stages_agg: dict[str, list[float]] = defaultdict(list)
    for r in buys:
        os_raw = r.get("offset_stages")
        if os_raw:
            os_data = json.loads(os_raw) if isinstance(os_raw, str) else os_raw
            for k, v in os_data.items():
                if v is not None and k != "schema_version":
                    try:
                        offset_stages_agg[k].append(float(v))
                    except (ValueError, TypeError):
                        pass

    # Cancel reasons
    cancel_counts: dict[str, int] = defaultdict(int)
    for r in records:
        if r.get("cancelled") and r.get("cancel_reason"):
            side = _get_side(r)
            cancel_counts[f"{side}:{r['cancel_reason']}"] += 1

    out.append(f"\n{'=' * 60}")
    out.append(f"  {label}")
    out.append(f"{'=' * 60}")
    out.append(f"  Cycles: {len(records)} | Fills: {len(fills)} ({_pct(len(fills), len(records))})")
    out.append(f"  BUY: {len(buys)} | SELL: {len(sells)}")
    out.append("")

    out.append("  30s PnL (bps):")
    if all_pnl:
        win = len([p for p in all_pnl if p > 0])
        out.append(f"    ALL:  avg={_safe_mean(all_pnl):+.3f} total={sum(all_pnl):+.1f} win={_pct(win, len(all_pnl))}")
    if buy_pnl:
        win = len([p for p in buy_pnl if p > 0])
        out.append(f"    BUY:  avg={_safe_mean(buy_pnl):+.3f} total={sum(buy_pnl):+.1f} win={_pct(win, len(buy_pnl))}")
    if sell_pnl:
        win = len([p for p in sell_pnl if p > 0])
        out.append(f"    SELL: avg={_safe_mean(sell_pnl):+.3f} total={sum(sell_pnl):+.1f} win={_pct(win, len(sell_pnl))}")

    out.append("")
    out.append("  120s PnL (bps):")
    if buy_120:
        out.append(f"    BUY:  avg={_safe_mean(buy_120):+.3f}")
    if sell_120:
        out.append(f"    SELL: avg={_safe_mean(sell_120):+.3f}")

    out.append("")
    out.append(f"  Adverse Selection: buy={_pct(buy_as, len(buys))} sell={_pct(sell_as, len(sells))}")
    out.append(f"  Spread Capture: buy={_safe_mean(buy_sc):+.3f} sell={_safe_mean(sell_sc):+.3f}")
    out.append(f"  Effective Offset: buy={_safe_mean(buy_off):.4f} sell={_safe_mean(sell_off):.4f}")
    out.append(f"  Avg Spread (bps): {_safe_mean(spreads):.2f}")

    out.append("")
    out.append("  Guards:")
    out.append(f"    spread_as_guard triggered: {sag_triggered}/{len(fills)} ({_pct(sag_triggered, len(fills))})")
    out.append(f"    entry_gate blocked: {eg_blocked} suppressed: {eg_suppressed} actual_blocks: {eg_blocked - eg_suppressed}")
    out.append(f"    skip_gate buy skips: {buy_skips}")
    if buy_sg_scores:
        out.append(f"    skip_gate buy score: avg={_safe_mean(buy_sg_scores):+.4f} med={_safe_median(buy_sg_scores):+.4f}")

    out.append("")
    out.append("  Side×Regime PnL:")
    for key in sorted(sr_metrics.keys()):
        vals = sr_metrics[key]["pnl"]
        out.append(f"    {key}: n={len(vals)} avg={_safe_mean(vals):+.3f} total={sum(vals):+.1f}")

    out.append("")
    out.append("  Buy Offset Stages (top components):")
    for k in sorted(offset_stages_agg.keys()):
        vals = offset_stages_agg[k]
        avg = _safe_mean(vals)
        if abs(avg) < 0.001:
            continue
        out.append(f"    {k}: avg={avg:.4f}")

    out.append("")
    out.append("  Worst Buy Hours (UTC, by avg 30s PnL):")
    sorted_hours = sorted(hour_pnl["buy"].items(), key=lambda x: _safe_mean(x[1]))
    for h, vals in sorted_hours[:5]:
        out.append(f"    {h:02d}h: avg={_safe_mean(vals):+.3f} n={len(vals)}")

    out.append("")
    out.append("  Top Cancel Reasons:")
    for reason, n in sorted(cancel_counts.items(), key=lambda x: -x[1])[:10]:
        out.append(f"    {reason}: {n}")

    return {
        "label": label,
        "cycles": len(records),
        "fills": len(fills),
        "buy_fills": len(buys),
        "sell_fills": len(sells),
        "buy_pnl_avg": _safe_mean(buy_pnl),
        "sell_pnl_avg": _safe_mean(sell_pnl),
        "buy_pnl_total": sum(buy_pnl) if buy_pnl else 0,
        "sell_pnl_total": sum(sell_pnl) if sell_pnl else 0,
        "buy_offset": _safe_mean(buy_off),
        "sell_offset": _safe_mean(sell_off),
        "avg_spread": _safe_mean(spreads),
        "buy_skips": buy_skips,
        "buy_sg_score": _safe_mean(buy_sg_scores),
        "sag_rate": sag_triggered / max(len(fills), 1),
        "actual_blocks": eg_blocked - eg_suppressed,
    }


def _compare_and_diagnose(
    pre: dict[str, Any], post: dict[str, Any], out: list[str]
) -> None:
    """Compare two periods and output summary + auto-diagnosis."""
    out.append(f"\n{'=' * 60}")
    out.append("  COMPARISON SUMMARY & AUTO-DIAGNOSIS")
    out.append(f"{'=' * 60}")

    def delta(key: str, fmt: str = "+.3f", reverse: bool = False) -> str:
        p_val = pre.get(key, 0)
        q_val = post.get(key, 0)
        d = q_val - p_val
        indicator = ""
        if isinstance(d, (int, float)):
            # For PnL: positive delta = improvement
            if reverse:
                indicator = " ✓" if d < 0 else " ✗" if d > 0 else ""
            else:
                indicator = " ✓" if d > 0 else " ✗" if d < 0 else ""
        return f"{d:{fmt}}{indicator}"

    out.append(f"  30s PnL (buy):  {pre['buy_pnl_avg']:+.3f} → {post['buy_pnl_avg']:+.3f} (Δ{delta('buy_pnl_avg')})")
    out.append(f"  30s PnL (sell): {pre['sell_pnl_avg']:+.3f} → {post['sell_pnl_avg']:+.3f} (Δ{delta('sell_pnl_avg')})")
    out.append(f"  Offset (buy):   {pre['buy_offset']:.4f} → {post['buy_offset']:.4f} (Δ{delta('buy_offset')})")
    out.append(f"  Offset (sell):  {pre['sell_offset']:.4f} → {post['sell_offset']:.4f}")
    out.append(f"  Spread (bps):   {pre['avg_spread']:.2f} → {post['avg_spread']:.2f}")
    out.append(f"  SAG trigger:    {pre['sag_rate']:.1%} → {post['sag_rate']:.1%}")
    out.append(f"  Buy skips:      {pre['buy_skips']} → {post['buy_skips']}")
    out.append(f"  SG score (buy): {pre['buy_sg_score']:+.4f} → {post['buy_sg_score']:+.4f}")
    out.append(f"  Actual blocks:  {pre['actual_blocks']} → {post['actual_blocks']}")

    out.append("")
    out.append("  Auto-diagnosis:")

    # Check sell improvement
    sell_delta = post["sell_pnl_total"] - pre["sell_pnl_total"]
    if sell_delta > 50:
        out.append("    [SELL IMPROVED] Sell PnL significantly improved — spread_as_guard/offset fixes effective")
    elif sell_delta > 0:
        out.append("    [SELL IMPROVED] Modest sell improvement")

    # Check buy degradation
    buy_delta = post["buy_pnl_total"] - pre["buy_pnl_total"]
    if buy_delta < -50:
        out.append("    [BUY DEGRADED] Buy PnL significantly degraded")
        # Root cause candidates
        spread_delta = post["avg_spread"] - pre["avg_spread"]
        if spread_delta < -0.2:
            out.append("      → Spread tightened significantly — offset compression likely")
        offset_delta = post["buy_offset"] - pre["buy_offset"]
        if offset_delta < -0.03:
            out.append("      → Buy offset dropped >3% — fills closer to mid, higher adverse risk")
        sg_delta = post["buy_sg_score"] - pre["buy_sg_score"]
        if sg_delta < -0.3:
            out.append("      → Skip gate score collapsed — model drift likely, not filtering marginal trades")
        if post["buy_skips"] == 0 and pre["buy_skips"] > 10:
            out.append("      → Skip gate completely stopped — zero buy skips (was {})".format(pre["buy_skips"]))
        if post["actual_blocks"] < 5:
            out.append("      → Entry gate auto-disabled — no effective blocking (check max_block_rate)")
    elif buy_delta < 0:
        out.append("    [BUY DEGRADED] Modest buy degradation")

    # Net assessment
    net_pre = pre["buy_pnl_total"] + pre["sell_pnl_total"]
    net_post = post["buy_pnl_total"] + post["sell_pnl_total"]
    out.append("")
    out.append(f"  Net PnL: {net_pre:+.1f} → {net_post:+.1f} (Δ{net_post - net_pre:+.1f})")
    if net_post > net_pre:
        out.append("  Overall: IMPROVED, but buy-side attention needed")
    else:
        out.append("  Overall: DETERIORATED")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare two fill_test periods (pre/post deployment)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--pre",
        required=True,
        help="Pre-deployment date range (YYYY-MM-DD:YYYY-MM-DD)",
    )
    parser.add_argument(
        "--post",
        required=True,
        help="Post-deployment date range (YYYY-MM-DD:YYYY-MM-DD)",
    )
    parser.add_argument(
        "--data-dir",
        default=DEFAULT_RESULTS_DIR,
        help=f"Data directory (default: {DEFAULT_RESULTS_DIR})",
    )
    parser.add_argument("--output", help="Output file path (optional, default: stdout)")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    pre_start, pre_end = _parse_date_range(args.pre)
    post_start, post_end = _parse_date_range(args.post)

    pre_records = _load_period(data_dir, pre_start, pre_end)
    post_records = _load_period(data_dir, post_start, post_end)

    out: list[str] = []
    out.append(f"Period Comparison Report — {datetime.now(tz=timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}")
    out.append(f"Pre:  {args.pre} ({len(pre_records)} records)")
    out.append(f"Post: {args.post} ({len(post_records)} records)")

    pre_metrics = _analyze_period(pre_records, f"PRE ({args.pre})", out)
    post_metrics = _analyze_period(post_records, f"POST ({args.post})", out)
    _compare_and_diagnose(pre_metrics, post_metrics, out)

    text = "\n".join(out)
    if args.output:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        Path(args.output).write_text(text, encoding="utf-8")
        print(f"Report written to {args.output}")
    else:
        print(text)


if __name__ == "__main__":
    main()
