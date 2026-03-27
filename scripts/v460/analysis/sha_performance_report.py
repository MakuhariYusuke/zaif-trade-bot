"""647# SHA別パフォーマンス分析スクリプト.

fill_records を読み込み、SHA毎のパフォーマンスを比較する。
645#/646# 修正の効果検証に使用。

機能:
  - SHA毎の fills/pnl30/pnl120/win率/AS率
  - Side別・Regime別内訳
  - Cancel reason 内訳 (no_feasible_quote, spread_too_narrow 等)
  - ATR閾値 vs 実スプレッドの乖離分析

Usage:
    python -m scripts.v460.analysis.sha_performance_report [--days N] [--output PATH]
"""

from __future__ import annotations

import collections
import datetime as dt
import json
import re
import sys
from pathlib import Path
from typing import Any

import numpy as np


def load_records(results_dir: Path, days: int = 7) -> list[dict[str, Any]]:
    """直近N日分のfill_recordsを読み込む."""
    records: list[dict[str, Any]] = []
    today = dt.date.today()
    for d in range(days):
        day = today - dt.timedelta(days=d)
        path = results_dir / f"fill_records_{day.strftime('%Y%m%d')}.jsonl"
        if path.exists():
            for line in path.open():
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    return records


def _ts_to_str(ts: float | str) -> str:
    if isinstance(ts, (int, float)) and ts > 1e9:
        return dt.datetime.fromtimestamp(ts).strftime("%m/%d %H:%M")
    return str(ts)[:16]


def _safe_pnl(records: list[dict[str, Any]], key: str = "post_fill_30s_pnl") -> list[float]:
    return [float(r[key]) for r in records if r.get(key) is not None]


def _decile_table(pnls: list[float]) -> str:
    if len(pnls) < 10:
        return "  (insufficient data)"
    arr = np.array(pnls)
    parts = []
    for pct in [5, 10, 25, 50, 75, 90, 95]:
        parts.append(f"p{pct}={np.percentile(arr, pct):+.3f}")
    return "  " + " ".join(parts)


def analyze_sha(
    records: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """SHA毎に集計."""
    sha_groups: dict[str, list[dict[str, Any]]] = collections.defaultdict(list)
    for r in records:
        sha = str(r.get("git_sha", "?"))[:10]
        sha_groups[sha].append(r)

    results = []
    for sha, recs in sha_groups.items():
        filled = [r for r in recs if r.get("filled")]
        pnl30 = _safe_pnl(filled)
        pnl120 = _safe_pnl(filled, "post_fill_120s_pnl")

        # Timestamps
        timestamps = [r.get("timestamp", 0) for r in recs if isinstance(r.get("timestamp"), (int, float))]
        ts_min = min(timestamps) if timestamps else 0
        ts_max = max(timestamps) if timestamps else 0

        # Side breakdown
        side_stats: dict[str, dict[str, Any]] = {}
        for side in ("buy", "sell"):
            sf = [r for r in filled if r.get("side") == side]
            sp30 = _safe_pnl(sf)
            sp120 = _safe_pnl(sf, "post_fill_120s_pnl")
            # Skip gate analysis
            sg_scores = [float(r["skip_gate_score"]) for r in sf if r.get("skip_gate_score") is not None]
            sg_skipped_all = [r for r in recs if r.get("side") == side and r.get("skip_gate_skipped")]
            # AS analysis
            as_cnt = sum(1 for r in sf if r.get("adverse_selected"))
            side_stats[side] = {
                "n_total": sum(1 for r in recs if r.get("side") == side),
                "n_filled": len(sf),
                "avg_pnl30": float(np.mean(sp30)) if sp30 else None,
                "avg_pnl120": float(np.mean(sp120)) if sp120 else None,
                "sum_pnl30": float(np.sum(sp30)) if sp30 else 0,
                "win_rate": sum(1 for x in sp30 if x > 0) / len(sp30) * 100 if sp30 else None,
                "as_rate": as_cnt / len(sf) * 100 if sf else None,
                "n_skipped": len(sg_skipped_all),
                "avg_sg_score": float(np.mean(sg_scores)) if sg_scores else None,
                "sg_score_std": float(np.std(sg_scores)) if len(sg_scores) > 1 else None,
                "pnl30_values": sp30,
            }

        # Regime breakdown
        regime_stats: dict[str, dict[str, Any]] = {}
        for regime in set(r.get("regime", "null") for r in filled):
            rf = [r for r in filled if r.get("regime") == regime]
            rp30 = _safe_pnl(rf)
            regime_stats[str(regime)] = {
                "n_filled": len(rf),
                "avg_pnl30": float(np.mean(rp30)) if rp30 else None,
                "sum_pnl30": float(np.sum(rp30)) if rp30 else 0,
            }

        # Cancel reason breakdown
        cancel_counts: dict[str, int] = collections.Counter()
        for r in recs:
            cr = r.get("cancel_reason") or ""
            if cr:
                cancel_counts[cr] += 1
        cancel_stats = {k: {"n": v, "pct": v / len(recs) * 100} for k, v in cancel_counts.most_common()}

        # ATR threshold analysis — extract from error_message
        atr_thresholds: list[float] = []
        actual_spreads_jpy: list[float] = []
        sigma_values: list[float] = []
        for r in recs:
            msg = str(r.get("error_message", "") or "")
            m_spread = re.search(r"Spread too narrow: (\d+) JPY < min (\d+)", msg)
            m_sigma = re.search(r"σ=([\d.]+)", msg)
            if m_spread:
                actual_spreads_jpy.append(float(m_spread.group(1)))
                atr_thresholds.append(float(m_spread.group(2)))
            if m_sigma:
                sigma_values.append(float(m_sigma.group(1)))

        spread_analysis: dict[str, Any] = {}
        if atr_thresholds:
            spread_analysis = {
                "n_spread_rejects": len(atr_thresholds),
                "avg_threshold_jpy": float(np.mean(atr_thresholds)),
                "avg_actual_spread_jpy": float(np.mean(actual_spreads_jpy)),
                "spread_gap_jpy": float(np.mean(atr_thresholds)) - float(np.mean(actual_spreads_jpy)),
                "avg_sigma": float(np.mean(sigma_values)) if sigma_values else None,
                "sigma_at_cap": all(abs(s - max(sigma_values)) < 1e-6 for s in sigma_values) if sigma_values else False,
            }

        results.append({
            "sha": sha,
            "ts_min": ts_min,
            "ts_max": ts_max,
            "ts_min_str": _ts_to_str(ts_min),
            "ts_max_str": _ts_to_str(ts_max),
            "n_orders": len(recs),
            "n_filled": len(filled),
            "fill_rate": len(filled) / len(recs) * 100 if recs else 0,
            "avg_pnl30": float(np.mean(pnl30)) if pnl30 else None,
            "avg_pnl120": float(np.mean(pnl120)) if pnl120 else None,
            "sum_pnl30": float(np.sum(pnl30)) if pnl30 else 0,
            "sum_pnl120": float(np.sum(pnl120)) if pnl120 else 0,
            "win_rate": sum(1 for x in pnl30 if x > 0) / len(pnl30) * 100 if pnl30 else None,
            "n_skipped": sum(1 for r in recs if r.get("skip_gate_skipped")),
            "pnl30_percentiles": _decile_table(pnl30),
            "side": side_stats,
            "regime": regime_stats,
            "cancel_reasons": cancel_stats,
            "spread_analysis": spread_analysis,
        })

    # Sort by ts_min
    results.sort(key=lambda x: x["ts_min"])
    return results


def print_report(sha_results: list[dict[str, Any]]) -> None:
    """コンソールレポート出力."""
    print("=" * 100)
    print("SHA別パフォーマンスレポート")
    print("=" * 100)

    for sr in sha_results:
        sha = sr["sha"]
        print(f"\n{'─' * 80}")
        print(
            f"SHA: {sha}  期間: {sr['ts_min_str']} ~ {sr['ts_max_str']}"
        )
        avg30 = sr["avg_pnl30"]
        avg120 = sr["avg_pnl120"]
        print(
            f"  orders={sr['n_orders']:4d}  fills={sr['n_filled']:3d}"
            f"({sr['fill_rate']:.0f}%)  skipped={sr['n_skipped']}"
            f"  avg_pnl30={avg30:+.3f}bps" if avg30 is not None else f"  avg_pnl30=N/A",
            end="",
        )
        if avg120 is not None:
            print(f"  avg_pnl120={avg120:+.3f}bps", end="")
        if sr["win_rate"] is not None:
            print(f"  win={sr['win_rate']:.0f}%", end="")
        print(f"  sum_pnl30={sr['sum_pnl30']:+.1f}bps")
        print(sr["pnl30_percentiles"])

        # Side
        for side in ("buy", "sell"):
            ss = sr["side"].get(side, {})
            if not ss or not ss.get("n_filled"):
                continue
            avg = ss["avg_pnl30"]
            avg_str = f"{avg:+.3f}" if avg is not None else "N/A"
            wr = ss["win_rate"]
            wr_str = f"{wr:.0f}%" if wr is not None else "N/A"
            asr = ss["as_rate"]
            asr_str = f"{asr:.0f}%" if asr is not None else "N/A"
            sg_std = ss.get("sg_score_std")
            sg_std_str = f"{sg_std:.4f}" if sg_std is not None else "N/A"
            print(
                f"  {side:4s}: fills={ss['n_filled']:3d} skip={ss['n_skipped']:3d}"
                f"  avg_p30={avg_str}bps  win={wr_str}  AS={asr_str}"
                f"  sg_std={sg_std_str}  sum={ss['sum_pnl30']:+.1f}bps"
            )

        # Regime
        if sr["regime"]:
            regimes = sorted(sr["regime"].items(), key=lambda x: x[1]["n_filled"], reverse=True)
            parts = []
            for regime, rs in regimes:
                avg_r = rs["avg_pnl30"]
                avg_str = f"{avg_r:+.3f}" if avg_r is not None else "N/A"
                parts.append(f"{regime}:{rs['n_filled']}({avg_str})")
            print(f"  regime: {' | '.join(parts)}")

        # Cancel reasons
        cr = sr.get("cancel_reasons", {})
        if cr:
            parts = []
            for reason, stats in sorted(cr.items(), key=lambda x: x[1]["n"], reverse=True)[:5]:
                parts.append(f"{reason}={stats['n']}({stats['pct']:.0f}%)")
            print(f"  cancel: {', '.join(parts)}")

        # Spread analysis
        sa = sr.get("spread_analysis", {})
        if sa:
            cap_str = " [σ AT CAP]" if sa.get("sigma_at_cap") else ""
            sigma_str = f"σ={sa['avg_sigma']:.6f}" if sa.get("avg_sigma") else ""
            print(
                f"  spread_reject: n={sa['n_spread_rejects']}"
                f"  thresh={sa['avg_threshold_jpy']:.0f}JPY"
                f"  actual={sa['avg_actual_spread_jpy']:.0f}JPY"
                f"  gap={sa['spread_gap_jpy']:+.0f}JPY"
                f"  {sigma_str}{cap_str}"
            )

    # Comparison table
    if len(sha_results) >= 2:
        print(f"\n{'=' * 100}")
        print("SHA比較サマリー")
        print(f"{'=' * 100}")
        print(f"{'SHA':<12} {'期間':<22} {'n':>4} {'fills':>5} {'fill%':>5} {'avg_p30':>9} {'win%':>5} {'sum_p30':>9} {'nfq':>4} {'stn':>4} {'mcb':>4}")
        print("-" * 105)
        for sr in sha_results:
            avg30 = f"{sr['avg_pnl30']:+.3f}" if sr['avg_pnl30'] is not None else "N/A"
            wr = f"{sr['win_rate']:.0f}" if sr['win_rate'] is not None else "?"
            cr = sr.get("cancel_reasons", {})
            nfq = cr.get("no_feasible_quote", {}).get("n", 0)
            stn = cr.get("spread_too_narrow", {}).get("n", 0)
            mcb = cr.get("mcb_halt", {}).get("n", 0)
            print(
                f"{sr['sha']:<12} {sr['ts_min_str']}~{sr['ts_max_str']:<5}"
                f" {sr['n_orders']:4d} {sr['n_filled']:5d} {sr['fill_rate']:5.1f}"
                f" {avg30:>9} {wr:>5}"
                f" {sr['sum_pnl30']:>+9.1f} {nfq:4d} {stn:4d} {mcb:4d}"
            )


def save_json(sha_results: list[dict[str, Any]], output: Path) -> None:
    """JSON出力 (pnl30_values は除外)."""
    cleaned = []
    for sr in sha_results:
        c = {k: v for k, v in sr.items()}
        # Remove raw values for JSON
        for side_key in c.get("side", {}):
            c["side"][side_key].pop("pnl30_values", None)
        cleaned.append(c)
    output.parent.mkdir(parents=True, exist_ok=True)
    with open(output, "w") as f:
        json.dump({"generated": dt.datetime.now().isoformat(), "sha_results": cleaned}, f, indent=2, default=str)
    print(f"\nJSON saved: {output}")


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="SHA別パフォーマンスレポート")
    parser.add_argument("--days", type=int, default=7, help="分析期間 (日)")
    parser.add_argument("--results-dir", default="results/v460/fill_test", help="fill_records ディレクトリ")
    parser.add_argument("--output", default=None, help="JSON出力先 (省略時はコンソールのみ)")
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    records = load_records(results_dir, days=args.days)
    if not records:
        print("No records found.")
        sys.exit(1)

    print(f"Loaded {len(records)} records from {args.days} days")
    sha_results = analyze_sha(records)
    print_report(sha_results)

    if args.output:
        save_json(sha_results, Path(args.output))


if __name__ == "__main__":
    main()
