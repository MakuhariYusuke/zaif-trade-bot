#!/usr/bin/env python3
"""122# B1: Gate 自動判定スクリプト.

fill_records JSONL から G1.1-quick / G1.2-full 判定を実行し、
結果を構造化レポートとして出力する。

118# §2.4 で提案された自動判定パイプライン。
116# g1_1_quick_judgment / g1_2_full_judgment を活用し、
122# B2 Holm-Bonferroni 補正済み PnL 多重比較を統合。
135# P0-07: per-run Gate 評価 + P0-12: CLI 統一.

Usage:
  python scripts/v460/gate_judgment.py
  python scripts/v460/gate_judgment.py --results-dir results/v460/fill_test
  python scripts/v460/gate_judgment.py --latest-run
  python scripts/v460/gate_judgment.py --run-id 1771669596_481369d6
  python scripts/v460/gate_judgment.py --output results/v460/fill_test/judgment.json
  python scripts/v460/gate_judgment.py --side-breakdown
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# プロジェクトルート解決
_SCRIPT_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _SCRIPT_DIR.parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))

from ztb.metrics.fill_quality import (
    FillRecord,
    compute_fill_metrics,
    filter_clean_records,
    g1_1_quick_judgment,
    g1_2_full_judgment,
    load_fill_records,
)


def _load_all_records(results_dir: Path) -> list[FillRecord]:
    """JSONL ファイルから全レコードを読み込み."""
    import glob

    pattern = str(results_dir / "fill_records_*.jsonl")
    files = sorted(glob.glob(pattern))
    all_records: list[FillRecord] = []
    for f in files:
        records = load_fill_records(f)
        all_records.extend(records)
    return all_records


def _filter_by_run_id(
    records: list[FillRecord],
    run_id: str | None = None,
    *,
    latest: bool = False,
) -> list[FillRecord]:
    """135# P0-07: run_id でレコードをフィルタ.

    Args:
        records: 全 FillRecord.
        run_id: 特定の run_id でフィルタ. None の場合は latest が優先.
        latest: True の場合、最新の run_id のレコードのみ返す.

    Returns:
        フィルタ済みレコードリスト.
    """
    if run_id:
        return [r for r in records if r.run_id == run_id]
    if latest:
        # 最新 run_id = 最大 timestamp のレコードの run_id
        valid = [r for r in records if r.run_id and r.run_id.strip()]
        if not valid:
            return records
        latest_record = max(valid, key=lambda r: r.timestamp)
        target_id = latest_record.run_id
        return [r for r in records if r.run_id == target_id]
    return records


def _get_unique_run_ids(records: list[FillRecord]) -> list[str]:
    """135# P0-07: ユニークな run_id をタイムスタンプ昇順で返す."""
    seen: dict[str, float] = {}  # run_id -> min_timestamp
    for r in records:
        rid = r.run_id
        if rid and rid.strip():
            if rid not in seen or r.timestamp < seen[rid]:
                seen[rid] = r.timestamp
    return sorted(seen, key=lambda k: seen[k])


def _side_metrics(records: list[FillRecord], side: str) -> dict:
    """side別メトリクス算出."""
    side_recs = [r for r in records if r.side == side]
    if not side_recs:
        return {"n": 0}
    m = compute_fill_metrics(side_recs)
    return {
        "n": m.total_orders,
        "filled": m.filled_orders,
        "fill_rate": round(m.attempted_fill_rate, 4),
        "pnl_30s_mean": round(m.post_fill_30s_pnl_mean, 4),
        "pnl_30s_pvalue": round(m.post_fill_30s_pnl_pvalue, 4),
        "pnl_60s_mean": round(m.post_fill_60s_pnl_mean, 4),
        "pnl_60s_pvalue": round(m.post_fill_60s_pnl_pvalue, 4),
        "pnl_120s_mean": round(m.post_fill_120s_pnl_mean, 4),
        "pnl_120s_pvalue": round(m.post_fill_120s_pnl_pvalue, 4),
        "as_ratio": round(m.adverse_selection_ratio, 4),
        "skip_gate_ratio": round(m.skip_gate_ratio, 4),
    }


def _format_check(name: str, check: dict) -> str:
    """チェック結果を1行フォーマット."""
    status = "PASS" if check.get("pass") else "FAIL"
    icon = "✓" if check.get("pass") else "✗"
    parts = [f"  {icon} {name}: {status}"]

    if "value" in check:
        val = check["value"]
        if isinstance(val, float):
            parts.append(f"value={val:.4f}")
        else:
            parts.append(f"value={val}")

    if "threshold" in check:
        parts.append(f"threshold={check['threshold']}")
    if "pvalue_holm" in check:
        parts.append(f"p_raw={check.get('pvalue_raw', 'N/A'):.4f}")
        parts.append(f"p_holm={check['pvalue_holm']:.4f}")
    elif "pvalue" in check:
        parts.append(f"p={check['pvalue']:.4f}")
    if "alpha" in check:
        parts.append(f"α={check['alpha']}")

    return "  ".join(parts)


def run_gate_judgment(
    records: list[FillRecord],
    gate_cfg: dict,
    *,
    side_breakdown: bool = False,
    monte_carlo: bool = False,
    mc_simulations: int = 10_000,
    mc_lot: float = 0.001,
) -> dict:
    """Gate 判定のコアロジック.

    Args:
        records: 全 FillRecord リスト
        gate_cfg: gate_thresholds YAML から読んだ設定 dict
        side_breakdown: buy/sell 別メトリクスを含めるか
        monte_carlo: Monte Carlo シミュレーションを実行するか
        mc_simulations: MC シミュレーション回数
        mc_lot: MC lot size (BTC)

    Returns:
        判定結果の dict (JSON-safe)
    """
    # Clean/quarantine filter
    clean, quarantine = filter_clean_records(records)

    # Compute metrics
    metrics = compute_fill_metrics(clean)

    # Gate judgments
    quick_thresholds = gate_cfg.get("g1_1_quick_exec", {})
    full_thresholds = gate_cfg.get("g1_2_full_exec", {})
    quick = g1_1_quick_judgment(metrics, quick_thresholds)
    full = g1_2_full_judgment(metrics, full_thresholds)

    # Elapsed hours estimation
    if clean:
        ts_min = min(r.timestamp for r in clean)
        ts_max = max(r.timestamp for r in clean)
        elapsed_h = (ts_max - ts_min) / 3600
    else:
        elapsed_h = 0.0

    # Build result
    result: dict = {
        "data_summary": {
            "total_records": len(records),
            "clean_records": len(clean),
            "quarantine_records": len(quarantine),
            "elapsed_hours": round(elapsed_h, 1),
            "measurement_days": metrics.measurement_days,
        },
        "metrics": {
            "attempted_fill_rate": round(metrics.attempted_fill_rate, 4),
            "overall_fill_rate": round(metrics.overall_fill_rate, 4),
            "attempted_cancel_ratio": round(metrics.attempted_cancel_ratio, 4),
            "queue_wait_median_sec": round(metrics.queue_wait_median_sec, 2),
            "pnl_30s_mean": round(metrics.post_fill_30s_pnl_mean, 4),
            "pnl_30s_pvalue": round(metrics.post_fill_30s_pnl_pvalue, 6),
            "pnl_60s_mean": round(metrics.post_fill_60s_pnl_mean, 4),
            "pnl_60s_pvalue": round(metrics.post_fill_60s_pnl_pvalue, 6),
            "pnl_120s_mean": round(metrics.post_fill_120s_pnl_mean, 4),
            "pnl_120s_pvalue": round(metrics.post_fill_120s_pnl_pvalue, 6),
            "pnl_ci_upper": round(metrics.post_fill_30s_pnl_ci_upper, 4),
            "as_ratio": round(metrics.adverse_selection_ratio, 4),
            "skip_gate_ratio": round(metrics.skip_gate_ratio, 4),
        },
        "g1_1_quick": quick,
        "g1_2_full": full,
    }

    if side_breakdown:
        result["side_breakdown"] = {
            "buy": _side_metrics(clean, "buy"),
            "sell": _side_metrics(clean, "sell"),
        }

    # E10: Monte Carlo PnL シミュレーション (014# T5 → 122# gate_judgment 統合)
    mc_result_obj = None
    if monte_carlo:
        try:
            from ztb.risk.pnl_monte_carlo import (
                MonteCarloConfig,
                PnLMonteCarloSimulator,
            )

            filled_prices = [
                r.order_price for r in clean
                if r.filled and r.order_price is not None
            ]
            btc_price = (
                sum(filled_prices) / len(filled_prices)
                if filled_prices
                else 10_300_000.0
            )

            mc_config = MonteCarloConfig(
                n_simulations=mc_simulations,
                lot_size_btc=mc_lot,
                btc_price_jpy=btc_price,
            )
            sim = PnLMonteCarloSimulator(clean, mc_config)
            mc_result_obj = sim.run()
            result["monte_carlo"] = mc_result_obj.to_dict()
        except Exception as e:
            result["monte_carlo"] = {"error": str(e)}

    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="122# B1: Gate 自動判定")
    parser.add_argument(
        "--results-dir",
        default="results/v460/fill_test",
        help="fill_records JSONL ディレクトリ",
    )
    parser.add_argument(
        "--gate-config",
        default=None,
        help="gate_thresholds.yaml パス (default: configs/v460/gate_thresholds.yaml)",
    )
    parser.add_argument(
        "--output", "-o",
        default=None,
        help="判定結果の JSON 出力先",
    )
    parser.add_argument(
        "--side-breakdown",
        action="store_true",
        help="buy/sell 別のメトリクスを出力",
    )
    parser.add_argument(
        "--run-id",
        default=None,
        help="135# P0-07: 特定 run_id のレコードのみで判定",
    )
    parser.add_argument(
        "--latest-run",
        action="store_true",
        help="135# P0-07: 最新 run のレコードのみで判定 (全体との対比表示)",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="JSON のみ出力 (人間可読レポートなし)",
    )
    parser.add_argument(
        "--monte-carlo",
        action="store_true",
        help="Monte Carlo PnL シミュレーションを実行 (E10)",
    )
    parser.add_argument(
        "--mc-simulations",
        type=int,
        default=10_000,
        help="Monte Carlo シミュレーション回数 (default: 10,000)",
    )
    parser.add_argument(
        "--mc-lot",
        type=float,
        default=0.001,
        help="Monte Carlo lot size BTC (default: 0.001)",
    )
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    if not results_dir.is_absolute():
        results_dir = _PROJECT_ROOT / results_dir

    # Gate thresholds
    from scripts.v460.lib.config_loader import load_gate_thresholds
    gate_cfg = load_gate_thresholds(args.gate_config)

    # Load records
    all_records = _load_all_records(results_dir)
    if not all_records:
        print(f"ERROR: No fill records found in {results_dir}", file=sys.stderr)
        sys.exit(1)

    # 135# P0-07: per-run フィルタリング
    target_records = all_records
    run_scope = "ALL"
    if args.run_id:
        target_records = _filter_by_run_id(all_records, run_id=args.run_id)
        run_scope = f"run={args.run_id}"
        if not target_records:
            print(f"ERROR: No records for run_id={args.run_id}", file=sys.stderr)
            available = _get_unique_run_ids(all_records)
            print(f"Available run_ids: {available}", file=sys.stderr)
            sys.exit(1)
    elif args.latest_run:
        target_records = _filter_by_run_id(all_records, latest=True)
        if target_records and target_records[0].run_id:
            run_scope = f"LATEST({target_records[0].run_id})"

    # Run judgment via core function
    result = run_gate_judgment(
        target_records,
        gate_cfg,
        side_breakdown=args.side_breakdown,
        monte_carlo=args.monte_carlo,
        mc_simulations=args.mc_simulations,
        mc_lot=args.mc_lot,
    )
    result["run_scope"] = run_scope

    # 135# P0-07: --latest-run の場合、全体判定も並列実行して対比
    all_result = None
    if args.latest_run and len(target_records) < len(all_records):
        all_result = run_gate_judgment(
            all_records,
            gate_cfg,
            side_breakdown=args.side_breakdown,
        )
        all_result["run_scope"] = "ALL"
        result["comparison"] = {
            "all_g1_2": all_result["g1_2_full"].get("gate_result", "N/A"),
            "latest_g1_2": result["g1_2_full"].get("gate_result", "N/A"),
            "all_records": len(all_records),
            "latest_records": len(target_records),
        }

    # Extract quick/full from result for report display
    quick = result["g1_1_quick"]
    full = result["g1_2_full"]

    if args.json:
        print(json.dumps(result, indent=2, ensure_ascii=False))
    else:
        # Human-readable report
        print("=" * 60)
        print(f"  v460 Gate 自動判定レポート (122# B1)  [{run_scope}]")
        print("=" * 60)
        print()
        ds = result["data_summary"]
        print(f"  Scope:   {run_scope}")
        print(f"  Records: {ds['clean_records']} clean / {ds['quarantine_records']} quarantine")
        print(f"  Elapsed: {ds['elapsed_hours']:.1f}h / 168h ({ds['elapsed_hours']/168*100:.0f}%)")
        print(f"  Days:    {ds['measurement_days']}")
        print()

        # Metrics summary
        ms = result["metrics"]
        print("--- Key Metrics ---")
        print(f"  Fill Rate (attempted): {ms['attempted_fill_rate']:.1%}")
        print(f"  Fill Rate (overall):   {ms['overall_fill_rate']:.1%}")
        print(f"  AS Ratio:              {ms['as_ratio']:.1%}")
        print(f"  SkipGate Ratio:        {ms['skip_gate_ratio']:.1%}")
        print(f"  Queue Wait Median:     {ms['queue_wait_median_sec']:.1f}s")
        print()
        print("--- PnL Multi-Timeframe (Holm-Bonferroni 補正済み) ---")
        print(f"  PnL 30s:  mean={ms['pnl_30s_mean']:+.3f} bps  p={ms['pnl_30s_pvalue']:.4f}")
        print(f"  PnL 60s:  mean={ms['pnl_60s_mean']:+.3f} bps  p={ms['pnl_60s_pvalue']:.4f}")
        print(f"  PnL 120s: mean={ms['pnl_120s_mean']:+.3f} bps  p={ms['pnl_120s_pvalue']:.4f}")
        print(f"  PnL CI upper: {ms['pnl_ci_upper']:+.3f} bps")
        print()

        # Side breakdown
        if args.side_breakdown and "side_breakdown" in result:
            print("--- Side Breakdown ---")
            for side in ["buy", "sell"]:
                sb = result["side_breakdown"][side]
                if sb["n"] == 0:
                    print(f"  {side.upper()}: no records")
                    continue
                print(f"  {side.upper()}: n={sb['n']}, filled={sb['filled']}, "
                      f"fill_rate={sb['fill_rate']:.1%}, "
                      f"PnL30={sb['pnl_30s_mean']:+.3f}bps (p={sb['pnl_30s_pvalue']:.4f}), "
                      f"AS={sb['as_ratio']:.1%}")
            print()

        # G1.1-quick
        print("--- G1.1-quick (72h Kill Gate) ---")
        q_result = quick.get("gate_result", "N/A")
        q_icon = "✓" if q_result == "PASS" else ("⚠" if q_result == "WATCH" else "✗")
        print(f"  Result: {q_icon} {q_result}")
        for name, check in quick.get("checks", {}).items():
            print(_format_check(name, check))
        print()

        # G1.2-full
        print("--- G1.2-full (168h Qualification Gate) ---")
        f_result = full.get("gate_result", "N/A")
        f_icon = "✓" if f_result == "PASS" else "✗"
        print(f"  Result: {f_icon} {f_result}")
        for name, check in full.get("checks", {}).items():
            print(_format_check(name, check))
        print()

        # 135# P0-07: ALL vs LATEST 対比表示
        cmp = result.get("comparison")
        if cmp:
            print("--- 135# P0-07: ALL vs LATEST 対比 (Simpson 型リスク検出) ---")
            all_icon = "✓" if cmp["all_g1_2"] == "PASS" else ("⚠" if cmp["all_g1_2"] == "WATCH" else "✗")
            lat_icon = "✓" if cmp["latest_g1_2"] == "PASS" else ("⚠" if cmp["latest_g1_2"] == "WATCH" else "✗")
            print(f"  [ALL]    G1.2={all_icon} {cmp['all_g1_2']}  (n={cmp['all_records']})")
            print(f"  [LATEST] G1.2={lat_icon} {cmp['latest_g1_2']}  (n={cmp['latest_records']})")
            if cmp["all_g1_2"] != "FAIL" and cmp["latest_g1_2"] == "FAIL":
                print("  ⚠ WARNING: 全体は WATCH/PASS だが最新 run は FAIL — ドリフト悪化の兆候")
            print()

        print("=" * 60)

        # Monte Carlo section
        mc_data = result.get("monte_carlo")
        if mc_data is not None and "error" not in mc_data:
            print()
            print("--- Monte Carlo PnL Simulation (E10, 014# T5) ---")
            print(f"  Simulations:  {mc_data['n_simulations']:,}")
            print(f"  Cycles/month: {mc_data['cycles_per_month']:,}")
            print()
            print(f"  E[PnL]:      {mc_data['pnl_mean_jpy']:+,.0f} JPY/mo")
            print(f"  σ[PnL]:      {mc_data['pnl_std_jpy']:,.0f} JPY/mo")
            for k, v in mc_data.get("pnl_percentiles_jpy", {}).items():
                print(f"    {k}: {v:+,.0f} JPY")
            print()
            print(f"  VaR 95%:     {mc_data['var_95_jpy']:+,.0f} JPY")
            print(f"  CVaR 95%:    {mc_data['cvar_95_jpy']:+,.0f} JPY")
            print(f"  P(loss):     {mc_data['prob_loss']:.1%}")
            print(f"  P(profit):   {mc_data['prob_profit']:.1%}")
            print()
            print(f"  Break-even fill rate: {mc_data['breakeven_fill_rate']:.0%}")
            print("=" * 60)
        elif mc_data is not None and "error" in mc_data:
            print()
            print(f"--- Monte Carlo: ERROR: {mc_data['error']} ---")

    # Save if requested
    if args.output:
        out_path = Path(args.output)
        if not out_path.is_absolute():
            out_path = _PROJECT_ROOT / out_path
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2, ensure_ascii=False)
        print(f"Saved to: {out_path}")


if __name__ == "__main__":
    main()
