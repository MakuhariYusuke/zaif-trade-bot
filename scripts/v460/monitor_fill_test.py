#!/usr/bin/env python3
"""
fill_test モニタリングスクリプト — 000# §3.9 継続中止ルール自動判定.

Usage:
  python scripts/v460/monitor_fill_test.py
  python scripts/v460/monitor_fill_test.py --results-dir results/v460/fill_test
  python scripts/v460/monitor_fill_test.py --watch --interval 300
"""

from __future__ import annotations

import argparse
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path
from typing import Callable, TypedDict

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))

from ztb.metrics.fill_quality import (
    FillMetrics,
    FillRecord,
    compute_fill_metrics,
    compute_hourly_metrics,
    compute_record_pnl_jpy,
    compute_regime_metrics,
    compute_round_trip_metrics,
    iter_fill_records_glob,
    g1_1_quick_judgment,
    g1_2_full_judgment,
)
from ztb.metrics.fill_record_integrity import partition_clean_records
from ztb.io.json_io import write_json
from scripts.v460.lib.config_loader import load_gate_thresholds


# ======================================================================
# §3.9 継続中止ルール
# ======================================================================

StopCheckFn = Callable[[FillMetrics, list[FillRecord]], bool]


StopRuleResult = TypedDict(
    "StopRuleResult",
    {
        "rule": str,
        "status": str,
        "triggered": bool,
        "reason": str,
        "min_n": int,
    },
)


GateCheckDetail = TypedDict(
    "GateCheckDetail",
    {
        "pass": bool,
        "value": float,
        "threshold": float,
    },
    total=False,
)


class GateResult(TypedDict, total=False):
    checks: dict[str, GateCheckDetail]
    gate_result: str


@lru_cache(maxsize=1)
def _gate_thresholds() -> tuple[dict[str, object], dict[str, object]]:
    """閾値 YAML は watch ループで再利用する."""
    cfg = load_gate_thresholds()
    quick = cfg.get("g1_1_quick_exec", {})
    full = cfg.get("g1_2_full_exec", {})
    quick_dict = quick if isinstance(quick, dict) else {}
    full_dict = full if isinstance(full, dict) else {}
    return quick_dict, full_dict


@dataclass(frozen=True)
class StopRule:
    """§3.9 中止ルール定義."""
    desc: str
    min_n: int
    check: StopCheckFn
    judgment: str  # STOP / PAUSE / CONTINUE


def _r1_fill_rate(metrics: FillMetrics, _records: list[FillRecord]) -> bool:
    return metrics.fill_rate_p90 < 0.70


def _r2_as_ratio(metrics: FillMetrics, _records: list[FillRecord]) -> bool:
    return metrics.adverse_selection_ratio > 0.50


def _r4_direction_ic(_metrics: FillMetrics, _records: list[FillRecord]) -> bool:
    return False  # G1再検証結果が必要


def _r5_cumulative_loss(_metrics: FillMetrics, records: list[FillRecord]) -> bool:
    return _check_cumulative_loss(records)


STOP_RULES: dict[str, StopRule] = {
    "R1_fill_rate": StopRule(
        desc="fill_rate < 70% → 中止",
        min_n=200,
        check=_r1_fill_rate,
        judgment="STOP",
    ),
    "R2_as_ratio": StopRule(
        desc="AS_ratio > spread/2 → 中止",
        min_n=500,
        check=_r2_as_ratio,
        judgment="STOP",
    ),
    "R4_direction_ic": StopRule(
        desc="方向IC > 0.04 → 続行",
        min_n=0,
        check=_r4_direction_ic,
        judgment="CONTINUE",
    ),
    "R5_cumulative_loss": StopRule(
        desc="累積実損 > 10,000 JPY → 一時停止",
        min_n=1,
        check=_r5_cumulative_loss,
        judgment="PAUSE",
    ),
}


def _check_cumulative_loss(records: list[FillRecord]) -> bool:
    """累積実損チェック (簡易: pnl_bps × quantity × price の概算)."""
    total_loss_jpy = 0.0
    for r in records:
        pnl_jpy = compute_record_pnl_jpy(r)
        if pnl_jpy is not None and pnl_jpy < 0:
            total_loss_jpy += pnl_jpy
    return abs(total_loss_jpy) > 10_000


def evaluate_stop_rules(
    metrics: FillMetrics,
    records: list[FillRecord],
) -> list[StopRuleResult]:
    """§3.9 全ルールを評価して結果を返す."""
    results: list[StopRuleResult] = []
    n = metrics.total_orders
    for rule_id, rule in STOP_RULES.items():
        if n < rule.min_n:
            status = "SKIP"
            reason = f"n={n} < min_n={rule.min_n}"
            triggered = False
        else:
            triggered = rule.check(metrics, records)
            status = rule.judgment if triggered else "OK"
            reason = rule.desc
        results.append({
            "rule": rule_id,
            "status": status,
            "triggered": triggered,
            "reason": reason,
            "min_n": rule.min_n,
        })
    return results


# ======================================================================
# 表示
# ======================================================================

def _fmt_pct(v: float) -> str:
    return f"{v * 100:.1f}%"


def _fmt_bps(v: float) -> str:
    return f"{v:+.3f} bps"


def _status_icon(status: str) -> str:
    icons = {"OK": "✅", "STOP": "🛑", "PAUSE": "⏸️", "SKIP": "⏭️", "CONTINUE": "➡️"}
    return icons.get(status, "❓")


def print_report(
    metrics: FillMetrics,
    records: list[FillRecord],
    gate_result: GateResult,
    stop_results: list[StopRuleResult],
    *,
    clean_count: int | None = None,
    quarantine_count: int | None = None,
    full_result: GateResult | None = None,
) -> None:
    """コンソールにモニタリングレポートを出力."""
    n = metrics.total_orders
    filled = metrics.filled_orders
    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")

    # 時間範囲
    if records:
        t_first = datetime.fromtimestamp(records[0].timestamp, tz=timezone.utc)
        t_last = datetime.fromtimestamp(records[-1].timestamp, tz=timezone.utc)
        elapsed_h = (records[-1].timestamp - records[0].timestamp) / 3600
        time_range = f"{t_first.strftime('%m/%d %H:%M')} — {t_last.strftime('%m/%d %H:%M')} ({elapsed_h:.1f}h)"
    else:
        time_range = "N/A"

    print("=" * 70)
    print(f"  fill_test MONITOR — {now}")
    print(f"  n={n} (filled={filled}, cancelled={metrics.cancelled_orders})")
    print(f"  期間: {time_range}")
    print("=" * 70)

    # 130# G1.1-quick Gate メトリクス (116# 二段階ゲート)
    print("\n📊 G1.1-quick (72h Kill Gate):")
    print("-" * 50)
    checks = gate_result.get("checks", {})
    for check_id, detail in checks.items():
        icon = "✅" if detail["pass"] else "❌"
        val = detail.get("value", 0.0)
        # K4 は複合条件 (threshold_p + threshold_mean)
        if "threshold" in detail:
            thr = detail["threshold"]
            if "pnl" in check_id.lower():
                val_str = _fmt_bps(val)
                thr_str = _fmt_bps(thr)
            elif "ratio" in check_id.lower() or "rate" in check_id.lower():
                val_str = _fmt_pct(val)
                thr_str = _fmt_pct(thr)
            else:
                val_str = f"{val:.1f}"
                thr_str = f"{thr:.1f}"
            print(f"  {icon} {check_id}: {val_str} (閾値: {thr_str})")
        elif "threshold_p" in detail:  # K4 PnL kill 複合条件
            val_str = _fmt_bps(val)
            p_val = detail.get("pvalue", 1.0)
            print(f"  {icon} {check_id}: mean={val_str}, p={p_val:.4f}")
        else:
            print(f"  {icon} {check_id}: {val:.3f}")

    gate_verdict = gate_result.get("gate_result", "???")
    gate_icon = {"PASS": "✅", "WATCH": "⚠️", "FAIL": "❌"}.get(gate_verdict, "❓")
    print(f"\n  {gate_icon} G1.1-quick: {gate_verdict}")

    # 130# G1.2-full 判定表示
    if full_result is not None:
        full_verdict = full_result.get("gate_result", "???")
        full_icon = {"PASS": "✅", "WATCH": "⚠️", "FAIL": "❌"}.get(full_verdict, "❓")
        print(f"\n📊 G1.2-full (168h Qualification):")
        print("-" * 50)
        full_checks = full_result.get("checks", {})
        for check_id, detail in full_checks.items():
            icon = "✅" if detail.get("pass", False) else "❌"
            val = detail.get("value", 0.0)
            thr = detail.get("threshold", 0.0)
            if "pnl" in check_id.lower():
                val_str = _fmt_bps(val)
                thr_str = _fmt_bps(thr)
            elif "ratio" in check_id.lower() or "rate" in check_id.lower():
                val_str = _fmt_pct(val)
                thr_str = _fmt_pct(thr)
            else:
                val_str = f"{val:.1f}"
                thr_str = f"{thr:.1f}"
            print(f"  {icon} {check_id}: {val_str} (閾値: {thr_str})")
        print(f"\n  {full_icon} G1.2-full: {full_verdict}")

    # §3.9 中止ルール
    print("\n🚦 §3.9 継続中止ルール:")
    print("-" * 50)
    any_triggered = False
    for r in stop_results:
        icon = _status_icon(r["status"])
        print(f"  {icon} {r['rule']}: {r['status']} — {r['reason']}")
        if r["triggered"]:
            any_triggered = True

    if any_triggered:
        print("\n  ⚠️  中止/一時停止条件に該当しています!")
    else:
        skipped = sum(1 for r in stop_results if r["status"] == "SKIP")
        if skipped > 0:
            print(f"\n  ℹ️  {skipped}件のルールはサンプル不足のためスキップ")
        print("  ✅ 現時点で中止条件には該当していません")

    # 補足統計
    print("\n📈 補足統計:")
    print("-" * 50)

    # 084# AS_raw vs AS_deadzone 並行表示
    if metrics.as_coverage > 0:
        print(f"  AS(deadzone): {_fmt_pct(metrics.adverse_selection_ratio)} (n={metrics.as_coverage})")
        if metrics.as_raw_coverage > 0:
            print(f"  AS(raw):      {_fmt_pct(metrics.adverse_selection_ratio_raw)} (n={metrics.as_raw_coverage})")
            gap_pp = (metrics.adverse_selection_ratio_raw - metrics.adverse_selection_ratio) * 100
            print(f"  deadzone差分: {gap_pp:+.1f}pp マスキング")

    if records:
        filled_recs = [r for r in records if r.filled]
        pnls = [r.post_fill_30s_pnl for r in filled_recs if r.post_fill_30s_pnl is not None]
        waits = [r.queue_wait_sec for r in filled_recs]

        if pnls:
            import numpy as np
            from ztb.utils.robust_stats import RobustStats
            pnls_arr = np.array(pnls, dtype=np.float64)
            pnls_robust = RobustStats.clip_outliers_mad(pnls_arr)
            print(f"  PnL mean:   {_fmt_bps(float(np.mean(pnls_arr)))}")
            print(f"  PnL median: {_fmt_bps(float(np.median(pnls_arr)))}")
            print(f"  PnL std:    {float(np.std(pnls_arr)):.3f} bps")
            # 632# RobustStats: MAD-clip 後のロバスト統計 (外れ値耐性)
            print(f"  PnL robust: mean={float(np.mean(pnls_robust)):.3f} std={float(np.std(pnls_robust)):.3f} bps")
            pos_ratio = sum(1 for p in pnls if p > 0) / len(pnls)
            print(f"  PnL 正率:   {_fmt_pct(pos_ratio)}")

        if waits:
            print(f"  Wait mean:  {float(np.mean(waits)):.1f}s")
            print(f"  Wait median:{float(np.median(waits)):.1f}s")

        # 累積損益概算
        total_pnl_jpy = 0.0
        for rec in filled_recs:
            pnl_jpy = compute_record_pnl_jpy(rec)
            if pnl_jpy is not None:
                total_pnl_jpy += pnl_jpy
        print(f"  累積PnL概算: {total_pnl_jpy:+.1f} JPY")

        # n到達までの推定時間
        if elapsed_h > 0 and n > 0:
            rate_per_h = n / elapsed_h
            for target in [200, 500]:
                if n < target:
                    remaining = target - n
                    eta_h = remaining / rate_per_h
                    print(f"  n={target}到達推定: {eta_h:.0f}h ({eta_h/24:.1f}日)")

    # 051# データ品質 (clean_rate)
    if clean_count is not None and quarantine_count is not None:
        total_all = clean_count + quarantine_count
        if total_all > 0:
            clean_rate = clean_count / total_all
            print(f"\n🧹 データ品質:")
            print("-" * 50)
            print(f"  clean:      {clean_count} ({clean_rate:.1%})")
            print(f"  quarantine: {quarantine_count} ({1 - clean_rate:.1%})")
            print(f"  合計:       {total_all}")

    # 051# P2-2 → 055# Fix: Round-trip 評価 (双方向ペアリング)
    if records:
        filled_recs_rt = [r for r in records if r.filled]
        if len(filled_recs_rt) >= 2:
            rt_metrics, _trips = compute_round_trip_metrics(records)
            if rt_metrics.total_pairs > 0:
                print(f"\n🔄 Round-trip 評価 (双方向 FIFO ペアリング):")
                print("-" * 50)
                print(f"  ペア数:     {rt_metrics.total_pairs}")
                print(f"  PnL mean:   {_fmt_bps(rt_metrics.pnl_mean_bps)}")
                print(f"  PnL median: {_fmt_bps(rt_metrics.pnl_median_bps)}")
                print(f"  PnL std:    {rt_metrics.pnl_std_bps:.3f} bps")
                print(f"  累積PnL:    {rt_metrics.pnl_total_jpy:+.1f} JPY")
                print(f"  勝率:       {_fmt_pct(rt_metrics.win_rate)}")
                print(f"  保持中央値: {rt_metrics.hold_sec_median:.0f}s")
                if rt_metrics.unpaired_buys > 0:
                    print(f"  未ペア buy:  {rt_metrics.unpaired_buys}")
                if rt_metrics.unpaired_sells > 0:
                    print(f"  未ペア sell: {rt_metrics.unpaired_sells}")
                if rt_metrics.net_inventory != 0:
                    print(f"  純在庫:      {rt_metrics.net_inventory:+d}")

    # 051# P2-4: レジーム別メトリクス
    if records:
        regime_list = compute_regime_metrics(records)
        # unknown のみの場合はスキップ
        if regime_list and not (len(regime_list) == 1 and regime_list[0].regime == "unknown"):
            print(f"\n🌊 レジーム別メトリクス:")
            print("-" * 50)
            print(f"  {'regime':<12} {'n':>5} {'fill%':>7} {'PnL':>10} {'AS%':>7} {'wait':>6}")
            for rm in regime_list:
                print(
                    f"  {rm.regime:<12} {rm.count:>5} "
                    f"{rm.fill_rate:>6.1%} "
                    f"{_fmt_bps(rm.pnl_mean_bps):>10} "
                    f"{rm.as_ratio:>6.1%} "
                    f"{rm.queue_wait_median_sec:>5.0f}s"
                )

    # 051# UTC 時間帯別分析
    if records:
        hourly_list = compute_hourly_metrics(records)
        if len(hourly_list) >= 3:
            print(f"\n🕐 UTC 時間帯別分析:")
            print("-" * 50)
            print(f"  {'UTC':>4} {'n':>5} {'fill':>5} {'PnL':>10} {'AS%':>7}")
            for hm in hourly_list:
                flag = " ⚠" if hm.as_ratio > 0.45 else ""
                print(
                    f"  {hm.utc_hour:>4} {hm.count:>5} {hm.filled:>5} "
                    f"{_fmt_bps(hm.pnl_mean_bps):>10} "
                    f"{hm.as_ratio:>6.1%}{flag}"
                )
    print("=" * 70)


# ======================================================================
# JSON 出力
# ======================================================================

def save_snapshot(
    metrics: FillMetrics,
    gate_result: GateResult,
    stop_results: list[StopRuleResult],
    output_dir: Path,
) -> Path:
    """モニタリング結果をJSONファイルとして保存."""
    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    snapshot = {
        "timestamp": ts,
        "metrics": metrics.to_dict(),
        "gate_result": gate_result,
        "stop_rules": stop_results,
    }
    # 092# round-trip 指標をスナップショットに追加
    checks = gate_result.get("checks", {})
    if "E6_round_trip_pnl" in checks:
        e6 = checks["E6_round_trip_pnl"]
        snapshot["round_trip"] = {
            "mean_bps": e6.get("value"),
            "median_bps": e6.get("median"),
            "total_jpy": e6.get("total_jpy"),
            "pairs": e6.get("pairs"),
        }
    if "E7_net_inventory" in checks:
        e7 = checks["E7_net_inventory"]
        snapshot["inventory"] = {
            "net": e7.get("net_inventory"),
            "unpaired_buys": e7.get("unpaired_buys"),
            "unpaired_sells": e7.get("unpaired_sells"),
        }
    out_path = output_dir / f"monitor_snapshot_{ts}.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    write_json(out_path, snapshot, ensure_ascii=False, indent=2)
    return out_path


# ======================================================================
# Main
# ======================================================================

def run_monitor(results_dir: Path, save_json: bool = True) -> dict[str, object]:
    """メインモニタリングロジック. watch モードからも呼び出し可能."""
    clean_records, quarantine_records = partition_clean_records(
        iter_fill_records_glob(results_dir),
    )
    if not clean_records and not quarantine_records:
        print(f"⚠️  {results_dir} にレコードが見つかりません")
        return {}

    # 051# clean/quarantine 分離
    clean_count = len(clean_records)
    quarantine_count = len(quarantine_records)

    # G1.1/G1.2 閾値読み込み (watch ループでキャッシュ)
    quick_thresholds, full_thresholds = _gate_thresholds()

    # clean レコードのみでメトリクス算出
    metrics = compute_fill_metrics(clean_records)

    # 130# 判定統一: g1_1_quick + g1_2_full を正とする (gate_judgment.py と同一系統)
    gate_result = g1_1_quick_judgment(metrics, quick_thresholds)
    full_result = g1_2_full_judgment(metrics, full_thresholds)
    stop_results = evaluate_stop_rules(metrics, clean_records)

    print_report(
        metrics, clean_records, gate_result, stop_results,
        clean_count=clean_count,
        quarantine_count=quarantine_count,
        full_result=full_result,
    )

    if save_json:
        snapshot_path = save_snapshot(metrics, gate_result, stop_results, results_dir)
        print(f"\n💾 スナップショット保存: {snapshot_path}")

    return {
        "metrics": metrics,
        "gate_result": gate_result,
        "full_result": full_result,
        "stop_results": stop_results,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="fill_test モニタリング")
    parser.add_argument(
        "--results-dir",
        type=str,
        default=str(_PROJECT_ROOT / "results" / "v460" / "fill_test"),
        help="fill_test 結果ディレクトリ",
    )
    parser.add_argument("--watch", action="store_true", help="定期実行モード")
    parser.add_argument(
        "--interval",
        type=int,
        default=300,
        help="watch モードの間隔 (秒)",
    )
    parser.add_argument("--no-save", action="store_true", help="JSON保存なし")
    args = parser.parse_args()

    results_dir = Path(args.results_dir)

    if args.watch:
        print(f"🔄 watch モード ({args.interval}s 間隔)")
        try:
            while True:
                run_monitor(results_dir, save_json=not args.no_save)
                print(f"\n次回: {args.interval}s 後...")
                time.sleep(args.interval)
        except KeyboardInterrupt:
            print("\n⏹ 監視終了")
    else:
        run_monitor(results_dir, save_json=not args.no_save)


if __name__ == "__main__":
    main()
