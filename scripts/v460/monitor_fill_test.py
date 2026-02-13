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
import json
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))

from ztb.metrics.fill_quality import (
    FillMetrics,
    FillRecord,
    compute_fill_metrics,
    g1_1_judgment,
    load_fill_records_glob,
)


# ======================================================================
# §3.9 継続中止ルール
# ======================================================================

StopCheckFn = Callable[[FillMetrics, list[FillRecord]], bool]


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
        if r.filled and r.post_fill_30s_pnl is not None and r.fill_price is not None:
            # bps → JPY: pnl_bps * 1e-4 * fill_price * quantity
            pnl_jpy = r.post_fill_30s_pnl * 1e-4 * r.fill_price * r.order_quantity
            if pnl_jpy < 0:
                total_loss_jpy += pnl_jpy
    return abs(total_loss_jpy) > 10_000


def evaluate_stop_rules(
    metrics: FillMetrics,
    records: list[FillRecord],
) -> list[dict[str, Any]]:
    """§3.9 全ルールを評価して結果を返す."""
    results: list[dict[str, Any]] = []
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
    gate_result: dict[str, Any],
    stop_results: list[dict[str, Any]],
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

    # G1.1 Gate メトリクス
    print("\n📊 G1.1 Gate メトリクス:")
    print("-" * 50)
    checks = gate_result.get("checks", {})
    for check_id, detail in checks.items():
        icon = "✅" if detail["pass"] else "❌"
        val = detail["value"]
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

    gate_verdict = gate_result.get("gate_result", "???")
    gate_icon = "✅" if gate_verdict == "PASS" else "❌"
    print(f"\n  {gate_icon} G1.1 総合: {gate_verdict}")

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
    if records:
        filled_recs = [r for r in records if r.filled]
        pnls = [r.post_fill_30s_pnl for r in filled_recs if r.post_fill_30s_pnl is not None]
        waits = [r.queue_wait_sec for r in filled_recs]

        if pnls:
            import numpy as np
            print(f"  PnL mean:   {_fmt_bps(float(np.mean(pnls)))}")
            print(f"  PnL median: {_fmt_bps(float(np.median(pnls)))}")
            print(f"  PnL std:    {float(np.std(pnls)):.3f} bps")
            pos_ratio = sum(1 for p in pnls if p > 0) / len(pnls)
            print(f"  PnL 正率:   {_fmt_pct(pos_ratio)}")

        if waits:
            import numpy as np
            print(f"  Wait mean:  {float(np.mean(waits)):.1f}s")
            print(f"  Wait median:{float(np.median(waits)):.1f}s")

        # 累積損益概算
        total_pnl_jpy = 0.0
        for rec in filled_recs:
            if rec.post_fill_30s_pnl is not None and rec.fill_price is not None:
                pnl_jpy = rec.post_fill_30s_pnl * 1e-4 * rec.fill_price * rec.order_quantity
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

    print("=" * 70)


# ======================================================================
# JSON 出力
# ======================================================================

def save_snapshot(
    metrics: FillMetrics,
    gate_result: dict[str, Any],
    stop_results: list[dict[str, Any]],
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
    out_path = output_dir / f"monitor_snapshot_{ts}.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(snapshot, f, ensure_ascii=False, indent=2)
    return out_path


# ======================================================================
# Main
# ======================================================================

def run_monitor(results_dir: Path, save_json: bool = True) -> dict[str, Any]:
    """メインモニタリングロジック. watch モードからも呼び出し可能."""
    records = load_fill_records_glob(results_dir)
    if not records:
        print(f"⚠️  {results_dir} にレコードが見つかりません")
        return {}

    # G1.1 閾値読み込み
    thresholds_path = _PROJECT_ROOT / "configs" / "v460" / "gate_thresholds.yaml"
    if thresholds_path.exists():
        import yaml  # type: ignore[import-untyped]
        with open(thresholds_path, "r") as f:
            cfg = yaml.safe_load(f)
        thresholds = cfg.get("g1_1_exec", {})
    else:
        thresholds = {}

    metrics = compute_fill_metrics(records)
    gate_result = g1_1_judgment(metrics, thresholds)
    stop_results = evaluate_stop_rules(metrics, records)

    print_report(metrics, records, gate_result, stop_results)

    if save_json:
        snapshot_path = save_snapshot(metrics, gate_result, stop_results, results_dir)
        print(f"\n💾 スナップショット保存: {snapshot_path}")

    return {
        "metrics": metrics,
        "gate_result": gate_result,
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
