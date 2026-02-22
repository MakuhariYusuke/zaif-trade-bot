"""Z2 Oracle テスト — maker 0% 理論上限の測定.

完全予測 (Oracle) エージェントが maker 0% 手数料で達成する PnL を算出。
fill_records の実データ (post_fill_30s_pnl, post_fill_120s_pnl) を使用。

Kill Switch 基準 (121# §6.3, 122# R7):
  Oracle PnL30s > +1.0bps  → ✅ ph3 SAC 訓練を進行
  0 < Oracle PnL30s ≤ +1.0 → ⚠️ 120s 保持を必須で含める
  Oracle PnL30s ≤ 0        → ❌ maker-only 30s に理論限界
  Oracle PnL120s ≤ 0       → ❌❌ 致命的 — 戦略再評価

Usage:
    .venv\\Scripts\\python.exe scripts/v460/analysis/oracle_test.py
    .venv\\Scripts\\python.exe scripts/v460/analysis/oracle_test.py --results-dir results/v460/fill_test
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import TypedDict

import numpy as np
import pandas as pd
from ztb.io.jsonl import append_jsonl

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent.parent))

from scripts.v460.ml.data_loader import load_fill_records
from scripts.v460.ml.feature_enricher import enrich_fill_records

logger = logging.getLogger(__name__)


class SideOracleStats(TypedDict):
    """Side別統計."""

    n: int
    mean_bps: float
    std_bps: float
    profitable_rate: float
    oracle_skip_mean_bps: float


class HorizonOracleResult(TypedDict, total=False):
    """1 horizon の Oracle 結果."""

    status: str
    n: int
    baseline_mean_bps: float
    baseline_std_bps: float
    oracle_skip_mean_bps: float
    oracle_skip_rate: float
    oracle_skip_improvement_bps: float
    oracle_flip_mean_bps: float
    profitable_rate: float
    side_analysis: dict[str, SideOracleStats]


class KillSwitchResult(TypedDict, total=False):
    """Kill Switch 判定."""

    pnl30: str
    pnl30_action: str
    oracle_pnl30_bps: float
    pnl120: str
    pnl120_action: str
    oracle_pnl120_bps: float


class OracleRunResult(TypedDict, total=False):
    """Oracle テスト実行結果."""

    timestamp: str
    results_dir: str
    status: str
    reason: str
    total_records: int
    filled_records: int
    oracle: dict[str, HorizonOracleResult]
    kill_switch: KillSwitchResult


def run_oracle_test(
    results_dir: str = "results/v460/fill_test",
    trades_fallback_recent_days: int = 1,
) -> OracleRunResult:
    """Oracle テストを実行.

    Returns:
        結果 dict: oracle_pnl30, oracle_pnl120, baseline, kill_switch 判定など.
    """
    results_path = Path(results_dir)
    result: OracleRunResult = {
        "timestamp": datetime.now().isoformat(),
        "results_dir": str(results_path),
    }

    # 1. データロード & エンリッチ (全 run)
    try:
        records = load_fill_records(results_path, exclude_missing_run_id=False)
    except FileNotFoundError:
        return {**result, "status": "error", "reason": "fill_records not found"}

    enriched = enrich_fill_records(
        records,
        trades_fallback_recent_days=trades_fallback_recent_days,
    )

    # filled のみ
    filled_mask = enriched["filled"].astype(bool)
    filled = enriched.loc[filled_mask].copy()
    result["total_records"] = int(len(enriched))
    result["filled_records"] = int(len(filled))

    # 2. PnL カラム取得
    pnl30_col = "post_fill_30s_pnl"
    pnl120_col = "post_fill_120s_pnl"

    horizons: dict[str, str] = {}
    if pnl30_col in filled.columns:
        horizons["pnl30"] = pnl30_col
    if pnl120_col in filled.columns:
        horizons["pnl120"] = pnl120_col

    if not horizons:
        return {**result, "status": "error", "reason": "PnL columns not found"}

    # 3. 各 horizon で Oracle 分析
    oracle_results: dict[str, HorizonOracleResult] = {}

    for label, col in horizons.items():
        pnl_values = filled[col].astype(float).dropna()
        n = len(pnl_values)

        if n == 0:
            oracle_results[label] = {"status": "no_data", "n": 0}
            continue

        pnl_arr = pnl_values.values

        # Baseline: 全トレードの平均 PnL (現実)
        baseline_mean = float(np.mean(pnl_arr))
        baseline_std = float(np.std(pnl_arr))

        # Oracle Skip: 負の PnL を完全にスキップ (perfect skip gate)
        profitable = pnl_arr[pnl_arr > 0]
        unprofitable = pnl_arr[pnl_arr <= 0]
        oracle_skip_mean = float(np.mean(profitable)) if len(profitable) > 0 else 0.0
        oracle_skip_rate = float(len(unprofitable) / n)

        # Oracle Side Flip: 常に正しい side を選択 (|pnl| の平均)
        oracle_flip_mean = float(np.mean(np.abs(pnl_arr)))

        # Oracle Skip の改善量 (bps)
        skip_improvement = oracle_skip_mean - baseline_mean

        # Side 別分析
        side_analysis: dict[str, SideOracleStats] = {}
        if "side" in filled.columns:
            for side in ["buy", "sell"]:
                side_mask = filled["side"] == side
                side_pnl = filled.loc[side_mask, col].astype(float).dropna()
                if len(side_pnl) > 0:
                    sp = side_pnl.values
                    side_profitable = sp[sp > 0]
                    side_analysis[side] = {
                        "n": int(len(sp)),
                        "mean_bps": float(np.mean(sp)),
                        "std_bps": float(np.std(sp)),
                        "profitable_rate": float(len(side_profitable) / len(sp)),
                        "oracle_skip_mean_bps": float(np.mean(side_profitable)) if len(side_profitable) > 0 else 0.0,
                    }

        oracle_results[label] = {
            "n": int(n),
            "baseline_mean_bps": round(baseline_mean, 4),
            "baseline_std_bps": round(baseline_std, 4),
            "oracle_skip_mean_bps": round(oracle_skip_mean, 4),
            "oracle_skip_rate": round(oracle_skip_rate, 4),
            "oracle_skip_improvement_bps": round(skip_improvement, 4),
            "oracle_flip_mean_bps": round(oracle_flip_mean, 4),
            "profitable_rate": round(1.0 - oracle_skip_rate, 4),
            "side_analysis": side_analysis,
        }

    result["oracle"] = oracle_results

    # 4. Kill Switch 判定 (121# §6.3, 122# R7)
    kill_switch: KillSwitchResult = {}

    if "pnl30" in oracle_results and oracle_results["pnl30"].get("n", 0) > 0:
        oracle_30 = oracle_results["pnl30"]["oracle_skip_mean_bps"]
        if oracle_30 > 1.0:
            kill_switch["pnl30"] = "PASS"
            kill_switch["pnl30_action"] = "✅ 天井は十分高い → ph3 SAC 訓練を進行"
        elif oracle_30 > 0:
            kill_switch["pnl30"] = "WARN"
            kill_switch["pnl30_action"] = "⚠️ 天井が低い → 120s 保持を必須で含める"
        else:
            kill_switch["pnl30"] = "FAIL"
            kill_switch["pnl30_action"] = "❌ maker-only 30s に理論限界 → ピボット検討"
        kill_switch["oracle_pnl30_bps"] = oracle_30

    if "pnl120" in oracle_results and oracle_results["pnl120"].get("n", 0) > 0:
        oracle_120 = oracle_results["pnl120"]["oracle_skip_mean_bps"]
        if oracle_120 > 1.0:
            kill_switch["pnl120"] = "PASS"
            kill_switch["pnl120_action"] = "✅ 天井は十分高い"
        elif oracle_120 > 0:
            kill_switch["pnl120"] = "WARN"
            kill_switch["pnl120_action"] = "⚠️ 天井が低い"
        else:
            kill_switch["pnl120"] = "FAIL"
            kill_switch["pnl120_action"] = "❌❌ 致命的 — 戦略全体の再評価"
        kill_switch["oracle_pnl120_bps"] = oracle_120

    result["kill_switch"] = kill_switch
    result["status"] = "completed"
    return result


def main() -> None:
    """CLI エントリポイント."""
    parser = argparse.ArgumentParser(description="Z2 Oracle テスト (理論上限)")
    parser.add_argument(
        "--results-dir", type=str, default="results/v460/fill_test",
        help="fill_records のディレクトリ",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    )

    result = run_oracle_test(results_dir=args.results_dir)

    # 結果表示
    print("\n" + "=" * 70)
    print("Z2 Oracle テスト結果 (121# §6.3, 122# R7)")
    print("=" * 70)

    if result.get("status") != "completed":
        print(f"ERROR: {result.get('reason', 'unknown')}")
        return

    print(f"  Total records: {result['total_records']}")
    print(f"  Filled records: {result['filled_records']}")

    for label, data in result["oracle"].items():
        print(f"\n  --- {label.upper()} ---")
        if data.get("status") == "no_data":
            print(f"    No data available")
            continue
        print(f"    Samples:             {data['n']}")
        print(f"    Baseline mean:       {data['baseline_mean_bps']:+.4f} bps")
        print(f"    Oracle Skip mean:    {data['oracle_skip_mean_bps']:+.4f} bps")
        print(f"    Oracle Flip mean:    {data['oracle_flip_mean_bps']:+.4f} bps")
        print(f"    Skip improvement:    {data['oracle_skip_improvement_bps']:+.4f} bps")
        print(f"    Profitable rate:     {data['profitable_rate']:.1%}")

        if data.get("side_analysis"):
            for side, sa in data["side_analysis"].items():
                print(f"    [{side:4s}] n={sa['n']:4d}, mean={sa['mean_bps']:+.4f}, "
                      f"profitable={sa['profitable_rate']:.1%}, "
                      f"oracle_skip={sa['oracle_skip_mean_bps']:+.4f}")

    print(f"\n  === Kill Switch ===")
    ks = result.get("kill_switch", {})
    for key in ["pnl30", "pnl120"]:
        status = ks.get(key, "N/A")
        oracle_val = ks.get(f"oracle_{key}_bps", "?")
        action = ks.get(f"{key}_action", "")
        print(f"    {key}: {status} (Oracle={oracle_val} bps)")
        if action:
            print(f"      → {action}")

    print("=" * 70)

    # JSONL ログ出力
    log_path = Path("logs") / "oracle_test.jsonl"
    append_jsonl(log_path, [result], ensure_ascii=False, default=str)
    print(f"\n  Result logged to {log_path}")

    # 全結果 JSON
    print(f"\n{json.dumps(result, indent=2, default=str)}")


if __name__ == "__main__":
    main()
