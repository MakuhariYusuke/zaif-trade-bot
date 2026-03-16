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


class ASCostAnalysis(TypedDict, total=False):
    """AS (Adverse Selection) コスト分析 — 158# P0-4 要件."""

    n_as: int
    n_non_as: int
    as_ratio: float
    as_avg_pnl30_bps: float
    non_as_avg_pnl30_bps: float
    as_cost_bps: float  # as_ratio × |as_avg_pnl30|
    oracle_net_of_as_bps: float  # oracle_flip - as_cost


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
    as_cost: ASCostAnalysis
    kill_switch: KillSwitchResult


def _to_finite_float_array(series: pd.Series) -> np.ndarray:
    """Series を有限 float 配列へ正規化する."""
    numeric = pd.to_numeric(series, errors="coerce").to_numpy(dtype=float, copy=False)
    return numeric[np.isfinite(numeric)]


def _summarize_pnl_array(pnl_arr: np.ndarray) -> tuple[float, float, float, float, float]:
    """PnL 配列から主要 Oracle 指標を算出する."""
    n = len(pnl_arr)
    if n == 0:
        return 0.0, 0.0, 0.0, 0.0, 0.0

    positive_mask = pnl_arr > 0
    positive_count = int(np.count_nonzero(positive_mask))
    positive_sum = float(pnl_arr[positive_mask].sum()) if positive_count > 0 else 0.0

    baseline_mean = float(pnl_arr.mean())
    baseline_std = float(pnl_arr.std())
    oracle_skip_mean = positive_sum / positive_count if positive_count > 0 else 0.0
    oracle_skip_rate = float((n - positive_count) / n)
    oracle_flip_mean = float(np.abs(pnl_arr).mean())
    return baseline_mean, baseline_std, oracle_skip_mean, oracle_skip_rate, oracle_flip_mean


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
        pnl_arr = _to_finite_float_array(filled[col])
        n = len(pnl_arr)

        if n == 0:
            oracle_results[label] = {"status": "no_data", "n": 0}
            continue

        baseline_mean, baseline_std, oracle_skip_mean, oracle_skip_rate, oracle_flip_mean = (
            _summarize_pnl_array(pnl_arr)
        )

        # Oracle Skip の改善量 (bps)
        skip_improvement = oracle_skip_mean - baseline_mean

        # Side 別分析
        side_analysis: dict[str, SideOracleStats] = {}
        if "side" in filled.columns:
            for side in ["buy", "sell"]:
                side_mask = filled["side"] == side
                side_pnl = _to_finite_float_array(filled.loc[side_mask, col])
                if len(side_pnl) > 0:
                    side_mean, side_std, side_skip_mean, side_skip_rate, _side_flip_mean_unused = (
                        _summarize_pnl_array(side_pnl)
                    )
                    side_analysis[side] = {
                        "n": int(len(side_pnl)),
                        "mean_bps": side_mean,
                        "std_bps": side_std,
                        "profitable_rate": round(1.0 - side_skip_rate, 4),
                        "oracle_skip_mean_bps": side_skip_mean,
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

    # 3.5. AS コスト分析 (158# P0-4: AS_ratio × avg_AS_loss)
    as_cost_result: ASCostAnalysis = {}
    as_col = "adverse_selected"
    pnl30_col_name = "post_fill_30s_pnl"
    if as_col in filled.columns and pnl30_col_name in filled.columns:
        pnl30_series = pd.to_numeric(filled[pnl30_col_name], errors="coerce")
        pnl30_array = pnl30_series.to_numpy(dtype=float, copy=False)
        valid_mask = np.isfinite(pnl30_array)
        as_mask = filled[as_col].fillna(False).astype(bool).to_numpy(dtype=bool, copy=False)

        valid_pnl30 = pnl30_array[valid_mask]
        valid_as_mask = as_mask[valid_mask]
        as_values = valid_pnl30[valid_as_mask]
        non_as_values = valid_pnl30[~valid_as_mask]

        n_as = int(len(as_values))
        n_non_as = int(len(non_as_values))
        n_total_valid = n_as + n_non_as

        if n_total_valid > 0 and n_as > 0:
            as_ratio = n_as / n_total_valid
            as_avg = float(as_values.mean())
            non_as_avg = float(non_as_values.mean()) if n_non_as > 0 else 0.0
            as_cost = as_ratio * abs(as_avg)  # AS コスト (bps)

            # Oracle Flip PnL30 から AS cost を差し引いた net
            oracle_flip_30 = oracle_results.get("pnl30", {}).get("oracle_flip_mean_bps", 0.0)
            oracle_net = oracle_flip_30 - as_cost

            as_cost_result = {
                "n_as": n_as,
                "n_non_as": n_non_as,
                "as_ratio": round(as_ratio, 4),
                "as_avg_pnl30_bps": round(as_avg, 4),
                "non_as_avg_pnl30_bps": round(non_as_avg, 4),
                "as_cost_bps": round(as_cost, 4),
                "oracle_net_of_as_bps": round(oracle_net, 4),
            }

    result["as_cost"] = as_cost_result

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

    print(f"\n  === AS Cost Analysis (158# P0-4) ===")
    as_cost = result.get("as_cost", {})
    if as_cost.get("n_as"):
        print(f"    AS records:   {as_cost['n_as']} ({as_cost['as_ratio']:.1%})")
        print(f"    Non-AS:       {as_cost['n_non_as']} ({1-as_cost['as_ratio']:.1%})")
        print(f"    AS avg PnL30: {as_cost['as_avg_pnl30_bps']:+.4f} bps")
        print(f"    Non-AS PnL30: {as_cost['non_as_avg_pnl30_bps']:+.4f} bps")
        print(f"    AS cost:      {as_cost['as_cost_bps']:+.4f} bps  (AS_ratio x |avg_AS_loss|)")
        print(f"    Oracle net:   {as_cost['oracle_net_of_as_bps']:+.4f} bps  (Oracle Flip - AS cost)")
        verdict = "PASS" if as_cost["oracle_net_of_as_bps"] > 0 else "FAIL"
        print(f"    158# P0-4 verdict: {verdict}")
    else:
        print("    No AS data available")

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
