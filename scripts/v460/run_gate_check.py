#!/usr/bin/env python3
"""
v460 Gate Check Runner — G0/G1/G1.1/G2/G3/G4 閾値照合ユーティリティ.

001# §4.1 / 000# §3 準拠.
009# P2-3: G1.1-exec 追加.
031# F7: G2/G3/G4 スタブ追加.

Usage:
  python scripts/v460/run_gate_check.py --gate G0 --data-path data/v460/features/btc_jpy_1m_v460_features.parquet
  python scripts/v460/run_gate_check.py --gate G1 --results-path results/v460/g1_results.json
  python scripts/v460/run_gate_check.py --gate G1.1 --results-dir results/v460/fill_test
  python scripts/v460/run_gate_check.py --gate G2 --results-path results/v460/g2_train_results.json
  python scripts/v460/run_gate_check.py --gate G3 --results-path results/v460/g3_pnl_results.json
  python scripts/v460/run_gate_check.py --gate G4 --results-path results/v460/g4_live_results.json
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

# Project root
_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))

from scripts.v460.lib.config_loader import load_gate_thresholds
from scripts.v460.lib.data_loader import check_nan_ratio, compute_data_hash, load_parquet
from scripts.v460.lib.manifest import ManifestWriter

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


# ======================================================================
# G0-data
# ======================================================================

def run_g0(
    data_path: str,
    expected_hash: str | None = None,
    thresholds: dict | None = None,
) -> dict:
    """G0-data チェック.

    000# §3.1:
      - データハッシュ一致
      - 特徴量カラム数 ≥ 4
      - NaN 比率 ≤ 1%
      - manifest.jsonl 記録 (存在チェックのみ)
    """
    if thresholds is None:
        thresholds = load_gate_thresholds().get("g0_data", {})

    min_cols = thresholds.get("min_feature_columns", 4)
    max_nan = thresholds.get("max_nan_ratio", 0.01)

    results: dict = {"gate": "G0-data", "checks": {}}

    # Hash
    actual_hash = compute_data_hash(data_path)
    if expected_hash:
        # Support prefix matching (e.g. 16-char prefix from config)
        cmp_len = min(len(expected_hash), len(actual_hash))
        hash_ok = actual_hash[:cmp_len] == expected_hash[:cmp_len]
    else:
        hash_ok = True  # No expected hash → skip (record for manifest)
        logger.warning("No expected hash provided. Recording actual hash only.")
    results["checks"]["data_hash"] = {
        "actual": actual_hash[:16],
        "expected": (expected_hash or "N/A")[:16],
        "pass": hash_ok,
    }

    # Column count — feature columns only (exclude target_, close, etc.)
    # 003# #18: use feature columns, not all columns
    df = load_parquet(data_path)
    feature_cols = [c for c in df.columns if not c.startswith("target_") and c != "close"]
    n_feature_cols = len(feature_cols)
    results["checks"]["feature_column_count"] = {
        "actual": n_feature_cols,
        "threshold": min_cols,
        "pass": n_feature_cols >= min_cols,
    }

    # NaN ratio
    nan_info = check_nan_ratio(df, max_nan)
    results["checks"]["nan_ratio"] = nan_info

    # Manifest existence
    mw = ManifestWriter()
    manifest_exists = mw.path.exists()
    results["checks"]["manifest_exists"] = {
        "path": str(mw.path),
        "pass": manifest_exists,
    }

    # Overall
    all_pass = all(c["pass"] for c in results["checks"].values())
    results["gate_result"] = "PASS" if all_pass else "FAIL"

    return results


# ======================================================================
# G1-info (判定のみ — 実験実行は run_experiment.py)
# ======================================================================

def run_g1_judgment(results_path: str, thresholds: dict | None = None) -> dict:
    """G1 judgment from pre-computed experiment results.

    003# #6: Also check min_ic, min_accuracy, min_significant_folds
    from gate_thresholds.yaml.

    Expects results JSON with fold_results structure per §5.3.
    """
    if thresholds is None:
        thresholds = load_gate_thresholds().get("g1_info", {})

    with open(results_path, "r", encoding="utf-8") as f:
        exp_results = json.load(f)

    # 007# F5: g1_judgment_cache があればそれを使用 (stats-only JSON 互換)
    # fold_results が生配列でない場合、g1_judgment() は構造不一致でクラッシュする
    cached_judgment = exp_results.get("g1_judgment_cache")
    if cached_judgment is not None:
        judgment = cached_judgment
        logger.info("Using cached g1_judgment result from experiment JSON")
    else:
        # Legacy: 生配列の fold_results がある場合のみ直接計算
        from ztb.metrics.gate_checks import g1_judgment
        fold_results = exp_results.get("fold_results", {})
        # Validate that fold_results contains raw arrays, not stats dicts
        is_raw = False
        if fold_results:
            first_val = next(iter(fold_results.values()), None)
            if isinstance(first_val, list) and first_val:
                is_raw = isinstance(first_val[0], (list, tuple))
        if is_raw:
            judgment = g1_judgment(
                fold_results=fold_results,
                alpha=thresholds.get("p_alpha", 0.05),
                min_effect=thresholds.get("min_cliff_d", 0.33),
            )
        else:
            logger.warning(
                "fold_results is stats-only and no g1_judgment_cache found. "
                "Cannot re-run g1_judgment. Treating as FAIL."
            )
            judgment = {"g1_pass": False, "passed_targets": [], "details": {}}

    # 003# #6, 007# F1/F2: Threshold checks per target — any() per 000# §3.2
    # "1 組合せでも PASS すれば G1 通過" → ∃ target, not ∀ target
    min_ic = thresholds.get("min_ic", 0.02)
    min_accuracy = thresholds.get("min_accuracy", 0.51)
    min_sig_folds = thresholds.get("min_significant_folds", 2)

    extra_checks: dict[str, dict] = {}
    xgb_results = exp_results.get("xgboost", {})
    for target_name, target_data in xgb_results.items():
        ic_mean = target_data.get("ic_mean", 0.0)
        acc_mean = target_data.get("accuracy_mean", 0.0)
        sig_count = target_data.get("ic_significant_count", 0)

        extra_checks[target_name] = {
            "ic_pass": ic_mean >= min_ic,
            "ic_mean": ic_mean,
            "ic_threshold": min_ic,
            "accuracy_pass": acc_mean >= min_accuracy,
            "accuracy_mean": acc_mean,
            "accuracy_threshold": min_accuracy,
            "sig_folds_pass": sig_count >= min_sig_folds,
            "sig_folds": sig_count,
            "sig_folds_threshold": min_sig_folds,
        }

    # 007# F1: any() — 1 target でも閾値クリアすれば PASS (000# §3.2 準拠)
    extra_any_pass = any(
        c["ic_pass"] and c["accuracy_pass"] and c["sig_folds_pass"]
        for c in extra_checks.values()
    ) if extra_checks else False

    final_pass = judgment["g1_pass"] and extra_any_pass

    return {
        "gate": "G1-info",
        "gate_result": "PASS" if final_pass else "FAIL",
        "details": judgment,
        "threshold_checks": extra_checks,
    }


# ======================================================================
# G1.1-exec (009# P2-3)
# ======================================================================

def run_g1_1(
    results_dir: str,
    thresholds: dict | None = None,
    with_mc: bool = False,
) -> dict:
    """G1.1-exec Gate チェック.

    000# §3.3 / 009# §2.1 準拠.
    fill_records JSONL からメトリクスを算出し、閾値照合を行う.

    Args:
        results_dir: fill_records JSONL ディレクトリ.
        thresholds: Gate 閾値 (None → gate_thresholds.yaml).
        with_mc: True → PnL モンテカルロシミュレーション結果を付加.
    """
    from ztb.metrics.fill_quality import (
        compute_fill_metrics,
        g1_1_judgment,
        load_fill_records_glob,
    )

    if thresholds is None:
        thresholds = load_gate_thresholds().get("g1_1_exec", {})

    records = load_fill_records_glob(results_dir)
    if not records:
        logger.error(f"No fill records found in {results_dir}")
        return {
            "gate": "G1.1-exec",
            "gate_result": "NO_DATA",
            "error": f"No fill records in {results_dir}",
        }

    metrics = compute_fill_metrics(records)
    judgment = g1_1_judgment(metrics, thresholds)

    # Monte Carlo PnL シミュレーション (014# T5 統合)
    # 027# 型統合: FillRecord 共通化によりフィールド変換不要
    if with_mc:
        try:
            from ztb.risk.pnl_monte_carlo import (
                MonteCarloConfig,
                PnLMonteCarloSimulator,
            )

            sim = PnLMonteCarloSimulator(records, MonteCarloConfig())
            mc_result = sim.run()
            judgment["monte_carlo"] = mc_result.to_dict()
            logger.info(
                f"MC: monthly PnL mean={mc_result.pnl_mean_jpy:+,.0f} JPY, "
                f"P(loss)={mc_result.prob_loss:.1%}"
            )
        except Exception as e:
            logger.warning(f"Monte Carlo simulation failed: {e}")
            judgment["monte_carlo"] = {"error": str(e)}

    return judgment


# ======================================================================
# G2-train (031# F7)
# ======================================================================

def run_g2_judgment(results_path: str, thresholds: dict | None = None) -> dict:
    """G2-train Gate チェック.

    000# §3.4: SAC 学習安定性検証.

    Expects results JSON with:
      - seed_results: [{seed, gross_roi, ic_mean}, ...]
      - convergence: {roi_variance_pct_after_30k}
    """
    import statistics as _stats

    if thresholds is None:
        thresholds = load_gate_thresholds().get("g2_train", {})

    with open(results_path, "r", encoding="utf-8") as f:
        exp_results = json.load(f)

    seed_results = exp_results.get("seed_results", [])
    if not seed_results:
        return {"gate": "G2-train", "gate_result": "NO_DATA", "error": "No seed results"}

    checks: dict[str, dict] = {}

    # E1: gross > 0 の seed 比率 >= 75%
    min_ratio = thresholds.get("min_positive_seed_ratio", 0.75)
    positive_seeds = sum(1 for s in seed_results if s.get("gross_roi", 0) > 0)
    ratio = positive_seeds / len(seed_results)
    checks["positive_seed_ratio"] = {
        "value": ratio,
        "threshold": min_ratio,
        "detail": f"{positive_seeds}/{len(seed_results)}",
        "pass": ratio >= min_ratio,
    }

    # E2: IC の seed 間標準偏差 <= 0.03
    max_ic_std = thresholds.get("max_ic_seed_std", 0.03)
    ic_values = [s.get("ic_mean", 0) for s in seed_results]
    ic_std = _stats.stdev(ic_values) if len(ic_values) >= 2 else 0.0
    checks["ic_seed_std"] = {
        "value": ic_std,
        "threshold": max_ic_std,
        "pass": ic_std <= max_ic_std,
    }

    # E3: 30K以降の ROI 変動 <= 5%
    max_roi_var = thresholds.get("max_roi_variance_pct", 5.0)
    convergence = exp_results.get("convergence", {})
    roi_var = convergence.get("roi_variance_pct_after_30k", 0.0)
    checks["convergence"] = {
        "value": roi_var,
        "threshold": max_roi_var,
        "pass": roi_var <= max_roi_var,
    }

    # E4: worst-seed ROI > -2%
    worst_min = thresholds.get("worst_seed_min_roi", -0.02)
    worst_roi = min(s.get("gross_roi", 0) for s in seed_results)
    checks["worst_seed_roi"] = {
        "value": worst_roi,
        "threshold": worst_min,
        "pass": worst_roi > worst_min,
    }

    all_pass = all(c["pass"] for c in checks.values())
    return {
        "gate": "G2-train",
        "gate_result": "PASS" if all_pass else "FAIL",
        "checks": checks,
        "n_seeds": len(seed_results),
    }


# ======================================================================
# G3-pnl (031# F7)
# ======================================================================

def run_g3_judgment(results_path: str, thresholds: dict | None = None) -> dict:
    """G3-pnl Gate チェック.

    000# §3.5: コスト込みの収益性検証.

    Expects results JSON with:
      - seed_metrics: [{pf, sharpe_annual, max_drawdown, avg_gross_per_trade, avg_fee_per_trade}, ...]
    """
    import statistics as _stats

    if thresholds is None:
        thresholds = load_gate_thresholds().get("g3_pnl", {})

    with open(results_path, "r", encoding="utf-8") as f:
        exp_results = json.load(f)

    seed_metrics = exp_results.get("seed_metrics", [])
    if not seed_metrics:
        return {"gate": "G3-pnl", "gate_result": "NO_DATA", "error": "No seed metrics"}

    checks: dict[str, dict] = {}

    # E1: PF median > 1.05
    min_pf_median = thresholds.get("min_pf_median", 1.05)
    pfs = sorted(s.get("pf", 0) for s in seed_metrics)
    pf_median = _stats.median(pfs)
    checks["pf_median"] = {
        "value": pf_median,
        "threshold": min_pf_median,
        "pass": pf_median > min_pf_median,
    }

    # E2: PF worst > 0.95
    min_pf_worst = thresholds.get("min_pf_worst", 0.95)
    pf_worst = min(pfs)
    checks["pf_worst"] = {
        "value": pf_worst,
        "threshold": min_pf_worst,
        "pass": pf_worst > min_pf_worst,
    }

    # E3: gross > fee
    gross_gt_fee_required = thresholds.get("gross_gt_fee", True)
    total_gross = sum(s.get("avg_gross_per_trade", 0) for s in seed_metrics)
    total_fee = sum(s.get("avg_fee_per_trade", 0) for s in seed_metrics)
    gross_gt_fee = total_gross > total_fee
    checks["gross_gt_fee"] = {
        "value": gross_gt_fee,
        "threshold": gross_gt_fee_required,
        "pass": gross_gt_fee if gross_gt_fee_required else True,
    }

    # E4: Max DD < 15%
    max_dd_threshold = thresholds.get("max_drawdown", 0.15)
    worst_dd = max(s.get("max_drawdown", 0) for s in seed_metrics)
    checks["max_drawdown"] = {
        "value": worst_dd,
        "threshold": max_dd_threshold,
        "pass": worst_dd < max_dd_threshold,
    }

    # E5: Sharpe annual median > 0.8
    min_sharpe = thresholds.get("min_sharpe_annual", 0.8)
    sharpes = [s.get("sharpe_annual", 0) for s in seed_metrics]
    sharpe_median = _stats.median(sharpes)
    checks["sharpe_annual"] = {
        "value": sharpe_median,
        "threshold": min_sharpe,
        "pass": sharpe_median > min_sharpe,
    }

    all_pass = all(c["pass"] for c in checks.values())
    return {
        "gate": "G3-pnl",
        "gate_result": "PASS" if all_pass else "FAIL",
        "checks": checks,
        "n_seeds": len(seed_metrics),
    }


# ======================================================================
# G4-live (031# F7)
# ======================================================================

def run_g4_judgment(results_path: str, thresholds: dict | None = None) -> dict:
    """G4-live Gate チェック.

    000# §3.6: Paper trading 運用検証.

    Expects results JSON with:
      - uptime_days, downtime_ratio, circuit_breaker_tested,
        g3_maintained, emergency_stop_response_sec
    """
    if thresholds is None:
        thresholds = load_gate_thresholds().get("g4_live", {})

    with open(results_path, "r", encoding="utf-8") as f:
        exp_results = json.load(f)

    checks: dict[str, dict] = {}

    # E1: 連続稼働 >= 7日
    min_days = thresholds.get("min_paper_days", 7)
    uptime_days = exp_results.get("uptime_days", 0)
    checks["uptime_days"] = {
        "value": uptime_days,
        "threshold": min_days,
        "pass": uptime_days >= min_days,
    }

    # E2: ダウンタイム < 1%
    max_downtime = thresholds.get("max_downtime_ratio", 0.01)
    downtime = exp_results.get("downtime_ratio", 1.0)
    checks["downtime_ratio"] = {
        "value": downtime,
        "threshold": max_downtime,
        "pass": downtime < max_downtime,
    }

    # E3: Circuit Breaker 発動確認
    cb_tested = exp_results.get("circuit_breaker_tested", False)
    checks["circuit_breaker"] = {
        "value": cb_tested,
        "threshold": True,
        "pass": cb_tested is True,
    }

    # E4: G3 指標維持
    g3_maintained = exp_results.get("g3_maintained", False)
    checks["g3_maintained"] = {
        "value": g3_maintained,
        "threshold": True,
        "pass": g3_maintained is True,
    }

    # E5: 緊急停止応答 < 1秒
    max_response = thresholds.get("max_emergency_stop_sec", 1.0)
    response_sec = exp_results.get("emergency_stop_response_sec", float("inf"))
    checks["emergency_stop"] = {
        "value": response_sec,
        "threshold": max_response,
        "pass": response_sec < max_response,
    }

    all_pass = all(c["pass"] for c in checks.values())
    return {
        "gate": "G4-live",
        "gate_result": "PASS" if all_pass else "FAIL",
        "checks": checks,
    }


# ======================================================================
# CLI
# ======================================================================

def main() -> None:
    parser = argparse.ArgumentParser(description="v460 Gate Check")
    parser.add_argument("--gate", required=True,
                        choices=["G0", "G1", "G1.1", "G2", "G3", "G4"],
                        help="Gate to check")
    parser.add_argument("--data-path", default=None,
                        help="Path to data file (G0)")
    parser.add_argument("--expected-hash", default=None,
                        help="Expected SHA-256 hash (G0)")
    parser.add_argument("--results-path", default=None,
                        help="Path to G1 results JSON")
    parser.add_argument("--results-dir", default=None,
                        help="Path to fill_records directory (G1.1)")
    parser.add_argument("--output", default=None,
                        help="Output JSON path")
    parser.add_argument("--with-mc", action="store_true", default=False,
                        help="Run Monte Carlo PnL simulation (G1.1 only)")
    args = parser.parse_args()

    if args.gate == "G0":
        if not args.data_path:
            parser.error("--data-path required for G0")
        result = run_g0(args.data_path, args.expected_hash)
    elif args.gate == "G1":
        if not args.results_path:
            parser.error("--results-path required for G1")
        result = run_g1_judgment(args.results_path)
    elif args.gate == "G1.1":
        results_dir = args.results_dir or "results/v460/fill_test"
        result = run_g1_1(results_dir, with_mc=args.with_mc)
    elif args.gate == "G2":
        if not args.results_path:
            parser.error("--results-path required for G2")
        result = run_g2_judgment(args.results_path)
    elif args.gate == "G3":
        if not args.results_path:
            parser.error("--results-path required for G3")
        result = run_g3_judgment(args.results_path)
    elif args.gate == "G4":
        if not args.results_path:
            parser.error("--results-path required for G4")
        result = run_g4_judgment(args.results_path)
    else:
        parser.error(f"Unknown gate: {args.gate}")
        return

    # Output
    out_str = json.dumps(result, indent=2, ensure_ascii=False)
    print(out_str)

    if args.output:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        with open(args.output, "w", encoding="utf-8") as f:
            f.write(out_str)
        logger.info(f"Saved: {args.output}")

    # Exit code
    sys.exit(0 if result["gate_result"] == "PASS" else 1)


if __name__ == "__main__":
    main()
