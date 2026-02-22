#!/usr/bin/env python3
"""
065# 公式 G1-info 再評価 — Holm 補正 + Cliff's Delta + accuracy 閾値.

064# の簡易 G1 PASS を、000# §3.2 / gate_thresholds.yaml の公式基準で再判定。

出力は run_gate_check.py --gate G1 互換形式。

Usage:
    python scripts/v460/run_065_g1_proper_eval.py
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))

from scripts.v460.lib.config_loader import load_gate_thresholds
from scripts.v460.lib.evaluator import (
    WalkForwardResult,
    make_logistic,
    make_ridge,
    make_xgboost_classifier,
    make_xgboost_regressor,
    walk_forward_eval,
)
from ztb.features.microstructure import MICROSTRUCTURE_FEATURES
from ztb.io.json_io import write_json
from ztb.metrics.gate_checks import cliffs_delta, g1_judgment, holm_bonferroni_gate

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)

# --- Config ---
FEATURES_PATH = (
    _PROJECT_ROOT / "data/v460/features/btc_jpy_1m_v460_real_features.parquet"
)
HORIZONS = [1, 5, 15]
TARGET_TYPES = ["direction", "magnitude", "volatility"]
N_FOLDS = 3  # 3 日分
REPORT_DIR = _PROJECT_ROOT / "docs/v460"
RESULTS_DIR = _PROJECT_ROOT / "results/v460"


def generate_targets(df: pd.DataFrame, horizons: list[int]) -> pd.DataFrame:
    """Generate direction/magnitude/volatility targets at multiple horizons."""
    df = df.copy()
    for h in horizons:
        fwd_ret = df["close"].pct_change(h).shift(-h)
        fwd_ret = fwd_ret.replace([np.inf, -np.inf], np.nan)
        df[f"target_direction_h{h}"] = (
            (fwd_ret > 0).where(fwd_ret.notna()).astype("Int64")
        )
        df[f"target_magnitude_h{h}"] = fwd_ret.abs()
        log_ret = np.log(df["close"] / df["close"].shift(1))
        log_ret = log_ret.replace([np.inf, -np.inf], np.nan)
        df[f"target_volatility_h{h}"] = (
            log_ret.rolling(h, min_periods=1).std().shift(-h)
        )
    return df


def _is_classification(ttype: str) -> bool:
    return ttype == "direction"


def collect_paired_signals(
    df: pd.DataFrame,
    feature_cols: list[str],
    horizons: list[int],
    target_types: list[str],
    n_folds: int,
    train_ratio: float = 0.80,
) -> dict[str, dict]:
    """Walk-forward eval with XGBoost + baseline, collecting per-fold signals.

    Returns:
        { target_name: {
            "xgb": WalkForwardResult,
            "baseline": WalkForwardResult,
            "fold_pairs": [(xgb_signal, baseline_signal), ...],
          }
        }
    """
    results: dict[str, dict] = {}

    for h in horizons:
        for ttype in target_types:
            target_col = f"target_{ttype}_h{h}"
            if target_col not in df.columns:
                logger.warning(f"Target {target_col} not found, skipping")
                continue

            mask = df[target_col].notna()
            df_clean = df.loc[mask].reset_index(drop=True)

            is_cls = _is_classification(ttype)
            if is_cls:
                xgb_factory = lambda: make_xgboost_classifier(seed=42)
                bl_factory = lambda: make_logistic(seed=42)
                df_clean[target_col] = df_clean[target_col].astype(int)
            else:
                xgb_factory = lambda: make_xgboost_regressor(seed=42)
                bl_factory = lambda: make_ridge(seed=42)

            logger.info(
                f"Evaluating {target_col}: {len(df_clean)} rows "
                f"({'cls' if is_cls else 'reg'})"
            )

            # Run both models to collect per-fold signals
            xgb_wf = walk_forward_eval(
                df_clean,
                feature_cols,
                target_col,
                xgb_factory,
                model_name="XGBoost",
                n_folds=n_folds,
                train_ratio=train_ratio,
                is_classification=is_cls,
            )
            bl_wf = walk_forward_eval(
                df_clean,
                feature_cols,
                target_col,
                bl_factory,
                model_name="Baseline",
                n_folds=n_folds,
                train_ratio=train_ratio,
                is_classification=is_cls,
            )

            # Pair per-fold signals
            fold_pairs: list[tuple[list[float], list[float]]] = []
            n_paired = min(len(xgb_wf.folds), len(bl_wf.folds))
            for i in range(n_paired):
                xgb_sig = xgb_wf.folds[i]._signal
                bl_sig = bl_wf.folds[i]._signal
                fold_pairs.append((xgb_sig, bl_sig))

            results[target_col] = {
                "xgb": xgb_wf,
                "baseline": bl_wf,
                "fold_pairs": fold_pairs,
            }

    return results


def apply_g1_criteria(
    paired_results: dict[str, dict],
    thresholds: dict,
) -> dict:
    """Apply official G1 criteria with Holm + Cliff's Delta.

    Returns:
        Full G1 judgment result compatible with run_gate_check.py.
    """
    min_ic = thresholds.get("min_ic", 0.02)
    min_accuracy = thresholds.get("min_accuracy", 0.51)
    min_sig_folds = thresholds.get("min_significant_folds", 2)
    p_alpha = thresholds.get("p_alpha", 0.05)
    min_cliff_d = thresholds.get("min_cliff_d", 0.33)

    # --- 1. Per-target threshold checks (IC, accuracy, sig_folds) ---
    threshold_checks: dict[str, dict] = {}
    for target_name, data in paired_results.items():
        xgb_wf: WalkForwardResult = data["xgb"]
        threshold_checks[target_name] = {
            "ic_pass": abs(xgb_wf.ic_mean) >= min_ic,
            "ic_mean": xgb_wf.ic_mean,
            "ic_threshold": min_ic,
            "accuracy_pass": xgb_wf.accuracy_mean >= min_accuracy,
            "accuracy_mean": xgb_wf.accuracy_mean,
            "accuracy_threshold": min_accuracy,
            "sig_folds_pass": xgb_wf.ic_significant_count >= min_sig_folds,
            "sig_folds": xgb_wf.ic_significant_count,
            "sig_folds_threshold": min_sig_folds,
            "n_folds": xgb_wf.n_folds,
        }

    # --- 2. Holm-Bonferroni + Cliff's Delta per target (holm_bonferroni_gate) ---
    holm_input: dict[str, tuple[list[float], list[float]]] = {}
    for target_name, data in paired_results.items():
        all_xgb: list[float] = []
        all_bl: list[float] = []
        for xgb_sig, bl_sig in data["fold_pairs"]:
            all_xgb.extend(xgb_sig)
            all_bl.extend(bl_sig)
        if all_xgb and all_bl:
            holm_input[target_name] = (all_xgb, all_bl)

    holm_results = holm_bonferroni_gate(
        holm_input, alpha=p_alpha, min_effect=min_cliff_d
    )

    # --- 3. g1_judgment (p-mean per target across folds → Holm → AND) ---
    fold_results_for_g1: dict[str, list[tuple[list[float], list[float]]]] = {}
    for target_name, data in paired_results.items():
        fold_results_for_g1[target_name] = data["fold_pairs"]

    g1_result = g1_judgment(
        fold_results_for_g1, alpha=p_alpha, min_effect=min_cliff_d
    )

    # --- 4. Combined judgment ---
    # 000# §3.2: any target passes ALL criteria → G1 PASS
    any_target_pass = any(
        tc["ic_pass"]
        and tc["accuracy_pass"]
        and tc["sig_folds_pass"]
        and holm_results.get(tn, {}).get("pass", False)
        for tn, tc in threshold_checks.items()
    )

    # Final = g1_judgment AND threshold_checks
    final_pass = g1_result["g1_pass"] and any_target_pass

    return {
        "gate": "G1-info",
        "gate_result": "PASS" if final_pass else "FAIL",
        "g1_judgment": g1_result,
        "holm_results": holm_results,
        "threshold_checks": threshold_checks,
        "criteria_used": {
            "min_ic": min_ic,
            "min_accuracy": min_accuracy,
            "min_significant_folds": min_sig_folds,
            "p_alpha": p_alpha,
            "min_cliff_d": min_cliff_d,
            "n_folds": N_FOLDS,
            "n_targets": len(paired_results),
        },
    }


def compute_raw_ic(df: pd.DataFrame, horizons: list[int]) -> pd.DataFrame:
    """Raw Spearman IC per feature (same as 064#)."""
    from scipy import stats as sp_stats

    rows = []
    for h in horizons:
        fwd_ret = df["close"].pct_change(h).shift(-h)
        mask = fwd_ret.notna()
        fwd_clean = fwd_ret[mask]
        for feat in MICROSTRUCTURE_FEATURES:
            if feat not in df.columns:
                continue
            feat_clean = df.loc[mask, feat]
            if feat_clean.std() < 1e-12:
                rows.append(
                    {"feature": feat, "horizon": h, "ic": 0.0, "pvalue": 1.0}
                )
                continue
            result = sp_stats.spearmanr(feat_clean, fwd_clean, nan_policy="omit")
            ic_val = (
                float(result.correlation)
                if not np.isnan(result.correlation)
                else 0.0
            )
            p_val = (
                float(result.pvalue) if not np.isnan(result.pvalue) else 1.0
            )
            rows.append(
                {
                    "feature": feat,
                    "horizon": h,
                    "ic": round(ic_val, 6),
                    "pvalue": round(p_val, 6),
                }
            )
    return pd.DataFrame(rows)


def generate_report(
    g1_result: dict,
    ic_df: pd.DataFrame,
    paired: dict[str, dict],
    data_shape: tuple[int, ...],
    n_features: int,
) -> str:
    """Generate Markdown report."""
    lines = [
        "# 065# 公式 G1-info 再評価結果",
        "",
        "**Phase**: ph1 (064# 再検証)",
        f"**Date**: {pd.Timestamp.now().strftime('%Y-%m-%d')}",
        f"**Data**: {data_shape[0]} rows, {n_features} features, 3 days (2/13-2/15)",
        f"**G1 Result**: **{g1_result['gate_result']}**",
        "",
        "## 1. 公式 G1 基準 (000# §3.2 / gate_thresholds.yaml)",
        "",
        "| 基準 | 値 |",
        "|---|---|",
    ]
    criteria = g1_result["criteria_used"]
    for k, v in criteria.items():
        lines.append(f"| {k} | {v} |")

    lines += [
        "",
        "## 2. Raw Feature IC (Spearman) — 064# 再掲",
        "",
        "| Feature | h1 IC | h5 IC | h15 IC |",
        "|---|---|---|---|",
    ]
    avail = ic_df["feature"].unique()
    for feat in avail:
        vals: dict[int, str] = {}
        for h in HORIZONS:
            row = ic_df[(ic_df["feature"] == feat) & (ic_df["horizon"] == h)]
            if len(row) > 0:
                ic_val = row.iloc[0]["ic"]
                p_val = row.iloc[0]["pvalue"]
                sig = (
                    "***"
                    if p_val < 0.01
                    else ("**" if p_val < 0.05 else "")
                )
                vals[h] = f"{ic_val:+.4f}{sig}"
            else:
                vals[h] = "N/A"
        lines.append(f"| {feat} | {vals[1]} | {vals[5]} | {vals[15]} |")

    # Per-target results
    lines += [
        "",
        "## 3. Walk-Forward Model Comparison (XGBoost vs Baseline)",
        "",
        "| Target | Model | Accuracy | IC_mean | Sig_folds | Folds |",
        "|---|---|---|---|---|---|",
    ]
    for target_name in sorted(paired.keys()):
        data = paired[target_name]
        xgb: WalkForwardResult = data["xgb"]
        bl: WalkForwardResult = data["baseline"]
        lines.append(
            f"| {target_name} | XGBoost | {xgb.accuracy_mean:.4f} | "
            f"{xgb.ic_mean:+.4f} | {xgb.ic_significant_count}/{xgb.n_folds} | "
            f"{xgb.n_folds} |"
        )
        lines.append(
            f"| {target_name} | Baseline | {bl.accuracy_mean:.4f} | "
            f"{bl.ic_mean:+.4f} | {bl.ic_significant_count}/{bl.n_folds} | "
            f"{bl.n_folds} |"
        )

    # Threshold checks
    lines += [
        "",
        "## 4. 公式 G1 閾値判定 (per target)",
        "",
        "| Target | IC pass | Acc pass | Sig pass | Holm pass | Cliff d |",
        "|---|---|---|---|---|---|",
    ]
    tc = g1_result["threshold_checks"]
    holm = g1_result["holm_results"]
    for tn in sorted(tc.keys()):
        c = tc[tn]
        h = holm.get(tn, {})
        lines.append(
            f"| {tn} | {'Y' if c['ic_pass'] else 'N'} (IC={c['ic_mean']:+.4f}) | "
            f"{'Y' if c['accuracy_pass'] else 'N'} (acc={c['accuracy_mean']:.4f}) | "
            f"{'Y' if c['sig_folds_pass'] else 'N'} ({c['sig_folds']}/{c['sig_folds_threshold']}) | "
            f"{'Y' if h.get('pass', False) else 'N'} | "
            f"{h.get('d', 0.0):+.4f} |"
        )

    # g1_judgment details
    lines += [
        "",
        "## 5. g1_judgment (p-mean + Holm + Cliff's Delta)",
        "",
        f"- **g1_pass**: {g1_result['g1_judgment']['g1_pass']}",
        f"- **passed_targets**: {g1_result['g1_judgment']['passed_targets']}",
        "",
        "| Target | p_geo | pmean_pass | holm_pass | cliff_d |",
        "|---|---|---|---|---|",
    ]
    for tn, d in g1_result["g1_judgment"]["details"].items():
        lines.append(
            f"| {tn} | {d['p_geo']:.6f} | {d['pmean_pass']} | "
            f"{d['holm_pass']} | {d['cliff_d']:+.4f} |"
        )

    # Final judgment
    lines += [
        "",
        "## 6. 最終判定",
        "",
        f"### G1-info: **{g1_result['gate_result']}**",
        "",
    ]
    if g1_result["gate_result"] == "FAIL":
        lines += [
            "> 公式 G1 基準では FAIL。065# レビューの指摘通り、064# の簡易 PASS は",
            "> 公式基準では通過しない。AS-LR SkipGate (060/061系) を ph2 主軸とする。",
            "",
            "### 推奨アクション",
            "",
            "1. AS-LR SkipGate (`mode=as`, `as_threshold=0.65`) で ph2 fill_test 再開",
            "2. 064# の有望特徴量 (vwap_deviation, order_flow_toxicity, ask_depth_slope) は",
            "   次回 SkipGate v3 候補として保持",
            "3. 064# XGBoost は shadow logging のみ（注文判定に使わない）",
        ]
    else:
        lines += [
            "> 公式 G1 基準 PASS。追加データ蓄積で信頼性を向上した後、",
            "> ph2 投入を検討。",
        ]

    return "\n".join(lines)


def main() -> None:
    # Load thresholds
    all_thresholds = load_gate_thresholds()
    g1_thresholds = all_thresholds.get("g1_info", {})
    logger.info(f"G1 thresholds: {g1_thresholds}")

    # Load features
    logger.info(f"Loading features from {FEATURES_PATH}")
    df = pd.read_parquet(FEATURES_PATH)
    logger.info(f"Loaded: {df.shape}")

    avail_features = [f for f in MICROSTRUCTURE_FEATURES if f in df.columns]
    logger.info(f"Available features ({len(avail_features)}): {avail_features}")

    # Step 1: Raw IC
    logger.info("=" * 60)
    logger.info("Step 1: Raw feature IC")
    ic_df = compute_raw_ic(df, HORIZONS)

    # Step 2: Generate targets
    logger.info("=" * 60)
    logger.info("Step 2: Targets")
    df = generate_targets(df, HORIZONS)

    # Step 3: Paired walk-forward (XGBoost + Baseline)
    logger.info("=" * 60)
    logger.info("Step 3: Walk-forward XGBoost vs Baseline (per-fold signal collection)")
    paired_results = collect_paired_signals(
        df,
        avail_features,
        HORIZONS,
        TARGET_TYPES,
        n_folds=N_FOLDS,
        train_ratio=0.80,
    )

    # Step 4: Apply full G1 criteria
    logger.info("=" * 60)
    logger.info("Step 4: Apply official G1 criteria")
    g1_result = apply_g1_criteria(paired_results, g1_thresholds)

    # Summary
    logger.info("=" * 60)
    logger.info(f"G1-info result: {g1_result['gate_result']}")
    for tn, tc in g1_result["threshold_checks"].items():
        logger.info(
            f"  {tn}: IC={'PASS' if tc['ic_pass'] else 'FAIL'} "
            f"({tc['ic_mean']:+.4f}), "
            f"Acc={'PASS' if tc['accuracy_pass'] else 'FAIL'} "
            f"({tc['accuracy_mean']:.4f}), "
            f"Sig={'PASS' if tc['sig_folds_pass'] else 'FAIL'} "
            f"({tc['sig_folds']}/{tc['sig_folds_threshold']})"
        )
    for tn, h in g1_result["holm_results"].items():
        logger.info(
            f"  {tn}: Holm={'PASS' if h['pass'] else 'FAIL'} "
            f"(d={h['d']:+.4f}, p_holm={h['p_holm']:.6f})"
        )

    # Step 5: Save
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    # Report
    report = generate_report(
        g1_result, ic_df, paired_results, df.shape, len(avail_features)
    )
    report_path = REPORT_DIR / "065_g1_proper_eval.md"
    report_path.write_text(report, encoding="utf-8")
    logger.info(f"Report saved: {report_path}")

    # JSON (run_gate_check.py compatible)
    # Build xgboost results in expected format
    xgb_for_gate: dict = {}
    for target_name, data in paired_results.items():
        xgb_wf: WalkForwardResult = data["xgb"]
        xgb_for_gate[target_name] = xgb_wf.to_dict()

    gate_json = {
        "g1_judgment_cache": g1_result["g1_judgment"],
        "xgboost": xgb_for_gate,
        "holm_results": g1_result["holm_results"],
        "threshold_checks": g1_result["threshold_checks"],
        "gate_result": g1_result["gate_result"],
        "criteria_used": g1_result["criteria_used"],
        "raw_ic": ic_df.to_dict(orient="records"),
    }
    json_path = RESULTS_DIR / "065_g1_proper_eval.json"
    write_json(json_path, gate_json, indent=2, ensure_ascii=False, default=str)
    logger.info(f"JSON saved: {json_path}")

    print(f"\n{'='*60}")
    print(f"  065# G1-info Proper Evaluation: {g1_result['gate_result']}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
