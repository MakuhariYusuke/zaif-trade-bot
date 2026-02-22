#!/usr/bin/env python3
"""
064# ph1 G1-info 再検証 — リアルOB/Trades microstructure IC検証.

蓄積3日分データ (2/13-2/15) でmicrostructure featureの予測力を検証。
Walk-forward XGBoost + Logistic baseline で direction IC を測定。

Usage:
    python scripts/v460/run_064_g1_info_verify.py
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))

from ztb.features.microstructure import MICROSTRUCTURE_FEATURES
from ztb.io.json_io import write_json
from scripts.v460.lib.evaluator import (
    make_xgboost_classifier,
    make_xgboost_regressor,
    make_logistic,
    make_ridge,
    walk_forward_eval,
    evaluate_multi_target,
)

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)

# --- Config ---
FEATURES_PATH = _PROJECT_ROOT / "data/v460/features/btc_jpy_1m_v460_real_features.parquet"
HORIZONS = [1, 5, 15]  # 1m, 5m, 15m forward
TARGET_TYPES = ["direction", "magnitude", "volatility"]
N_FOLDS = 3  # 3日分 → 3 folds
REPORT_PATH = _PROJECT_ROOT / "docs/v460/064_ph1_g1_info_verify.md"


def generate_targets(df: pd.DataFrame, horizons: list[int]) -> pd.DataFrame:
    """Generate direction/magnitude/volatility targets at multiple horizons."""
    df = df.copy()
    for h in horizons:
        fwd_ret = df["close"].pct_change(h).shift(-h)
        # Replace inf with NaN (can occur when close=0 from ffill)
        fwd_ret = fwd_ret.replace([np.inf, -np.inf], np.nan)

        # Direction: 1 if positive, 0 if negative
        df[f"target_direction_h{h}"] = (fwd_ret > 0).where(fwd_ret.notna()).astype("Int64")

        # Magnitude: absolute return
        df[f"target_magnitude_h{h}"] = fwd_ret.abs()

        # Volatility: rolling std of returns shifted forward
        log_ret = np.log(df["close"] / df["close"].shift(1))
        log_ret = log_ret.replace([np.inf, -np.inf], np.nan)
        df[f"target_volatility_h{h}"] = (
            log_ret.rolling(h, min_periods=1).std().shift(-h)
        )

    return df


def compute_feature_ic(df: pd.DataFrame, horizons: list[int]) -> pd.DataFrame:
    """Raw Spearman IC per feature (no model, direct correlation)."""
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
                rows.append({"feature": feat, "horizon": h, "ic": 0.0, "pvalue": 1.0})
                continue
            result = sp_stats.spearmanr(feat_clean, fwd_clean, nan_policy="omit")
            ic_val = float(result.correlation) if not np.isnan(result.correlation) else 0.0
            p_val = float(result.pvalue) if not np.isnan(result.pvalue) else 1.0
            rows.append({"feature": feat, "horizon": h, "ic": round(ic_val, 6), "pvalue": round(p_val, 6)})
    return pd.DataFrame(rows)


def main() -> None:
    logger.info(f"Loading features from {FEATURES_PATH}")
    df = pd.read_parquet(FEATURES_PATH)
    logger.info(f"Loaded: {df.shape}, index: {df.index[0]} ~ {df.index[-1]}")

    # Check available features
    avail_features = [f for f in MICROSTRUCTURE_FEATURES if f in df.columns]
    logger.info(f"Available features ({len(avail_features)}): {avail_features}")

    # Step 1: Raw IC per feature
    logger.info("=" * 60)
    logger.info("Step 1: Raw feature IC (Spearman)")
    ic_df = compute_feature_ic(df, HORIZONS)
    for h in HORIZONS:
        subset = ic_df[ic_df["horizon"] == h].sort_values("ic", key=abs, ascending=False)
        logger.info(f"\n  h={h}:")
        for _, row in subset.iterrows():
            sig = "***" if row["pvalue"] < 0.01 else ("**" if row["pvalue"] < 0.05 else "")
            logger.info(f"    {row['feature']:30s}  IC={row['ic']:+.6f}  p={row['pvalue']:.4f}  {sig}")

    # Step 2: Generate targets
    logger.info("=" * 60)
    logger.info("Step 2: Generate targets")
    df = generate_targets(df, HORIZONS)

    # Step 3: Walk-forward evaluation 
    logger.info("=" * 60)
    logger.info("Step 3: Walk-forward XGBoost evaluation")

    xgb_results = evaluate_multi_target(
        df, avail_features, HORIZONS, TARGET_TYPES,
        model_factory=lambda: make_xgboost_classifier(seed=42),
        model_name="XGBoost",
        n_folds=N_FOLDS,
        train_ratio=0.80,
        regression_factory=lambda: make_xgboost_regressor(seed=42),
    )

    # Step 4: Baseline (Logistic / Ridge)
    logger.info("=" * 60)
    logger.info("Step 4: Walk-forward Logistic/Ridge baseline")

    baseline_results = evaluate_multi_target(
        df, avail_features, HORIZONS, TARGET_TYPES,
        model_factory=lambda: make_logistic(seed=42),
        model_name="Logistic",
        n_folds=N_FOLDS,
        train_ratio=0.80,
        regression_factory=lambda: make_ridge(seed=42),
    )

    # Step 5: Summary + Report
    logger.info("=" * 60)
    logger.info("Step 5: Summary")

    report_lines = [
        "# 064# ph1 G1-info 再検証結果",
        "",
        "**Phase**: ph1 (検証フェーズ)",
        f"**Date**: {pd.Timestamp.now().strftime('%Y-%m-%d')}",
        f"**Data**: {df.shape[0]} rows, {len(avail_features)} features, 3 days (2/13-2/15)",
        "",
        "## 1. Raw Feature IC (Spearman rank correlation)",
        "",
        "| Feature | h1 IC | h5 IC | h15 IC |",
        "|---|---|---|---|",
    ]

    for feat in avail_features:
        vals = {}
        for h in HORIZONS:
            row = ic_df[(ic_df["feature"] == feat) & (ic_df["horizon"] == h)]
            if len(row) > 0:
                ic_val = row.iloc[0]["ic"]
                p_val = row.iloc[0]["pvalue"]
                sig = "***" if p_val < 0.01 else ("**" if p_val < 0.05 else "")
                vals[h] = f"{ic_val:+.4f}{sig}"
            else:
                vals[h] = "N/A"
        report_lines.append(f"| {feat} | {vals[1]} | {vals[5]} | {vals[15]} |")

    report_lines += [
        "",
        "## 2. Walk-Forward Results",
        "",
        "| Target | Model | Acc/MAE | IC_mean | IC_sig | Folds |",
        "|---|---|---|---|---|---|",
    ]

    all_results = {}
    all_results.update({k: ("XGBoost", v) for k, v in xgb_results.items()})
    all_results.update({k + "_bl": ("Logistic/Ridge", v) for k, v in baseline_results.items()})

    for key in sorted(xgb_results.keys()):
        xgb_r = xgb_results[key]
        bl_r = baseline_results.get(key)
        is_cls = "direction" in key
        if is_cls:
            xgb_metric = f"acc={xgb_r.accuracy_mean:.4f}"
            bl_metric = f"acc={bl_r.accuracy_mean:.4f}" if bl_r else "N/A"
        else:
            xgb_mae = np.mean([f.mae for f in xgb_r.folds if f.mae is not None])
            xgb_metric = f"mae={xgb_mae:.6f}"
            bl_mae = np.mean([f.mae for f in bl_r.folds if f.mae is not None]) if bl_r else float("nan")
            bl_metric = f"mae={bl_mae:.6f}" if bl_r else "N/A"

        report_lines.append(
            f"| {key} | XGBoost | {xgb_metric} | {xgb_r.ic_mean:+.4f} | "
            f"{xgb_r.ic_significant_count}/{xgb_r.n_folds} | {xgb_r.n_folds} |"
        )
        if bl_r:
            report_lines.append(
                f"| {key} | Baseline | {bl_metric} | {bl_r.ic_mean:+.4f} | "
                f"{bl_r.ic_significant_count}/{bl_r.n_folds} | {bl_r.n_folds} |"
            )

    # G1-info judgment
    report_lines += ["", "## 3. G1-info 判定", ""]

    direction_ics = []
    for key, r in xgb_results.items():
        if "direction" in key:
            direction_ics.append(r.ic_mean)
    avg_direction_ic = np.mean(direction_ics) if direction_ics else 0.0

    pass_criteria = abs(avg_direction_ic) > 0.02
    report_lines.append(f"- **Direction IC 平均**: {avg_direction_ic:+.6f}")
    report_lines.append(f"- **G1-info 基準 (|IC| > 0.02)**: {'**PASS**' if pass_criteria else '**FAIL**'}")
    report_lines.append("")

    if pass_criteria:
        report_lines.append("> Microstructure features are informative. "
                          "Proceed to SkipGate live integration.")
    else:
        report_lines.append("> Microstructure features show weak signal. "
                          "Consider feature engineering improvements or more data.")

    # Save report
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text("\n".join(report_lines), encoding="utf-8")
    logger.info(f"Report saved: {REPORT_PATH}")

    # Also save JSON for programmatic access
    json_path = REPORT_PATH.with_suffix(".json")
    json_data = {
        "raw_ic": ic_df.to_dict(orient="records"),
        "xgb_results": {k: v.to_dict() for k, v in xgb_results.items()},
        "baseline_results": {k: v.to_dict() for k, v in baseline_results.items()},
        "g1_info_pass": pass_criteria,
        "avg_direction_ic": round(avg_direction_ic, 6),
    }
    write_json(json_path, json_data, indent=2, ensure_ascii=False, default=str)
    logger.info(f"JSON saved: {json_path}")

    # Final print
    print("\n" + "=" * 60)
    print("  064# G1-info Re-verification Results")
    print("=" * 60)
    print(f"  Data:       {df.shape[0]} rows, 3 days")
    print(f"  Features:   {len(avail_features)}")
    print(f"  Direction IC avg: {avg_direction_ic:+.6f}")
    print(f"  G1-info:    {'PASS' if pass_criteria else 'FAIL'}")
    print("=" * 60)


if __name__ == "__main__":
    main()
