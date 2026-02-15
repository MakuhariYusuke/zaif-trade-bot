"""057# Run ML Pipeline: AS分類器 + Fill分類器のオフライン評価.

使い方:
    python scripts/v460/ml/run_ml_pipeline.py [--output-dir reports/v460/ml]
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from dataclasses import asdict
from pathlib import Path

# プロジェクトルートを追加
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent.parent))

from scripts.v460.ml.as_classifier import (
    ASModelMetrics,
    evaluate_skip_policy,
    train_as_classifier,
)
from scripts.v460.ml.data_loader import (
    build_as_features,
    build_fill_features,
    load_fill_records,
)
from scripts.v460.ml.fill_classifier import train_fill_classifier

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
)
logger = logging.getLogger(__name__)


def run_as_pipeline(df: "pd.DataFrame", output_dir: Path) -> dict:
    """AS 分類器パイプライン."""
    import pandas as pd

    logger.info("=" * 60)
    logger.info("ML-1: AS Classifier Pipeline")
    logger.info("=" * 60)

    X, y = build_as_features(df)

    # PnL for skip simulation
    filled_mask = df["filled"].astype(bool) & df["adverse_selected_raw"].notna()
    pnl = df.loc[filled_mask, "post_fill_30s_pnl"].astype(float)
    pnl = pnl.reindex(X.index)

    results: dict = {}

    # --- GradientBoosting ---
    logger.info("\n--- GradientBoosting ---")
    gb_metrics, gb_model, gb_scaler = train_as_classifier(
        X, y, pnl, model_type="gb", n_splits=5
    )
    results["gb"] = asdict(gb_metrics)
    _print_as_metrics("GB", gb_metrics)

    # --- LogisticRegression ---
    logger.info("\n--- LogisticRegression ---")
    lr_metrics, lr_model, lr_scaler = train_as_classifier(
        X, y, pnl, model_type="lr", n_splits=5
    )
    results["lr"] = asdict(lr_metrics)
    _print_as_metrics("LR", lr_metrics)

    # --- Skip policy simulation (best model) ---
    best = "gb" if gb_metrics.pr_auc_mean >= lr_metrics.pr_auc_mean else "lr"
    best_model = gb_model if best == "gb" else lr_model
    best_scaler = gb_scaler if best == "gb" else lr_scaler
    logger.info(f"\nBest model: {best.upper()}")

    skip_df = evaluate_skip_policy(X, y, pnl, best_model, best_scaler)
    logger.info("\nSkip Policy Simulation:")
    logger.info(skip_df.to_string(index=False, float_format="%.4f"))
    results["skip_policy"] = skip_df.to_dict(orient="records")
    results["best_model"] = best

    # 保存
    out_file = output_dir / "as_classifier_results.json"
    out_file.parent.mkdir(parents=True, exist_ok=True)
    with open(out_file, "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False, default=str)
    logger.info(f"\nResults saved to {out_file}")

    return results


def run_fill_pipeline(df: "pd.DataFrame", output_dir: Path) -> dict:
    """Fill 分類器パイプライン."""
    logger.info("=" * 60)
    logger.info("ML-2: Fill Classifier Pipeline")
    logger.info("=" * 60)

    X, y = build_fill_features(df)

    results: dict = {}

    # --- GradientBoosting ---
    logger.info("\n--- GradientBoosting ---")
    gb_metrics, gb_model, gb_scaler = train_fill_classifier(
        X, y, model_type="gb", n_splits=5
    )
    results["gb"] = asdict(gb_metrics)
    _print_fill_metrics("GB", gb_metrics)

    # --- LogisticRegression ---
    logger.info("\n--- LogisticRegression ---")
    lr_metrics, lr_model, lr_scaler = train_fill_classifier(
        X, y, model_type="lr", n_splits=5
    )
    results["lr"] = asdict(lr_metrics)
    _print_fill_metrics("LR", lr_metrics)

    best = "gb" if gb_metrics.roc_auc_mean >= lr_metrics.roc_auc_mean else "lr"
    results["best_model"] = best
    logger.info(f"\nBest model: {best.upper()}")

    # 保存
    out_file = output_dir / "fill_classifier_results.json"
    out_file.parent.mkdir(parents=True, exist_ok=True)
    with open(out_file, "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False, default=str)
    logger.info(f"\nResults saved to {out_file}")

    return results


def _print_as_metrics(name: str, m: ASModelMetrics) -> None:
    logger.info(f"  Samples:    {m.n_samples}")
    logger.info(f"  ROC-AUC:    {m.roc_auc_mean:.4f} ± {m.roc_auc_std:.4f}")
    logger.info(f"  PR-AUC:     {m.pr_auc_mean:.4f} ± {m.pr_auc_std:.4f}")
    logger.info(f"  Brier:      {m.brier_mean:.4f} ± {m.brier_std:.4f}")
    logger.info(f"  Naive PR:   {m.naive_pr_auc:.4f}")
    logger.info(f"  Improvement:{m.improvement_over_naive:+.4f}")
    logger.info(f"  Skip20% →   {m.skip_top20_pnl_improvement_bps:+.3f} bps")
    logger.info(f"  Skip10% →   {m.skip_top10_pnl_improvement_bps:+.3f} bps")
    if m.feature_importances:
        sorted_fi = sorted(
            m.feature_importances.items(), key=lambda x: x[1], reverse=True
        )
        logger.info(f"  Top features: {sorted_fi[:5]}")


def _print_fill_metrics(name: str, m: "FillModelMetrics") -> None:
    from scripts.v460.ml.fill_classifier import FillModelMetrics

    logger.info(f"  Samples:    {m.n_samples}")
    logger.info(f"  Fill rate:  {m.fill_rate:.1%}")
    logger.info(f"  ROC-AUC:    {m.roc_auc_mean:.4f} ± {m.roc_auc_std:.4f}")
    logger.info(f"  PR-AUC:     {m.pr_auc_mean:.4f} ± {m.pr_auc_std:.4f}")
    logger.info(f"  Brier:      {m.brier_mean:.4f} ± {m.brier_std:.4f}")
    if m.feature_importances:
        sorted_fi = sorted(
            m.feature_importances.items(), key=lambda x: x[1], reverse=True
        )
        logger.info(f"  Top features: {sorted_fi[:5]}")


def main() -> None:
    parser = argparse.ArgumentParser(description="057# ML Pipeline")
    parser.add_argument(
        "--output-dir",
        type=str,
        default="reports/v460/ml",
        help="Output directory for results",
    )
    parser.add_argument(
        "--results-dir",
        type=str,
        default=None,
        help="Fill records directory (default: results/v460/fill_test)",
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    results_dir = Path(args.results_dir) if args.results_dir else None

    df = load_fill_records(results_dir)
    logger.info(f"Total records: {len(df)}")

    # ML-1: AS Classifier
    as_results = run_as_pipeline(df, output_dir)

    print()

    # ML-2: Fill Classifier
    fill_results = run_fill_pipeline(df, output_dir)

    # Summary
    print("\n" + "=" * 60)
    print("057# ML Pipeline Summary")
    print("=" * 60)
    print(f"  AS Classifier best:   {as_results['best_model'].upper()}")
    as_best = as_results[as_results["best_model"]]
    print(f"    ROC-AUC: {as_best['roc_auc_mean']:.4f}, PR-AUC: {as_best['pr_auc_mean']:.4f}")
    print(f"    Skip20% PnL: {as_best['skip_top20_pnl_improvement_bps']:+.3f} bps")
    print(f"  Fill Classifier best: {fill_results['best_model'].upper()}")
    fill_best = fill_results[fill_results["best_model"]]
    print(f"    ROC-AUC: {fill_best['roc_auc_mean']:.4f}, PR-AUC: {fill_best['pr_auc_mean']:.4f}")


if __name__ == "__main__":
    main()
