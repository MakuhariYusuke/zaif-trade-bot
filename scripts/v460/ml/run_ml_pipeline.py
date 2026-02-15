"""058# Run ML Pipeline: AS分類器 + Fill分類器のオフライン評価.

使い方:
    python scripts/v460/ml/run_ml_pipeline.py [--output-dir reports/v460/ml]
    python scripts/v460/ml/run_ml_pipeline.py --enriched  # マイクロストラクチャ特徴量付き
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


def run_as_pipeline(
    df: "pd.DataFrame",
    output_dir: Path,
    *,
    enriched: bool = False,
    raw_dir: Path | None = None,
) -> dict:
    """AS 分類器パイプライン."""
    import pandas as pd

    logger.info("=" * 60)
    logger.info(f"ML-1: AS Classifier Pipeline {'(ENRICHED)' if enriched else '(baseline)'}")
    logger.info("=" * 60)

    if enriched:
        from scripts.v460.ml.feature_enricher import (
            build_enriched_as_features,
            enrich_fill_records,
        )
        enriched_df = enrich_fill_records(df, raw_dir=raw_dir)
        X, y = build_enriched_as_features(enriched_df)
    else:
        X, y = build_as_features(df)

    # PnL for skip simulation
    filled_mask = df["filled"].astype(bool) & df["adverse_selected_raw"].notna()
    pnl = df.loc[filled_mask, "post_fill_30s_pnl"].astype(float)
    pnl = pnl.reindex(X.index)

    results: dict = {}

    # --- GradientBoosting ---
    logger.info("\n--- GradientBoosting ---")
    gb_metrics, gb_model, gb_scaler, gb_oof = train_as_classifier(
        X, y, pnl, model_type="gb", n_splits=5
    )
    results["gb"] = asdict(gb_metrics)
    _print_as_metrics("GB", gb_metrics)

    # --- LogisticRegression ---
    logger.info("\n--- LogisticRegression ---")
    lr_metrics, lr_model, lr_scaler, lr_oof = train_as_classifier(
        X, y, pnl, model_type="lr", n_splits=5
    )
    results["lr"] = asdict(lr_metrics)
    _print_as_metrics("LR", lr_metrics)

    # --- Skip policy simulation (best model, OOF only — 059# P0-3) ---
    best = "gb" if gb_metrics.pr_auc_mean >= lr_metrics.pr_auc_mean else "lr"
    best_model = gb_model if best == "gb" else lr_model
    best_scaler = gb_scaler if best == "gb" else lr_scaler
    best_oof = gb_oof if best == "gb" else lr_oof
    logger.info(f"\nBest model: {best.upper()}")

    skip_df = evaluate_skip_policy(
        X, y, pnl, best_model, best_scaler, oof_probs=best_oof
    )
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


def run_pnl_pipeline(
    df: "pd.DataFrame",
    output_dir: Path,
    *,
    raw_dir: Path | None = None,
) -> dict:
    """ML-1b: PnL 回帰パイプライン.

    AS 分類の代わりに post_fill_30s_pnl (bps) を直接予測。
    目的変数が連続値 → Ridge / GradientBoostingRegressor.
    """
    import pandas as pd
    from sklearn.linear_model import Ridge
    from sklearn.ensemble import GradientBoostingRegressor
    from sklearn.impute import SimpleImputer
    from sklearn.model_selection import TimeSeriesSplit
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
    from scipy.stats import spearmanr
    import numpy as np

    from scripts.v460.ml.feature_enricher import (
        build_pnl_features,
        enrich_fill_records,
    )

    logger.info("=" * 60)
    logger.info("ML-1b: PnL Regressor Pipeline")
    logger.info("=" * 60)

    enriched_df = enrich_fill_records(df, raw_dir=raw_dir)
    X, y = build_pnl_features(enriched_df)

    if len(X) < 30:
        logger.warning(f"Too few PnL samples: {len(X)}, skipping")
        return {}

    tscv = TimeSeriesSplit(n_splits=5)
    results: dict = {}

    for model_name, make_model in [
        ("ridge", lambda: Ridge(alpha=10.0)),
        ("gbr", lambda: GradientBoostingRegressor(
            n_estimators=50, max_depth=2, learning_rate=0.05,
            subsample=0.8, random_state=42,
        )),
    ]:
        ics: list[float] = []
        maes: list[float] = []
        skip_improvements: list[float] = []
        oof_preds = np.full(len(X), np.nan)

        for fold, (train_idx, test_idx) in enumerate(tscv.split(X)):
            # 059# P0-1: Pipeline化 — 補完・スケーリングを fold 内で実施
            pipe = Pipeline([
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
                ("model", make_model()),
            ])
            pipe.fit(X.iloc[train_idx], y.iloc[train_idx].values)
            preds = pipe.predict(X.iloc[test_idx])
            y_te = y.iloc[test_idx].values

            oof_preds[test_idx] = preds
            mae = float(np.mean(np.abs(preds - y_te)))
            maes.append(mae)

            # Spearman IC (signal quality)
            if len(y_te) > 5:
                ic, _ = spearmanr(preds, y_te)
                if np.isnan(ic):
                    ic = 0.0
                ics.append(ic)

        # OOF skip simulation: skip predicted PnL < 0
        valid = ~np.isnan(oof_preds)
        if valid.sum() > 20:
            pnl_vals = y.values[valid]
            pred_vals = oof_preds[valid]
            baseline_pnl = float(np.mean(pnl_vals))

            # Skip predicted negative PnL
            keep_mask = pred_vals >= 0
            n_keep = int(keep_mask.sum())
            n_skip = int((~keep_mask).sum())
            if n_keep > 0:
                kept_pnl = float(np.mean(pnl_vals[keep_mask]))
                pnl_improvement = kept_pnl - baseline_pnl
            else:
                kept_pnl = 0.0
                pnl_improvement = 0.0
        else:
            n_keep, n_skip = len(X), 0
            baseline_pnl = float(y.mean())
            kept_pnl = baseline_pnl
            pnl_improvement = 0.0

        ic_mean = float(np.mean(ics)) if ics else 0.0
        mae_mean = float(np.mean(maes))

        # 059# P1-4: OOF有効件数を報告
        n_oof_valid = int(valid.sum())
        model_result = {
            "n_samples": len(X),
            "n_oof_valid": n_oof_valid,
            "ic_mean": ic_mean,
            "ic_std": float(np.std(ics)) if ics else 0.0,
            "mae_mean": mae_mean,
            "mae_std": float(np.std(maes)),
            "baseline_pnl_bps": baseline_pnl,
            "skip_neg_keep": n_keep,
            "skip_neg_skip": n_skip,
            "skip_neg_kept_pnl_bps": kept_pnl,
            "skip_neg_improvement_bps": pnl_improvement,
        }

        # Feature importances — 059# P0-1: Pipeline化
        final_pipe = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("model", make_model()),
        ])
        final_pipe.fit(X, y.values)
        final_model = final_pipe.named_steps["model"]

        if hasattr(final_model, "feature_importances_"):
            fi = dict(zip(X.columns, final_model.feature_importances_.tolist()))
        elif hasattr(final_model, "coef_"):
            fi = dict(zip(X.columns, np.abs(final_model.coef_).tolist()))
        else:
            fi = {}

        model_result["feature_importances"] = fi
        results[model_name] = model_result

        logger.info(f"\n--- {model_name.upper()} ---")
        logger.info(f"  Samples:       {len(X)} total, {n_oof_valid} OOF valid")
        logger.info(f"  IC (Spearman): {ic_mean:.4f}")
        logger.info(f"  MAE:           {mae_mean:.4f} bps")
        logger.info(f"  Baseline PnL:  {baseline_pnl:.4f} bps")
        logger.info(f"  Skip(PnL<0):   keep={n_keep}, skip={n_skip}")
        logger.info(f"  Kept PnL:      {kept_pnl:.4f} bps")
        logger.info(f"  Improvement:   {pnl_improvement:+.4f} bps")
        if fi:
            sorted_fi = sorted(fi.items(), key=lambda x: x[1], reverse=True)
            logger.info(f"  Top features:  {sorted_fi[:5]}")

    # Save
    out_file = output_dir / "pnl_regressor_results.json"
    out_file.parent.mkdir(parents=True, exist_ok=True)
    with open(out_file, "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False, default=str)
    logger.info(f"\nPnL results saved to {out_file}")

    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="058# ML Pipeline")
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
    parser.add_argument(
        "--enriched",
        action="store_true",
        help="Use microstructure-enriched features from raw data",
    )
    parser.add_argument(
        "--raw-dir",
        type=str,
        default=None,
        help="Raw data directory (default: data/v460/raw)",
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    results_dir = Path(args.results_dir) if args.results_dir else None
    raw_dir = Path(args.raw_dir) if args.raw_dir else None

    df = load_fill_records(results_dir)
    logger.info(f"Total records: {len(df)}")

    # ML-1: AS Classifier
    as_results = run_as_pipeline(
        df, output_dir, enriched=args.enriched, raw_dir=raw_dir
    )

    print()

    # ML-1b: PnL Regressor (enriched only)
    pnl_results: dict = {}
    if args.enriched:
        pnl_results = run_pnl_pipeline(df, output_dir, raw_dir=raw_dir)
        print()

    # ML-2: Fill Classifier
    fill_results = run_fill_pipeline(df, output_dir)

    # Summary
    print("\n" + "=" * 60)
    print(f"058# ML Pipeline Summary {'(ENRICHED)' if args.enriched else '(baseline)'}")
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
