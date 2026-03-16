"""061# Walk-Forward Validation for AS Classifier.

TimeSeriesSplit (固定 n_splits=5) に加え、
expanding-window walk-forward + embargo で AS 分類器のロバスト性を検証する。

Usage:
    python scripts/v460/ml/walk_forward_as.py [--output-dir reports/v460/ml_061]
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent.parent))

from scripts.v460.ml.cache_cleanup import clear_ml_data_caches_with_log
from scripts.v460.ml.data_loader import load_fill_records
from scripts.v460.ml.feature_enricher import (
    build_preorder_as_features,
    enrich_fill_records,
)
from ztb.io.json_io import write_json

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
)
logger = logging.getLogger(__name__)


def expanding_window_splits(
    n: int,
    *,
    min_train: int = 50,
    step: int = 20,
    embargo: int = 2,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Expanding-window walk-forward splitter with embargo.

    Args:
        n: 総サンプル数.
        min_train: 最小学習サンプル数.
        step: テストウィンドウサイズ (ステップ).
        embargo: Train/Test 間のギャップ (時間的リーク防止).

    Returns:
        List of (train_idx, test_idx) tuples.
    """
    splits: list[tuple[np.ndarray, np.ndarray]] = []
    idx = np.arange(n)

    start = min_train
    while start + embargo + step <= n:
        train_end = start
        test_start = start + embargo
        test_end = min(test_start + step, n)

        train_idx = idx[:train_end]
        test_idx = idx[test_start:test_end]

        if len(train_idx) >= min_train and len(test_idx) > 0:
            splits.append((train_idx, test_idx))

        start += step

    # 残りがあれば最終 fold に含める
    if splits and test_end < n:
        last_train, _ = splits[-1]
        final_test_start = test_end
        if final_test_start < n:
            final_train = idx[:test_end - embargo]
            final_test = idx[final_test_start:]
            if len(final_test) >= 5:
                splits.append((final_train, final_test))

    return splits


def run_walk_forward(
    X: pd.DataFrame,
    y: pd.Series,
    pnl: pd.Series | None = None,
    *,
    min_train: int = 50,
    step: int = 20,
    embargo: int = 2,
    k: int = 8,
) -> dict:
    """Walk-forward validation for AS LR classifier.

    Args:
        X: 特徴量 DataFrame.
        y: ラベル (0/1).
        pnl: PnL (bps), optional.
        min_train: 最小学習サンプル数.
        step: テストウィンドウサイズ.
        embargo: 学習/テスト間のギャップ.
        k: SelectKBest の k.

    Returns:
        Dict with per-fold and aggregated results.
    """
    splits = expanding_window_splits(
        len(X), min_train=min_train, step=step, embargo=embargo
    )
    X_values = X.to_numpy(dtype=np.float32, copy=False)
    y_values = y.to_numpy(copy=False)

    if not splits:
        logger.warning("Not enough data for walk-forward validation")
        return {"error": "insufficient_data"}

    logger.info(f"Walk-forward: {len(splits)} folds, "
                f"min_train={min_train}, step={step}, embargo={embargo}")

    fold_results: list[dict] = []
    oof_probs = np.full(len(X), np.nan)

    for fold_i, (train_idx, test_idx) in enumerate(splits):
        X_train = X_values[train_idx]
        y_train = y_values[train_idx]
        X_test = X_values[test_idx]
        y_test = y_values[test_idx]

        # 061# tuned: LR(C=0.01, l2, k=8)
        k_actual = min(k, X_values.shape[1])
        pipe = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("selector", SelectKBest(f_classif, k=k_actual)),
            ("scaler", StandardScaler()),
            ("model", LogisticRegression(
                C=0.01, max_iter=2000, class_weight="balanced", random_state=42
            )),
        ])
        pipe.fit(X_train, y_train)
        probs = pipe.predict_proba(X_test)[:, 1]
        oof_probs[test_idx] = probs

        fold_result: dict = {
            "fold": fold_i,
            "n_train": len(train_idx),
            "n_test": len(test_idx),
            "train_range": f"[0:{train_idx[-1]}]",
            "test_range": f"[{test_idx[0]}:{test_idx[-1]}]",
            "as_rate_train": float(y_train.mean()),
            "as_rate_test": float(y_test.mean()),
        }

        if len(np.unique(y_test)) > 1:
            fold_result["roc_auc"] = float(roc_auc_score(y_test, probs))
            fold_result["pr_auc"] = float(average_precision_score(y_test, probs))
        else:
            fold_result["roc_auc"] = None
            fold_result["pr_auc"] = None

        fold_result["brier"] = float(brier_score_loss(y_test, probs))

        # Selected features
        # SimpleImputer may drop all-NaN columns, so track surviving columns
        imputer = pipe.named_steps["imputer"]
        # imputer.statistics_ has NaN for dropped columns, finite for kept
        survived_mask = np.isfinite(imputer.statistics_)
        survived_cols = X.columns[survived_mask]
        selector = pipe.named_steps["selector"]
        selected = survived_cols[selector.get_support()].tolist()
        fold_result["selected_features"] = selected

        logger.info(
            f"  Fold {fold_i}: train={len(train_idx)}, test={len(test_idx)}, "
            f"ROC={fold_result['roc_auc']}, "
            f"features={selected[:3]}..."
        )

        fold_results.append(fold_result)

    # --- Aggregated metrics ---
    valid_rocs = [f["roc_auc"] for f in fold_results if f["roc_auc"] is not None]
    valid_prs = [f["pr_auc"] for f in fold_results if f["pr_auc"] is not None]
    briers = [f["brier"] for f in fold_results]

    agg: dict = {
        "n_folds": len(splits),
        "roc_auc_mean": float(np.mean(valid_rocs)) if valid_rocs else None,
        "roc_auc_std": float(np.std(valid_rocs)) if valid_rocs else None,
        "pr_auc_mean": float(np.mean(valid_prs)) if valid_prs else None,
        "brier_mean": float(np.mean(briers)),
    }

    # --- Skip simulation on OOF ---
    skip_sim: dict = {}
    if pnl is not None:
        valid_mask = ~np.isnan(oof_probs) & ~np.isnan(pnl.values)
        if valid_mask.sum() > 10:
            valid_probs = oof_probs[valid_mask]
            valid_pnl = pnl.values[valid_mask]
            baseline_pnl = float(np.mean(valid_pnl))

            for pct_label, pct_val in [("skip20", 80), ("skip10", 90)]:
                threshold = np.percentile(valid_probs, pct_val)
                keep = valid_probs < threshold
                if keep.sum() > 0:
                    kept_pnl = float(np.mean(valid_pnl[keep]))
                    improvement = kept_pnl - baseline_pnl
                else:
                    improvement = 0.0
                skip_sim[f"{pct_label}_improvement_bps"] = improvement

            skip_sim["baseline_pnl_bps"] = baseline_pnl
            skip_sim["n_valid"] = int(valid_mask.sum())

    # --- Feature stability ---
    all_selected = [set(f["selected_features"]) for f in fold_results]
    if all_selected:
        intersection = set.intersection(*all_selected)
        union = set.union(*all_selected)
        stability = len(intersection) / len(union) if union else 0.0
    else:
        intersection = set()
        union = set()
        stability = 0.0

    feature_stability: dict = {
        "jaccard_stability": stability,
        "always_selected": sorted(intersection),
        "ever_selected": sorted(union),
        "n_always": len(intersection),
        "n_ever": len(union),
    }

    return {
        "config": {
            "min_train": min_train,
            "step": step,
            "embargo": embargo,
            "k": k,
            "model": "LR(C=0.01, l2)",
        },
        "folds": fold_results,
        "aggregate": agg,
        "skip_simulation": skip_sim,
        "feature_stability": feature_stability,
    }


def main() -> None:
    try:
        _run_walk_forward_as_main()
    finally:
        clear_ml_data_caches_with_log(
            logger,
            context="walk_forward_as",
            collect_garbage=True,
        )


def _run_walk_forward_as_main() -> None:
    parser = argparse.ArgumentParser(description="061# Walk-Forward AS Validation")
    parser.add_argument(
        "--output-dir",
        type=str,
        default="reports/v460/ml_061",
        help="Output directory",
    )
    parser.add_argument("--min-train", type=int, default=50)
    parser.add_argument("--step", type=int, default=20)
    parser.add_argument("--embargo", type=int, default=2)
    parser.add_argument("--k", type=int, default=8)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)

    # --- Load data ---
    df = load_fill_records()
    enriched_df = enrich_fill_records(df)
    X, y = build_preorder_as_features(enriched_df)

    # PnL for skip simulation
    filled_mask = df["filled"].astype(bool) & df["adverse_selected_raw"].notna()
    pnl = df.loc[filled_mask, "post_fill_30s_pnl"].astype(float)
    pnl = pnl.reindex(X.index)

    logger.info(f"Data: {len(X)} samples, {X.shape[1]} features")

    # --- Run walk-forward ---
    results = run_walk_forward(
        X, y, pnl,
        min_train=args.min_train,
        step=args.step,
        embargo=args.embargo,
        k=args.k,
    )

    # --- Print summary ---
    agg = results.get("aggregate", {})
    skip = results.get("skip_simulation", {})
    feat = results.get("feature_stability", {})

    logger.info("\n" + "=" * 60)
    logger.info("Walk-Forward Results Summary")
    logger.info("=" * 60)
    logger.info(f"  Folds:           {agg.get('n_folds', 0)}")
    logger.info(f"  ROC-AUC (mean):  {agg.get('roc_auc_mean', 'N/A')}")
    logger.info(f"  ROC-AUC (std):   {agg.get('roc_auc_std', 'N/A')}")
    logger.info(f"  PR-AUC (mean):   {agg.get('pr_auc_mean', 'N/A')}")
    logger.info(f"  Brier (mean):    {agg.get('brier_mean', 'N/A')}")
    if skip:
        logger.info(f"  Skip20% →        {skip.get('skip20_improvement_bps', 0):+.3f} bps")
        logger.info(f"  Skip10% →        {skip.get('skip10_improvement_bps', 0):+.3f} bps")
        logger.info(f"  Baseline PnL:    {skip.get('baseline_pnl_bps', 0):.3f} bps")
    logger.info(f"  Feature stability (Jaccard): {feat.get('jaccard_stability', 0):.3f}")
    logger.info(f"  Always selected: {feat.get('always_selected', [])}")

    # TSCV vs Walk-Forward comparison header
    logger.info("\n--- Comparison: TSCV (060f) vs Walk-Forward (061) ---")
    logger.info(f"  WF ROC-AUC: {agg.get('roc_auc_mean', 'N/A')}")
    logger.info(f"  WF Skip20%: {skip.get('skip20_improvement_bps', 0):+.3f} bps")

    # --- Save ---
    output_dir.mkdir(parents=True, exist_ok=True)
    out_file = output_dir / "walk_forward_as_results.json"
    write_json(out_file, results, indent=2, ensure_ascii=False, default=str)
    logger.info(f"\nResults saved to {out_file}")


if __name__ == "__main__":
    main()
