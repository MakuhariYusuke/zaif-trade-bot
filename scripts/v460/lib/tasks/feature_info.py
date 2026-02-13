"""
G1 Feature Information Test — task implementation.

003# #16: run_experiment.py から分離.
003# #1: XGBoost vs Logistic/Ridge ペア化 (baseline ゼロベクトル廃止).
003# #2: direction=classifier, magnitude/volatility=regressor 自動切替.
003# #3: XGB パラメータ二重指定を _RESERVED_XGB_KEYS で防止.
"""

from __future__ import annotations

import logging
from typing import Any

from scripts.v460.lib.data_loader import generate_targets, load_parquet
from scripts.v460.lib.evaluator import (
    evaluate_multi_target,
    make_logistic,
    make_ridge,
    make_xgboost_classifier,
    make_xgboost_regressor,
)

logger = logging.getLogger(__name__)


def task_feature_info(cfg: dict) -> dict:
    """G1 feature information test — XGBoost walk-forward.

    003# #1: XGBoost fold signals を Logistic/Ridge fold signals とペアにして
    g1_judgment に渡す (baseline ゼロベクトル問題の解消).

    003# #2: direction → classifier, magnitude/volatility → regressor.

    007# F4: fold_results の保存を統計量のみに変更。
    debug=True 時のみ生配列を保存する。

    Returns:
        results dict with fold_results for g1_judgment.
    """
    data_cfg = cfg["data"]
    feat_cfg = cfg["features"]
    wf_cfg = cfg.get("walk_forward", {})
    xgb_cfg = cfg.get("xgboost", {})
    seed = cfg.get("seed", 42)

    # Load data
    data_path = data_cfg.get("v460_features_path") or data_cfg.get("ohlcv_path")
    feature_cols = feat_cfg["selected"]

    df = load_parquet(data_path, feature_cols)

    # Generate targets
    g1_cfg = cfg.get("g1", {})
    horizons = g1_cfg.get("horizons", [1, 5, 15])
    target_types = g1_cfg.get("targets", ["direction", "magnitude", "volatility"])

    df = generate_targets(df, horizons, target_types)

    # Ensure features are float32
    for col in feature_cols:
        df[col] = df[col].astype("float32")

    # Walk-forward params
    n_folds = wf_cfg.get("n_folds", 5)
    train_ratio = wf_cfg.get("train_ratio", 0.80)

    # --- XGBoost evaluation ---
    # #3: _RESERVED_XGB_KEYS in factory handles double-specification prevention
    def xgb_cls_factory() -> Any:
        return make_xgboost_classifier(seed=seed, **xgb_cfg)

    def xgb_reg_factory() -> Any:
        return make_xgboost_regressor(seed=seed, **xgb_cfg)

    multi_results = evaluate_multi_target(
        df, feature_cols, horizons, target_types,
        model_factory=xgb_cls_factory,
        model_name="XGBoost",
        n_folds=n_folds,
        train_ratio=train_ratio,
        regression_factory=xgb_reg_factory,
    )

    # --- Baseline evaluation ---
    # #1: Logistic (classification) / Ridge (regression) as real baseline
    baseline_results = evaluate_multi_target(
        df, feature_cols, horizons, target_types,
        model_factory=lambda: make_logistic(seed=seed),
        model_name="Logistic",
        n_folds=n_folds,
        train_ratio=train_ratio,
        regression_factory=lambda: make_ridge(seed=seed),
    )

    # --- Build fold_results for g1_judgment ---
    # #1: Pair XGBoost fold signal vs Logistic/Ridge fold signal
    fold_results: dict[str, list[tuple[list[float], list[float]]]] = {}
    for target_name, wf_result in multi_results.items():
        bl_result = baseline_results.get(target_name)
        fold_pairs: list[tuple[list[float], list[float]]] = []

        for i, xgb_fold in enumerate(wf_result.folds):
            # Get baseline fold signal (paired)
            if bl_result and i < len(bl_result.folds):
                bl_signal = bl_result.folds[i]._signal
            else:
                # Fallback: empty baseline (should not happen normally)
                bl_signal = [0.0] * len(xgb_fold._signal)
                logger.warning(
                    f"No baseline fold {i} for {target_name}, using zeros"
                )
            fold_pairs.append((xgb_fold._signal, bl_signal))

        fold_results[target_name] = fold_pairs

    # 007# F4: fold_results 統計量サマリ (生配列保存を回避 → ~142MB → ~数KB)
    debug_mode = cfg.get("debug", False)
    fold_results_for_save: dict[str, Any] = {}
    if debug_mode:
        # Debug: 生配列をそのまま保存 (Gate 判定再現用)
        fold_results_for_save = fold_results
    else:
        # Default: 統計量のみ保存 (n_samples, mean, std per fold)
        for tgt, pairs in fold_results.items():
            fold_stats = []
            for model_s, baseline_s in pairs:
                fold_stats.append({
                    "n_model": len(model_s),
                    "n_baseline": len(baseline_s),
                    "model_mean": float(np.mean(model_s)) if model_s else 0.0,
                    "model_std": float(np.std(model_s)) if model_s else 0.0,
                    "baseline_mean": float(np.mean(baseline_s)) if baseline_s else 0.0,
                    "baseline_std": float(np.std(baseline_s)) if baseline_s else 0.0,
                })
            fold_results_for_save[tgt] = fold_stats

    # Summary metrics
    summary: dict = {
        "xgboost": {k: v.to_dict() for k, v in multi_results.items()},
        "logistic": {k: v.to_dict() for k, v in baseline_results.items()},
        "fold_results": fold_results,  # Full data for g1_judgment (in-memory only)
        "fold_results_saved": fold_results_for_save,  # Slimmed for JSON serialization
    }

    return summary
