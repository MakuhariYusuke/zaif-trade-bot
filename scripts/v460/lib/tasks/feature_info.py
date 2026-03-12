"""
G1 Feature Information Test — task implementation.

003# #16: run_experiment.py から分離.
003# #1: XGBoost vs Logistic/Ridge ペア化 (baseline ゼロベクトル廃止).
003# #2: direction=classifier, magnitude/volatility=regressor 自動切替.
003# #3: XGB パラメータ二重指定を _RESERVED_XGB_KEYS で防止.
"""

from __future__ import annotations

import logging

import numpy as np

from scripts.v460.lib.config_access import as_bool, as_float, as_int, section
from scripts.v460.lib.data_loader import generate_targets, load_parquet
from scripts.v460.lib.evaluator import (
    FitPredictModel,
    evaluate_multi_target,
    make_logistic,
    make_ridge,
    make_xgboost_classifier,
    make_xgboost_regressor,
)
from ztb.types.common import ConfigSection

logger = logging.getLogger(__name__)


def task_feature_info(cfg: ConfigSection) -> dict[str, object]:
    """G1 feature information test — XGBoost walk-forward.

    003# #1: XGBoost fold signals を Logistic/Ridge fold signals とペアにして
    g1_judgment に渡す (baseline ゼロベクトル問題の解消).

    003# #2: direction → classifier, magnitude/volatility → regressor.

    007# F4: fold_results の保存を統計量のみに変更。
    debug=True 時のみ生配列を保存する。

    Returns:
        results dict with fold_results for g1_judgment.
    """
    data_cfg = section(cfg, "data")
    feat_cfg = section(cfg, "features")
    wf_cfg = section(cfg, "walk_forward")
    xgb_cfg = section(cfg, "xgboost")
    seed = cfg.get("seed", 42)
    seed_value = as_int(seed, 42)

    # Load data
    data_path = data_cfg.get("v460_features_path") or data_cfg.get("ohlcv_path")
    if not isinstance(data_path, str) or not data_path:
        raise ValueError("data.v460_features_path or data.ohlcv_path is required")
    selected_raw = feat_cfg.get("selected")
    if not isinstance(selected_raw, list) or not selected_raw:
        raise ValueError("features.selected must be a non-empty list")
    feature_cols = [str(col) for col in selected_raw]

    df = load_parquet(data_path, feature_cols)

    # Generate targets
    g1_cfg = section(cfg, "g1")
    horizons_raw = g1_cfg.get("horizons", [1, 5, 15])
    horizons = [int(h) for h in horizons_raw] if isinstance(horizons_raw, list) else [1, 5, 15]
    target_types_raw = g1_cfg.get("targets", ["direction", "magnitude", "volatility"])
    target_types = (
        [str(t) for t in target_types_raw]
        if isinstance(target_types_raw, list)
        else ["direction", "magnitude", "volatility"]
    )

    df = generate_targets(df, horizons, target_types)

    # Ensure features are float32
    for col in feature_cols:
        df[col] = df[col].astype("float32")

    # Walk-forward params
    n_folds = wf_cfg.get("n_folds", 5)
    train_ratio = wf_cfg.get("train_ratio", 0.80)
    n_folds_value = as_int(n_folds, 5)
    train_ratio_value = as_float(train_ratio, 0.80)
    xgb_kwargs: dict[str, object] = dict(xgb_cfg)

    # --- XGBoost evaluation ---
    # #3: _RESERVED_XGB_KEYS in factory handles double-specification prevention
    def xgb_cls_factory() -> FitPredictModel:
        return make_xgboost_classifier(seed=seed_value, **xgb_kwargs)

    def xgb_reg_factory() -> FitPredictModel:
        return make_xgboost_regressor(seed=seed_value, **xgb_kwargs)

    multi_results = evaluate_multi_target(
        df, feature_cols, horizons, target_types,
        model_factory=xgb_cls_factory,
        model_name="XGBoost",
        n_folds=n_folds_value,
        train_ratio=train_ratio_value,
        regression_factory=xgb_reg_factory,
    )

    # --- Baseline evaluation ---
    # #1: Logistic (classification) / Ridge (regression) as real baseline
    baseline_results = evaluate_multi_target(
        df, feature_cols, horizons, target_types,
        model_factory=lambda: make_logistic(seed=seed_value),
        model_name="Logistic",
        n_folds=n_folds_value,
        train_ratio=train_ratio_value,
        regression_factory=lambda: make_ridge(seed=seed_value),
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
    debug_raw = cfg.get("debug", False)
    debug_mode = as_bool(debug_raw, default=False)
    fold_results_for_save: dict[str, object] = {}
    if debug_mode:
        # Debug: gate判定に使う生配列をJSON互換形で保存
        for tgt, pairs in fold_results.items():
            fold_results_for_save[tgt] = [
                {"model_signal": model_s, "baseline_signal": baseline_s}
                for model_s, baseline_s in pairs
            ]
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
    summary: dict[str, object] = {
        "xgboost": {k: v.to_dict() for k, v in multi_results.items()},
        "logistic": {k: v.to_dict() for k, v in baseline_results.items()},
        "fold_results": fold_results,  # Full data for g1_judgment (in-memory only)
        "fold_results_saved": fold_results_for_save,  # Slimmed for JSON serialization
    }

    return summary
