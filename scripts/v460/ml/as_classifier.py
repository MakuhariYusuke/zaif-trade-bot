"""057# AS Classifier: P(adverse_selection) を予測.

055# §6 ML-1 の実装。
fill records から特徴量を構築し、AS 発生確率を推定する。
高リスク注文のスキップ判定に使用。
"""

from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    classification_report,
    roc_auc_score,
)
from sklearn.model_selection import TimeSeriesSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)


@dataclass
class ASModelMetrics:
    """AS 分類器の評価指標."""

    n_samples: int = 0
    n_folds: int = 0
    roc_auc_mean: float = 0.0
    roc_auc_std: float = 0.0
    pr_auc_mean: float = 0.0
    pr_auc_std: float = 0.0
    brier_mean: float = 0.0
    brier_std: float = 0.0
    # スキップ閾値ごとの期待値改善
    skip_top20_pnl_improvement_bps: float = 0.0
    skip_top10_pnl_improvement_bps: float = 0.0
    # 特徴量重要度
    feature_importances: dict[str, float] | None = None
    # Baseline (naive)
    naive_pr_auc: float = 0.0
    improvement_over_naive: float = 0.0


def train_as_classifier(
    X: pd.DataFrame,
    y: pd.Series,
    pnl: pd.Series | None = None,
    *,
    n_splits: int = 5,
    model_type: str = "gb",
    n_features_select: int | None = None,
) -> tuple[ASModelMetrics, Any, Pipeline, np.ndarray]:
    """AS 分類器の学習と時系列 CV 評価.

    Args:
        X: 特徴量 DataFrame.
        y: ラベル (0/1).
        pnl: post_fill_30s_pnl (スキップ効果計算用, optional).
        n_splits: TimeSeriesSplit の fold 数.
        model_type: "lr" (LogisticRegression) or "gb" (GradientBoosting).
        n_features_select: 060# v2: SelectKBest(f_classif) で選択する特徴量数.
            None の場合は全特徴量を使用.

    Returns:
        (metrics, model, pipeline, oof_probs) タプル.
        pipeline は Imputer+Scaler+Model の完全 Pipeline.
    """
    tscv = TimeSeriesSplit(n_splits=n_splits)

    roc_aucs: list[float] = []
    pr_aucs: list[float] = []
    briers: list[float] = []
    oof_probs = np.full(len(X), np.nan)
    oof_indices: list[int] = []

    for fold, (train_idx, test_idx) in enumerate(tscv.split(X)):
        X_train = X.iloc[train_idx]
        y_train = y.iloc[train_idx]
        X_test = X.iloc[test_idx]
        y_test = y.iloc[test_idx]

        # 059# P0-1: Imputer+Scaler+Model を fold 内で fit (リーク防止)
        # 060# tuned: LR C=0.01, GB n=30/d=3/lr=0.05 (feature selection時)
        if model_type == "lr":
            # 060#: 強正則化 (C=0.01) — 高次元特徴量の過学習抑制
            c_val = 0.01 if n_features_select else 1.0
            clf = LogisticRegression(
                C=c_val, max_iter=2000, class_weight="balanced", random_state=42
            )
        else:
            # 060#: 小さい木 (n=30, lr=0.05) — 過学習防止
            n_est = 30 if n_features_select else 100
            lr_val = 0.05 if n_features_select else 0.1
            clf = GradientBoostingClassifier(
                n_estimators=n_est,
                max_depth=3,
                learning_rate=lr_val,
                subsample=0.8,
                random_state=42,
            )

        # 060# v2: Feature selection (CV内で特徴量選択 → リーク防止)
        k = min(n_features_select, X_train.shape[1]) if n_features_select else None
        steps: list[tuple[str, Any]] = [
            ("imputer", SimpleImputer(strategy="median")),
        ]
        if k is not None:
            steps.append(("selector", SelectKBest(f_classif, k=k)))
        steps.extend([
            ("scaler", StandardScaler()),
            ("model", clf),
        ])
        pipe = Pipeline(steps)
        pipe.fit(X_train, y_train)
        probs = pipe.predict_proba(X_test)[:, 1]

        # Metrics
        if len(np.unique(y_test)) > 1:
            roc_aucs.append(roc_auc_score(y_test, probs))
            pr_aucs.append(average_precision_score(y_test, probs))
        briers.append(brier_score_loss(y_test, probs))

        oof_probs[test_idx] = probs
        oof_indices.extend(test_idx.tolist())

    # Naive baseline (always predict mean)
    naive_pr_auc = float(y.mean())  # PR-AUC baseline for imbalanced

    # Final model on all data (for feature importance extraction only)
    if model_type == "lr":
        c_val = 0.01 if n_features_select else 1.0
        final_clf = LogisticRegression(
            C=c_val, max_iter=2000, class_weight="balanced", random_state=42
        )
    else:
        n_est = 30 if n_features_select else 100
        lr_val = 0.05 if n_features_select else 0.1
        final_clf = GradientBoostingClassifier(
            n_estimators=n_est,
            max_depth=3,
            learning_rate=lr_val,
            subsample=0.8,
            random_state=42,
        )
    k_final = min(n_features_select, X.shape[1]) if n_features_select else None
    final_steps: list[tuple[str, Any]] = [
        ("imputer", SimpleImputer(strategy="median")),
    ]
    if k_final is not None:
        final_steps.append(("selector", SelectKBest(f_classif, k=k_final)))
    final_steps.extend([
        ("scaler", StandardScaler()),
        ("model", final_clf),
    ])
    final_pipe = Pipeline(final_steps)
    final_pipe.fit(X, y)
    final_model = final_pipe.named_steps["model"]
    scaler_final = final_pipe  # Return Pipeline instead of bare scaler

    # Feature importances (selected features only if selection is active)
    if "selector" in final_pipe.named_steps:
        selector = final_pipe.named_steps["selector"]
        selected_mask = selector.get_support()
        selected_cols = X.columns[selected_mask].tolist()
    else:
        selected_cols = X.columns.tolist()

    if hasattr(final_model, "feature_importances_"):
        importances = dict(
            zip(selected_cols, final_model.feature_importances_.tolist())
        )
    elif hasattr(final_model, "coef_"):
        importances = dict(
            zip(selected_cols, np.abs(final_model.coef_[0]).tolist())
        )
    else:
        importances = None

    # Skip simulation on OOF predictions
    skip_20_improvement = 0.0
    skip_10_improvement = 0.0
    if pnl is not None:
        # 059# NEW-07: PnL の NaN もフィルタ
        valid_mask = ~np.isnan(oof_probs) & ~np.isnan(pnl.values)
        if valid_mask.sum() > 20:
            valid_probs = oof_probs[valid_mask]
            valid_pnl = pnl.values[valid_mask]
            valid_y = y.values[valid_mask]

            baseline_pnl = float(np.mean(valid_pnl))

            # Skip top 20% highest AS risk
            threshold_20 = np.percentile(valid_probs, 80)
            keep_mask_20 = valid_probs < threshold_20
            if keep_mask_20.sum() > 0:
                skip_20_improvement = float(np.mean(valid_pnl[keep_mask_20])) - baseline_pnl

            # Skip top 10% highest AS risk
            threshold_10 = np.percentile(valid_probs, 90)
            keep_mask_10 = valid_probs < threshold_10
            if keep_mask_10.sum() > 0:
                skip_10_improvement = float(np.mean(valid_pnl[keep_mask_10])) - baseline_pnl

    pr_auc_mean = float(np.mean(pr_aucs)) if pr_aucs else 0.0

    metrics = ASModelMetrics(
        n_samples=len(X),
        n_folds=n_splits,
        roc_auc_mean=float(np.mean(roc_aucs)) if roc_aucs else 0.0,
        roc_auc_std=float(np.std(roc_aucs)) if roc_aucs else 0.0,
        pr_auc_mean=pr_auc_mean,
        pr_auc_std=float(np.std(pr_aucs)) if pr_aucs else 0.0,
        brier_mean=float(np.mean(briers)),
        brier_std=float(np.std(briers)),
        skip_top20_pnl_improvement_bps=skip_20_improvement,
        skip_top10_pnl_improvement_bps=skip_10_improvement,
        feature_importances=importances,
        naive_pr_auc=naive_pr_auc,
        improvement_over_naive=pr_auc_mean - naive_pr_auc,
    )

    return metrics, final_model, scaler_final, oof_probs


def evaluate_skip_policy(
    X: pd.DataFrame,
    y: pd.Series,
    pnl: pd.Series,
    model: Any,
    scaler: Any,
    thresholds: list[float] | None = None,
    *,
    oof_probs: np.ndarray | None = None,
) -> pd.DataFrame:
    """スキップ閾値ごとの効果をシミュレーション.

    059# P0-3: OOF 予測のみで評価 (in-sample 評価廃止).

    Args:
        X, y, pnl: テストデータ.
        model, scaler: 学習済みモデル (OOF 非提供時のフォールバック用).
        thresholds: スキップ閾値のリスト.
        oof_probs: Out-of-Fold 予測確率 (推奨). None の場合 model で推論.

    Returns:
        閾値ごとの結果 DataFrame.
    """
    if thresholds is None:
        thresholds = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8]

    if oof_probs is not None:
        # OOF 予測を使用 (リークなし)
        valid_mask = ~np.isnan(oof_probs)
        probs = oof_probs[valid_mask]
        pnl_valid = pnl.values[valid_mask]
        y_valid = y.values[valid_mask]
        eval_source = "OOF"
    else:
        # フォールバック: model で推論 (in-sample, 非推奨)
        if hasattr(scaler, "transform"):
            X_scaled = scaler.transform(X)
        else:
            X_scaled = X.values
        probs = model.predict_proba(X_scaled)[:, 1]
        pnl_valid = pnl.values
        y_valid = y.values
        eval_source = "in-sample"
        logger.warning("evaluate_skip_policy: using in-sample (non-OOF) evaluation")

    results = []
    baseline_pnl = float(np.mean(pnl_valid))
    baseline_as_rate = float(np.mean(y_valid))

    for th in thresholds:
        keep = probs < th
        n_keep = int(keep.sum())
        n_skip = int((~keep).sum())
        skip_rate = n_skip / len(probs) if len(probs) > 0 else 0

        if n_keep > 0:
            kept_pnl = float(np.mean(pnl_valid[keep]))
            kept_as_rate = float(np.mean(y_valid[keep]))
        else:
            kept_pnl = 0.0
            kept_as_rate = 0.0

        results.append({
            "threshold": th,
            "n_keep": n_keep,
            "n_skip": n_skip,
            "skip_rate": skip_rate,
            "kept_pnl_bps": kept_pnl,
            "pnl_improvement_bps": kept_pnl - baseline_pnl,
            "kept_as_rate": kept_as_rate,
            "as_reduction": baseline_as_rate - kept_as_rate,
            "eval_source": eval_source,
        })

    return pd.DataFrame(results)
