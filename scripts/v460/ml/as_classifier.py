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
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    classification_report,
    roc_auc_score,
)
from sklearn.model_selection import TimeSeriesSplit
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
) -> tuple[ASModelMetrics, Any, StandardScaler]:
    """AS 分類器の学習と時系列 CV 評価.

    Args:
        X: 特徴量 DataFrame.
        y: ラベル (0/1).
        pnl: post_fill_30s_pnl (スキップ効果計算用, optional).
        n_splits: TimeSeriesSplit の fold 数.
        model_type: "lr" (LogisticRegression) or "gb" (GradientBoosting).

    Returns:
        (metrics, model, scaler) タプル.
    """
    tscv = TimeSeriesSplit(n_splits=n_splits)

    roc_aucs: list[float] = []
    pr_aucs: list[float] = []
    briers: list[float] = []
    oof_probs = np.full(len(X), np.nan)
    oof_indices: list[int] = []

    scaler = StandardScaler()

    for fold, (train_idx, test_idx) in enumerate(tscv.split(X)):
        X_train = X.iloc[train_idx]
        y_train = y.iloc[train_idx]
        X_test = X.iloc[test_idx]
        y_test = y.iloc[test_idx]

        # Scale
        X_train_s = scaler.fit_transform(X_train)
        X_test_s = scaler.transform(X_test)

        # Model
        if model_type == "lr":
            model = LogisticRegression(
                C=1.0, max_iter=1000, class_weight="balanced", random_state=42
            )
        else:
            model = GradientBoostingClassifier(
                n_estimators=100,
                max_depth=3,
                learning_rate=0.1,
                subsample=0.8,
                random_state=42,
            )

        model.fit(X_train_s, y_train)
        probs = model.predict_proba(X_test_s)[:, 1]

        # Metrics
        if len(np.unique(y_test)) > 1:
            roc_aucs.append(roc_auc_score(y_test, probs))
            pr_aucs.append(average_precision_score(y_test, probs))
        briers.append(brier_score_loss(y_test, probs))

        oof_probs[test_idx] = probs
        oof_indices.extend(test_idx.tolist())

    # Naive baseline (always predict mean)
    naive_pr_auc = float(y.mean())  # PR-AUC baseline for imbalanced

    # Final model on all data
    scaler_final = StandardScaler()
    X_scaled = scaler_final.fit_transform(X)

    if model_type == "lr":
        final_model = LogisticRegression(
            C=1.0, max_iter=1000, class_weight="balanced", random_state=42
        )
    else:
        final_model = GradientBoostingClassifier(
            n_estimators=100,
            max_depth=3,
            learning_rate=0.1,
            subsample=0.8,
            random_state=42,
        )
    final_model.fit(X_scaled, y)

    # Feature importances
    if hasattr(final_model, "feature_importances_"):
        importances = dict(
            zip(X.columns, final_model.feature_importances_.tolist())
        )
    elif hasattr(final_model, "coef_"):
        importances = dict(
            zip(X.columns, np.abs(final_model.coef_[0]).tolist())
        )
    else:
        importances = None

    # Skip simulation on OOF predictions
    skip_20_improvement = 0.0
    skip_10_improvement = 0.0
    if pnl is not None:
        valid_mask = ~np.isnan(oof_probs)
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

    return metrics, final_model, scaler_final


def evaluate_skip_policy(
    X: pd.DataFrame,
    y: pd.Series,
    pnl: pd.Series,
    model: Any,
    scaler: StandardScaler,
    thresholds: list[float] | None = None,
) -> pd.DataFrame:
    """スキップ閾値ごとの効果をシミュレーション.

    Args:
        X, y, pnl: テストデータ.
        model, scaler: 学習済みモデル.
        thresholds: スキップ閾値のリスト (AS確率がこれ以上ならスキップ).

    Returns:
        閾値ごとの結果 DataFrame.
    """
    if thresholds is None:
        thresholds = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8]

    X_scaled = scaler.transform(X)
    probs = model.predict_proba(X_scaled)[:, 1]

    results = []
    baseline_pnl = float(np.mean(pnl))
    baseline_as_rate = float(y.mean())

    for th in thresholds:
        keep = probs < th
        n_keep = int(keep.sum())
        n_skip = int((~keep).sum())
        skip_rate = n_skip / len(probs) if len(probs) > 0 else 0

        if n_keep > 0:
            kept_pnl = float(np.mean(pnl[keep]))
            kept_as_rate = float(y[keep].mean())
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
        })

    return pd.DataFrame(results)
