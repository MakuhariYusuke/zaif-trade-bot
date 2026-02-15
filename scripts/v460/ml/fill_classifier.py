"""057# Fill Probability Model: P(fill <= timeout) を予測.

055# §6 ML-2 の実装。
Fill/Timeout を予測し、offset の期待値最適化に使用。
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    roc_auc_score,
)
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)


@dataclass
class FillModelMetrics:
    """Fill 分類器の評価指標."""

    n_samples: int = 0
    n_folds: int = 0
    roc_auc_mean: float = 0.0
    roc_auc_std: float = 0.0
    pr_auc_mean: float = 0.0
    pr_auc_std: float = 0.0
    brier_mean: float = 0.0
    brier_std: float = 0.0
    fill_rate: float = 0.0
    feature_importances: dict[str, float] | None = None


def train_fill_classifier(
    X: pd.DataFrame,
    y: pd.Series,
    *,
    n_splits: int = 5,
    model_type: str = "gb",
) -> tuple[FillModelMetrics, Any, StandardScaler]:
    """Fill/Timeout 分類器の学習と時系列 CV 評価.

    Args:
        X: 特徴量 DataFrame.
        y: ラベル (0=timeout, 1=filled).
        n_splits: TimeSeriesSplit の fold 数.
        model_type: "lr" or "gb".

    Returns:
        (metrics, model, scaler) タプル.
    """
    tscv = TimeSeriesSplit(n_splits=n_splits)

    roc_aucs: list[float] = []
    pr_aucs: list[float] = []
    briers: list[float] = []

    scaler = StandardScaler()

    for fold, (train_idx, test_idx) in enumerate(tscv.split(X)):
        X_train = X.iloc[train_idx]
        y_train = y.iloc[train_idx]
        X_test = X.iloc[test_idx]
        y_test = y.iloc[test_idx]

        X_train_s = scaler.fit_transform(X_train)
        X_test_s = scaler.transform(X_test)

        if model_type == "lr":
            model = LogisticRegression(
                C=1.0, max_iter=1000, class_weight="balanced", random_state=42
            )
        else:
            model = GradientBoostingClassifier(
                n_estimators=80,
                max_depth=3,
                learning_rate=0.1,
                subsample=0.8,
                random_state=42,
            )

        model.fit(X_train_s, y_train)
        probs = model.predict_proba(X_test_s)[:, 1]

        if len(np.unique(y_test)) > 1:
            roc_aucs.append(roc_auc_score(y_test, probs))
            pr_aucs.append(average_precision_score(y_test, probs))
        briers.append(brier_score_loss(y_test, probs))

    # Final model
    scaler_final = StandardScaler()
    X_scaled = scaler_final.fit_transform(X)

    if model_type == "lr":
        final_model = LogisticRegression(
            C=1.0, max_iter=1000, class_weight="balanced", random_state=42
        )
    else:
        final_model = GradientBoostingClassifier(
            n_estimators=80,
            max_depth=3,
            learning_rate=0.1,
            subsample=0.8,
            random_state=42,
        )
    final_model.fit(X_scaled, y)

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

    metrics = FillModelMetrics(
        n_samples=len(X),
        n_folds=n_splits,
        roc_auc_mean=float(np.mean(roc_aucs)) if roc_aucs else 0.0,
        roc_auc_std=float(np.std(roc_aucs)) if roc_aucs else 0.0,
        pr_auc_mean=float(np.mean(pr_aucs)) if pr_aucs else 0.0,
        pr_auc_std=float(np.std(pr_aucs)) if pr_aucs else 0.0,
        brier_mean=float(np.mean(briers)),
        brier_std=float(np.std(briers)),
        fill_rate=float(y.mean()),
        feature_importances=importances,
    )

    return metrics, final_model, scaler_final
