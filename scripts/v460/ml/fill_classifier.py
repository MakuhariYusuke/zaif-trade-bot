"""057# Fill Probability Model: P(fill <= timeout) を予測.

055# §6 ML-2 の実装。
Fill/Timeout を予測し、offset の期待値最適化に使用。
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import cast

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

from scripts.v460.ml.data_loader import make_preprocessing_pipeline
from scripts.v460.ml.model_protocols import (
    FeatureTransformer,
    ProbabilisticEstimator,
)

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
) -> tuple[FillModelMetrics, ProbabilisticEstimator, FeatureTransformer]:
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
    X_values = X.to_numpy(dtype=np.float32, copy=False)
    y_values = y.to_numpy(copy=False)

    roc_aucs: list[float] = []
    pr_aucs: list[float] = []
    briers: list[float] = []

    def _make_model() -> ProbabilisticEstimator:
        if model_type == "lr":
            return LogisticRegression(
                C=1.0, max_iter=1000, class_weight="balanced", random_state=42
            )
        return GradientBoostingClassifier(
            n_estimators=80,
            max_depth=3,
            learning_rate=0.1,
            subsample=0.8,
            random_state=42,
        )

    for train_idx, test_idx in tscv.split(X_values):
        # 059# P0-1: Pipeline化 — 補完・スケーリングを fold 内で実施
        pipe = make_preprocessing_pipeline(_make_model())
        pipe.fit(X_values[train_idx], y_values[train_idx])

        y_test = y_values[test_idx]
        probs = pipe.predict_proba(X_values[test_idx])[:, 1]

        if len(np.unique(y_test)) > 1:
            roc_aucs.append(roc_auc_score(y_test, probs))
            pr_aucs.append(average_precision_score(y_test, probs))
        briers.append(brier_score_loss(y_test, probs))
        del pipe

    # Final model — 059# P0-1: Pipeline化
    final_pipe = make_preprocessing_pipeline(_make_model())
    final_pipe.fit(X_values, y_values)

    final_model = cast(ProbabilisticEstimator, final_pipe.named_steps["model"])
    scaler_final = cast(FeatureTransformer, final_pipe.named_steps["scaler"])

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
        fill_rate=float(np.mean(y_values)),
        feature_importances=importances,
    )

    return metrics, cast(ProbabilisticEstimator, final_pipe), scaler_final
