"""
v460 XGBoost Walk-Forward Evaluator.

K2 (scripts/v459/run_k2_nonrl_upper_bound.py) の walk_forward_eval() を
ライブラリ化。G1-info 判定の評価エンジン。

001# §6.4 準拠: K2 の walk_forward_eval() を evaluator.py にライブラリ化.
"""

from __future__ import annotations

import logging
import warnings
from dataclasses import dataclass, field
from typing import Any, Callable, Optional

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import accuracy_score, f1_score
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)
warnings.filterwarnings("ignore", category=FutureWarning)

# ---------------------------------------------------------------
# Model factories
# ---------------------------------------------------------------

def make_xgboost(
    n_estimators: int = 200,
    max_depth: int = 6,
    learning_rate: float = 0.05,
    subsample: float = 0.8,
    colsample_bytree: float = 0.8,
    seed: int = 42,
    **kwargs: Any,
) -> Any:
    """XGBoost classifier factory."""
    from xgboost import XGBClassifier
    return XGBClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        learning_rate=learning_rate,
        subsample=subsample,
        colsample_bytree=colsample_bytree,
        random_state=seed,
        eval_metric="logloss",
        verbosity=0,
        n_jobs=-1,
        **kwargs,
    )


def make_logistic(seed: int = 42) -> Any:
    """Logistic regression factory (baseline)."""
    from sklearn.linear_model import LogisticRegression
    return LogisticRegression(max_iter=500, C=1.0, solver="lbfgs", random_state=seed)


# ---------------------------------------------------------------
# Result types
# ---------------------------------------------------------------

@dataclass
class FoldResult:
    """Single fold evaluation result."""
    fold: int
    model_name: str
    train_size: int
    test_size: int
    accuracy: float
    f1_macro: float
    ic_spearman: float
    ic_pvalue: float
    ic_high_conf: Optional[float] = None
    n_high_conf: int = 0
    target_rate: float = 0.5

    # For p-mean / Holm: raw model & baseline scores
    model_scores: list[float] = field(default_factory=list)
    baseline_scores: list[float] = field(default_factory=list)


@dataclass
class WalkForwardResult:
    """Aggregated walk-forward result."""
    model_name: str
    target_name: str
    n_folds: int
    folds: list[FoldResult]
    accuracy_mean: float = 0.0
    accuracy_std: float = 0.0
    ic_mean: float = 0.0
    ic_std: float = 0.0
    ic_significant_count: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "model_name": self.model_name,
            "target_name": self.target_name,
            "n_folds": self.n_folds,
            "accuracy_mean": round(self.accuracy_mean, 6),
            "accuracy_std": round(self.accuracy_std, 6),
            "ic_mean": round(self.ic_mean, 6),
            "ic_std": round(self.ic_std, 6),
            "ic_significant_count": self.ic_significant_count,
            "folds": [
                {
                    "fold": f.fold, "accuracy": f.accuracy,
                    "ic_spearman": f.ic_spearman, "ic_pvalue": f.ic_pvalue,
                }
                for f in self.folds
            ],
        }


# ---------------------------------------------------------------
# Core evaluation
# ---------------------------------------------------------------

def walk_forward_eval(
    df: pd.DataFrame,
    feature_cols: list[str],
    target_col: str,
    model_factory: Callable[[], Any],
    model_name: str = "XGBoost",
    n_folds: int = 5,
    train_ratio: float = 0.80,
) -> WalkForwardResult:
    """Walk-forward N-fold evaluation.

    K2 walk_forward_eval のライブラリ版。Blocked time-split.

    Args:
        df: Full DataFrame with feature and target columns.
        feature_cols: Feature column names.
        target_col: Target column name.
        model_factory: Callable that returns a scikit-learn compatible model.
        model_name: Name for reporting.
        n_folds: Number of folds.
        train_ratio: Train proportion within each fold.

    Returns:
        WalkForwardResult with per-fold metrics.
    """
    n = len(df)
    fold_size = n // n_folds
    folds: list[FoldResult] = []

    for fold_i in range(n_folds):
        fold_start = fold_i * fold_size
        fold_end = min((fold_i + 1) * fold_size, n)
        fold_data = df.iloc[fold_start:fold_end]

        train_size = int(len(fold_data) * train_ratio)
        train = fold_data.iloc[:train_size]
        test = fold_data.iloc[train_size:]

        if len(test) < 100:
            continue

        X_train = train[feature_cols].values
        y_train = train[target_col].values
        X_test = test[feature_cols].values
        y_test = test[target_col].values

        # Scaling
        scaler = StandardScaler()
        X_train_s = scaler.fit_transform(X_train)
        X_test_s = scaler.transform(X_test)

        # Train
        model = model_factory()
        model.fit(X_train_s, y_train)

        # Predict
        y_pred = model.predict(X_test_s)

        # Probabilities (for IC)
        if hasattr(model, "predict_proba"):
            y_prob = model.predict_proba(X_test_s)[:, 1]
        else:
            y_prob = y_pred.astype(float)

        # Metrics
        acc = float(accuracy_score(y_test, y_pred))
        f1 = float(f1_score(y_test, y_pred, average="macro"))

        # IC: P(up) → [-1, 1] vs price_change
        signal = y_prob * 2 - 1
        price_changes = df["close"].iloc[fold_start:fold_end].iloc[train_size:].diff().values
        if len(price_changes) == len(signal):
            ic_result = stats.spearmanr(signal, price_changes)
            ic = float(ic_result.correlation) if not np.isnan(ic_result.correlation) else 0.0
            ic_p = float(ic_result.pvalue) if not np.isnan(ic_result.pvalue) else 1.0
        else:
            ic, ic_p = 0.0, 1.0

        # High-confidence IC
        high_conf_mask = np.abs(signal) > 0.3
        n_high = int(high_conf_mask.sum())
        ic_high = None
        if n_high > 50:
            hc_result = stats.spearmanr(signal[high_conf_mask], price_changes[high_conf_mask])
            ic_high = float(hc_result.correlation) if not np.isnan(hc_result.correlation) else None

        # Mann-Whitney scores for gate tests
        model_scores = signal.tolist()
        baseline_scores = np.zeros_like(signal).tolist()

        folds.append(FoldResult(
            fold=fold_i,
            model_name=model_name,
            train_size=len(train),
            test_size=len(test),
            accuracy=round(acc, 6),
            f1_macro=round(f1, 6),
            ic_spearman=round(ic, 6),
            ic_pvalue=round(ic_p, 6),
            ic_high_conf=round(ic_high, 6) if ic_high is not None else None,
            n_high_conf=n_high,
            target_rate=round(float(y_test.mean()), 4),
            model_scores=model_scores,
            baseline_scores=baseline_scores,
        ))
        logger.info(
            f"[{model_name}] fold={fold_i}: acc={acc:.4f} ic={ic:.6f} p={ic_p:.4f}"
        )

    # Aggregate
    result = WalkForwardResult(
        model_name=model_name,
        target_name=target_col,
        n_folds=len(folds),
        folds=folds,
    )
    if folds:
        accs = [f.accuracy for f in folds]
        ics = [f.ic_spearman for f in folds]
        result.accuracy_mean = round(float(np.mean(accs)), 6)
        result.accuracy_std = round(float(np.std(accs)), 6)
        result.ic_mean = round(float(np.mean(ics)), 6)
        result.ic_std = round(float(np.std(ics)), 6)
        result.ic_significant_count = sum(1 for f in folds if f.ic_pvalue < 0.05)

    return result


def evaluate_multi_target(
    df: pd.DataFrame,
    feature_cols: list[str],
    horizons: list[int],
    target_types: list[str],
    model_factory: Callable[[], Any],
    model_name: str = "XGBoost",
    n_folds: int = 5,
    train_ratio: float = 0.80,
) -> dict[str, WalkForwardResult]:
    """Run walk-forward for all horizon × target combinations.

    Args:
        df: DataFrame with target columns (target_{type}_h{horizon}).
        feature_cols: Feature column names.
        horizons: List of horizons (1, 5, 15).
        target_types: List of target types (direction, magnitude, volatility).
        model_factory: Model factory callable.
        model_name: Model name for reporting.
        n_folds: Number of folds.
        train_ratio: Train ratio within each fold.

    Returns:
        Dict of {target_name: WalkForwardResult}.
    """
    results: dict[str, WalkForwardResult] = {}

    for h in horizons:
        for ttype in target_types:
            target_col = f"target_{ttype}_h{h}"
            if target_col not in df.columns:
                logger.warning(f"Target column {target_col} not found, skipping")
                continue

            # Drop rows where target is NaN
            mask = df[target_col].notna()
            df_clean = df.loc[mask].reset_index(drop=True)

            # For classification targets, ensure int type
            if ttype == "direction":
                df_clean[target_col] = df_clean[target_col].astype(int)

            logger.info(f"Evaluating {target_col}: {len(df_clean)} rows")
            wf = walk_forward_eval(
                df_clean, feature_cols, target_col,
                model_factory, model_name, n_folds, train_ratio,
            )
            wf.target_name = target_col
            results[target_col] = wf

    return results
