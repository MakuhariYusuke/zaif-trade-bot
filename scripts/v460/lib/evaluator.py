"""
v460 XGBoost Walk-Forward Evaluator.

K2 (scripts/v459/run_k2_nonrl_upper_bound.py) の walk_forward_eval() を
ライブラリ化。G1-info 判定の評価エンジン。

001# §6.4 準拠: K2 の walk_forward_eval() を evaluator.py にライブラリ化.

003# レビュー反映:
  #1: baseline をゼロベクトル → Logistic ペア化
  #2: magnitude/volatility に回帰器 (XGBRegressor) 分離
  #3: XGB パラメータ二重指定修正 (factory 側で _RESERVED_KEYS を除外)
  #17: fold_results に統計量のみ保持 (生配列廃止)
"""

from __future__ import annotations

import logging
import warnings
from dataclasses import dataclass, field
from typing import Callable, Optional, Protocol, cast

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import accuracy_score, f1_score, mean_absolute_error
from sklearn.preprocessing import StandardScaler
from ztb.types.common import JSONDict

logger = logging.getLogger(__name__)
warnings.filterwarnings("ignore", category=FutureWarning)

# Keys managed internally by factory — excluded from config pass-through
_RESERVED_XGB_KEYS = frozenset({
    "eval_metric", "n_jobs", "verbosity", "random_state", "seed",
})


Array = np.ndarray


class FitPredictModel(Protocol):
    """Minimal protocol for sklearn-like estimators used in evaluator."""

    def fit(self, X: Array, y: Array) -> object:
        ...

    def predict(self, X: Array) -> Array:
        ...


class ProbabilisticModel(FitPredictModel, Protocol):
    """Classifier protocol with probability output."""

    def predict_proba(self, X: Array) -> Array:
        ...


ModelFactory = Callable[[], FitPredictModel]


# ---------------------------------------------------------------
# Model factories
# ---------------------------------------------------------------

def make_xgboost_classifier(
    seed: int = 42,
    **kwargs: object,
) -> FitPredictModel:
    """XGBoost classifier factory (direction targets)."""
    from xgboost import XGBClassifier
    # Filter out reserved keys to avoid TypeError from double specification
    filtered: dict[str, object] = {
        k: v for k, v in kwargs.items() if k not in _RESERVED_XGB_KEYS
    }
    return XGBClassifier(
        n_estimators=filtered.pop("n_estimators", 200),
        max_depth=filtered.pop("max_depth", 6),
        learning_rate=filtered.pop("learning_rate", 0.05),
        subsample=filtered.pop("subsample", 0.8),
        colsample_bytree=filtered.pop("colsample_bytree", 0.8),
        random_state=seed,
        eval_metric="logloss",
        verbosity=0,
        n_jobs=-1,
        **filtered,
    )


def make_xgboost_regressor(
    seed: int = 42,
    **kwargs: object,
) -> FitPredictModel:
    """XGBoost regressor factory (magnitude/volatility targets)."""
    from xgboost import XGBRegressor
    filtered: dict[str, object] = {
        k: v for k, v in kwargs.items() if k not in _RESERVED_XGB_KEYS
    }
    return XGBRegressor(
        n_estimators=filtered.pop("n_estimators", 200),
        max_depth=filtered.pop("max_depth", 6),
        learning_rate=filtered.pop("learning_rate", 0.05),
        subsample=filtered.pop("subsample", 0.8),
        colsample_bytree=filtered.pop("colsample_bytree", 0.8),
        random_state=seed,
        eval_metric="rmse",
        verbosity=0,
        n_jobs=-1,
        **filtered,
    )


# Backward compat alias
def make_xgboost(seed: int = 42, **kwargs: object) -> FitPredictModel:
    """Alias for make_xgboost_classifier (backward compat)."""
    return make_xgboost_classifier(seed=seed, **kwargs)


def make_logistic(seed: int = 42) -> FitPredictModel:
    """Logistic regression factory (baseline for classification)."""
    from sklearn.linear_model import LogisticRegression
    return LogisticRegression(max_iter=500, C=1.0, solver="lbfgs", random_state=seed)


def make_ridge(seed: int = 42) -> FitPredictModel:
    """Ridge regression factory (baseline for regression)."""
    from sklearn.linear_model import Ridge
    return Ridge(alpha=1.0, random_state=seed)


def _is_classification(target_type: str) -> bool:
    """Determine if a target type requires classification or regression."""
    return target_type == "direction"


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
    mae: Optional[float] = None  # For regression targets
    is_classification: bool = True
    # Transient: per-fold signal for g1_judgment pairing.
    # Not serialized in to_dict() — only used in-memory.
    _signal: list[float] = field(default_factory=list, repr=False)


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

    def to_dict(self) -> JSONDict:
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
                    "mae": f.mae,
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
    model_factory: ModelFactory,
    model_name: str = "XGBoost",
    n_folds: int = 5,
    train_ratio: float = 0.80,
    is_classification: bool = True,
) -> WalkForwardResult:
    """Walk-forward N-fold evaluation.

    K2 walk_forward_eval のライブラリ版。Blocked time-split.
    分類/回帰の両方に対応。

    Args:
        df: Full DataFrame with feature and target columns.
        feature_cols: Feature column names.
        target_col: Target column name.
        model_factory: Callable that returns a scikit-learn compatible model.
        model_name: Name for reporting.
        n_folds: Number of folds.
        train_ratio: Train proportion within each fold.
        is_classification: True for classification, False for regression.

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
        y_pred = np.asarray(model.predict(X_test_s))

        # Signal for IC calculation
        if is_classification and hasattr(model, "predict_proba"):
            proba_model = cast(ProbabilisticModel, model)
            y_prob = np.asarray(proba_model.predict_proba(X_test_s))[:, 1]
            signal = y_prob * 2 - 1  # [0,1] → [-1,1]
        else:
            # Regression: predictions are the signal directly
            signal = np.asarray(y_pred, dtype=float)

        # Metrics
        if is_classification:
            acc = float(accuracy_score(y_test, y_pred))
            f1 = float(f1_score(y_test, y_pred, average="macro"))
            mae = None
        else:
            # Regression: accuracy = direction agreement
            acc = float(np.mean(np.sign(y_pred) == np.sign(y_test)))
            f1 = 0.0
            mae = float(mean_absolute_error(y_test, y_pred))

        # IC: Spearman rank correlation between signal and actual target.
        # Classification: signal (continuous [-1,1]) vs binary y_test → rank biserial.
        # Regression: predicted vs actual forward return → standard IC definition.
        # BUG FIX: 旧コードは df["close"].diff() を使用 → 先頭 NaN が spearmanr を
        # 完全に NaN 化していた。y_test を直接使用することで修正。
        ic_target = y_test.astype(float)
        if len(ic_target) >= 10:
            ic_result = stats.spearmanr(signal, ic_target, nan_policy="omit")
            ic = float(ic_result.correlation) if not np.isnan(ic_result.correlation) else 0.0
            ic_p = float(ic_result.pvalue) if not np.isnan(ic_result.pvalue) else 1.0
        else:
            ic, ic_p = 0.0, 1.0

        # High-confidence IC
        high_conf_mask = np.abs(signal) > 0.3
        n_high = int(high_conf_mask.sum())
        ic_high = None
        if n_high > 50:
            hc_result = stats.spearmanr(
                signal[high_conf_mask], ic_target[high_conf_mask],
                nan_policy="omit",
            )
            ic_high = float(hc_result.correlation) if not np.isnan(hc_result.correlation) else None

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
            mae=round(mae, 6) if mae is not None else None,
            is_classification=is_classification,
            _signal=signal.tolist(),
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
    model_factory: ModelFactory,
    model_name: str = "XGBoost",
    n_folds: int = 5,
    train_ratio: float = 0.80,
    regression_factory: Optional[ModelFactory] = None,
) -> dict[str, WalkForwardResult]:
    """Run walk-forward for all horizon × target combinations.

    分類 (direction) と回帰 (magnitude/volatility) を自動切替。

    Args:
        df: DataFrame with target columns (target_{type}_h{horizon}).
        feature_cols: Feature column names.
        horizons: List of horizons (1, 5, 15).
        target_types: List of target types (direction, magnitude, volatility).
        model_factory: Classifier factory (for direction targets).
        model_name: Model name for reporting.
        n_folds: Number of folds.
        train_ratio: Train ratio within each fold.
        regression_factory: Regressor factory (for magnitude/volatility).
            If None, uses model_factory for all targets.

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

            is_cls = _is_classification(ttype)

            # Select appropriate factory
            if is_cls:
                factory = model_factory
                df_clean[target_col] = df_clean[target_col].astype(int)
            else:
                factory = regression_factory or model_factory

            logger.info(
                f"Evaluating {target_col}: {len(df_clean)} rows "
                f"({'classification' if is_cls else 'regression'})"
            )
            wf = walk_forward_eval(
                df_clean, feature_cols, target_col,
                factory, model_name, n_folds, train_ratio,
                is_classification=is_cls,
            )
            wf.target_name = target_col
            results[target_col] = wf

    return results
