"""124# SkipGate v3: 多角的モデル探索パイプライン.

LR 線形制約 + AS30 ターゲットを突破し、より儲かるモデルを追求。

探索軸:
  1. 非線形モデル (LightGBM, XGBoost, RandomForest)
  2. ターゲット変数再設計 (PnL120 > 0, PnL 回帰)
  3. 逆転 SG (逆選別を逆手に取る)
  4. 特徴量工学 (インタラクション、ローリング統計)
  5. Dual-horizon 評価 (PnL30 + PnL120 両方で改善計測)

Usage:
    .venv\\Scripts\\python.exe scripts/v460/ml/train_sg_v3.py
"""

from __future__ import annotations

import logging
import sys
import warnings
import gc
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import cast

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import roc_auc_score

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent.parent))

from scripts.v460.ml.data_loader import load_fill_records
from scripts.v460.ml.feature_enricher import (
    build_preorder_as_features,
    enrich_fill_records,
)
from scripts.v460.ml.model_protocols import ProbabilisticEstimator, RegressorEstimator
from scripts.v460.ml.walk_forward_as import expanding_window_splits
from ztb.io.json_io import write_json

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
)
logger = logging.getLogger(__name__)

REPORT_DIR = Path("reports/v460/ml_124")
MODEL_DIR = Path("models/v460")


# ============================================================
# 1. 特徴量工学
# ============================================================

def engineer_features(
    enriched_df: pd.DataFrame,
    X_base: pd.DataFrame,
    y_base: pd.Series,
) -> pd.DataFrame:
    """既存特徴量にインタラクション・ローリング統計を追加.

    追加する特徴量:
      - hour × spread interaction
      - hour × side interaction
      - regime × spread interaction
      - spread_percentile (過去 window 内でのスプレッド位置)
      - recent_as_rate (直近 N fill の AS 率)
      - recent_pnl_mean (直近 N fill の平均 PnL)
      - time_since_last_fill (前回 fill からの経過秒)
      - cycle_index_in_run (run 内での何番目のサイクルか)
    """
    X = X_base.copy()
    idx = X.index

    # --- インタラクション特徴量 ---
    if "hour_cos" in X.columns and "spread_jpy" in X.columns:
        X["hour_x_spread"] = X["hour_cos"] * X["spread_jpy"]
    if "hour_cos" in X.columns and "side_buy" in X.columns:
        X["hour_x_side"] = X["hour_cos"] * X["side_buy"]
    if "spread_jpy" in X.columns and "side_buy" in X.columns:
        X["spread_x_side"] = X["spread_jpy"] * X["side_buy"]

    # regime × spread
    for reg in ["regime_trending", "regime_ranging", "regime_high_vol"]:
        if reg in X.columns and "spread_jpy" in X.columns:
            X[f"{reg}_x_spread"] = X[reg] * X["spread_jpy"]

    # --- ローリング統計 (時系列順を前提) ---
    # 直近 N fill の AS 率・平均 PnL
    if "adverse_selected_raw" in enriched_df.columns:
        as_raw = enriched_df.loc[idx, "adverse_selected_raw"].astype(float)
        for w in [10, 30]:
            X[f"recent_as_rate_{w}"] = as_raw.rolling(w, min_periods=3).mean().shift(1)

    if "post_fill_30s_pnl" in enriched_df.columns:
        pnl30 = enriched_df.loc[idx, "post_fill_30s_pnl"].astype(float)
        for w in [10, 30]:
            X[f"recent_pnl30_mean_{w}"] = pnl30.rolling(w, min_periods=3).mean().shift(1)

    if "post_fill_120s_pnl" in enriched_df.columns:
        pnl120 = enriched_df.loc[idx, "post_fill_120s_pnl"].astype(float)
        for w in [10, 30]:
            X[f"recent_pnl120_mean_{w}"] = pnl120.rolling(w, min_periods=3).mean().shift(1)

    # spread percentile (過去 50 件でのランク)
    if "spread_jpy" in X.columns:
        X["spread_percentile"] = (
            X["spread_jpy"]
            .rolling(50, min_periods=10)
            .rank(pct=True)
            .shift(1)
        )

    # time_since_last_fill
    if "timestamp" in enriched_df.columns:
        ts = enriched_df.loc[idx, "timestamp"].astype(float)
        X["time_since_last_fill"] = ts.diff().clip(upper=7200)  # cap at 2h

    # VPIN × side interaction
    if "vpin_60s" in X.columns and "side_buy" in X.columns:
        X["vpin_x_side"] = X["vpin_60s"] * X["side_buy"]

    logger.info(f"Engineered features: {X_base.shape[1]} → {X.shape[1]} columns")
    return X


# ============================================================
# 2. ターゲット変数
# ============================================================

def build_targets(
    enriched_df: pd.DataFrame,
    X_index: pd.Index,
) -> dict[str, pd.Series]:
    """複数のターゲット変数を構築."""
    targets: dict[str, pd.Series] = {}

    # T1: AS30 (現行: pnl30 < 0)
    if "adverse_selected_raw" in enriched_df.columns:
        targets["as30"] = enriched_df.loc[X_index, "adverse_selected_raw"].astype(int)

    # T2: PnL30 profitable (pnl30 > 0)
    if "post_fill_30s_pnl" in enriched_df.columns:
        pnl30 = enriched_df.loc[X_index, "post_fill_30s_pnl"].astype(float)
        targets["profitable30"] = (pnl30 > 0).astype(int)
        targets["pnl30_regression"] = pnl30

    # T3: PnL120 profitable (pnl120 > 0) — THE ACTUAL EDGE
    if "post_fill_120s_pnl" in enriched_df.columns:
        pnl120 = enriched_df.loc[X_index, "post_fill_120s_pnl"].astype(float)
        valid_120 = pnl120.notna()
        if valid_120.sum() > 100:
            targets["profitable120"] = (pnl120 > 0).astype(int)
            targets["pnl120_regression"] = pnl120

    # T4: Multi-horizon consensus (30s good AND 120s good)
    if "profitable30" in targets and "profitable120" in targets:
        targets["profitable_both"] = (
            (targets["profitable30"] == 1) & (targets["profitable120"] == 1)
        ).astype(int)

    # T5: Really bad trade (pnl30 < -1.0 bps) — avoid only the worst
    if "post_fill_30s_pnl" in enriched_df.columns:
        pnl30 = enriched_df.loc[X_index, "post_fill_30s_pnl"].astype(float)
        targets["really_bad30"] = (pnl30 < -1.0).astype(int)

    for name, t in targets.items():
        if hasattr(t, "mean"):
            rate = t.mean()
            logger.info(f"  Target '{name}': mean={rate:.3f}, n={len(t)}")

    return targets


# ============================================================
# 3. モデル定義
# ============================================================

def get_models() -> dict[str, object]:
    """評価するモデル候補を返す."""
    models: dict[str, object] = {}

    # M1: 現行 LR (baseline)
    models["LR_C001"] = LogisticRegression(
        C=0.01, max_iter=2000, class_weight="balanced", random_state=42,
    )
    # M2: LR with higher C (less regularization)
    models["LR_C01"] = LogisticRegression(
        C=0.1, max_iter=2000, class_weight="balanced", random_state=42,
    )
    # M3: LightGBM
    try:
        import lightgbm as lgb
        models["LGBM"] = lgb.LGBMClassifier(
            n_estimators=200,
            max_depth=4,
            learning_rate=0.05,
            num_leaves=15,
            min_child_samples=20,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_alpha=1.0,
            reg_lambda=1.0,
            class_weight="balanced",
            random_state=42,
            verbose=-1,
            n_jobs=1,
        )
        # M3b: Conservative LGBM (fewer trees, more regularization)
        models["LGBM_conservative"] = lgb.LGBMClassifier(
            n_estimators=80,
            max_depth=3,
            learning_rate=0.03,
            num_leaves=8,
            min_child_samples=30,
            subsample=0.7,
            colsample_bytree=0.7,
            reg_alpha=2.0,
            reg_lambda=2.0,
            class_weight="balanced",
            random_state=42,
            verbose=-1,
            n_jobs=1,
        )
    except ImportError:
        logger.warning("LightGBM not available")

    # M4: XGBoost
    try:
        import xgboost as xgb
        models["XGB"] = xgb.XGBClassifier(
            n_estimators=200,
            max_depth=4,
            learning_rate=0.05,
            min_child_weight=20,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_alpha=1.0,
            reg_lambda=1.0,
            scale_pos_weight=1.0,  # will adjust per fold
            random_state=42,
            eval_metric="logloss",
            verbosity=0,
            n_jobs=1,
        )
    except ImportError:
        logger.warning("XGBoost not available")

    # M5: Random Forest
    models["RF"] = RandomForestClassifier(
        n_estimators=200,
        max_depth=5,
        min_samples_leaf=20,
        class_weight="balanced",
        random_state=42,
        n_jobs=1,
    )

    # M6: Gradient Boosting (sklearn)
    models["GBM_sklearn"] = GradientBoostingClassifier(
        n_estimators=150,
        max_depth=3,
        learning_rate=0.05,
        min_samples_leaf=20,
        subsample=0.8,
        random_state=42,
    )

    return models


def get_regression_models() -> dict[str, object]:
    """回帰モデル候補."""
    models: dict[str, object] = {}

    models["Ridge"] = Ridge(alpha=10.0)

    try:
        import lightgbm as lgb
        models["LGBM_reg"] = lgb.LGBMRegressor(
            n_estimators=150,
            max_depth=4,
            learning_rate=0.05,
            num_leaves=15,
            min_child_samples=20,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_alpha=1.0,
            reg_lambda=1.0,
            random_state=42,
            verbose=-1,
            n_jobs=1,
        )
    except ImportError:
        pass

    return models


# ============================================================
# 4. Walk-Forward 評価 (Dual Horizon)
# ============================================================

@dataclass
class WFResult:
    """Walk-Forward 結果."""
    experiment: str
    model_name: str
    target_name: str
    feature_set: str
    n_samples: int
    n_folds: int
    auc_mean: float | None
    auc_std: float | None
    # PnL30 improvement when skipping top 20% AS-probability
    skip20_pnl30_improvement: float
    skip10_pnl30_improvement: float
    # PnL120 improvement (the real profit metric)
    skip20_pnl120_improvement: float
    skip10_pnl120_improvement: float
    # Reverse selection
    reverse_30: bool
    reverse_120: bool
    # Baseline
    baseline_pnl30: float
    baseline_pnl120: float
    # Inverted (skip LOWEST probability instead)
    inv_skip20_pnl30_improvement: float
    inv_skip20_pnl120_improvement: float
    # Net profit score: weighted combination
    profit_score: float
    notes: str = ""

    def to_dict(self) -> dict:
        return {k: v for k, v in self.__dict__.items()}


def _dual_horizon_skip_sim(
    probs: np.ndarray,
    pnl30: np.ndarray,
    pnl120: np.ndarray,
    *,
    invert: bool = False,
) -> dict[str, float]:
    """PnL30 + PnL120 の dual-horizon skip simulation.

    Args:
        probs: 予測確率 (高い = skip 候補)
        pnl30: PnL30s (bps)
        pnl120: PnL120s (bps)
        invert: True の場合、低確率を skip (逆転 SG)
    """
    valid = ~np.isnan(probs) & ~np.isnan(pnl30)
    if valid.sum() < 20:
        return {"skip20_pnl30": 0, "skip10_pnl30": 0,
                "skip20_pnl120": 0, "skip10_pnl120": 0,
                "baseline_pnl30": 0, "baseline_pnl120": 0}

    p = probs[valid]
    p30 = pnl30[valid]
    p120_all = pnl120[valid]

    baseline_30 = _safe_nanmean(p30)
    baseline_120 = _safe_nanmean(p120_all)

    result: dict[str, float] = {
        "baseline_pnl30": baseline_30,
        "baseline_pnl120": baseline_120,
    }

    for label, pct in [("skip20", 80), ("skip10", 90)]:
        if invert:
            # Skip LOWEST probability → keep high probability
            threshold = np.percentile(p, 100 - pct)
            keep = p >= threshold
        else:
            # Skip HIGHEST probability → keep low probability (standard)
            threshold = np.percentile(p, pct)
            keep = p < threshold

        if keep.sum() > 0:
            kept_30 = _safe_nanmean(p30[keep])
            kept_120 = _safe_nanmean(p120_all[keep])
            result[f"{label}_pnl30"] = kept_30 - baseline_30
            result[f"{label}_pnl120"] = kept_120 - baseline_120
        else:
            result[f"{label}_pnl30"] = 0.0
            result[f"{label}_pnl120"] = 0.0

    return result


def _safe_nanmean(values: np.ndarray) -> float:
    """NaN/inf のみの場合でも 0.0 を返す安全 mean."""
    if values.size == 0:
        return 0.0
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return 0.0
    return float(np.mean(finite))


def run_classification_wf(
    X: pd.DataFrame,
    y: pd.Series,
    pnl30: pd.Series,
    pnl120: pd.Series,
    model: object,
    *,
    experiment: str,
    model_name: str,
    target_name: str,
    feature_set: str,
    min_train: int = 50,
    step: int = 25,
    embargo: int = 2,
) -> WFResult:
    """分類モデルの Walk-Forward 評価."""
    # y と pnl のインデックスを同期
    common_idx = y.dropna().index.intersection(pnl30.dropna().index)
    X_wf = X.loc[common_idx].to_numpy(dtype=np.float32, copy=False)
    y_wf = y.loc[common_idx].to_numpy(copy=False)
    pnl30_wf = pnl30.loc[common_idx]
    pnl120_wf = pnl120.reindex(common_idx)

    splits = expanding_window_splits(
        len(X_wf), min_train=min_train, step=step, embargo=embargo,
    )
    if not splits:
        return WFResult(
            experiment=experiment, model_name=model_name,
            target_name=target_name, feature_set=feature_set,
            n_samples=len(X_wf), n_folds=0,
            auc_mean=None, auc_std=None,
            skip20_pnl30_improvement=0, skip10_pnl30_improvement=0,
            skip20_pnl120_improvement=0, skip10_pnl120_improvement=0,
            reverse_30=True, reverse_120=True,
            baseline_pnl30=0, baseline_pnl120=0,
            inv_skip20_pnl30_improvement=0, inv_skip20_pnl120_improvement=0,
            profit_score=-999, notes="insufficient_data",
        )

    oof_probs = np.full(len(X_wf), np.nan)
    fold_aucs: list[float] = []

    for fold_i, (train_idx, test_idx) in enumerate(splits):
        X_tr = X_wf[train_idx]
        y_tr = y_wf[train_idx]
        X_te = X_wf[test_idx]
        y_te = y_wf[test_idx]

        # NaN 処理
        imputer = SimpleImputer(strategy="median")
        X_tr_imp = imputer.fit_transform(X_tr)
        X_te_imp = imputer.transform(X_te)

        try:
            m = cast(ProbabilisticEstimator, clone(model))
            m.fit(X_tr_imp, y_tr)
            probs = m.predict_proba(X_te_imp)[:, 1]
            oof_probs[test_idx] = probs

            if len(np.unique(y_te)) > 1:
                fold_aucs.append(float(roc_auc_score(y_te, probs)))
        except Exception as e:
            logger.warning(f"  Fold {fold_i} failed: {e}")
            continue
        finally:
            del X_tr_imp, X_te_imp

    # Skip simulation (dual horizon)
    pnl30_arr = pnl30_wf.values.astype(float)
    pnl120_arr = pnl120_wf.values.astype(float)

    sim = _dual_horizon_skip_sim(oof_probs, pnl30_arr, pnl120_arr, invert=False)
    sim_inv = _dual_horizon_skip_sim(oof_probs, pnl30_arr, pnl120_arr, invert=True)

    auc_mean = float(np.mean(fold_aucs)) if fold_aucs else None
    auc_std = float(np.std(fold_aucs)) if fold_aucs else None

    # Profit score: weighted combination favoring PnL120 improvement
    # PnL120 is where the real money is
    s20_30 = sim.get("skip20_pnl30", 0.0)
    s20_120 = sim.get("skip20_pnl120", 0.0)
    inv_s20_30 = sim_inv.get("skip20_pnl30", 0.0)
    inv_s20_120 = sim_inv.get("skip20_pnl120", 0.0)

    # Take best of normal/inverted
    best_30 = max(s20_30, inv_s20_30)
    best_120 = max(s20_120, inv_s20_120)
    profit_score = best_30 * 0.3 + best_120 * 0.7  # 120s weighted higher

    return WFResult(
        experiment=experiment,
        model_name=model_name,
        target_name=target_name,
        feature_set=feature_set,
        n_samples=len(X_wf),
        n_folds=len(splits),
        auc_mean=auc_mean,
        auc_std=auc_std,
        skip20_pnl30_improvement=s20_30,
        skip10_pnl30_improvement=sim.get("skip10_pnl30", 0.0),
        skip20_pnl120_improvement=s20_120,
        skip10_pnl120_improvement=sim.get("skip10_pnl120", 0.0),
        reverse_30=s20_30 < 0,
        reverse_120=s20_120 < 0,
        baseline_pnl30=sim.get("baseline_pnl30", 0.0),
        baseline_pnl120=sim.get("baseline_pnl120", 0.0),
        inv_skip20_pnl30_improvement=inv_s20_30,
        inv_skip20_pnl120_improvement=inv_s20_120,
        profit_score=profit_score,
    )


def run_regression_wf(
    X: pd.DataFrame,
    y_reg: pd.Series,
    pnl30: pd.Series,
    pnl120: pd.Series,
    model: object,
    *,
    experiment: str,
    model_name: str,
    target_name: str,
    feature_set: str,
    min_train: int = 50,
    step: int = 25,
    embargo: int = 2,
) -> WFResult:
    """回帰モデルの Walk-Forward 評価.

    予測値を確率に変換: 高い予測 PnL = keep (低い確率), 低い予測 PnL = skip (高い確率)
    """
    common_idx = y_reg.dropna().index.intersection(pnl30.dropna().index)
    X_wf = X.loc[common_idx].to_numpy(dtype=np.float32, copy=False)
    y_wf = y_reg.loc[common_idx].to_numpy(dtype=np.float64, copy=False)
    pnl30_wf = pnl30.loc[common_idx]
    pnl120_wf = pnl120.reindex(common_idx)

    splits = expanding_window_splits(
        len(X_wf), min_train=min_train, step=step, embargo=embargo,
    )
    if not splits:
        return WFResult(
            experiment=experiment, model_name=model_name,
            target_name=target_name, feature_set=feature_set,
            n_samples=len(X_wf), n_folds=0,
            auc_mean=None, auc_std=None,
            skip20_pnl30_improvement=0, skip10_pnl30_improvement=0,
            skip20_pnl120_improvement=0, skip10_pnl120_improvement=0,
            reverse_30=True, reverse_120=True,
            baseline_pnl30=0, baseline_pnl120=0,
            inv_skip20_pnl30_improvement=0, inv_skip20_pnl120_improvement=0,
            profit_score=-999, notes="insufficient_data",
        )

    oof_preds = np.full(len(X_wf), np.nan)

    for fold_i, (train_idx, test_idx) in enumerate(splits):
        X_tr = X_wf[train_idx]
        y_tr = y_wf[train_idx]
        X_te = X_wf[test_idx]

        imputer = SimpleImputer(strategy="median")
        X_tr_imp = imputer.fit_transform(X_tr)
        X_te_imp = imputer.transform(X_te)

        try:
            m = cast(RegressorEstimator, clone(model))
            m.fit(X_tr_imp, y_tr)
            preds = m.predict(X_te_imp)
            oof_preds[test_idx] = preds
        except Exception as e:
            logger.warning(f"  Fold {fold_i} failed: {e}")
            continue
        finally:
            del X_tr_imp, X_te_imp

    # 回帰予測 → "skip probability" 変換
    # 低い予測 PnL = 高い skip probability (悪い trade = skip したい)
    valid = ~np.isnan(oof_preds)
    if valid.sum() > 20:
        # Invert: skip_prob = -predicted_pnl (低 PnL → 高 skip_prob)
        skip_prob = np.full_like(oof_preds, np.nan)
        skip_prob[valid] = -oof_preds[valid]
        # Normalize to [0, 1] for consistency
        vmin = np.nanmin(skip_prob[valid])
        vmax = np.nanmax(skip_prob[valid])
        if vmax > vmin:
            skip_prob[valid] = (skip_prob[valid] - vmin) / (vmax - vmin)
    else:
        skip_prob = oof_preds

    pnl30_arr = pnl30_wf.values.astype(float)
    pnl120_arr = pnl120_wf.values.astype(float)

    sim = _dual_horizon_skip_sim(skip_prob, pnl30_arr, pnl120_arr, invert=False)
    sim_inv = _dual_horizon_skip_sim(skip_prob, pnl30_arr, pnl120_arr, invert=True)

    s20_30 = sim.get("skip20_pnl30", 0.0)
    s20_120 = sim.get("skip20_pnl120", 0.0)
    inv_s20_30 = sim_inv.get("skip20_pnl30", 0.0)
    inv_s20_120 = sim_inv.get("skip20_pnl120", 0.0)

    best_30 = max(s20_30, inv_s20_30)
    best_120 = max(s20_120, inv_s20_120)
    profit_score = best_30 * 0.3 + best_120 * 0.7

    return WFResult(
        experiment=experiment,
        model_name=model_name,
        target_name=target_name,
        feature_set=feature_set,
        n_samples=len(X_wf),
        n_folds=len(splits),
        auc_mean=None,  # regression has no AUC
        auc_std=None,
        skip20_pnl30_improvement=s20_30,
        skip10_pnl30_improvement=sim.get("skip10_pnl30", 0.0),
        skip20_pnl120_improvement=s20_120,
        skip10_pnl120_improvement=sim.get("skip10_pnl120", 0.0),
        reverse_30=s20_30 < 0,
        reverse_120=s20_120 < 0,
        baseline_pnl30=sim.get("baseline_pnl30", 0.0),
        baseline_pnl120=sim.get("baseline_pnl120", 0.0),
        inv_skip20_pnl30_improvement=inv_s20_30,
        inv_skip20_pnl120_improvement=inv_s20_120,
        profit_score=profit_score,
    )


# ============================================================
# 5. Percentile-based Skip Simulation (ルールベース)
# ============================================================

def rule_based_experiment(
    enriched_df: pd.DataFrame,
    X_index: pd.Index,
    pnl30: pd.Series,
    pnl120: pd.Series,
) -> list[WFResult]:
    """ルールベースの skip 戦略を評価."""
    results: list[WFResult] = []
    idx = X_index

    # Rule 1: Skip when spread < P25 (narrow spread = more AS)
    if "spread_at_order" in enriched_df.columns:
        spread = enriched_df.loc[idx, "spread_at_order"].astype(float)
        valid = spread.notna() & pnl30.notna()
        if valid.sum() > 100:
            # Use rolling percentile as threshold
            oof_probs = np.full(len(idx), 0.5)
            for i in range(50, len(idx)):
                past_spreads = spread.iloc[:i]
                p25 = past_spreads.quantile(0.25)
                current = spread.iloc[i]
                # Lower spread → higher skip probability
                if not np.isnan(current) and not np.isnan(p25):
                    if p25 > 0:
                        oof_probs[i] = max(0, min(1, 1.0 - current / (p25 * 4)))

            sim = _dual_horizon_skip_sim(
                oof_probs, pnl30.values.astype(float),
                pnl120.reindex(idx).values.astype(float),
            )
            s20_30 = sim.get("skip20_pnl30", 0.0)
            s20_120 = sim.get("skip20_pnl120", 0.0)
            results.append(WFResult(
                experiment="Rule_narrow_spread", model_name="rule",
                target_name="spread", feature_set="rule",
                n_samples=int(valid.sum()), n_folds=0,
                auc_mean=None, auc_std=None,
                skip20_pnl30_improvement=s20_30,
                skip10_pnl30_improvement=sim.get("skip10_pnl30", 0.0),
                skip20_pnl120_improvement=s20_120,
                skip10_pnl120_improvement=sim.get("skip10_pnl120", 0.0),
                reverse_30=s20_30 < 0, reverse_120=s20_120 < 0,
                baseline_pnl30=sim.get("baseline_pnl30", 0.0),
                baseline_pnl120=sim.get("baseline_pnl120", 0.0),
                inv_skip20_pnl30_improvement=0, inv_skip20_pnl120_improvement=0,
                profit_score=s20_30 * 0.3 + s20_120 * 0.7,
                notes="Skip narrow spread trades",
            ))

    # Rule 2: Skip sell-side in unknown regime
    if "regime" in enriched_df.columns and "side" in enriched_df.columns:
        regime = enriched_df.loc[idx, "regime"].fillna("unknown")
        side = enriched_df.loc[idx, "side"]
        oof_probs = np.where(
            (regime == "unknown") & (side == "sell"), 0.9,
            np.where(regime == "unknown", 0.6, 0.3)
        ).astype(float)

        sim = _dual_horizon_skip_sim(
            oof_probs, pnl30.values.astype(float),
            pnl120.reindex(idx).values.astype(float),
        )
        s20_30 = sim.get("skip20_pnl30", 0.0)
        s20_120 = sim.get("skip20_pnl120", 0.0)
        results.append(WFResult(
            experiment="Rule_skip_unknown_sell", model_name="rule",
            target_name="regime_side", feature_set="rule",
            n_samples=len(idx), n_folds=0,
            auc_mean=None, auc_std=None,
            skip20_pnl30_improvement=s20_30,
            skip10_pnl30_improvement=sim.get("skip10_pnl30", 0.0),
            skip20_pnl120_improvement=s20_120,
            skip10_pnl120_improvement=sim.get("skip10_pnl120", 0.0),
            reverse_30=s20_30 < 0, reverse_120=s20_120 < 0,
            baseline_pnl30=sim.get("baseline_pnl30", 0.0),
            baseline_pnl120=sim.get("baseline_pnl120", 0.0),
            inv_skip20_pnl30_improvement=0, inv_skip20_pnl120_improvement=0,
            profit_score=s20_30 * 0.3 + s20_120 * 0.7,
            notes="Skip sell trades in unknown regime",
        ))

    return results


# ============================================================
# 6. メイン実行
# ============================================================

def main() -> None:
    """全実験を実行し、最良モデルを選定."""
    logger.info("=" * 80)
    logger.info("124# SkipGate v3: Comprehensive Model Exploration")
    logger.info("=" * 80)

    # --- データ読み込み ---
    logger.info("\n--- Step 1: Data Loading ---")
    df = load_fill_records()
    logger.info(f"Total records: {len(df)}")

    enriched_df = enrich_fill_records(df)

    # 基本特徴量
    X_base, y_base = build_preorder_as_features(enriched_df)
    logger.info(f"Base features: {X_base.shape}")

    # PnL series
    filled_mask = enriched_df["filled"].astype(bool) & enriched_df["adverse_selected_raw"].notna()
    pnl30 = enriched_df.loc[filled_mask, "post_fill_30s_pnl"].astype(float).reindex(X_base.index)
    pnl120_col = "post_fill_120s_pnl"
    if pnl120_col in enriched_df.columns:
        pnl120 = enriched_df.loc[filled_mask, pnl120_col].astype(float).reindex(X_base.index)
    else:
        logger.warning("post_fill_120s_pnl not found, using pnl30 as fallback")
        pnl120 = pnl30

    logger.info(f"PnL30 baseline: {pnl30.mean():.3f} bps (n={pnl30.notna().sum()})")
    logger.info(f"PnL120 baseline: {pnl120.mean():.3f} bps (n={pnl120.notna().sum()})")

    # --- 特徴量エンジニアリング ---
    logger.info("\n--- Step 2: Feature Engineering ---")
    X_engineered = engineer_features(enriched_df, X_base, y_base)

    # OB 特徴量追加版
    ob_cols = ["spread_bps_ob", "depth_imbalance_ob"]
    has_ob = all(c in enriched_df.columns for c in ob_cols)
    if has_ob:
        X_full = X_engineered.copy()
        for col in ob_cols:
            X_full[col] = enriched_df.loc[X_base.index, col].astype(float)
        if "depth_imbalance_ob" in enriched_df.columns:
            side_sign = enriched_df.loc[X_base.index, "side"].map(
                {"buy": 1.0, "sell": -1.0}
            ).astype(float)
            X_full["side_aligned_imbalance_ob"] = (
                enriched_df.loc[X_base.index, "depth_imbalance_ob"].astype(float)
                * side_sign
            ).fillna(0.0)
        logger.info(f"Full features (base+engineered+OB): {X_full.shape}")
    else:
        X_full = X_engineered
        logger.info("No OB features available")

    # --- ターゲット変数 ---
    logger.info("\n--- Step 3: Target Variables ---")
    targets = build_targets(enriched_df, X_base.index)

    # --- モデル ---
    cls_models = get_models()
    reg_models = get_regression_models()

    # --- 実験実行 ---
    logger.info("\n--- Step 4: Walk-Forward Experiments ---")
    all_results: list[WFResult] = []

    # Feature sets to test
    feature_sets: dict[str, pd.DataFrame] = {
        "base": X_base,
        "engineered": X_engineered,
    }
    if has_ob:
        feature_sets["full"] = X_full

    # Classification targets
    cls_targets = ["as30", "profitable30", "profitable120", "really_bad30"]
    if "profitable_both" in targets:
        cls_targets.append("profitable_both")

    experiment_count = 0
    total_experiments = (
        len(feature_sets) * len(cls_models) * len([t for t in cls_targets if t in targets])
        + len(feature_sets) * len(reg_models) * len([t for t in ["pnl30_regression", "pnl120_regression"] if t in targets])
    )
    logger.info(f"Total experiments planned: ~{total_experiments}")

    # 4a. Classification experiments
    for feat_name, X_feat in feature_sets.items():
        for target_name in cls_targets:
            if target_name not in targets:
                continue
            y_target = targets[target_name]

            for model_name, model in cls_models.items():
                experiment_count += 1
                exp_name = f"{model_name}_{target_name}_{feat_name}"
                logger.info(f"\n[{experiment_count}] {exp_name}")

                try:
                    result = run_classification_wf(
                        X_feat, y_target, pnl30, pnl120, model,
                        experiment=exp_name,
                        model_name=model_name,
                        target_name=target_name,
                        feature_set=feat_name,
                    )
                    all_results.append(result)
                    logger.info(
                        f"  AUC={result.auc_mean}, "
                        f"Skip20%_30={result.skip20_pnl30_improvement:+.3f}, "
                        f"Skip20%_120={result.skip20_pnl120_improvement:+.3f}, "
                        f"InvSkip20%_30={result.inv_skip20_pnl30_improvement:+.3f}, "
                        f"InvSkip20%_120={result.inv_skip20_pnl120_improvement:+.3f}, "
                        f"score={result.profit_score:+.3f}"
                    )
                except Exception as e:
                    logger.error(f"  FAILED: {e}")
                if experiment_count % 25 == 0:
                    gc.collect()

    # 4b. Regression experiments
    for feat_name, X_feat in feature_sets.items():
        for target_name in ["pnl30_regression", "pnl120_regression"]:
            if target_name not in targets:
                continue
            y_reg = targets[target_name]

            for model_name, model in reg_models.items():
                experiment_count += 1
                exp_name = f"{model_name}_{target_name}_{feat_name}"
                logger.info(f"\n[{experiment_count}] {exp_name}")

                try:
                    result = run_regression_wf(
                        X_feat, y_reg, pnl30, pnl120, model,
                        experiment=exp_name,
                        model_name=model_name,
                        target_name=target_name,
                        feature_set=feat_name,
                    )
                    all_results.append(result)
                    logger.info(
                        f"  Skip20%_30={result.skip20_pnl30_improvement:+.3f}, "
                        f"Skip20%_120={result.skip20_pnl120_improvement:+.3f}, "
                        f"InvSkip20%_30={result.inv_skip20_pnl30_improvement:+.3f}, "
                        f"InvSkip20%_120={result.inv_skip20_pnl120_improvement:+.3f}, "
                        f"score={result.profit_score:+.3f}"
                    )
                except Exception as e:
                    logger.error(f"  FAILED: {e}")
                if experiment_count % 25 == 0:
                    gc.collect()

    # 4c. Rule-based experiments
    logger.info("\n--- Rule-based experiments ---")
    rule_results = rule_based_experiment(enriched_df, X_base.index, pnl30, pnl120)
    for r in rule_results:
        all_results.append(r)
        logger.info(
            f"  {r.experiment}: "
            f"Skip20%_30={r.skip20_pnl30_improvement:+.3f}, "
            f"Skip20%_120={r.skip20_pnl120_improvement:+.3f}, "
            f"score={r.profit_score:+.3f}"
        )

    # --- 結果比較 ---
    logger.info("\n" + "=" * 120)
    logger.info("COMPREHENSIVE COMPARISON TABLE")
    logger.info("=" * 120)
    logger.info(
        f"{'Experiment':<50} {'AUC':>6} "
        f"{'S20%_30':>8} {'S20%_120':>9} "
        f"{'Inv30':>7} {'Inv120':>8} "
        f"{'Score':>7} {'Rev30':>6} {'Rev120':>7}"
    )
    logger.info("-" * 120)

    # Sort by profit score
    sorted_results = sorted(all_results, key=lambda r: r.profit_score, reverse=True)

    for r in sorted_results:
        auc_str = f"{r.auc_mean:.3f}" if r.auc_mean is not None else "  N/A"
        logger.info(
            f"  {r.experiment:<48} {auc_str:>6} "
            f"{r.skip20_pnl30_improvement:>+8.3f} {r.skip20_pnl120_improvement:>+9.3f} "
            f"{r.inv_skip20_pnl30_improvement:>+7.3f} {r.inv_skip20_pnl120_improvement:>+8.3f} "
            f"{r.profit_score:>+7.3f} "
            f"{'YES' if r.reverse_30 else ' no':>6} "
            f"{'YES' if r.reverse_120 else ' no':>7}"
        )

    # --- Top 10 ---
    logger.info("\n" + "=" * 80)
    logger.info("TOP 10 BY PROFIT SCORE")
    logger.info("=" * 80)
    for i, r in enumerate(sorted_results[:10], 1):
        logger.info(
            f"  #{i}: {r.experiment}\n"
            f"       Score={r.profit_score:+.3f}, "
            f"PnL30 skip20={r.skip20_pnl30_improvement:+.3f}, "
            f"PnL120 skip20={r.skip20_pnl120_improvement:+.3f}\n"
            f"       Inv30={r.inv_skip20_pnl30_improvement:+.3f}, "
            f"Inv120={r.inv_skip20_pnl120_improvement:+.3f}, "
            f"AUC={r.auc_mean}"
        )

    # --- 非逆選別のベスト ---
    non_reverse_30 = [r for r in sorted_results if not r.reverse_30]
    non_reverse_120 = [r for r in sorted_results if not r.reverse_120]
    non_reverse_both = [r for r in sorted_results if not r.reverse_30 and not r.reverse_120]

    logger.info("\n--- Non-reverse selection (PnL30) top 5 ---")
    for r in non_reverse_30[:5]:
        logger.info(f"  {r.experiment}: S20%_30={r.skip20_pnl30_improvement:+.3f}")

    logger.info("\n--- Non-reverse selection (PnL120) top 5 ---")
    for r in non_reverse_120[:5]:
        logger.info(f"  {r.experiment}: S20%_120={r.skip20_pnl120_improvement:+.3f}")

    logger.info(f"\n--- Non-reverse BOTH: {len(non_reverse_both)} experiments ---")
    for r in non_reverse_both[:5]:
        logger.info(
            f"  {r.experiment}: S20%_30={r.skip20_pnl30_improvement:+.3f}, "
            f"S20%_120={r.skip20_pnl120_improvement:+.3f}, "
            f"score={r.profit_score:+.3f}"
        )

    # --- Inverted (逆転 SG) のベスト ---
    inv_results = []
    for r in sorted_results:
        inv_score = r.inv_skip20_pnl30_improvement * 0.3 + r.inv_skip20_pnl120_improvement * 0.7
        inv_results.append((r, inv_score))
    inv_results.sort(key=lambda x: x[1], reverse=True)

    logger.info("\n--- Inverted SG (reverse the model) top 5 ---")
    for r, inv_score in inv_results[:5]:
        logger.info(
            f"  {r.experiment}: Inv30={r.inv_skip20_pnl30_improvement:+.3f}, "
            f"Inv120={r.inv_skip20_pnl120_improvement:+.3f}, "
            f"inv_score={inv_score:+.3f}"
        )

    # --- レポート保存 ---
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    report = {
        "generated_at": datetime.now().isoformat(),
        "source": "124# train_sg_v3.py",
        "data_summary": {
            "total_records": len(df),
            "filled_with_label": len(X_base),
            "n_features_base": X_base.shape[1],
            "n_features_engineered": X_engineered.shape[1],
            "n_features_full": X_full.shape[1] if has_ob else X_engineered.shape[1],
            "pnl30_baseline_bps": float(pnl30.mean()),
            "pnl120_baseline_bps": float(pnl120.mean()),
            "has_ob": has_ob,
        },
        "experiments": [r.to_dict() for r in sorted_results],
        "top10": [r.to_dict() for r in sorted_results[:10]],
        "non_reverse_both_top5": [r.to_dict() for r in non_reverse_both[:5]],
        "inverted_top5": [
            {**r.to_dict(), "inv_profit_score": inv_s}
            for r, inv_s in inv_results[:5]
        ],
    }

    report_path = REPORT_DIR / "sg_v3_comprehensive.json"
    write_json(report_path, report, indent=2, ensure_ascii=False, default=str)
    logger.info(f"\nReport saved to {report_path}")

    # --- 最良モデルの判定 ---
    logger.info("\n" + "=" * 80)
    logger.info("DEPLOYMENT DECISION")
    logger.info("=" * 80)

    best = sorted_results[0] if sorted_results else None
    if best and best.profit_score > 0.05:
        logger.info(f"*** BEST: {best.experiment} (score={best.profit_score:+.3f}) ***")
        logger.info(f"    PnL30 improvement: {best.skip20_pnl30_improvement:+.3f} bps")
        logger.info(f"    PnL120 improvement: {best.skip20_pnl120_improvement:+.3f} bps")
        logger.info(f"    Inverted30: {best.inv_skip20_pnl30_improvement:+.3f} bps")
        logger.info(f"    Inverted120: {best.inv_skip20_pnl120_improvement:+.3f} bps")

        # Check if inverted version is better
        inv_best_r, inv_best_score = inv_results[0]
        if inv_best_score > best.profit_score:
            logger.info(f"\n*** INVERTED MODEL IS BETTER: {inv_best_r.experiment} ***")
            logger.info(f"    Inv score: {inv_best_score:+.3f}")
            logger.info("    → Deploy with INVERTED skip logic")
    else:
        logger.info("No experiment meets deployment threshold (profit_score > 0.05)")
        logger.info("Consider: rule-based approach or parameter-only optimization")


if __name__ == "__main__":
    main()
