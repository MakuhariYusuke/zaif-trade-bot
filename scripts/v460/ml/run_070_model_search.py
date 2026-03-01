"""070# 網羅的モデル探索: ph2 fill test に投入すべき収益モデルの発見.

全491サイクル (filled 373, AS-labeled 284) を使い、
多角的にモデル構成を比較。OOF walk-forward で信頼性を担保。

評価軸:
  1. OOF skip simulation PnL improvement (bps) — 第一指標
  2. ROC-AUC — 分類精度
  3. Feature stability — 安定性
  4. Rule-based baseline との比較 — ML は本当に必要か

構成:
  Part A: Rule-based baselines (ML不要)
  Part B: AS分類器 (LR/GB × feature sets × k values)
  Part C: PnL回帰器 (Ridge/GBR — 直接PnL予測)
  Part D: Side別モデル (buy/sell 分離)
  Part E: 閾値最適化 (best model の最適 threshold)
"""

from __future__ import annotations

import logging
import sys
import warnings
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.ensemble import (
    GradientBoostingClassifier,
    GradientBoostingRegressor,
    RandomForestClassifier,
)
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
from sklearn.model_selection import TimeSeriesSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent.parent))

from scripts.v460.ml.data_loader import build_as_features, load_fill_records
from scripts.v460.ml.feature_enricher import (
    build_enriched_as_features,
    build_pnl_features,
    enrich_fill_records,
)
from scripts.v460.ml.frame_utils import compute_utc_hour
from scripts.v460.ml.walk_forward_as import expanding_window_splits
from ztb.io.json_io import write_json

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════
# Part A: Rule-based baselines
# ═══════════════════════════════════════════════════


def eval_rule_baselines(df: pd.DataFrame) -> list[dict]:
    """ルールベースのスキップ戦略を評価."""
    filled = df[df["filled"].astype(bool)].copy()
    pnl = filled["post_fill_30s_pnl"].astype(float)
    baseline_pnl = float(pnl.mean())
    n_total = len(filled)

    results = []

    def _eval(name: str, skip_mask: pd.Series) -> dict:
        keep_mask = ~skip_mask
        n_keep = int(keep_mask.sum())
        n_skip = int(skip_mask.sum())
        if n_keep == 0:
            return {"name": name, "skip_rate": 1.0, "kept_pnl": 0.0,
                    "improvement": 0.0, "n_keep": 0, "n_skip": n_total}
        kept_pnl = float(pnl[keep_mask].mean())
        return {
            "name": name,
            "n_keep": n_keep,
            "n_skip": n_skip,
            "skip_rate": n_skip / n_total,
            "kept_pnl_bps": round(kept_pnl, 4),
            "improvement_bps": round(kept_pnl - baseline_pnl, 4),
            "baseline_pnl_bps": round(baseline_pnl, 4),
        }

    # R0: No skip (baseline)
    results.append(_eval("R0_no_skip", pd.Series(False, index=filled.index)))

    # R1: Skip all sells
    results.append(_eval("R1_skip_all_sell", filled["side"] == "sell"))

    # R2: Skip all buys
    results.append(_eval("R2_skip_all_buy", filled["side"] == "buy"))

    # R3-R5: time-based skips (hour bins)
    hours_utc = compute_utc_hour(filled["timestamp"])

    # R3: Skip "bad" UTC hours (from fill_test.yaml time filter)
    bad_hours = {1, 2, 8, 9, 12, 13, 14, 16, 17, 18, 19, 21}
    results.append(_eval("R3_skip_bad_hours", hours_utc.isin(bad_hours)))

    # R4: Skip night UTC (0-8)
    results.append(_eval("R4_skip_night_utc", hours_utc.isin(range(9))))

    # R5: Skip based on PnL stats per hour
    hour_pnl = filled.groupby(hours_utc)["post_fill_30s_pnl"].mean()
    neg_hours = set(hour_pnl[hour_pnl < 0].index)
    results.append(_eval("R5_skip_neg_pnl_hours",
                         hours_utc.isin(neg_hours)))

    # R6: Skip sell + bad hours
    results.append(_eval("R6_skip_sell_or_bad_hours",
                         (filled["side"] == "sell") | hours_utc.isin(bad_hours)))

    # R7: Skip when queue_wait < 3s (too fast, likely adverse)
    if "queue_wait_sec" in filled.columns:
        qw = filled["queue_wait_sec"].astype(float)
        results.append(_eval("R7_skip_fast_fill_3s", qw < 3.0))
        results.append(_eval("R8_skip_fast_fill_5s", qw < 5.0))
        results.append(_eval("R9_skip_fast_fill_10s", qw < 10.0))

    # R10: Skip when spread is narrow (< median)
    if "spread_at_order" in filled.columns:
        spread = filled["spread_at_order"].astype(float)
        spread_valid = spread.notna()
        if spread_valid.sum() > 20:
            med = spread[spread_valid].median()
            # Only skip among those with spread data
            skip = spread_valid & (spread < med)
            results.append(_eval("R10_skip_narrow_spread", skip))

    # R11: Combined: skip sell + fast fill
    if "queue_wait_sec" in filled.columns:
        results.append(_eval("R11_skip_sell_and_fast5s",
                             (filled["side"] == "sell") & (qw < 5.0)))

    return results


# ═══════════════════════════════════════════════════
# Part B: AS Classifier sweep
# ═══════════════════════════════════════════════════


def _build_pipeline(model_type: str, k: int | None, **kwargs) -> Pipeline:
    """分類器パイプラインを構築."""
    steps: list = [("imputer", SimpleImputer(strategy="median"))]
    if k is not None:
        steps.append(("selector", SelectKBest(f_classif, k=k)))
    steps.append(("scaler", StandardScaler()))

    if model_type == "lr":
        c = kwargs.get("C", 0.01)
        steps.append(("model", LogisticRegression(
            C=c, max_iter=2000, class_weight="balanced",
            penalty="l2", random_state=42,
        )))
    elif model_type == "lr_l1":
        c = kwargs.get("C", 0.01)
        steps.append(("model", LogisticRegression(
            C=c, max_iter=2000, class_weight="balanced",
            penalty="l1", solver="saga", random_state=42,
        )))
    elif model_type == "gb":
        n_est = kwargs.get("n_estimators", 30)
        lr_val = kwargs.get("learning_rate", 0.05)
        md = kwargs.get("max_depth", 3)
        steps.append(("model", GradientBoostingClassifier(
            n_estimators=n_est, max_depth=md, learning_rate=lr_val,
            subsample=0.8, random_state=42,
        )))
    elif model_type == "rf":
        n_est = kwargs.get("n_estimators", 50)
        md = kwargs.get("max_depth", 3)
        steps.append(("model", RandomForestClassifier(
            n_estimators=n_est, max_depth=md, class_weight="balanced",
            random_state=42,
        )))
    return Pipeline(steps)


def _oof_walk_forward_clf(
    X: pd.DataFrame,
    y: pd.Series,
    pnl: pd.Series,
    config: dict,
    *,
    min_train: int = 50,
    step: int = 20,
    embargo: int = 2,
) -> dict:
    """Walk-forward OOF evaluation for a classifier config."""
    splits = expanding_window_splits(
        len(X), min_train=min_train, step=step, embargo=embargo
    )
    if not splits:
        return {"error": "insufficient_data", "config": config}

    oof_probs = np.full(len(X), np.nan)
    roc_aucs = []
    pr_aucs = []

    for train_idx, test_idx in splits:
        X_tr, y_tr = X.iloc[train_idx], y.iloc[train_idx]
        X_te, y_te = X.iloc[test_idx], y.iloc[test_idx]

        k = config.get("k")
        if k is not None:
            k = min(k, X_tr.shape[1])
        pipe = _build_pipeline(config["model_type"], k, **config.get("params", {}))
        pipe.fit(X_tr, y_tr)
        probs = pipe.predict_proba(X_te)[:, 1]
        oof_probs[test_idx] = probs

        if len(np.unique(y_te)) > 1:
            roc_aucs.append(roc_auc_score(y_te, probs))
            pr_aucs.append(average_precision_score(y_te, probs))

    # Skip simulation
    valid = ~np.isnan(oof_probs) & ~np.isnan(pnl.values)
    result = {
        "config": config,
        "n_folds": len(splits),
        "roc_auc_mean": round(float(np.mean(roc_aucs)), 4) if roc_aucs else None,
        "roc_auc_std": round(float(np.std(roc_aucs)), 4) if roc_aucs else None,
        "pr_auc_mean": round(float(np.mean(pr_aucs)), 4) if pr_aucs else None,
        "n_oof_valid": int(valid.sum()),
    }

    if valid.sum() > 10:
        vp = oof_probs[valid]
        vpnl = pnl.values[valid]
        baseline = float(np.mean(vpnl))
        result["baseline_pnl_bps"] = round(baseline, 4)

        # Multiple skip percentiles
        for pct_name, pct_val in [("skip10", 90), ("skip20", 80),
                                   ("skip30", 70), ("skip40", 60)]:
            th = np.percentile(vp, pct_val)
            keep = vp < th
            if keep.sum() > 0:
                kept = float(np.mean(vpnl[keep]))
                result[f"{pct_name}_improvement_bps"] = round(kept - baseline, 4)
                result[f"{pct_name}_kept_pnl_bps"] = round(kept, 4)
                result[f"{pct_name}_n_keep"] = int(keep.sum())
            else:
                result[f"{pct_name}_improvement_bps"] = 0.0

        # Threshold sweep
        best_th = None
        best_kept_pnl = baseline
        for th_val in np.arange(0.3, 0.8, 0.05):
            keep = vp < th_val
            if keep.sum() >= 10:
                kept = float(np.mean(vpnl[keep]))
                if kept > best_kept_pnl:
                    best_kept_pnl = kept
                    best_th = round(float(th_val), 2)
        result["best_threshold"] = best_th
        result["best_threshold_pnl_bps"] = round(best_kept_pnl, 4) if best_th else None
        result["best_threshold_improvement_bps"] = round(best_kept_pnl - baseline, 4) if best_th else None

    return result


def run_classifier_sweep(
    X: pd.DataFrame, y: pd.Series, pnl: pd.Series, label: str
) -> list[dict]:
    """多数の分類器構成をWalk-Forward評価."""
    configs = [
        # LR variants
        {"model_type": "lr", "k": None, "params": {"C": 0.01}, "label": f"{label}_LR_C0.01_kAll"},
        {"model_type": "lr", "k": None, "params": {"C": 0.1}, "label": f"{label}_LR_C0.1_kAll"},
        {"model_type": "lr", "k": None, "params": {"C": 1.0}, "label": f"{label}_LR_C1.0_kAll"},
        {"model_type": "lr", "k": 3, "params": {"C": 0.01}, "label": f"{label}_LR_C0.01_k3"},
        {"model_type": "lr", "k": 5, "params": {"C": 0.01}, "label": f"{label}_LR_C0.01_k5"},
        {"model_type": "lr", "k": 8, "params": {"C": 0.01}, "label": f"{label}_LR_C0.01_k8"},
        {"model_type": "lr", "k": 12, "params": {"C": 0.01}, "label": f"{label}_LR_C0.01_k12"},
        {"model_type": "lr", "k": 5, "params": {"C": 0.1}, "label": f"{label}_LR_C0.1_k5"},
        {"model_type": "lr", "k": 8, "params": {"C": 0.1}, "label": f"{label}_LR_C0.1_k8"},
        {"model_type": "lr", "k": 5, "params": {"C": 1.0}, "label": f"{label}_LR_C1.0_k5"},
        # L1 variants
        {"model_type": "lr_l1", "k": None, "params": {"C": 0.01}, "label": f"{label}_LR_L1_C0.01_kAll"},
        {"model_type": "lr_l1", "k": None, "params": {"C": 0.1}, "label": f"{label}_LR_L1_C0.1_kAll"},
        {"model_type": "lr_l1", "k": 5, "params": {"C": 0.1}, "label": f"{label}_LR_L1_C0.1_k5"},
        # GB variants
        {"model_type": "gb", "k": None, "params": {"n_estimators": 30, "learning_rate": 0.05, "max_depth": 2}, "label": f"{label}_GB_n30_d2_kAll"},
        {"model_type": "gb", "k": None, "params": {"n_estimators": 30, "learning_rate": 0.05, "max_depth": 3}, "label": f"{label}_GB_n30_d3_kAll"},
        {"model_type": "gb", "k": None, "params": {"n_estimators": 50, "learning_rate": 0.05, "max_depth": 2}, "label": f"{label}_GB_n50_d2_kAll"},
        {"model_type": "gb", "k": None, "params": {"n_estimators": 100, "learning_rate": 0.03, "max_depth": 2}, "label": f"{label}_GB_n100_d2_kAll"},
        {"model_type": "gb", "k": 5, "params": {"n_estimators": 30, "learning_rate": 0.05, "max_depth": 3}, "label": f"{label}_GB_n30_d3_k5"},
        {"model_type": "gb", "k": 8, "params": {"n_estimators": 30, "learning_rate": 0.05, "max_depth": 3}, "label": f"{label}_GB_n30_d3_k8"},
        {"model_type": "gb", "k": 12, "params": {"n_estimators": 50, "learning_rate": 0.05, "max_depth": 2}, "label": f"{label}_GB_n50_d2_k12"},
        # RF variants
        {"model_type": "rf", "k": None, "params": {"n_estimators": 50, "max_depth": 3}, "label": f"{label}_RF_n50_d3_kAll"},
        {"model_type": "rf", "k": None, "params": {"n_estimators": 100, "max_depth": 2}, "label": f"{label}_RF_n100_d2_kAll"},
        {"model_type": "rf", "k": 5, "params": {"n_estimators": 50, "max_depth": 3}, "label": f"{label}_RF_n50_d3_k5"},
    ]

    results = []
    for cfg in configs:
        try:
            r = _oof_walk_forward_clf(X, y, pnl, cfg)
            r["label"] = cfg["label"]
            results.append(r)
            skip20 = r.get("skip20_improvement_bps", "N/A")
            roc = r.get("roc_auc_mean", "N/A")
            logger.info(f"  {cfg['label']:45s}  ROC={roc}  skip20={skip20}")
        except Exception as e:
            logger.warning(f"  {cfg['label']}: FAILED ({e})")
    return results


# ═══════════════════════════════════════════════════
# Part C: PnL Regressor
# ═══════════════════════════════════════════════════


def _oof_walk_forward_reg(
    X: pd.DataFrame,
    y: pd.Series,  # PnL target
    config: dict,
    *,
    min_train: int = 50,
    step: int = 20,
    embargo: int = 2,
) -> dict:
    """Walk-forward OOF for regressor."""
    splits = expanding_window_splits(
        len(X), min_train=min_train, step=step, embargo=embargo
    )
    if not splits:
        return {"error": "insufficient_data"}

    oof_preds = np.full(len(X), np.nan)
    ics = []
    maes = []

    for train_idx, test_idx in splits:
        steps: list = [
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
        k = config.get("k")
        if k is not None:
            k_actual = min(k, X.iloc[train_idx].shape[1])
            steps.insert(1, ("selector", SelectKBest(f_classif, k=k_actual)))

        model_type = config["model_type"]
        params = config.get("params", {})
        if model_type == "ridge":
            steps.append(("model", Ridge(alpha=params.get("alpha", 10.0))))
        elif model_type == "gbr":
            steps.append(("model", GradientBoostingRegressor(
                n_estimators=params.get("n_estimators", 50),
                max_depth=params.get("max_depth", 2),
                learning_rate=params.get("learning_rate", 0.05),
                subsample=0.8, random_state=42,
            )))
        pipe = Pipeline(steps)
        pipe.fit(X.iloc[train_idx], y.iloc[train_idx])
        preds = pipe.predict(X.iloc[test_idx])
        oof_preds[test_idx] = preds

        y_te = y.iloc[test_idx].values
        maes.append(float(np.mean(np.abs(preds - y_te))))
        if len(y_te) > 5:
            ic, _ = spearmanr(preds, y_te)
            if not np.isnan(ic):
                ics.append(ic)

    valid = ~np.isnan(oof_preds)
    baseline_pnl = float(y.values[valid].mean()) if valid.sum() > 0 else 0.0

    result = {
        "config": config,
        "n_folds": len(splits),
        "ic_mean": round(float(np.mean(ics)), 4) if ics else None,
        "mae_mean": round(float(np.mean(maes)), 4),
        "baseline_pnl_bps": round(baseline_pnl, 4),
        "n_oof_valid": int(valid.sum()),
    }

    if valid.sum() > 10:
        vpred = oof_preds[valid]
        vpnl = y.values[valid]

        # Skip predicted negative PnL
        keep = vpred >= 0
        n_keep = int(keep.sum())
        if n_keep > 0 and n_keep < valid.sum():
            kept_pnl = float(np.mean(vpnl[keep]))
            result["skip_neg_kept_pnl_bps"] = round(kept_pnl, 4)
            result["skip_neg_improvement_bps"] = round(kept_pnl - baseline_pnl, 4)
            result["skip_neg_n_keep"] = n_keep
            result["skip_neg_skip_rate"] = round(1 - n_keep / valid.sum(), 4)
        else:
            result["skip_neg_improvement_bps"] = 0.0

        # Skip bottom N%
        for pct_name, pct_val in [("skip10", 10), ("skip20", 20), ("skip30", 30)]:
            th = np.percentile(vpred, pct_val)
            keep = vpred >= th
            if keep.sum() > 0:
                kept = float(np.mean(vpnl[keep]))
                result[f"{pct_name}_improvement_bps"] = round(kept - baseline_pnl, 4)
                result[f"{pct_name}_kept_pnl_bps"] = round(kept, 4)

    return result


def run_regressor_sweep(
    X: pd.DataFrame, y: pd.Series, label: str
) -> list[dict]:
    """PnL回帰器のsweep."""
    configs = [
        {"model_type": "ridge", "k": None, "params": {"alpha": 1.0}, "label": f"{label}_Ridge_a1"},
        {"model_type": "ridge", "k": None, "params": {"alpha": 10.0}, "label": f"{label}_Ridge_a10"},
        {"model_type": "ridge", "k": None, "params": {"alpha": 100.0}, "label": f"{label}_Ridge_a100"},
        {"model_type": "ridge", "k": 5, "params": {"alpha": 10.0}, "label": f"{label}_Ridge_a10_k5"},
        {"model_type": "ridge", "k": 8, "params": {"alpha": 10.0}, "label": f"{label}_Ridge_a10_k8"},
        {"model_type": "gbr", "k": None, "params": {"n_estimators": 30, "max_depth": 2, "learning_rate": 0.05}, "label": f"{label}_GBR_n30_d2"},
        {"model_type": "gbr", "k": None, "params": {"n_estimators": 50, "max_depth": 2, "learning_rate": 0.05}, "label": f"{label}_GBR_n50_d2"},
        {"model_type": "gbr", "k": None, "params": {"n_estimators": 50, "max_depth": 3, "learning_rate": 0.03}, "label": f"{label}_GBR_n50_d3"},
        {"model_type": "gbr", "k": 5, "params": {"n_estimators": 50, "max_depth": 2, "learning_rate": 0.05}, "label": f"{label}_GBR_n50_d2_k5"},
        {"model_type": "gbr", "k": 8, "params": {"n_estimators": 30, "max_depth": 2, "learning_rate": 0.05}, "label": f"{label}_GBR_n30_d2_k8"},
    ]

    results = []
    for cfg in configs:
        try:
            r = _oof_walk_forward_reg(X, y, cfg)
            r["label"] = cfg["label"]
            results.append(r)
            skip_neg = r.get("skip_neg_improvement_bps", "N/A")
            ic = r.get("ic_mean", "N/A")
            logger.info(f"  {cfg['label']:45s}  IC={ic}  skip_neg={skip_neg}")
        except Exception as e:
            logger.warning(f"  {cfg['label']}: FAILED ({e})")
    return results


# ═══════════════════════════════════════════════════
# Part D: Side-specific models
# ═══════════════════════════════════════════════════


def run_side_specific(
    X: pd.DataFrame, y: pd.Series, pnl: pd.Series, side_col: pd.Series
) -> list[dict]:
    """buy/sell 別にトップ構成で評価."""
    results = []
    for side in ["buy", "sell"]:
        mask = side_col == side
        Xs = X[mask].reset_index(drop=True)
        ys = y[mask].reset_index(drop=True)
        ps = pnl[mask].reset_index(drop=True)

        if len(Xs) < 60:
            logger.warning(f"  Side {side}: only {len(Xs)} samples, skipping")
            continue

        for cfg in [
            {"model_type": "lr", "k": 5, "params": {"C": 0.01}, "label": f"side_{side}_LR_k5"},
            {"model_type": "lr", "k": None, "params": {"C": 0.01}, "label": f"side_{side}_LR_kAll"},
            {"model_type": "gb", "k": None, "params": {"n_estimators": 30, "max_depth": 2, "learning_rate": 0.05}, "label": f"side_{side}_GB_d2"},
        ]:
            try:
                r = _oof_walk_forward_clf(Xs, ys, ps, cfg, min_train=30)
                r["label"] = cfg["label"]
                r["side"] = side
                results.append(r)
                skip20 = r.get("skip20_improvement_bps", "N/A")
                roc = r.get("roc_auc_mean", "N/A")
                logger.info(f"  {cfg['label']:45s}  ROC={roc}  skip20={skip20}")
            except Exception as e:
                logger.warning(f"  {cfg['label']}: FAILED ({e})")
    return results


# ═══════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════


def main() -> None:
    output_dir = Path("reports/v460/model_search_070")
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("=" * 70)
    logger.info("070# 網羅的モデル探索")
    logger.info("=" * 70)

    # --- Load data ---
    df = load_fill_records()
    logger.info(f"Total records: {len(df)}")

    filled = df[df["filled"].astype(bool)].copy()
    logger.info(f"Filled: {len(filled)}")

    # ═══════════════════════════════════════════════════
    # Part A: Rule-based baselines
    # ═══════════════════════════════════════════════════
    logger.info("\n" + "=" * 70)
    logger.info("Part A: Rule-based Baselines")
    logger.info("=" * 70)

    rule_results = eval_rule_baselines(df)
    for r in rule_results:
        logger.info(f"  {r['name']:35s}  kept_pnl={r.get('kept_pnl_bps', 'N/A'):>8}  "
                     f"improvement={r.get('improvement_bps', 'N/A'):>8}  "
                     f"skip_rate={r.get('skip_rate', 0):.1%}")

    # ═══════════════════════════════════════════════════
    # Part B: AS Classifier sweep — Trade-only features (n=284)
    # ═══════════════════════════════════════════════════
    logger.info("\n" + "=" * 70)
    logger.info("Part B-1: AS Classifier (Trade-only, n=284)")
    logger.info("=" * 70)

    X_base, y_base = build_as_features(df, require_spread=False)
    pnl_base = df.loc[X_base.index, "post_fill_30s_pnl"].astype(float)
    logger.info(f"  Features: {list(X_base.columns)}")
    logger.info(f"  Samples: {len(X_base)}, AS rate: {y_base.mean():.1%}")

    clf_base_results = run_classifier_sweep(X_base, y_base, pnl_base, "base")

    # ═══════════════════════════════════════════════════
    # Part B-2: AS Classifier sweep — Enriched features (n≈166)
    # ═══════════════════════════════════════════════════
    logger.info("\n" + "=" * 70)
    logger.info("Part B-2: AS Classifier (Enriched, spread-required)")
    logger.info("=" * 70)

    try:
        enriched_df = enrich_fill_records(df)
        X_enr, y_enr = build_enriched_as_features(enriched_df)
        pnl_enr = df.loc[X_enr.index, "post_fill_30s_pnl"].astype(float)
        logger.info(f"  Features: {X_enr.shape[1]}, Samples: {len(X_enr)}")

        clf_enr_results = run_classifier_sweep(X_enr, y_enr, pnl_enr, "enriched")
    except Exception as e:
        logger.warning(f"  Enriched features failed: {e}")
        clf_enr_results = []

    # ═══════════════════════════════════════════════════
    # Part B-3: Curated trade features (065# specification)
    # ═══════════════════════════════════════════════════
    logger.info("\n" + "=" * 70)
    logger.info("Part B-3: AS Classifier (Curated trade features)")
    logger.info("=" * 70)

    # Build enriched without requiring spread
    try:
        X_enr_all, y_enr_all = build_enriched_as_features(enriched_df)
        CURATED = [
            "log_queue_wait", "edge_bps", "vpin_60s", "vpin_30s", "vpin_300s",
            "price_velocity_bps", "buy_ratio", "tfi_300s", "hour_cos", "hour_sin",
            "side_aligned_tfi", "side_aligned_velocity",
        ]
        available_curated = [c for c in CURATED if c in X_enr_all.columns]
        if available_curated:
            X_cur = X_enr_all[available_curated]
            pnl_cur = df.loc[X_cur.index, "post_fill_30s_pnl"].astype(float)
            logger.info(f"  Curated features ({len(available_curated)}): {available_curated}")
            logger.info(f"  Samples: {len(X_cur)}")
            clf_cur_results = run_classifier_sweep(X_cur, y_enr_all, pnl_cur, "curated")
        else:
            logger.warning("  No curated features available")
            clf_cur_results = []
    except Exception as e:
        logger.warning(f"  Curated features failed: {e}")
        clf_cur_results = []

    # ═══════════════════════════════════════════════════
    # Part C: PnL Regressor
    # ═══════════════════════════════════════════════════
    logger.info("\n" + "=" * 70)
    logger.info("Part C: PnL Regressor")
    logger.info("=" * 70)

    # Use base features for regression on PnL
    X_reg = X_base.copy()
    y_reg = pnl_base.reindex(X_reg.index)
    valid_reg = ~y_reg.isna()
    X_reg = X_reg[valid_reg]
    y_reg = y_reg[valid_reg]
    logger.info(f"  Features: {X_reg.shape[1]}, Samples: {len(X_reg)}")

    reg_base_results = run_regressor_sweep(X_reg, y_reg, "base")

    # Enriched regressor
    try:
        X_pnl, y_pnl = build_pnl_features(enriched_df)
        logger.info(f"  Enriched PnL features: {X_pnl.shape[1]}, Samples: {len(X_pnl)}")
        reg_enr_results = run_regressor_sweep(X_pnl, y_pnl, "enriched")
    except Exception as e:
        logger.warning(f"  Enriched PnL regression failed: {e}")
        reg_enr_results = []

    # ═══════════════════════════════════════════════════
    # Part D: Side-specific models
    # ═══════════════════════════════════════════════════
    logger.info("\n" + "=" * 70)
    logger.info("Part D: Side-specific Models")
    logger.info("=" * 70)

    side_col = df.loc[X_base.index, "side"]
    side_results = run_side_specific(X_base, y_base, pnl_base, side_col)

    # ═══════════════════════════════════════════════════
    # Summary — Best models
    # ═══════════════════════════════════════════════════
    logger.info("\n" + "=" * 70)
    logger.info("SUMMARY: Top Models by skip20% PnL Improvement")
    logger.info("=" * 70)

    all_clf = clf_base_results + clf_enr_results + clf_cur_results
    all_clf_valid = [r for r in all_clf if r.get("skip20_improvement_bps") is not None]
    all_clf_valid.sort(key=lambda x: x.get("skip20_improvement_bps", -999), reverse=True)

    logger.info("\n--- Top 10 Classifiers (by skip20% improvement) ---")
    for i, r in enumerate(all_clf_valid[:10]):
        label = r.get("label", "?")
        roc = r.get("roc_auc_mean", "?")
        s20 = r.get("skip20_improvement_bps", "?")
        s30 = r.get("skip30_improvement_bps", "?")
        best_th = r.get("best_threshold", "?")
        best_pnl = r.get("best_threshold_pnl_bps", "?")
        logger.info(f"  #{i+1:2d} {label:50s}  ROC={roc}  skip20={s20:>7}  "
                     f"skip30={s30}  best_th={best_th}  best_pnl={best_pnl}")

    logger.info("\n--- Top 5 Regressors (by skip_neg improvement) ---")
    all_reg = reg_base_results + reg_enr_results
    all_reg_valid = [r for r in all_reg if r.get("skip_neg_improvement_bps") is not None]
    all_reg_valid.sort(key=lambda x: x.get("skip_neg_improvement_bps", -999), reverse=True)
    for i, r in enumerate(all_reg_valid[:5]):
        label = r.get("label", "?")
        ic = r.get("ic_mean", "?")
        sn = r.get("skip_neg_improvement_bps", "?")
        s20 = r.get("skip20_improvement_bps", "?")
        logger.info(f"  #{i+1:2d} {label:50s}  IC={ic}  skip_neg={sn}  skip20={s20}")

    logger.info("\n--- Side-specific ---")
    for r in side_results:
        label = r.get("label", "?")
        roc = r.get("roc_auc_mean", "?")
        s20 = r.get("skip20_improvement_bps", "?")
        logger.info(f"  {label:45s}  ROC={roc}  skip20={s20}")

    logger.info("\n--- Rule-based (top 5) ---")
    rule_sorted = sorted(rule_results, key=lambda x: x.get("improvement_bps", -999), reverse=True)
    for r in rule_sorted[:5]:
        logger.info(f"  {r['name']:35s}  improvement={r.get('improvement_bps', 0):>8} bps  "
                     f"skip_rate={r.get('skip_rate', 0):.1%}")

    # ═══════════════════════════════════════════════════
    # Save all results
    # ═══════════════════════════════════════════════════
    all_results = {
        "timestamp": datetime.now().isoformat(),
        "data_stats": {
            "total_records": len(df),
            "filled": int(filled.sum() if hasattr(filled, "sum") else len(filled)),
            "as_labeled": int(len(X_base)),
            "enriched_samples": int(len(X_enr)) if clf_enr_results else 0,
        },
        "rule_baselines": rule_results,
        "classifiers_base": clf_base_results,
        "classifiers_enriched": clf_enr_results,
        "classifiers_curated": clf_cur_results,
        "regressors_base": reg_base_results,
        "regressors_enriched": reg_enr_results,
        "side_specific": side_results,
    }

    out_file = output_dir / "model_search_results.json"
    write_json(out_file, all_results, indent=2, ensure_ascii=False, default=str)
    logger.info(f"\nAll results saved to {out_file}")


if __name__ == "__main__":
    main()
