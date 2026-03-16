"""070# Part 3: 最有力候補の詳細検証 + 再学習.

Part 1/2 で判明した知見を基に:
1. 最有力候補 (enriched_LR_C0.1_k5) の詳細 walk-forward
2. PnL 回帰器 (enriched_GBR_n30_d2_k8, IC=0.062) の実用性検証
3. Queue-wait による分層分析 (唯一の正のPnLセグメント)
4. 複合戦略の検証 (rule + ML ハイブリッド)
5. 最終候補モデルの再学習と保存
"""

from __future__ import annotations

import logging
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from scipy.stats import spearmanr

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent.parent))

from scripts.v460.ml.cache_cleanup import clear_ml_data_caches_with_log
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


def detailed_walk_forward_analysis(
    X: pd.DataFrame, y: pd.Series, pnl: pd.Series, label: str,
    make_pipeline_fn, *, min_train: int = 50, step: int = 15, embargo: int = 2,
    is_regressor: bool = False,
) -> dict:
    """最有力候補の詳細 WF 分析."""
    splits = expanding_window_splits(len(X), min_train=min_train, step=step, embargo=embargo)
    if not splits:
        return {"error": "insufficient_data"}

    oof_preds = np.full(len(X), np.nan)
    fold_details = []

    for fold_i, (train_idx, test_idx) in enumerate(splits):
        pipe = make_pipeline_fn()
        X_tr, y_tr = X.iloc[train_idx], y.iloc[train_idx]
        X_te, y_te = X.iloc[test_idx], y.iloc[test_idx]
        pipe.fit(X_tr, y_tr)

        if is_regressor:
            preds = pipe.predict(X_te)
            oof_preds[test_idx] = preds
            ic, _ = spearmanr(preds, y_te.values)
            mae = float(np.mean(np.abs(preds - y_te.values)))
            fold_details.append({
                "fold": fold_i, "n_train": len(train_idx), "n_test": len(test_idx),
                "ic": round(float(ic) if not np.isnan(ic) else 0.0, 4),
                "mae": round(mae, 4),
            })
        else:
            probs = pipe.predict_proba(X_te)[:, 1]
            oof_preds[test_idx] = probs
            roc = roc_auc_score(y_te, probs) if len(np.unique(y_te)) > 1 else 0.5
            fold_details.append({
                "fold": fold_i, "n_train": len(train_idx), "n_test": len(test_idx),
                "roc_auc": round(float(roc), 4),
            })

    valid = ~np.isnan(oof_preds) & ~np.isnan(pnl.values)
    vp = oof_preds[valid]
    vpnl = pnl.values[valid]
    baseline = float(np.mean(vpnl))

    logger.info(f"\n{'='*60}")
    logger.info(f"Detailed WF: {label}")
    logger.info(f"{'='*60}")
    logger.info(f"  Folds: {len(splits)}, OOF valid: {valid.sum()}")
    logger.info(f"  Baseline PnL: {baseline:.4f} bps")

    for fd in fold_details:
        metric_key = "ic" if is_regressor else "roc_auc"
        logger.info(f"  Fold {fd['fold']}: train={fd['n_train']}, test={fd['n_test']}, "
                     f"{metric_key}={fd[metric_key]}")

    # Threshold sweep (fine-grained)
    logger.info(f"\n  Threshold sweep:")
    best_improvement = 0.0
    best_config = None
    sweep_results = []

    if is_regressor:
        # For regressor: skip predicted PnL < threshold
        for th in np.arange(-3.0, 3.0, 0.25):
            keep = vp >= th
            n_keep = int(keep.sum())
            skip_rate = 1 - n_keep / len(vp)
            if n_keep >= 10 and skip_rate >= 0.05 and skip_rate <= 0.8:
                kept = float(np.mean(vpnl[keep]))
                impr = kept - baseline
                sweep_results.append({
                    "threshold": round(float(th), 2),
                    "n_keep": n_keep,
                    "skip_rate": round(skip_rate, 3),
                    "kept_pnl": round(kept, 4),
                    "improvement": round(impr, 4),
                })
                if impr > best_improvement:
                    best_improvement = impr
                    best_config = {"threshold": round(float(th), 2), "improvement": round(impr, 4),
                                   "kept_pnl": round(kept, 4), "skip_rate": round(skip_rate, 3)}
                logger.info(f"    th={th:+6.2f}: skip={skip_rate:.1%}, n_keep={n_keep}, "
                             f"kept_pnl={kept:+.4f}, improvement={impr:+.4f}")
    else:
        # For classifier: skip P(AS) > threshold
        for th in np.arange(0.30, 0.75, 0.025):
            keep = vp < th
            n_keep = int(keep.sum())
            skip_rate = 1 - n_keep / len(vp)
            if n_keep >= 10 and skip_rate >= 0.05 and skip_rate <= 0.8:
                kept = float(np.mean(vpnl[keep]))
                impr = kept - baseline
                sweep_results.append({
                    "threshold": round(float(th), 3),
                    "n_keep": n_keep,
                    "skip_rate": round(skip_rate, 3),
                    "kept_pnl": round(kept, 4),
                    "improvement": round(impr, 4),
                })
                if impr > best_improvement:
                    best_improvement = impr
                    best_config = {"threshold": round(float(th), 3), "improvement": round(impr, 4),
                                   "kept_pnl": round(kept, 4), "skip_rate": round(skip_rate, 3)}
                logger.info(f"    th={th:.3f}: skip={skip_rate:.1%}, n_keep={n_keep}, "
                             f"kept_pnl={kept:+.4f}, improvement={impr:+.4f}")

    if best_config:
        logger.info(f"\n  BEST: th={best_config['threshold']}, "
                     f"improvement={best_config['improvement']:+.4f} bps, "
                     f"skip_rate={best_config['skip_rate']:.1%}")

    return {
        "label": label,
        "n_folds": len(splits),
        "n_oof_valid": int(valid.sum()),
        "baseline_pnl": round(baseline, 4),
        "fold_details": fold_details,
        "sweep_results": sweep_results,
        "best_config": best_config,
    }


def hybrid_strategy_test(df: pd.DataFrame) -> dict:
    """Rule-based + ML ハイブリッド戦略の OOF 検証."""
    logger.info(f"\n{'='*60}")
    logger.info("Hybrid Strategy Test")
    logger.info(f"{'='*60}")

    filled = df[df["filled"].astype(bool)].copy().sort_values("timestamp").reset_index(drop=True)
    pnl = filled["post_fill_30s_pnl"].astype(float)
    n = len(filled)
    baseline = float(pnl.mean())

    results = []

    # Strategy 1: Buy-only (skip all sells)
    keep_buy = filled["side"] == "buy"
    kept_pnl_buy = float(pnl[keep_buy].mean())
    results.append({
        "strategy": "buy_only",
        "n_keep": int(keep_buy.sum()),
        "skip_rate": round(1 - keep_buy.mean(), 3),
        "kept_pnl": round(kept_pnl_buy, 4),
        "improvement": round(kept_pnl_buy - baseline, 4),
    })
    logger.info(f"  buy_only: kept_pnl={kept_pnl_buy:+.4f}, improvement={kept_pnl_buy - baseline:+.4f}")

    # Strategy 2: Skip sells during bad sell hours (sell PnL < -2 bps hours)
    hours_utc = compute_utc_hour(filled["timestamp"])

    # WF test: train on first half
    mid = n // 2
    train_window = filled.iloc[:mid]
    train_sell = train_window[train_window["side"] == "sell"]
    train_sell_hours = hours_utc.iloc[:mid].loc[train_sell.index]
    train_sell_pnl = train_sell["post_fill_30s_pnl"].astype(float)
    sell_hour_pnl = train_sell_pnl.groupby(train_sell_hours).mean()
    bad_sell_hours = set(sell_hour_pnl[sell_hour_pnl < -1.5].index.tolist())

    test_data = filled.iloc[mid:]
    test_pnl = pnl.iloc[mid:]
    test_hours = hours_utc.iloc[mid:]
    test_sides = test_data["side"]

    skip_mask = (test_sides == "sell") & test_hours.isin(bad_sell_hours)
    keep_mask = ~skip_mask
    n_keep = int(keep_mask.sum())
    if n_keep > 0:
        kept = float(test_pnl[keep_mask].mean())
        test_baseline = float(test_pnl.mean())
        results.append({
            "strategy": "skip_sell_bad_hours_WF",
            "bad_sell_hours": sorted(bad_sell_hours),
            "n_keep": n_keep,
            "skip_rate": round(1 - n_keep / len(test_data), 3),
            "test_baseline": round(test_baseline, 4),
            "kept_pnl": round(kept, 4),
            "improvement": round(kept - test_baseline, 4),
        })
        logger.info(f"  skip_sell_bad_hours: bad={sorted(bad_sell_hours)}, "
                     f"kept_pnl={kept:+.4f}, improvement={kept - test_baseline:+.4f}")

    # Strategy 3: Queue wait > 60s only
    qw = filled["queue_wait_sec"].astype(float)
    keep_qw = qw >= 60
    n_keep_qw = int(keep_qw.sum())
    if n_keep_qw >= 10:
        kept_qw = float(pnl[keep_qw].mean())
        results.append({
            "strategy": "queue_wait_60s+",
            "n_keep": n_keep_qw,
            "skip_rate": round(1 - n_keep_qw / n, 3),
            "kept_pnl": round(kept_qw, 4),
            "improvement": round(kept_qw - baseline, 4),
        })
        logger.info(f"  queue_wait_60s+: n={n_keep_qw}, kept={kept_qw:+.4f}, imp={kept_qw-baseline:+.4f}")

    # Strategy 4: Wider offset simulation
    # If we increase offset, we expect:
    # - Fewer fills (lower fill rate)
    # - Each fill has more edge
    # Proxy: edge_bps at order time
    if "mid_at_fill" in filled.columns and "fill_price" in filled.columns:
        mid_at_fill = filled["mid_at_fill"].astype(float)
        fill_price = filled["fill_price"].astype(float)
        side_sign = filled["side"].map({"buy": -1, "sell": 1}).astype(float)
        edge = (fill_price - mid_at_fill) / mid_at_fill * 10000 * side_sign

        # Current edge stats
        logger.info(f"\n  Edge at fill: mean={edge.mean():.4f}, median={edge.median():.4f}")

        # If we only keep fills with edge > X bps
        for edge_th in [0, 1, 2, 3]:
            keep = edge >= edge_th
            n_k = int(keep.sum())
            if n_k >= 10:
                kp = float(pnl[keep].mean())
                logger.info(f"  edge>={edge_th}: n={n_k}, kept_pnl={kp:+.4f}, "
                             f"improvement={kp-baseline:+.4f}")
                results.append({
                    "strategy": f"edge_filter_{edge_th}bps",
                    "n_keep": n_k,
                    "skip_rate": round(1 - n_k / n, 3),
                    "kept_pnl": round(kp, 4),
                    "improvement": round(kp - baseline, 4),
                })

    return {"strategies": results}


def train_and_save_best_models(df: pd.DataFrame) -> dict:
    """最有力モデル候補を学習して保存."""
    logger.info(f"\n{'='*60}")
    logger.info("Train & Save Best Model Candidates")
    logger.info(f"{'='*60}")

    output_dir = Path("models/v460")
    output_dir.mkdir(parents=True, exist_ok=True)

    results = {}

    # --- Candidate 1: enriched_LR_C0.1_k5 (AS分類器, best OOF) ---
    try:
        enriched_df = enrich_fill_records(df)
        X_enr, y_enr = build_enriched_as_features(enriched_df)

        pipe1 = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("selector", SelectKBest(f_classif, k=5)),
            ("scaler", StandardScaler()),
            ("model", LogisticRegression(C=0.1, max_iter=2000, class_weight="balanced", random_state=42)),
        ])
        pipe1.fit(X_enr, y_enr)

        # Selected features
        imputer = pipe1.named_steps["imputer"]
        survived = np.isfinite(imputer.statistics_)
        survived_cols = X_enr.columns[survived]
        selected = survived_cols[pipe1.named_steps["selector"].get_support()].tolist()

        # Coefficients
        coefs = pipe1.named_steps["model"].coef_[0]
        coef_dict = dict(zip(selected, coefs.tolist()))

        path1 = output_dir / "skip_gate_as_070_candidate1.pkl"
        with open(path1, "wb") as f:
            pickle.dump(pipe1, f)
        logger.info(f"  Candidate 1 saved: {path1}")
        logger.info(f"  Selected features: {selected}")
        logger.info(f"  Coefficients: {coef_dict}")
        logger.info(f"  Samples: {len(X_enr)}, AS rate: {y_enr.mean():.1%}")

        results["candidate1"] = {
            "path": str(path1),
            "type": "AS_classifier",
            "model": "LR(C=0.1, k=5)",
            "features": selected,
            "coefficients": {k: round(v, 4) for k, v in coef_dict.items()},
            "n_samples": len(X_enr),
            "as_rate": round(float(y_enr.mean()), 4),
        }
    except Exception as e:
        logger.warning(f"  Candidate 1 failed: {e}")

    # --- Candidate 2: base_LR_C0.01_kAll (AS分類器, trade-only) ---
    X_base, y_base = build_as_features(df, require_spread=False)
    pipe2 = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("model", LogisticRegression(C=0.01, max_iter=2000, class_weight="balanced", random_state=42)),
    ])
    pipe2.fit(X_base, y_base)

    coefs2 = pipe2.named_steps["model"].coef_[0]
    coef_dict2 = dict(zip(X_base.columns.tolist(), coefs2.tolist()))

    path2 = output_dir / "skip_gate_as_070_candidate2.pkl"
    with open(path2, "wb") as f:
        pickle.dump(pipe2, f)
    logger.info(f"  Candidate 2 saved: {path2}")
    logger.info(f"  Features: {X_base.columns.tolist()}")
    logger.info(f"  Coefficients: {coef_dict2}")

    results["candidate2"] = {
        "path": str(path2),
        "type": "AS_classifier",
        "model": "LR(C=0.01, all features, trade-only)",
        "features": X_base.columns.tolist(),
        "coefficients": {k: round(v, 4) for k, v in coef_dict2.items()},
        "n_samples": len(X_base),
    }

    # --- Candidate 3: PnL Regressor (enriched_GBR_n30_d2_k8) ---
    try:
        X_pnl, y_pnl = build_pnl_features(enriched_df)
        pipe3 = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("selector", SelectKBest(f_classif, k=8)),
            ("scaler", StandardScaler()),
            ("model", GradientBoostingRegressor(
                n_estimators=30, max_depth=2, learning_rate=0.05,
                subsample=0.8, random_state=42,
            )),
        ])
        pipe3.fit(X_pnl, y_pnl)

        # Selected features
        imp3 = pipe3.named_steps["imputer"]
        surv3 = np.isfinite(imp3.statistics_)
        surv_cols3 = X_pnl.columns[surv3]
        sel3 = surv_cols3[pipe3.named_steps["selector"].get_support()].tolist()

        path3 = output_dir / "pnl_regressor_070_candidate3.pkl"
        with open(path3, "wb") as f:
            pickle.dump(pipe3, f)
        logger.info(f"  Candidate 3 saved: {path3}")
        logger.info(f"  Selected features: {sel3}")
        logger.info(f"  Samples: {len(X_pnl)}")

        # Feature importances
        fi = dict(zip(sel3, pipe3.named_steps["model"].feature_importances_.tolist()))

        results["candidate3"] = {
            "path": str(path3),
            "type": "PnL_regressor",
            "model": "GBR(n=30, d=2, k=8)",
            "features": sel3,
            "feature_importances": {k: round(v, 4) for k, v in sorted(fi.items(), key=lambda x: -x[1])},
            "n_samples": len(X_pnl),
        }
    except Exception as e:
        logger.warning(f"  Candidate 3 failed: {e}")

    return results


def main() -> None:
    try:
        _run_070_final_analysis_main()
    finally:
        clear_ml_data_caches_with_log(
            logger,
            context="run_070_final_analysis",
            collect_garbage=True,
        )


def _run_070_final_analysis_main() -> None:
    output_dir = Path("reports/v460/model_search_070")
    output_dir.mkdir(parents=True, exist_ok=True)

    df = load_fill_records()
    enriched_df = enrich_fill_records(df)

    all_results = {}

    # 1. Detailed WF for top AS classifier
    X_enr, y_enr = build_enriched_as_features(enriched_df)
    pnl_enr = df.loc[X_enr.index, "post_fill_30s_pnl"].astype(float)

    def make_lr_pipe():
        return Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("selector", SelectKBest(f_classif, k=5)),
            ("scaler", StandardScaler()),
            ("model", LogisticRegression(C=0.1, max_iter=2000, class_weight="balanced", random_state=42)),
        ])

    all_results["wf_lr_c01_k5"] = detailed_walk_forward_analysis(
        X_enr, y_enr, pnl_enr, "enriched_LR_C0.1_k5", make_lr_pipe, step=15
    )

    # 2. Detailed WF for PnL regressor
    X_pnl, y_pnl = build_pnl_features(enriched_df)

    # Need PnL for simulation (use itself since it's the target)
    def make_gbr_pipe():
        return Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("selector", SelectKBest(f_classif, k=8)),
            ("scaler", StandardScaler()),
            ("model", GradientBoostingRegressor(
                n_estimators=30, max_depth=2, learning_rate=0.05,
                subsample=0.8, random_state=42,
            )),
        ])

    all_results["wf_gbr_k8"] = detailed_walk_forward_analysis(
        X_pnl, y_pnl, y_pnl, "enriched_GBR_n30_d2_k8", make_gbr_pipe,
        step=20, is_regressor=True
    )

    # 3. Base LR with all features
    X_base, y_base = build_as_features(df, require_spread=False)
    pnl_base = df.loc[X_base.index, "post_fill_30s_pnl"].astype(float)

    def make_base_lr():
        return Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("model", LogisticRegression(C=0.01, max_iter=2000, class_weight="balanced", random_state=42)),
        ])

    all_results["wf_base_lr"] = detailed_walk_forward_analysis(
        X_base, y_base, pnl_base, "base_LR_C0.01_kAll", make_base_lr, step=20
    )

    # 4. Hybrid strategies
    all_results["hybrid"] = hybrid_strategy_test(df)

    # 5. Train and save candidates
    all_results["saved_models"] = train_and_save_best_models(df)

    # Save results
    out_file = output_dir / "final_analysis_results.json"
    write_json(out_file, all_results, indent=2, ensure_ascii=False, default=str)
    logger.info(f"\nResults saved to {out_file}")


if __name__ == "__main__":
    main()
