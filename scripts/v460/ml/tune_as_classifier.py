"""060# Model Tuning: AS分類器のハイパーパラメータ探索.

複数の (model, regularization, feature_selection) 組み合わせを TSCV で評価.
Skip policy PnL improvement を最適化目標とする.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent.parent))

import logging

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import TimeSeriesSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from scripts.v460.ml.data_loader import load_fill_records
from scripts.v460.ml.feature_enricher import (
    build_enriched_as_features,
    enrich_fill_records,
)
from ztb.io.json_io import write_json

logging.basicConfig(
    level=logging.WARNING,
    format="%(asctime)s %(levelname)s: %(message)s",
)
logger = logging.getLogger(__name__)


def evaluate_config(
    X: pd.DataFrame,
    y: pd.Series,
    pnl: pd.Series,
    splits: list[tuple[np.ndarray, np.ndarray]],
    *,
    model_type: str,
    C: float | None = None,
    n_estimators: int | None = None,
    max_depth: int | None = None,
    learning_rate: float | None = None,
    k_select: int | None = None,
    penalty: str = "l2",
) -> dict:
    """単一構成の TSCV 評価."""
    X_values = X.to_numpy(dtype=np.float32, copy=False)
    y_values = y.to_numpy(copy=False)
    pnl_values = pnl.to_numpy(dtype=np.float64, copy=False)
    roc_aucs: list[float] = []
    pr_aucs: list[float] = []
    oof_probs = np.full(len(X), np.nan)

    for train_idx, test_idx in splits:
        if model_type == "lr":
            clf = LogisticRegression(
                C=C or 1.0,
                penalty=penalty,
                solver="saga" if penalty == "l1" else "lbfgs",
                max_iter=2000,
                class_weight="balanced",
                random_state=42,
            )
        else:
            clf = GradientBoostingClassifier(
                n_estimators=n_estimators or 50,
                max_depth=max_depth or 2,
                learning_rate=learning_rate or 0.05,
                subsample=0.8,
                random_state=42,
            )

        steps: list[tuple[str, object]] = [("imputer", SimpleImputer(strategy="median"))]
        if k_select is not None:
            k = min(k_select, X_values.shape[1])
            steps.append(("selector", SelectKBest(f_classif, k=k)))
        steps.extend([
            ("scaler", StandardScaler()),
            ("model", clf),
        ])

        pipe = Pipeline(steps)
        pipe.fit(X_values[train_idx], y_values[train_idx])
        probs = pipe.predict_proba(X_values[test_idx])[:, 1]

        y_test = y_values[test_idx]
        if len(np.unique(y_test)) > 1:
            roc_aucs.append(roc_auc_score(y_test, probs))
            pr_aucs.append(average_precision_score(y_test, probs))
        oof_probs[test_idx] = probs
        del pipe

    # Skip policy simulation on OOF
    valid_mask = ~np.isnan(oof_probs) & ~np.isnan(pnl_values)
    if valid_mask.sum() <= 1:
        return {
            "roc_auc": float(np.mean(roc_aucs)) if roc_aucs else 0.0,
            "pr_auc": float(np.mean(pr_aucs)) if pr_aucs else 0.0,
            "skip20_improvement": 0.0,
            "skip10_improvement": 0.0,
        }

    oof_valid = oof_probs[valid_mask]
    pnl_valid = pnl_values[valid_mask]

    # Skip 20%
    n_skip_20 = max(1, int(len(oof_valid) * 0.2))
    top_20 = np.argsort(oof_valid)[-n_skip_20:]
    keep_20 = np.ones(len(oof_valid), dtype=bool)
    keep_20[top_20] = False
    pnl_kept_20 = float(np.nanmean(pnl_valid[keep_20]))
    pnl_baseline = float(np.nanmean(pnl_valid))
    improvement_20 = pnl_kept_20 - pnl_baseline

    # Skip 10%
    n_skip_10 = max(1, int(len(oof_valid) * 0.1))
    top_10 = np.argsort(oof_valid)[-n_skip_10:]
    keep_10 = np.ones(len(oof_valid), dtype=bool)
    keep_10[top_10] = False
    pnl_kept_10 = float(np.nanmean(pnl_valid[keep_10]))
    improvement_10 = pnl_kept_10 - pnl_baseline

    return {
        "roc_auc": float(np.mean(roc_aucs)) if roc_aucs else 0.0,
        "pr_auc": float(np.mean(pr_aucs)) if pr_aucs else 0.0,
        "skip20_improvement": improvement_20,
        "skip10_improvement": improvement_10,
    }


def main() -> None:
    import warnings
    warnings.filterwarnings("ignore")

    df = load_fill_records()
    enriched = enrich_fill_records(df)
    X, y = build_enriched_as_features(enriched, require_spread=True)

    # PnL for skip simulation
    filled_mask = df["filled"].astype(bool) & df["adverse_selected_raw"].notna()
    pnl = df.loc[filled_mask, "post_fill_30s_pnl"].astype(float).reindex(X.index)

    print(f"Data: {X.shape[0]} samples, {X.shape[1]} features")
    print(f"AS rate: {y.mean():.1%}, Baseline PnL: {pnl.mean():.3f} bps")
    print()

    # CV split は全構成で共通: 毎回生成し直さない
    splits = list(TimeSeriesSplit(n_splits=5).split(X))

    configs = []

    # LR configs
    for C_val in [0.01, 0.1, 0.5, 1.0, 5.0]:
        for penalty in ["l1", "l2"]:
            for k in [None, 8, 12, 15]:
                configs.append({
                    "model_type": "lr",
                    "C": C_val,
                    "penalty": penalty,
                    "k_select": k,
                    "label": f"LR(C={C_val},{penalty},k={k})",
                })

    # GB configs
    for n_est in [30, 50, 100]:
        for max_d in [2, 3]:
            for lr_val in [0.03, 0.05, 0.1]:
                for k in [None, 10, 15]:
                    configs.append({
                        "model_type": "gb",
                        "n_estimators": n_est,
                        "max_depth": max_d,
                        "learning_rate": lr_val,
                        "k_select": k,
                        "label": f"GB(n={n_est},d={max_d},lr={lr_val},k={k})",
                    })

    print(f"Evaluating {len(configs)} configurations...")
    print("-" * 90)
    print(f"{'Config':<40} {'ROC-AUC':>8} {'PR-AUC':>8} {'Skip20%':>10} {'Skip10%':>10}")
    print("-" * 90)

    results = []
    best_skip20 = -999.0
    best_config = ""

    for cfg in configs:
        label = cfg.pop("label")
        try:
            r = evaluate_config(X, y, pnl, splits, **cfg)
        except Exception as e:
            cfg["label"] = label
            continue

        skip20 = r["skip20_improvement"]
        skip10 = r["skip10_improvement"]
        marker = " ***" if skip20 > best_skip20 else ""
        if skip20 > best_skip20:
            best_skip20 = skip20
            best_config = label

        print(f"{label:<40} {r['roc_auc']:>8.4f} {r['pr_auc']:>8.4f} {skip20:>+10.3f} {skip10:>+10.3f}{marker}")

        cfg["label"] = label
        results.append({"config": label, **r})

    print("-" * 90)
    print(f"\nBest by Skip20%: {best_config} → {best_skip20:+.3f} bps")

    # Top 10
    top10 = sorted(results, key=lambda x: x["skip20_improvement"], reverse=True)[:10]
    print(f"\nTop 10 configs by Skip20%:")
    for i, r in enumerate(top10):
        print(f"  {i+1}. {r['config']}: Skip20%={r['skip20_improvement']:+.3f}, ROC={r['roc_auc']:.4f}")

    out = Path("reports/v460/ml_060e")
    out.mkdir(parents=True, exist_ok=True)
    write_json(out / "tuning_results.json", results, indent=2)
    print(f"\nResults saved to {out / 'tuning_results.json'}")


if __name__ == "__main__":
    main()
