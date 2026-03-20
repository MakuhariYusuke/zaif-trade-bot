"""121# Track B + D: SkipGate v2 再訓練パイプライン.

759 filled records で SkipGate AS 分類器を再訓練し、
B1 (baseline)・B2 (regime k=12)・B3 (buy/sell split)・D2 (OB features) を比較。

Usage:
    .venv\\Scripts\\python.exe scripts/v460/ml/train_sg_v2.py

出力:
    reports/v460/ml_121/sg_v2_comparison.json   — 全実験結果
    models/v460/skip_gate_as_v2.pkl             — 最良モデル (全体)
    models/v460/skip_gate_buy_v2.pkl            — buy 分割モデル (B3)
    models/v460/skip_gate_sell_v2.pkl           — sell 分割モデル (B3)
"""

from __future__ import annotations

import logging
import pickle
import sys
from dataclasses import asdict
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent.parent))

from scripts.v460.ml.cache_cleanup import clear_ml_data_caches_with_log
from scripts.v460.ml.data_loader import load_fill_records
from scripts.v460.ml.feature_enricher import (
    build_preorder_as_features,
    enrich_fill_records,
)
from scripts.v460.ml.walk_forward_as import expanding_window_splits, run_walk_forward
from ztb.io.json_io import write_json
from ztb.ml.skip_gate import SkipGate, SkipGateConfig
from ztb.ml.metadata_utils import current_iso_timestamp

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
)
logger = logging.getLogger(__name__)

REPORT_DIR = Path("reports/v460/ml_121")
MODEL_DIR = Path("models/v460")

# --- 121# §5.6 デプロイ判定基準 ---
DEPLOY_AUC_THRESHOLD = 0.55
DEPLOY_SKIP20_THRESHOLD = 0.3  # bps


def _make_pnl_series(fill_df: pd.DataFrame, index: pd.Index) -> pd.Series:
    """Walk-forward 用の PnL series を構築."""
    filled_mask = fill_df["filled"].astype(bool) & fill_df["adverse_selected_raw"].notna()
    pnl = fill_df.loc[filled_mask, "post_fill_30s_pnl"].astype(float)
    return pnl.reindex(index)


def _train_final_model(
    X: pd.DataFrame,
    y: pd.Series,
    k: int,
) -> "Pipeline":
    """全データで最終モデルを訓練 (デプロイ用)."""
    from sklearn.feature_selection import SelectKBest, f_classif
    from sklearn.impute import SimpleImputer
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    k_actual = min(k, X.shape[1])
    pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("selector", SelectKBest(f_classif, k=k_actual)),
        ("scaler", StandardScaler()),
        ("model", LogisticRegression(
            C=0.01, max_iter=2000, class_weight="balanced", random_state=42
        )),
    ])
    pipe.fit(X, y)
    return pipe


def _save_skip_gate(
    pipe: "Pipeline",
    X: pd.DataFrame,
    y: pd.Series,
    output_path: Path,
    experiment_name: str,
    k: int,
) -> Path:
    """Pipeline を SkipGate 形式で保存."""
    model = pipe.named_steps["model"]
    scaler = pipe.named_steps["scaler"]

    # Selected features 取得
    imputer = pipe.named_steps["imputer"]
    survived_mask = np.isfinite(imputer.statistics_)
    survived_cols = X.columns[survived_mask]
    selector = pipe.named_steps["selector"]
    selected_cols = survived_cols[selector.get_support()].tolist()

    fi = dict(zip(selected_cols, np.abs(model.coef_[0]).tolist()))
    sorted_fi = sorted(fi.items(), key=lambda x: x[1], reverse=True)

    gate = SkipGate(
        model=model,
        scaler=scaler,
        feature_cols=X.columns.tolist(),
        config=SkipGateConfig(
            mode="as",
            as_threshold=0.50,
            threshold_bps=0.0,
        ),
        metadata={
            "n_samples": len(X),
            "as_rate": float(y.mean()),
            "k": k,
            "selected_features": selected_cols,
            "feature_importances": dict(sorted_fi),
            "experiment": experiment_name,
            "trained_at": current_iso_timestamp(),
            "source": "121# Track B train_sg_v2.py",
        },
        pipeline=pipe,
    )
    return gate.save(output_path)


def _check_reverse_selection(
    X: pd.DataFrame,
    y: pd.Series,
    pnl: pd.Series,
    wf_result: dict,
) -> dict:
    """逆選別チェック: skip 群の PnL が keep 群より良い場合は逆選別."""
    skip_sim = wf_result.get("skip_simulation", {})
    skip20_improvement = skip_sim.get("skip20_improvement_bps", 0.0)
    is_reverse = skip20_improvement < 0
    return {
        "reverse_selection": is_reverse,
        "skip20_improvement_bps": skip20_improvement,
    }


def run_experiment(
    name: str,
    X: pd.DataFrame,
    y: pd.Series,
    pnl: pd.Series,
    *,
    k: int = 10,
    min_train: int = 50,
    step: int = 30,
    embargo: int = 2,
) -> dict:
    """単一実験の Walk-Forward 評価を実行."""
    logger.info(f"\n{'='*60}")
    logger.info(f"Experiment: {name}")
    logger.info(f"  Samples: {len(X)}, Features: {X.shape[1]}, k={k}")
    logger.info(f"{'='*60}")

    wf = run_walk_forward(
        X, y, pnl,
        min_train=min_train,
        step=step,
        embargo=embargo,
        k=k,
    )

    agg = wf.get("aggregate", {})
    skip = wf.get("skip_simulation", {})
    feat = wf.get("feature_stability", {})

    # 逆選別チェック
    reverse = _check_reverse_selection(X, y, pnl, wf)

    result = {
        "name": name,
        "n_samples": len(X),
        "n_features": X.shape[1],
        "k": k,
        "roc_auc_mean": agg.get("roc_auc_mean"),
        "roc_auc_std": agg.get("roc_auc_std"),
        "n_folds": agg.get("n_folds"),
        "skip20_improvement_bps": skip.get("skip20_improvement_bps", 0.0),
        "skip10_improvement_bps": skip.get("skip10_improvement_bps", 0.0),
        "baseline_pnl_bps": skip.get("baseline_pnl_bps", 0.0),
        "feature_stability": feat.get("jaccard_stability", 0.0),
        "always_selected": feat.get("always_selected", []),
        "reverse_selection": reverse["reverse_selection"],
        "deploy_candidate": (
            (agg.get("roc_auc_mean") or 0) > DEPLOY_AUC_THRESHOLD
            and skip.get("skip20_improvement_bps", 0.0) > DEPLOY_SKIP20_THRESHOLD
            and not reverse["reverse_selection"]
        ),
        "wf_detail": wf,
    }

    logger.info(f"  ROC-AUC: {result['roc_auc_mean']}")
    logger.info(f"  Skip20%: {result['skip20_improvement_bps']:+.3f} bps")
    logger.info(f"  Reverse selection: {result['reverse_selection']}")
    logger.info(f"  Deploy candidate: {result['deploy_candidate']}")

    return result


def main() -> None:
    try:
        _run_train_sg_v2_main()
    finally:
        clear_ml_data_caches_with_log(
            logger,
            context="train_sg_v2",
            collect_garbage=True,
        )


def _run_train_sg_v2_main() -> None:
    """121# Track B+D 全実験を実行."""

    # --- Step 0: データ読み込み + OB エンリッチ (D2) ---
    logger.info("Loading fill records...")
    df = load_fill_records()
    logger.info(f"Total records: {len(df)}")

    logger.info("Enriching with OB/trade features (Track D2)...")
    enriched_df = enrich_fill_records(df)

    # --- B1: Baseline (k=10, 現行構成) ---
    X_base, y_base = build_preorder_as_features(enriched_df)
    pnl_base = _make_pnl_series(df, X_base.index)
    experiments: list[dict] = []

    b1 = run_experiment("B1_baseline_k10", X_base, y_base, pnl_base, k=10)
    experiments.append(b1)

    # --- B2: regime 特徴量強制 include (k=12) ---
    b2 = run_experiment("B2_regime_k12", X_base, y_base, pnl_base, k=12)
    experiments.append(b2)

    # --- B2b: regime k=14 (追加探索) ---
    b2b = run_experiment("B2b_regime_k14", X_base, y_base, pnl_base, k=14)
    experiments.append(b2b)

    # --- B3: buy/sell 分割 ---
    side_col = enriched_df.loc[X_base.index, "side"]

    # Buy split
    buy_mask = side_col == "buy"
    if buy_mask.sum() >= 100:
        X_buy = X_base.loc[buy_mask]
        y_buy = y_base.loc[buy_mask]
        pnl_buy = pnl_base.loc[buy_mask]
        b3_buy = run_experiment(
            "B3_buy_only_k10", X_buy, y_buy, pnl_buy,
            k=10, min_train=40, step=20,
        )
        experiments.append(b3_buy)
    else:
        logger.warning(f"Buy samples too few: {buy_mask.sum()}")
        b3_buy = None

    # Sell split
    sell_mask = side_col == "sell"
    if sell_mask.sum() >= 100:
        X_sell = X_base.loc[sell_mask]
        y_sell = y_base.loc[sell_mask]
        pnl_sell = pnl_base.loc[sell_mask]
        b3_sell = run_experiment(
            "B3_sell_only_k10", X_sell, y_sell, pnl_sell,
            k=10, min_train=40, step=20,
        )
        experiments.append(b3_sell)
    else:
        logger.warning(f"Sell samples too few: {sell_mask.sum()}")
        b3_sell = None

    # --- D2: OB 特徴量の追加効果検証 ---
    # build_preorder_as_features は OB 列がある場合に自動で含める
    # ただし現行の build_preorder_as_features は OB 特徴量を追加していない
    # enrich_fill_records の出力に spread_bps_ob, depth_imbalance_ob がある場合、
    # 手動で追加して効果を確認
    ob_cols = ["spread_bps_ob", "depth_imbalance_ob"]
    has_ob = all(c in enriched_df.columns for c in ob_cols)
    ob_match_rate = 0.0
    if has_ob:
        ob_match_rate = float(enriched_df.loc[X_base.index, "spread_bps_ob"].notna().mean())
        logger.info(f"OB match rate: {ob_match_rate:.1%}")

        if ob_match_rate > 0.3:
            # OB 特徴量を X_base に追加
            X_ob = X_base.copy()
            for col in ob_cols:
                X_ob[col] = enriched_df.loc[X_base.index, col].astype(float)

            # side_aligned_imbalance も追加
            if "depth_imbalance_ob" in enriched_df.columns:
                side_sign = enriched_df.loc[X_base.index, "side"].map(
                    {"buy": 1.0, "sell": -1.0}
                ).astype(float)
                X_ob["side_aligned_imbalance"] = (
                    enriched_df.loc[X_base.index, "depth_imbalance_ob"].astype(float)
                    * side_sign
                ).fillna(0.0)

            d2 = run_experiment("D2_with_ob_k12", X_ob, y_base, pnl_base, k=12)
            experiments.append(d2)

            d2b = run_experiment("D2b_with_ob_k14", X_ob, y_base, pnl_base, k=14)
            experiments.append(d2b)
        else:
            logger.warning(f"OB match rate too low ({ob_match_rate:.1%}), skipping D2")

    # --- 結果比較テーブル ---
    logger.info("\n" + "=" * 80)
    logger.info("COMPARISON TABLE")
    logger.info("=" * 80)
    logger.info(f"{'Experiment':<25} {'AUC':>8} {'Skip20%':>10} {'Rev?':>5} {'Deploy?':>8}")
    logger.info("-" * 60)
    for exp in experiments:
        auc = exp.get("roc_auc_mean")
        auc_str = f"{auc:.4f}" if auc is not None else "N/A"
        s20 = exp.get("skip20_improvement_bps", 0.0)
        rev = "YES" if exp.get("reverse_selection") else "no"
        dep = "YES" if exp.get("deploy_candidate") else "no"
        logger.info(f"  {exp['name']:<23} {auc_str:>8} {s20:>+10.3f} {rev:>5} {dep:>8}")

    # --- 最良モデルの選択とデプロイ ---
    deploy_candidates = [e for e in experiments if e.get("deploy_candidate")]

    if deploy_candidates:
        # Skip20% PnL improvement が最大のものを選択
        best = max(deploy_candidates, key=lambda e: e.get("skip20_improvement_bps", 0.0))
        logger.info(f"\n*** Best deploy candidate: {best['name']} ***")
        logger.info(f"    AUC={best['roc_auc_mean']:.4f}, "
                     f"Skip20%={best['skip20_improvement_bps']:+.3f}bps")

        # 全体モデル (B1/B2/D2 系) の保存
        if "buy_only" not in best["name"] and "sell_only" not in best["name"]:
            pipe = _train_final_model(X_base, y_base, k=best["k"])
            # OB 特徴量付きの場合
            if "ob" in best["name"].lower() and has_ob:
                X_final = X_base.copy()
                for col in ob_cols:
                    X_final[col] = enriched_df.loc[X_base.index, col].astype(float)
                if "depth_imbalance_ob" in enriched_df.columns:
                    side_sign = enriched_df.loc[X_base.index, "side"].map(
                        {"buy": 1.0, "sell": -1.0}
                    ).astype(float)
                    X_final["side_aligned_imbalance"] = (
                        enriched_df.loc[X_base.index, "depth_imbalance_ob"].astype(float)
                        * side_sign
                    ).fillna(0.0)
                pipe = _train_final_model(X_final, y_base, k=best["k"])
                _save_skip_gate(
                    pipe, X_final, y_base,
                    MODEL_DIR / "skip_gate_as_v2.pkl",
                    best["name"], best["k"],
                )
            else:
                _save_skip_gate(
                    pipe, X_base, y_base,
                    MODEL_DIR / "skip_gate_as_v2.pkl",
                    best["name"], best["k"],
                )
            logger.info(f"Saved: models/v460/skip_gate_as_v2.pkl")
    else:
        logger.warning("No experiment meets deploy criteria!")
        # 最良のものでも保存 (比較用)
        if experiments:
            best_overall = max(experiments, key=lambda e: e.get("roc_auc_mean") or 0)
            logger.info(f"Best overall (not deployed): {best_overall['name']}")

    # --- B3 分割モデル保存 ---
    if b3_buy is not None and not b3_buy.get("reverse_selection", True):
        pipe_buy = _train_final_model(X_buy, y_buy, k=10)
        _save_skip_gate(
            pipe_buy, X_buy, y_buy,
            MODEL_DIR / "skip_gate_buy_v2.pkl",
            "B3_buy", 10,
        )
        logger.info("Saved: models/v460/skip_gate_buy_v2.pkl")

    if b3_sell is not None and not b3_sell.get("reverse_selection", True):
        pipe_sell = _train_final_model(X_sell, y_sell, k=10)
        _save_skip_gate(
            pipe_sell, X_sell, y_sell,
            MODEL_DIR / "skip_gate_sell_v2.pkl",
            "B3_sell", 10,
        )
        logger.info("Saved: models/v460/skip_gate_sell_v2.pkl")
    elif b3_sell is not None:
        logger.warning("B3 sell model has reverse selection — NOT deploying")

    # --- レポート保存 ---
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    report = {
        "generated_at": current_iso_timestamp(),
        "source": "121# Track B+D train_sg_v2.py",
        "data_summary": {
            "total_records": len(df),
            "filled_records": int(enriched_df["filled"].astype(bool).sum()),
            "n_features_base": X_base.shape[1],
            "ob_match_rate": ob_match_rate,
        },
        "deploy_criteria": {
            "roc_auc_threshold": DEPLOY_AUC_THRESHOLD,
            "skip20_threshold_bps": DEPLOY_SKIP20_THRESHOLD,
        },
        "experiments": [
            {k: v for k, v in exp.items() if k != "wf_detail"}
            for exp in experiments
        ],
    }
    report_path = REPORT_DIR / "sg_v2_comparison.json"
    write_json(report_path, report, indent=2, ensure_ascii=False, default=str)
    logger.info(f"\nReport saved to {report_path}")

    # 詳細 WF 結果も保存
    detail_path = REPORT_DIR / "sg_v2_wf_details.json"
    write_json(
        detail_path,
        {exp["name"]: exp["wf_detail"] for exp in experiments},
        indent=2,
        ensure_ascii=False,
        default=str,
    )
    logger.info(f"WF details saved to {detail_path}")


if __name__ == "__main__":
    main()
