"""125# SkipGate v4 デプロイ: LGBM 回帰 (PnL120 直接予測) + OB 特徴量.

WF 評価結果 (train_sg_v3.py):
  LGBM_reg_pnl120_regression_full (base + OB):
    S20%_30=+0.075 bps, S20%_120=+0.324 bps (non-reverse both)
    profit_score=+0.249

  比較: 現行 GBM_sklearn_really_bad30_base:
    S20%_30=+0.117, S20%_120=+0.221, score=+0.190

  改善: score +31%, PnL120 skip improvement +47%

Usage:
    .venv\\Scripts\\python.exe scripts/v460/ml/deploy_sg_v4.py
"""

from __future__ import annotations

import logging
import pickle
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent.parent))

from scripts.v460.ml.data_loader import load_fill_records
from scripts.v460.ml.feature_enricher import (
    build_preorder_as_features,
    enrich_fill_records,
)
from scripts.v460.ml.skip_eval_utils import compute_skip_slice_metrics
from scripts.v460.ml.skip_gate import (
    SkipGate,
    SkipGateConfig,
    _BASE_FEATURE_COLS,
    _OB_FEATURE_COLS,
    get_gate_feature_cols,
)
from ztb.io.json_io import write_json

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
)
logger = logging.getLogger(__name__)

MODEL_DIR = Path("models/v460")
REPORT_DIR = Path("reports/v460/ml_125")

# OB 特徴量込みの全特徴量
_FULL_FEATURE_COLS = get_gate_feature_cols(use_ob=True)


def load_and_prepare_data() -> tuple[pd.DataFrame, pd.Series, pd.DataFrame]:
    """Fill records をロードし、特徴量 (OB 込み) と PnL120 ターゲットを構築.

    Returns:
        (X_full, y_pnl120, enriched_df)
    """
    records = load_fill_records()
    logger.info(f"Loaded {len(records)} records")

    enriched = enrich_fill_records(records)
    logger.info(f"Enriched: {len(enriched)} records")

    # base features
    X_base, y_as = build_preorder_as_features(enriched)
    logger.info(f"Base features: {X_base.shape}, AS rate={y_as.mean():.3f}")

    # OB 特徴量を追加 (学習データ: enriched_df から取得)
    X_full = X_base.copy()
    ob_cols_in_enriched = {
        "spread_bps_ob": "spread_bps_ob",
        "depth_imbalance_ob": "depth_imbalance_ob",
    }
    for feat_col, df_col in ob_cols_in_enriched.items():
        if df_col in enriched.columns:
            X_full[feat_col] = enriched.loc[X_base.index, df_col].astype(float)
        else:
            X_full[feat_col] = np.nan
            logger.warning(f"OB feature '{df_col}' not in enriched, filling NaN")

    # side_aligned_imbalance: OB depth_imbalance × side_sign
    if "depth_imbalance_ob" in enriched.columns:
        side_sign = enriched.loc[X_base.index, "side"].map(
            {"buy": 1.0, "sell": -1.0}
        ).astype(float)
        X_full["side_aligned_imbalance"] = (
            enriched.loc[X_base.index, "depth_imbalance_ob"].astype(float)
            * side_sign
        ).fillna(0.0)
    else:
        X_full["side_aligned_imbalance"] = 0.0

    # 特徴量契約: _FULL_FEATURE_COLS と一致させる
    for col in _FULL_FEATURE_COLS:
        if col not in X_full.columns:
            X_full[col] = np.nan
            logger.warning(f"Missing feature '{col}', filling NaN")
    X_full = X_full[_FULL_FEATURE_COLS]
    logger.info(f"Full feature matrix: {X_full.shape}")

    # OB 特徴量の充足率
    for col in _OB_FEATURE_COLS:
        notna = X_full[col].notna().sum()
        logger.info(f"  OB feature '{col}': {notna}/{len(X_full)} ({notna/len(X_full):.1%})")

    # ターゲット: PnL120 (回帰)
    pnl120_col = "post_fill_120s_pnl"
    if pnl120_col in enriched.columns:
        y_pnl120 = enriched.loc[X_base.index, pnl120_col].astype(float)
    else:
        raise ValueError("post_fill_120s_pnl not available in enriched data")

    valid_120 = y_pnl120.notna()
    logger.info(
        f"PnL120 target: {valid_120.sum()}/{len(y_pnl120)} ({valid_120.mean():.1%}) available"
    )
    logger.info(
        f"PnL120 stats: mean={y_pnl120.mean():.3f}, median={y_pnl120.median():.3f}, "
        f"std={y_pnl120.std():.3f}"
    )

    # PnL120 の無い行は除外
    valid_mask = valid_120
    X_full = X_full.loc[valid_mask]
    y_pnl120 = y_pnl120.loc[valid_mask]
    logger.info(f"After PnL120 filter: {X_full.shape}")

    return X_full, y_pnl120, enriched


def train_and_save_model(
    X: pd.DataFrame,
    y: pd.Series,
) -> Path:
    """LGBM 回帰 Pipeline を訓練し SkipGate 形式で保存.

    Returns:
        保存先パス.
    """
    try:
        import lightgbm as lgb
    except ImportError:
        raise RuntimeError("LightGBM required: pip install lightgbm")

    # LGBM ハイパーパラメータ (WF 評価と同一)
    lgbm = lgb.LGBMRegressor(
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

    # Pipeline: Imputer → Scaler → LGBM
    pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("model", lgbm),
    ])

    logger.info("Training LGBM regression pipeline on full data...")
    pipeline.fit(X, y)

    # 訓練データでの予測分布確認
    preds = pipeline.predict(X)
    logger.info(
        f"Predictions on train: "
        f"mean={np.mean(preds):.3f}, "
        f"median={np.median(preds):.3f}, "
        f"std={np.std(preds):.3f}, "
        f"P10={np.percentile(preds, 10):.3f}, "
        f"P25={np.percentile(preds, 25):.3f}, "
        f"P75={np.percentile(preds, 75):.3f}, "
        f"P90={np.percentile(preds, 90):.3f}"
    )

    # SkipGateConfig: mode=pnl, predicted_pnl < threshold → skip
    # 閾値: PnL120 予測が負の場合にスキップ (threshold_bps=0.0)
    config = SkipGateConfig(
        mode="pnl",
        enabled=True,
        buy_enabled=True,
        sell_enabled=True,
        threshold_bps=0.0,  # PnL予測 < 0 → skip
        as_threshold=0.50,  # mode=pnl では未使用
        as_threshold_buy=0.50,
        as_threshold_sell=0.50,
        max_skip_rate=0.3,
        use_ob_features=True,  # 125#: OB 特徴量有効化
        # 088# 動的較正: mode=pnl ではパーセンタイルベースで調整
        adaptive_threshold=True,
        target_skip_rate_buy=0.15,
        target_skip_rate_sell=0.20,
        adaptive_window=50,
        adaptive_min_samples=20,
        adaptive_step=0.05,
        adaptive_floor=0.35,
        adaptive_ceiling=0.80,
    )

    metadata = {
        "version": "v4_lgbm_pnl120",
        "trained_at": datetime.now().isoformat(),
        "n_samples": len(X),
        "n_features": len(_FULL_FEATURE_COLS),
        "target": "pnl120_regression (post_fill_120s_pnl bps)",
        "model_class": "LGBMRegressor",
        "wf_results": {
            "skip20_pnl30_improvement_bps": 0.075,
            "skip20_pnl120_improvement_bps": 0.324,
            "profit_score": 0.249,
            "feature_set": "full (base + OB)",
            "reverse_30": False,
            "reverse_120": False,
        },
        "feature_cols": list(_FULL_FEATURE_COLS),
        "pipeline_steps": [
            "SimpleImputer(median)",
            "StandardScaler",
            "LGBMRegressor(150,4,0.05)",
        ],
        "predecessor": "skip_gate_rb30.pkl (GBM_really_bad30, score=+0.190)",
    }

    # SkipGate オブジェクト生成 & 保存
    gate = SkipGate(
        model=pipeline.named_steps["model"],
        scaler=pipeline.named_steps["scaler"],
        feature_cols=list(_FULL_FEATURE_COLS),
        config=config,
        metadata=metadata,
        pipeline=pipeline,
    )

    save_path = MODEL_DIR / "skip_gate_lgbm_pnl120.pkl"
    saved_path = gate.save(save_path)
    logger.info(f"Model saved to {saved_path}")

    # 検証: ロードして predict が動くか
    gate_loaded = SkipGate.load(save_path)
    test_features = {col: 0.0 for col in _FULL_FEATURE_COLS}
    test_features["spread_jpy"] = 2000.0
    test_features["side_buy"] = 1.0
    decision = gate_loaded.evaluate(test_features, side="buy")
    logger.info(
        f"Verification: predicted_pnl={decision.predicted_pnl_bps:.3f}, "
        f"skip={decision.should_skip}, reason={decision.reason}"
    )

    return saved_path


def evaluate_on_train(
    X: pd.DataFrame,
    y: pd.Series,
    enriched_df: pd.DataFrame,
    model_path: Path,
) -> dict:
    """訓練データでのスキップシミュレーション (参考値)."""
    gate = SkipGate.load(model_path)

    preds = gate._pipeline.predict(X)

    pnl30_col = "post_fill_30s_pnl"
    pnl120_col = "post_fill_120s_pnl"

    filled_mask = enriched_df["filled"].astype(bool) & enriched_df["adverse_selected_raw"].notna()
    pnl30 = enriched_df.loc[filled_mask, pnl30_col].astype(float).reindex(X.index).values
    pnl120 = enriched_df.loc[filled_mask, pnl120_col].astype(float).reindex(X.index).values

    results = {
        "pred_mean": float(np.mean(preds)),
        "pred_median": float(np.median(preds)),
    }

    for i, (skip_pct_label, skip_pct) in enumerate((("skip20", 20), ("skip10", 10))):
        stats = compute_skip_slice_metrics(
            preds,
            pnl30,
            pnl120,
            skip_pct=skip_pct,
            skip_low_scores=True,
        )
        if i == 0:
            results["baseline_pnl30_bps"] = stats.baseline_pnl30
            results["baseline_pnl120_bps"] = stats.baseline_pnl120
        results[f"{skip_pct_label}_threshold"] = stats.threshold
        results[f"{skip_pct_label}_n_keep"] = stats.n_keep
        results[f"{skip_pct_label}_pnl30_bps"] = stats.kept_pnl30
        results[f"{skip_pct_label}_pnl120_bps"] = stats.kept_pnl120
        results[f"{skip_pct_label}_pnl30_improvement"] = stats.pnl30_improvement
        results[f"{skip_pct_label}_pnl120_improvement"] = stats.pnl120_improvement

    # Side別評価
    side_data = enriched_df.loc[X.index, "side"] if "side" in enriched_df.columns else None
    if side_data is not None:
        for s in ["buy", "sell"]:
            s_mask = (side_data == s).values
            if s_mask.sum() > 10:
                s_pnl30 = pnl30[s_mask]
                s_pnl120 = pnl120[s_mask]
                s_preds = preds[s_mask]
                side_stats = compute_skip_slice_metrics(
                    s_preds,
                    s_pnl30,
                    s_pnl120,
                    skip_pct=20,
                    skip_low_scores=True,
                )
                results[f"{s}_n"] = int(s_mask.sum())
                results[f"{s}_baseline_pnl30"] = side_stats.baseline_pnl30
                results[f"{s}_baseline_pnl120"] = side_stats.baseline_pnl120
                # Skip lowest 20% predicted PnL per side
                if s_mask.sum() > 20:
                    results[f"{s}_skip20_pnl30_improvement"] = side_stats.pnl30_improvement
                    results[f"{s}_skip20_pnl120_improvement"] = side_stats.pnl120_improvement

    return results


def main() -> None:
    """メインエントリポイント."""
    logger.info("=== 125# SkipGate v4 Deploy: LGBM PnL120 Regression + OB Features ===")

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    # Step 1: データ準備
    X, y_pnl120, enriched = load_and_prepare_data()

    # Step 2: 訓練 & 保存
    model_path = train_and_save_model(X, y_pnl120)

    # Step 3: 全データでの動作確認
    eval_results = evaluate_on_train(X, y_pnl120, enriched, model_path)
    logger.info("=== Full-data evaluation (reference, NOT OOS) ===")
    for k, v in eval_results.items():
        if isinstance(v, float):
            logger.info(f"  {k}: {v:+.4f}")
        else:
            logger.info(f"  {k}: {v}")

    # Step 4: レポート保存
    report = {
        "generated_at": datetime.now().isoformat(),
        "source": "125# deploy_sg_v4.py",
        "model_path": str(model_path),
        "model_type": "LGBM_pnl120_regression_full",
        "training_data_n": len(X),
        "target": "post_fill_120s_pnl (bps)",
        "full_data_eval": eval_results,
        "wf_oos_results": {
            "skip20_pnl30_improvement": 0.075,
            "skip20_pnl120_improvement": 0.324,
            "profit_score": 0.249,
            "n_folds": "from train_sg_v3.py",
            "feature_set": "full (base + OB)",
            "note": "Non-reverse on both horizons",
        },
        "deployment_config": {
            "mode": "pnl",
            "use_ob_features": True,
            "buy_enabled": True,
            "sell_enabled": True,
            "adaptive_threshold": True,
            "note": "predicted_pnl < threshold → skip. Low PnL = bad trade = skip.",
        },
        "predecessor": {
            "model": "skip_gate_rb30.pkl",
            "type": "GBM_really_bad30",
            "score": 0.190,
        },
    }
    report_path = REPORT_DIR / "deploy_lgbm_pnl120_report.json"
    write_json(report_path, report, indent=2, ensure_ascii=False, default=str)
    logger.info(f"Report saved to {report_path}")

    logger.info("=== Deploy complete ===")
    logger.info(f"Model: {model_path}")
    logger.info(
        "Next steps:\n"
        "  1. Update configs/v460/fill_test.yaml:\n"
        "     skip_gate.model_path: models/v460/skip_gate_lgbm_pnl120.pkl\n"
        "     skip_gate.mode: pnl\n"
        "     skip_gate.use_ob_features: true\n"
        "  2. Restart fill_test\n"
        "  3. Monitor for 48h before next change"
    )


if __name__ == "__main__":
    main()
