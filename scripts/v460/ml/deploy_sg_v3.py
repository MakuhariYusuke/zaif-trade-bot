"""124# SkipGate v3 デプロイ: 最終モデルの訓練・保存.

GBM_sklearn_really_bad30 を全データで訓練し、SkipGate 互換 pickle で保存。
Rule_skip_unknown_sell はコード側で実装 (YAML フラグ制御)。

WF 評価結果:
  GBM_sklearn_really_bad30_base:
    S20%_30=+0.114 bps, S20%_120=+0.224 bps (both positive, NO reverse selection)
    AUC=0.521, 27 folds, 712 samples

  Rule_skip_unknown_sell:
    S20%_30=+0.198 bps, S20%_120=+0.140 bps

Usage:
    .venv\\Scripts\\python.exe scripts/v460/ml/deploy_sg_v3.py
"""

from __future__ import annotations

import logging
import pickle
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent.parent))

from scripts.v460.ml.cache_cleanup import clear_ml_data_caches_with_log
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
)
from ztb.io.json_io import write_json

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
)
logger = logging.getLogger(__name__)

MODEL_DIR = Path("models/v460")
REPORT_DIR = Path("reports/v460/ml_124")


def load_and_prepare_data() -> tuple[pd.DataFrame, pd.Series, pd.DataFrame]:
    """Fill records をロードし、特徴量・ターゲットを構築.

    train_sg_v3.py と同じデータパイプライン:
      load_fill_records → enrich_fill_records → build_preorder_as_features

    Returns:
        (X, y_really_bad30, enriched_df)
    """
    records = load_fill_records()
    logger.info(f"Loaded {len(records)} records")

    enriched = enrich_fill_records(records)
    logger.info(f"Enriched: {len(enriched)} records")

    # build_preorder_as_features: filled かつ AS ラベル有のみ → (X, y_as)
    X_base, y_as = build_preorder_as_features(enriched)
    logger.info(f"Base features: {X_base.shape}, AS rate={y_as.mean():.3f}")

    # base 列のみ抽出 (ランタイムと完全一致保証)
    base_cols = [c for c in _BASE_FEATURE_COLS if c in X_base.columns]
    missing = [c for c in _BASE_FEATURE_COLS if c not in X_base.columns]
    if missing:
        logger.warning(f"Missing base features (will be NaN): {missing}")
        for c in missing:
            X_base[c] = np.nan
    X = X_base[_BASE_FEATURE_COLS]
    logger.info(f"Feature matrix (base only): {X.shape}")

    # ターゲット: really_bad30 (PnL30 < -1.0 bps → 1)
    filled_mask = enriched["filled"].astype(bool) & enriched["adverse_selected_raw"].notna()
    pnl30 = enriched.loc[filled_mask, "post_fill_30s_pnl"].astype(float).reindex(X.index)
    y_really_bad = (pnl30 < -1.0).astype(int)
    logger.info(
        f"Target really_bad30: positive_rate={y_really_bad.mean():.3f} "
        f"({y_really_bad.sum()}/{len(y_really_bad)})"
    )

    return X, y_really_bad, enriched


def train_and_save_model(
    X: pd.DataFrame,
    y: pd.Series,
) -> Path:
    """Pipeline (Imputer + Scaler + GBM) を訓練し SkipGate 形式で保存.

    Returns:
        保存先パス.
    """
    # GBM ハイパーパラメータ (WF 評価と同一)
    gbm = GradientBoostingClassifier(
        n_estimators=150,
        max_depth=3,
        learning_rate=0.05,
        min_samples_leaf=20,
        subsample=0.8,
        random_state=42,
    )

    # Pipeline: Imputer → Scaler → GBM
    pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("model", gbm),
    ])

    logger.info("Training GBM pipeline on full data...")
    pipeline.fit(X, y)

    # 訓練データでの P(really_bad) 分布確認
    probs = pipeline.predict_proba(X)[:, 1]
    logger.info(
        f"P(really_bad) distribution on train: "
        f"mean={np.mean(probs):.3f}, "
        f"median={np.median(probs):.3f}, "
        f"P25={np.percentile(probs, 25):.3f}, "
        f"P75={np.percentile(probs, 75):.3f}, "
        f"P90={np.percentile(probs, 90):.3f}"
    )

    # SkipGateConfig: mode=as, really_bad30 確率 > threshold → skip
    config = SkipGateConfig(
        mode="as",
        enabled=True,
        buy_enabled=True,
        sell_enabled=True,  # 124#: 逆選別なし → 両 side 有効化
        as_threshold=0.50,
        as_threshold_buy=0.50,
        as_threshold_sell=0.50,
        threshold_bps=0.0,
        max_skip_rate=0.3,
        use_ob_features=False,
        # 088# 動的較正はそのまま維持
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
        "version": "v3_really_bad30",
        "trained_at": datetime.now().isoformat(),
        "n_samples": len(X),
        "n_features": len(_BASE_FEATURE_COLS),
        "target": "really_bad30 (pnl30 < -1.0 bps)",
        "model_class": "GradientBoostingClassifier",
        "wf_results": {
            "skip20_pnl30_improvement_bps": 0.114,
            "skip20_pnl120_improvement_bps": 0.224,
            "auc_mean": 0.521,
            "profit_score": 0.191,
            "reverse_selection": False,
        },
        "feature_cols": list(_BASE_FEATURE_COLS),
        "pipeline_steps": ["SimpleImputer(median)", "StandardScaler", "GBM(150,3,0.05)"],
    }

    # SkipGate オブジェクト生成 & 保存
    gate = SkipGate(
        model=pipeline.named_steps["model"],
        scaler=pipeline.named_steps["scaler"],
        feature_cols=list(_BASE_FEATURE_COLS),
        config=config,
        metadata=metadata,
        pipeline=pipeline,
    )

    save_path = MODEL_DIR / "skip_gate_rb30.pkl"
    saved_path = gate.save(save_path)
    logger.info(f"Model saved to {saved_path}")

    # 検証: ロードして predict_proba が動くか
    gate_loaded = SkipGate.load(save_path)
    test_features = {col: 0.0 for col in _BASE_FEATURE_COLS}
    test_features["spread_jpy"] = 2000.0
    test_features["side_buy"] = 1.0
    decision = gate_loaded.evaluate(test_features, side="buy")
    logger.info(
        f"Verification: P(really_bad)={decision.as_probability:.3f}, "
        f"skip={decision.should_skip}, reason={decision.reason}"
    )

    return saved_path


def evaluate_on_train(
    X: pd.DataFrame,
    y: pd.Series,
    enriched_df: pd.DataFrame,
    model_path: Path,
) -> dict:
    """訓練データでのスキップシミュレーション (参考値).

    ※ Walk-Forward OOS 評価は train_sg_v3.py で実施済み。
    ここは最終モデルの全データでの動作確認。
    """
    gate = SkipGate.load(model_path)

    probs = gate._pipeline.predict_proba(X)[:, 1]

    filled_mask = enriched_df["filled"].astype(bool) & enriched_df["adverse_selected_raw"].notna()
    pnl30 = enriched_df.loc[filled_mask, "post_fill_30s_pnl"].astype(float).reindex(X.index).values
    pnl120_col = "post_fill_120s_pnl"
    if pnl120_col in enriched_df.columns:
        pnl120 = enriched_df.loc[filled_mask, pnl120_col].astype(float).reindex(X.index).values
    else:
        pnl120 = np.full(len(pnl30), np.nan)

    results = {
        "prob_mean": float(np.mean(probs)),
        "prob_median": float(np.median(probs)),
    }

    for i, (skip_pct_label, skip_pct) in enumerate((("skip20", 20), ("skip10", 10))):
        stats = compute_skip_slice_metrics(
            probs,
            pnl30,
            pnl120,
            skip_pct=skip_pct,
            skip_low_scores=False,
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

    return results


def main() -> None:
    try:
        _run_deploy_sg_v3_main()
    finally:
        clear_ml_data_caches_with_log(
            logger,
            context="deploy_sg_v3",
            collect_garbage=True,
        )


def _run_deploy_sg_v3_main() -> None:
    """メインエントリポイント."""
    logger.info("=== 124# SkipGate v3 Deploy ===")

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    # Step 1: データ準備
    X, y_rb30, enriched = load_and_prepare_data()

    # Step 2: 訓練 & 保存
    model_path = train_and_save_model(X, y_rb30)

    # Step 3: 全データでの動作確認
    eval_results = evaluate_on_train(X, y_rb30, enriched, model_path)
    logger.info("=== Full-data evaluation (reference, NOT OOS) ===")
    for k, v in eval_results.items():
        if isinstance(v, float):
            logger.info(f"  {k}: {v:+.4f}")
        else:
            logger.info(f"  {k}: {v}")

    # Step 4: レポート保存
    report = {
        "generated_at": datetime.now().isoformat(),
        "source": "124# deploy_sg_v3.py",
        "model_path": str(model_path),
        "model_type": "GBM_sklearn_really_bad30",
        "training_data_n": len(X),
        "target_positive_rate": float(y_rb30.mean()),
        "full_data_eval": eval_results,
        "wf_oos_results": {
            "skip20_pnl30_improvement": 0.114,
            "skip20_pnl120_improvement": 0.224,
            "auc_mean": 0.521,
            "profit_score": 0.191,
            "n_folds": 27,
            "note": "From train_sg_v3.py Walk-Forward evaluation",
        },
        "deployment_config": {
            "mode": "as",
            "buy_enabled": True,
            "sell_enabled": True,
            "adaptive_threshold": True,
            "note": "P(really_bad30) replaces P(AS) — high prob = catastrophic trade = skip",
        },
    }
    report_path = REPORT_DIR / "deploy_rb30_report.json"
    write_json(report_path, report, indent=2, ensure_ascii=False, default=str)
    logger.info(f"Report saved to {report_path}")

    logger.info("=== Deploy complete ===")
    logger.info(f"Model: {model_path}")
    logger.info("Next: Update configs/v460/fill_test.yaml skip_gate.model_path")


if __name__ == "__main__":
    main()
