"""189# Alt Horizon モデル訓練スクリプト — ev_weighted 用.

Phase C (188#) で構築した ev_weighted SkipGate 基盤に必要な
副 horizon モデルを訓練・デプロイする。

訓練対象:
  - buy_long:   buy 側 pnl120 回帰モデル (alt=長期 horizon)
  - sell_short:  sell 側 pnl30 回帰モデル  (alt=短期 horizon)

既存 primary モデルとの関係:
  - buy primary:  skip_gate_lgbm_pnl30_buy.pkl  (短期)
  - sell primary: skip_gate_lgbm_pnl120_sell.pkl (長期)
  - buy alt:      skip_gate_lgbm_pnl120_buy.pkl  ← 本スクリプトで訓練
  - sell alt:     skip_gate_lgbm_pnl30_sell.pkl   ← 本スクリプトで訓練

Usage:
    .venv\\Scripts\\python.exe scripts/v460/ml/train_alt_horizon.py
    .venv\\Scripts\\python.exe scripts/v460/ml/train_alt_horizon.py --side buy
    .venv\\Scripts\\python.exe scripts/v460/ml/train_alt_horizon.py --side sell
"""

from __future__ import annotations

import argparse
import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Literal, TypedDict

import numpy as np
import pandas as pd
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
from ztb.ml.skip_gate import (
    SkipGate,
    SkipGateConfig,
    _OB_FEATURE_COLS,
    get_gate_feature_cols,
)
from ztb.io.json_io import write_json
from ztb.ml.metadata_utils import current_iso_timestamp

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
)
logger = logging.getLogger(__name__)

MODEL_DIR = Path("models/v460")
REPORT_DIR = Path("reports/v460/ml_189")

_FULL_FEATURE_COLS = get_gate_feature_cols(use_ob=True)


class AltSpec(TypedDict):
    side: str
    pnl_col: str
    target_label: str
    model_file: str
    description: str


class DataStats(TypedDict):
    n_samples: int
    target_mean: float
    target_median: float
    target_std: float
    target_positive_rate: float


class PredStats(TypedDict):
    pred_mean: float
    pred_median: float
    pred_std: float
    pred_p10: float
    pred_p25: float
    pred_p75: float
    pred_p90: float
    feature_importance: dict[str, float]


EvalMetricValue = float | int
EvalResults = dict[str, EvalMetricValue]


class TrainReport(TypedDict):
    generated_at: str
    source: str
    side: str
    model_path: str
    spec: AltSpec
    data_stats: DataStats
    pred_stats: PredStats
    eval_results: EvalResults


class ErrorReport(TypedDict):
    status: Literal["error"]
    reason: str

# --- Alt horizon 定義テーブル ---
# (side, pnl_col, target_label, output_filename)
_ALT_SPECS: dict[str, AltSpec] = {
    "buy": {
        "side": "buy",
        "pnl_col": "post_fill_120s_pnl",
        "target_label": "pnl120",
        "model_file": "skip_gate_lgbm_pnl120_buy.pkl",
        "description": "buy alt (長期 horizon pnl120 回帰)",
    },
    "sell": {
        "side": "sell",
        "pnl_col": "post_fill_30s_pnl",
        "target_label": "pnl30",
        "model_file": "skip_gate_lgbm_pnl30_sell.pkl",
        "description": "sell alt (短期 horizon pnl30 回帰)",
    },
}


def load_side_data(
    spec: AltSpec,
) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame, DataStats]:
    """side フィルタ + ターゲット構築."""
    records = load_fill_records()
    logger.info(f"Loaded {len(records)} total records")

    enriched = enrich_fill_records(records)
    logger.info(f"Enriched: {len(enriched)} records")

    # base features
    X_base, y_as = build_preorder_as_features(enriched)
    logger.info(f"Base features: {X_base.shape}")

    # OB 特徴量を追加
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

    # side_aligned_imbalance
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

    # 特徴量契約
    for col in _FULL_FEATURE_COLS:
        if col not in X_full.columns:
            X_full[col] = np.nan
    X_full = X_full[_FULL_FEATURE_COLS]

    # side フィルタ
    side = spec["side"]
    side_mask = enriched.loc[X_full.index, "side"] == side
    X_side = X_full.loc[side_mask]
    logger.info(f"After side={side} filter: {len(X_side)} records")

    # ターゲット: pnl 回帰
    pnl_col = spec["pnl_col"]
    if pnl_col not in enriched.columns:
        raise ValueError(f"Target column '{pnl_col}' not in enriched data")

    y_pnl = enriched.loc[X_side.index, pnl_col].astype(float)
    valid_mask = y_pnl.notna()
    X_side = X_side.loc[valid_mask]
    y_pnl = y_pnl.loc[valid_mask]
    logger.info(f"After NaN filter: {len(X_side)} records with valid {pnl_col}")

    if len(X_side) < 50:
        raise ValueError(
            f"Insufficient data: {len(X_side)} < 50 (side={side}, target={pnl_col})"
        )

    # 統計情報
    stats: DataStats = {
        "n_samples": len(X_side),
        "target_mean": float(y_pnl.mean()),
        "target_median": float(y_pnl.median()),
        "target_std": float(y_pnl.std()),
        "target_positive_rate": float((y_pnl > 0).mean()),
    }
    logger.info(
        f"Target stats: mean={stats['target_mean']:.3f}, "
        f"median={stats['target_median']:.3f}, "
        f"std={stats['target_std']:.3f}, "
        f"positive_rate={stats['target_positive_rate']:.1%}"
    )

    return X_side, y_pnl, enriched, stats


def train_lgbm_pipeline(
    X: pd.DataFrame,
    y: pd.Series,
    spec: AltSpec,
) -> tuple[Pipeline, PredStats]:
    """LGBM 回帰 Pipeline を訓練."""
    try:
        import lightgbm as lgb
    except ImportError:
        raise RuntimeError("LightGBM required: pip install lightgbm")

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

    pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("model", lgbm),
    ])

    logger.info(f"Training LGBM regression pipeline for {spec['description']}...")
    pipeline.fit(X, y)

    preds = pipeline.predict(X)
    pred_stats: PredStats = {
        "pred_mean": float(np.mean(preds)),
        "pred_median": float(np.median(preds)),
        "pred_std": float(np.std(preds)),
        "pred_p10": float(np.percentile(preds, 10)),
        "pred_p25": float(np.percentile(preds, 25)),
        "pred_p75": float(np.percentile(preds, 75)),
        "pred_p90": float(np.percentile(preds, 90)),
    }
    logger.info(
        f"Predictions on train: mean={pred_stats['pred_mean']:.3f}, "
        f"median={pred_stats['pred_median']:.3f}, std={pred_stats['pred_std']:.3f}"
    )

    # feature importance
    feat_importance: dict[str, float] = {}
    model = pipeline.named_steps["model"]
    if hasattr(model, "feature_importances_"):
        for col, imp in zip(X.columns, model.feature_importances_):
            feat_importance[col] = float(imp)
    pred_stats["feature_importance"] = feat_importance

    return pipeline, pred_stats


def evaluate_skip_quality(
    pipeline: Pipeline,
    X: pd.DataFrame,
    enriched: pd.DataFrame,
) -> EvalResults:
    """スキップシミュレーション (参考値 — 訓練データ上)."""
    preds = pipeline.predict(X)
    idx = X.index

    result: EvalResults = {}
    pnl30 = (
        enriched.loc[idx, "post_fill_30s_pnl"].astype(float).values
        if "post_fill_30s_pnl" in enriched.columns
        else np.full(len(idx), np.nan, dtype=np.float64)
    )
    pnl120 = (
        enriched.loc[idx, "post_fill_120s_pnl"].astype(float).values
        if "post_fill_120s_pnl" in enriched.columns
        else np.full(len(idx), np.nan, dtype=np.float64)
    )

    for i, skip_pct in enumerate((10, 20)):
        stats = compute_skip_slice_metrics(
            preds,
            pnl30,
            pnl120,
            skip_pct=skip_pct,
            skip_low_scores=True,
        )
        if i == 0:
            if "post_fill_30s_pnl" in enriched.columns:
                result["baseline_pnl30"] = stats.baseline_pnl30
            if "post_fill_120s_pnl" in enriched.columns:
                result["baseline_pnl120"] = stats.baseline_pnl120
        result[f"skip{skip_pct}_n_keep"] = stats.n_keep
        if "post_fill_30s_pnl" in enriched.columns:
            result[f"skip{skip_pct}_pnl30_improvement"] = stats.pnl30_improvement
            result[f"skip{skip_pct}_pnl30_kept_mean"] = stats.kept_pnl30
        if "post_fill_120s_pnl" in enriched.columns:
            result[f"skip{skip_pct}_pnl120_improvement"] = stats.pnl120_improvement
            result[f"skip{skip_pct}_pnl120_kept_mean"] = stats.kept_pnl120

    return result


def save_skipgate(
    pipeline: Pipeline,
    spec: AltSpec,
    data_stats: DataStats,
    pred_stats: PredStats,
) -> Path:
    """SkipGate 形式で保存."""
    config = SkipGateConfig(
        mode="pnl",
        enabled=True,
        buy_enabled=True,
        sell_enabled=True,
        threshold_bps=0.0,
        as_threshold=0.50,
        as_threshold_buy=0.50,
        as_threshold_sell=0.50,
        max_skip_rate=0.3,
        use_ob_features=True,
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
        "version": f"v4_lgbm_{spec['target_label']}_{spec['side']}_alt",
        "trained_at": current_iso_timestamp(),
        "n_samples": data_stats["n_samples"],
        "n_features": len(_FULL_FEATURE_COLS),
        "target": f"{spec['target_label']} regression ({spec['pnl_col']})",
        "model_class": "LGBMRegressor",
        "side": spec["side"],
        "horizon_label": "alt",
        "description": spec["description"],
        "data_stats": data_stats,
        "pred_stats": {k: v for k, v in pred_stats.items() if k != "feature_importance"},
        "feature_cols": list(_FULL_FEATURE_COLS),
        "pipeline_steps": [
            "SimpleImputer(median)",
            "StandardScaler",
            "LGBMRegressor(150,4,0.05)",
        ],
        "session": "189#",
    }

    gate = SkipGate(
        model=pipeline.named_steps["model"],
        scaler=pipeline.named_steps["scaler"],
        feature_cols=list(_FULL_FEATURE_COLS),
        config=config,
        metadata=metadata,
        pipeline=pipeline,
    )

    save_path = MODEL_DIR / spec["model_file"]
    saved = gate.save(save_path)
    logger.info(f"Model saved to {saved}")

    # 検証: ロード + predict
    gate_loaded = SkipGate.load(save_path)
    test_features = {col: 0.0 for col in _FULL_FEATURE_COLS}
    test_features["spread_jpy"] = 2000.0
    test_features["side_buy"] = 1.0 if spec["side"] == "buy" else 0.0
    decision = gate_loaded.evaluate(test_features, side=spec["side"])
    logger.info(
        f"Verification: predicted_pnl={decision.predicted_pnl_bps:.3f}, "
        f"skip={decision.should_skip}, reason={decision.reason}"
    )

    return save_path


def train_one(side: str) -> TrainReport:
    """1 side の alt モデルを訓練して保存."""
    spec = _ALT_SPECS[side]
    logger.info(f"\n{'='*60}")
    logger.info(f"189# Training alt horizon: {spec['description']}")
    logger.info(f"{'='*60}")

    X, y, enriched, data_stats = load_side_data(spec)
    pipeline, pred_stats = train_lgbm_pipeline(X, y, spec)
    eval_results = evaluate_skip_quality(pipeline, X, enriched)
    model_path = save_skipgate(pipeline, spec, data_stats, pred_stats)

    report: TrainReport = {
        "generated_at": current_iso_timestamp(),
        "source": "189# train_alt_horizon.py",
        "side": side,
        "model_path": str(model_path),
        "spec": spec,
        "data_stats": data_stats,
        "pred_stats": pred_stats,
        "eval_results": eval_results,
    }

    report_path = REPORT_DIR / f"alt_{spec['target_label']}_{side}_report.json"
    write_json(report_path, report, indent=2, ensure_ascii=False, default=str)
    logger.info(f"Report saved to {report_path}")

    return report


def main() -> None:
    try:
        _run_train_alt_horizon_main()
    finally:
        clear_ml_data_caches_with_log(
            logger,
            context="train_alt_horizon",
            collect_garbage=True,
        )


def _run_train_alt_horizon_main() -> None:
    parser = argparse.ArgumentParser(description="189# Alt Horizon Model Training")
    parser.add_argument(
        "--side",
        choices=["buy", "sell", "both"],
        default="both",
        help="Which side to train (default: both)",
    )
    args = parser.parse_args()

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    sides = ["buy", "sell"] if args.side == "both" else [args.side]
    results: dict[str, TrainReport | ErrorReport] = {}

    for side in sides:
        try:
            results[side] = train_one(side)
        except Exception as e:
            logger.error(f"Training failed for {side}: {e}", exc_info=True)
            results[side] = {"status": "error", "reason": str(e)}

    # Summary
    logger.info("\n" + "=" * 60)
    logger.info("189# Alt Horizon Training Summary")
    logger.info("=" * 60)
    for side, result in results.items():
        if "status" in result and result["status"] == "error":
            logger.error(f"  {side}: FAILED — {result['reason']}")
        else:
            stats = result.get("data_stats", {})
            ev = result.get("eval_results", {})
            logger.info(
                f"  {side}: n={stats.get('n_samples', '?')}, "
                f"target_mean={stats.get('target_mean', 0):.3f}, "
                f"skip20_pnl30_imp={ev.get('skip20_pnl30_improvement', 0):.3f}, "
                f"skip20_pnl120_imp={ev.get('skip20_pnl120_improvement', 0):.3f}"
            )

    logger.info(
        "\nNext steps:\n"
        "  1. Update configs/v460/fill_test.yaml:\n"
        "     skip_gate.model_path_buy_long: models/v460/skip_gate_lgbm_pnl120_buy.pkl\n"
        "     skip_gate.model_path_sell_short: models/v460/skip_gate_lgbm_pnl30_sell.pkl\n"
        "     skip_gate.ev_weighted_enabled: true\n"
        "  2. Restart fill_test\n"
        "  3. Monitor ev_weighted decisions in logs"
    )


if __name__ == "__main__":
    main()
