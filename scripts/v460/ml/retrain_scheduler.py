"""126# SkipGate 定期再学習スケジューラ.

fill_test 稼働中にバックグラウンドで定期的にモデルを再学習し、
アトミックにモデルファイルを差し替える。

設計:
  - fill_records_*.jsonl の蓄積データで定期的に再学習
  - Walk-Forward OOS 評価で品質チェック (regression gate)
  - アトミック書き込み (tmp → rename) で pkl を差し替え
  - SkipGateEvaluator 側の hot-reload が新モデルを検出・ロード
  - 品質劣化時はスキップ (既存モデルを維持)

Usage:
    # fill_test と並行して別プロセスで実行
    .venv\\Scripts\\python.exe scripts/v460/ml/retrain_scheduler.py

    # ワンショット実行 (テスト用)
    .venv\\Scripts\\python.exe scripts/v460/ml/retrain_scheduler.py --once
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import pickle
import shutil
import sys
import tempfile
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent.parent))

from scripts.v460.ml.data_loader import load_fill_records
from scripts.v460.ml.feature_enricher import (
    build_preorder_as_features,
    enrich_fill_records,
)
from scripts.v460.ml.skip_gate import (
    SkipGate,
    SkipGateConfig,
    _BASE_FEATURE_COLS,
    _OB_FEATURE_COLS,
    get_gate_feature_cols,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("logs/retrain_scheduler.log", encoding="utf-8"),
    ],
)
logger = logging.getLogger(__name__)

# デフォルト設定
_DEFAULT_CONFIG = {
    "interval_sec": 3600,           # 再学習間隔 (秒) — 1時間
    "min_new_samples": 30,          # 再学習に必要な最小新規サンプル数
    "min_total_samples": 100,       # 再学習に必要な最小合計サンプル数
    "model_path": "models/v460/skip_gate_lgbm_pnl120.pkl",
    "results_dir": "results/v460/fill_test",
    "target": "pnl120",            # "pnl120" or "as30"
    "use_ob_features": True,
    # Walk-Forward 品質ゲート
    "quality_gate_enabled": True,
    "min_score_improvement": -0.05,  # 前モデル比でこれ以下なら棄却
    "wf_test_ratio": 0.2,           # WF テスト比率
    # LGBM ハイパーパラメータ
    "lgbm_n_estimators": 150,
    "lgbm_max_depth": 4,
    "lgbm_learning_rate": 0.05,
    "lgbm_num_leaves": 15,
    "lgbm_min_child_samples": 20,
    # SkipGate config
    "adaptive_threshold": True,
    "target_skip_rate_buy": 0.15,
    "target_skip_rate_sell": 0.20,
}


def load_retrain_config(config_path: Path | None = None) -> dict[str, Any]:
    """YAML retrain セクションから設定を読み込む."""
    cfg = dict(_DEFAULT_CONFIG)
    yaml_path = config_path or Path("configs/v460/fill_test.yaml")
    if yaml_path.exists():
        try:
            import yaml
            with open(yaml_path) as f:
                yaml_data = yaml.safe_load(f) or {}
            retrain_cfg = yaml_data.get("retrain", {})
            if retrain_cfg:
                for key in cfg:
                    if key in retrain_cfg:
                        cfg[key] = retrain_cfg[key]
                logger.info(f"Retrain config loaded from {yaml_path}")
        except Exception as e:
            logger.warning(f"Failed to load retrain config: {e}, using defaults")
    return cfg


def _build_full_features(
    enriched: pd.DataFrame,
    X_base: pd.DataFrame,
    use_ob: bool = True,
) -> pd.DataFrame:
    """base + OB 特徴量を構築."""
    X = X_base.copy()
    if use_ob:
        ob_cols = {"spread_bps_ob": "spread_bps_ob", "depth_imbalance_ob": "depth_imbalance_ob"}
        for feat_col, df_col in ob_cols.items():
            if df_col in enriched.columns:
                X[feat_col] = enriched.loc[X_base.index, df_col].astype(float)
            else:
                X[feat_col] = np.nan

        if "depth_imbalance_ob" in enriched.columns:
            side_sign = enriched.loc[X_base.index, "side"].map(
                {"buy": 1.0, "sell": -1.0}
            ).astype(float)
            X["side_aligned_imbalance"] = (
                enriched.loc[X_base.index, "depth_imbalance_ob"].astype(float) * side_sign
            ).fillna(0.0)
        else:
            X["side_aligned_imbalance"] = 0.0

    feature_cols = get_gate_feature_cols(use_ob=use_ob)
    for col in feature_cols:
        if col not in X.columns:
            X[col] = np.nan
    return X[feature_cols]


def _evaluate_wf(
    X: pd.DataFrame,
    y: pd.Series,
    enriched: pd.DataFrame,
    cfg: dict[str, Any],
) -> dict[str, float]:
    """Walk-Forward OOS 評価で品質スコアを算出.

    直近 test_ratio をテストセットとし、残りで訓練→テスト予測の skip simulation。

    Returns:
        {"score": float, "pnl30_improvement": float, "pnl120_improvement": float}
    """
    from sklearn.impute import SimpleImputer
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    try:
        import lightgbm as lgb
    except ImportError:
        raise RuntimeError("LightGBM required")

    test_ratio = cfg.get("wf_test_ratio", 0.2)
    n = len(X)
    split_idx = int(n * (1.0 - test_ratio))
    if split_idx < 50 or (n - split_idx) < 20:
        logger.warning(f"Insufficient data for WF eval: train={split_idx}, test={n - split_idx}")
        return {"score": 0.0, "pnl30_improvement": 0.0, "pnl120_improvement": 0.0}

    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]

    lgbm = lgb.LGBMRegressor(
        n_estimators=cfg.get("lgbm_n_estimators", 150),
        max_depth=cfg.get("lgbm_max_depth", 4),
        learning_rate=cfg.get("lgbm_learning_rate", 0.05),
        num_leaves=cfg.get("lgbm_num_leaves", 15),
        min_child_samples=cfg.get("lgbm_min_child_samples", 20),
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=1.0,
        reg_lambda=1.0,
        random_state=42,
        verbose=-1,
        n_jobs=1,
    )
    pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("model", lgbm),
    ])
    pipe.fit(X_train, y_train)
    preds_test = pipe.predict(X_test)

    # OOS PnL 参照 (pnl30/pnl120)
    pnl30_col = "post_fill_30s_pnl"
    pnl120_col = "post_fill_120s_pnl"
    test_idx = X_test.index

    pnl30 = enriched.loc[test_idx, pnl30_col].astype(float).values if pnl30_col in enriched.columns else np.full(len(test_idx), np.nan)
    pnl120 = enriched.loc[test_idx, pnl120_col].astype(float).values if pnl120_col in enriched.columns else np.full(len(test_idx), np.nan)

    baseline_30 = float(np.nanmean(pnl30))
    baseline_120 = float(np.nanmean(pnl120))

    # Skip bottom 20% predicted PnL
    threshold = np.percentile(preds_test, 20)
    keep_mask = preds_test >= threshold
    kept_30 = float(np.nanmean(pnl30[keep_mask]))
    kept_120 = float(np.nanmean(pnl120[keep_mask]))

    imp_30 = kept_30 - baseline_30
    imp_120 = kept_120 - baseline_120
    score = imp_120 - max(0, -imp_30)  # 125# profit_score: pnl120 改善 - pnl30 悪化ペナルティ

    return {
        "score": score,
        "pnl30_improvement": imp_30,
        "pnl120_improvement": imp_120,
        "n_test": int(len(X_test)),
        "n_train": int(len(X_train)),
    }


def retrain_model(cfg: dict[str, Any]) -> dict[str, Any]:
    """モデル再学習 → 品質評価 → アトミック差し替え.

    Returns:
        再学習結果のサマリー dict。
        "status" が "deployed" ならモデルが差し替えられた。
    """
    from sklearn.impute import SimpleImputer
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    try:
        import lightgbm as lgb
    except ImportError:
        return {"status": "error", "reason": "lightgbm not installed"}

    model_path = Path(cfg["model_path"])
    results_dir = Path(cfg["results_dir"])
    target = cfg["target"]
    use_ob = cfg.get("use_ob_features", True)

    result: dict[str, Any] = {
        "timestamp": datetime.now().isoformat(),
        "status": "pending",
    }

    # Step 1: データロード
    try:
        records = load_fill_records(results_dir)
    except FileNotFoundError:
        return {**result, "status": "skipped", "reason": "no fill_records found"}

    enriched = enrich_fill_records(records)
    try:
        X_base, y_as = build_preorder_as_features(enriched)
    except ValueError as exc:
        return {**result, "status": "skipped", "reason": str(exc)}
    X_full = _build_full_features(enriched, X_base, use_ob=use_ob)

    # ターゲット選定
    if target == "pnl120":
        pnl_col = "post_fill_120s_pnl"
    else:
        pnl_col = "post_fill_30s_pnl"

    if pnl_col not in enriched.columns:
        return {**result, "status": "skipped", "reason": f"{pnl_col} not available"}

    y_target = enriched.loc[X_base.index, pnl_col].astype(float)
    valid_mask = y_target.notna()
    X_valid = X_full.loc[valid_mask]
    y_valid = y_target.loc[valid_mask]

    result["total_samples"] = int(len(X_full))
    result["valid_target_samples"] = int(len(X_valid))

    # Step 2: 最小サンプルチェック
    min_total = cfg.get("min_total_samples", 100)
    if len(X_valid) < min_total:
        return {
            **result,
            "status": "skipped",
            "reason": f"insufficient samples: {len(X_valid)} < {min_total}",
        }

    # Step 3: 新規サンプルチェック (前回学習時の n_samples と比較)
    min_new = cfg.get("min_new_samples", 30)
    prev_n_samples = 0
    if model_path.exists():
        try:
            prev_gate = SkipGate.load(model_path)
            prev_n_samples = prev_gate.metadata.get("n_samples", 0)
        except Exception:
            pass
    new_samples = len(X_valid) - prev_n_samples
    result["new_samples"] = int(new_samples)

    if new_samples < min_new:
        return {
            **result,
            "status": "skipped",
            "reason": f"insufficient new samples: {new_samples} < {min_new}",
        }

    logger.info(
        f"Retraining: {len(X_valid)} samples ({new_samples} new), "
        f"target={target}, use_ob={use_ob}"
    )

    # Step 4: Walk-Forward 品質評価
    if cfg.get("quality_gate_enabled", True):
        wf_result = _evaluate_wf(X_valid, y_valid, enriched, cfg)
        result["wf_eval"] = wf_result
        logger.info(
            f"WF eval: score={wf_result['score']:.4f}, "
            f"pnl30_imp={wf_result['pnl30_improvement']:.4f}, "
            f"pnl120_imp={wf_result['pnl120_improvement']:.4f}"
        )

        # 品質ゲート: 前モデルの score と比較
        min_improvement = cfg.get("min_score_improvement", -0.05)
        prev_score = 0.0
        if model_path.exists():
            try:
                prev_gate = SkipGate.load(model_path)
                prev_wf = prev_gate.metadata.get("wf_results", {})
                prev_score = prev_wf.get("profit_score", 0.0)
            except Exception:
                pass

        improvement = wf_result["score"] - prev_score
        result["score_improvement"] = improvement
        if improvement < min_improvement:
            logger.warning(
                f"Quality gate REJECT: improvement={improvement:.4f} < {min_improvement}. "
                f"Keeping existing model."
            )
            return {**result, "status": "rejected", "reason": "quality_gate"}

    # Step 5: 全データで訓練
    lgbm = lgb.LGBMRegressor(
        n_estimators=cfg.get("lgbm_n_estimators", 150),
        max_depth=cfg.get("lgbm_max_depth", 4),
        learning_rate=cfg.get("lgbm_learning_rate", 0.05),
        num_leaves=cfg.get("lgbm_num_leaves", 15),
        min_child_samples=cfg.get("lgbm_min_child_samples", 20),
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=1.0,
        reg_lambda=1.0,
        random_state=42,
        verbose=-1,
        n_jobs=1,
    )
    feature_cols = list(get_gate_feature_cols(use_ob=use_ob))
    pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("model", lgbm),
    ])
    pipeline.fit(X_valid, y_valid)

    # SkipGateConfig
    sg_config = SkipGateConfig(
        mode="pnl",
        enabled=True,
        buy_enabled=True,
        sell_enabled=True,
        threshold_bps=0.0,
        use_ob_features=use_ob,
        adaptive_threshold=cfg.get("adaptive_threshold", True),
        target_skip_rate_buy=cfg.get("target_skip_rate_buy", 0.15),
        target_skip_rate_sell=cfg.get("target_skip_rate_sell", 0.20),
        adaptive_window=50,
        adaptive_min_samples=20,
        adaptive_step=0.05,
        adaptive_floor=0.35,
        adaptive_ceiling=0.80,
    )

    wf_results_meta = {}
    if cfg.get("quality_gate_enabled"):
        wf_results_meta = {
            "profit_score": wf_result["score"],
            "skip20_pnl30_improvement_bps": wf_result["pnl30_improvement"],
            "skip20_pnl120_improvement_bps": wf_result["pnl120_improvement"],
        }

    metadata = {
        "version": f"v4_lgbm_{target}_retrained",
        "trained_at": datetime.now().isoformat(),
        "n_samples": len(X_valid),
        "n_features": len(feature_cols),
        "target": f"{target} regression",
        "model_class": "LGBMRegressor",
        "retrained": True,
        "prev_n_samples": prev_n_samples,
        "new_samples": new_samples,
        "wf_results": wf_results_meta,
        "feature_cols": feature_cols,
    }

    gate = SkipGate(
        model=pipeline.named_steps["model"],
        scaler=pipeline.named_steps["scaler"],
        feature_cols=feature_cols,
        config=sg_config,
        metadata=metadata,
        pipeline=pipeline,
    )

    # Step 6: アトミック書き込み
    model_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = model_path.with_suffix(".pkl.tmp")
    try:
        gate.save(tmp_path)
        # アトミック rename (Windows: os.replace)
        os.replace(str(tmp_path), str(model_path))
        # SHA256 も更新 (save が tmp に書いたハッシュを本体パスに移動)
        tmp_hash = tmp_path.with_suffix(".pkl.tmp.sha256")
        real_hash = model_path.with_suffix(".pkl.sha256")
        if tmp_hash.exists():
            os.replace(str(tmp_hash), str(real_hash))
        logger.info(f"Model atomically deployed to {model_path}")
    except Exception as e:
        # tmp 残留防止
        for p in [tmp_path, tmp_path.with_suffix(".pkl.tmp.sha256")]:
            if p.exists():
                p.unlink()
        return {**result, "status": "error", "reason": f"atomic_write_failed: {e}"}

    result["status"] = "deployed"
    result["model_path"] = str(model_path)
    logger.info(
        f"Retrain complete: {len(X_valid)} samples → {model_path} "
        f"(+{new_samples} new, score={wf_results_meta.get('profit_score', 'N/A')})"
    )
    return result


def run_scheduler(cfg: dict[str, Any]) -> None:
    """定期再学習ループ."""
    interval = cfg.get("interval_sec", 3600)
    logger.info(
        f"=== 126# Retrain Scheduler started ===\n"
        f"  interval: {interval}s ({interval / 3600:.1f}h)\n"
        f"  model_path: {cfg['model_path']}\n"
        f"  target: {cfg['target']}\n"
        f"  min_new_samples: {cfg['min_new_samples']}\n"
        f"  quality_gate: {cfg.get('quality_gate_enabled', True)}"
    )

    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)
    history_path = log_dir / "retrain_history.jsonl"

    while True:
        try:
            result = retrain_model(cfg)
            logger.info(f"Retrain cycle: status={result['status']}")
            # 履歴ファイルに記録
            with open(history_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(result, default=str) + "\n")
        except Exception as e:
            logger.error(f"Retrain cycle failed: {e}", exc_info=True)

        logger.info(f"Next retrain in {interval}s ({interval / 3600:.1f}h)")
        time.sleep(interval)


def main() -> None:
    """CLI エントリポイント."""
    parser = argparse.ArgumentParser(description="126# SkipGate retrain scheduler")
    parser.add_argument(
        "--once", action="store_true",
        help="ワンショット実行 (スケジューラループなし)",
    )
    parser.add_argument(
        "--config", type=str, default="configs/v460/fill_test.yaml",
        help="YAML 設定ファイルパス",
    )
    args = parser.parse_args()

    Path("logs").mkdir(exist_ok=True)
    cfg = load_retrain_config(Path(args.config))

    if args.once:
        logger.info("=== One-shot retrain ===")
        result = retrain_model(cfg)
        logger.info(f"Result: {json.dumps(result, indent=2, default=str)}")
    else:
        run_scheduler(cfg)


if __name__ == "__main__":
    main()
