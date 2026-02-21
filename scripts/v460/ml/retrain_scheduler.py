"""126# SkipGate 定期再学習スケジューラ.

fill_test 稼働中にバックグラウンドで定期的にモデルを再学習し、
アトミックにモデルファイルを差し替える。

127# レビュー反映:
  - C1: model_path / mode / use_ob_features を skip_gate: セクションから継承
  - H1: PnL 用特徴量ビルダー (AS ラベル非依存)
  - H2: run_id フィルタ (latest_run_only)
  - M1: absolute_min_score (初回モデル品質保証)
  - M2: target 命名 pnl30/pnl120 に統一
  - X1: module-level FileHandler レース修正
  - X2: モデル二重ロード解消

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
import json
import logging
import os
import sys
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

# 127# X1: module-level FileHandler を廃止。main() 内で初期化する。
logger = logging.getLogger(__name__)

# デフォルト設定 (127# C1: model_path/mode/use_ob は skip_gate: から継承)
_DEFAULT_CONFIG: dict[str, Any] = {
    "interval_sec": 3600,           # 再学習間隔 (秒) — 1時間
    "min_new_samples": 30,          # 再学習に必要な最小新規サンプル数
    "min_total_samples": 100,       # 再学習に必要な最小合計サンプル数
    # 130# Bootstrap Phase: run切替直後のスタベーション回避
    "bootstrap_min_total_samples": 30,   # Bootstrap 段階での最小合計
    "bootstrap_min_new_samples": 10,     # Bootstrap 段階での最小新規
    "bootstrap_threshold": 100,          # total < この値なら Bootstrap 段階と判定
    "target": "pnl120",            # 127# M2: "pnl120" or "pnl30"
    # 127# H2: run_id 分離
    "latest_run_only": True,        # 最新 run_id のみ学習
    "exclude_missing_run_id": True, # run_id 欠損行除外
    # Walk-Forward 品質ゲート
    "quality_gate_enabled": True,
    "min_score_improvement": -0.05,  # 前モデル比でこれ以下なら棄却
    "absolute_min_score": -0.10,    # 127# M1: prev_score 不在時の絶対最低 score
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
    """YAML retrain + skip_gate セクションから設定を読み込む.

    127# C1: model_path / mode / use_ob_features は skip_gate: から継承。
    retrain: セクションで重複定義せずに single source of truth を保証。
    """
    cfg = dict(_DEFAULT_CONFIG)
    yaml_path = config_path or Path("configs/v460/fill_test.yaml")
    if yaml_path.exists():
        try:
            import yaml
            with open(yaml_path) as f:
                yaml_data = yaml.safe_load(f) or {}

            # 127# C1: skip_gate セクションから model_path / mode / use_ob_features を継承
            sg_cfg = yaml_data.get("skip_gate", {})
            cfg["model_path"] = sg_cfg.get("model_path", "models/v460/skip_gate_lgbm_pnl120.pkl")
            cfg["mode"] = sg_cfg.get("mode", "pnl")
            cfg["use_ob_features"] = sg_cfg.get("use_ob_features", True)
            # results_dir はトップレベルから継承
            cfg["results_dir"] = yaml_data.get("results_dir", "results/v460/fill_test")

            retrain_cfg = yaml_data.get("retrain", {})
            if retrain_cfg:
                for key in _DEFAULT_CONFIG:
                    if key in retrain_cfg:
                        cfg[key] = retrain_cfg[key]
                logger.info(f"Retrain config loaded from {yaml_path}")
        except Exception as e:
            logger.warning(f"Failed to load retrain config: {e}, using defaults")

    # 127# C1: フォールバック (YAML に skip_gate セクションがない場合)
    cfg.setdefault("model_path", "models/v460/skip_gate_lgbm_pnl120.pkl")
    cfg.setdefault("mode", "pnl")
    cfg.setdefault("use_ob_features", True)
    cfg.setdefault("results_dir", "results/v460/fill_test")

    # 127# C1: mode/target 整合性バリデーション
    _validate_config(cfg)
    return cfg


def _validate_config(cfg: dict[str, Any]) -> None:
    """127# C1: 設定の整合性を検証 (fail-fast)."""
    mode = cfg.get("mode", "pnl")
    target = cfg.get("target", "pnl120")

    if mode != "pnl":
        raise ValueError(
            f"retrain_scheduler requires skip_gate.mode='pnl' but got '{mode}'. "
            f"retrain は PnL 回帰モデルのみ対応。skip_gate.mode を 'pnl' に変更してください。"
        )
    if target not in ("pnl120", "pnl30"):
        raise ValueError(
            f"retrain.target must be 'pnl120' or 'pnl30', got '{target}'. "
            f"127# M2: 'as30' は廃止。PnL 回帰 target を指定してください。"
        )
    model_path = cfg.get("model_path", "")
    if not model_path:
        raise ValueError("model_path is empty. skip_gate.model_path を設定してください。")


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

    127# レビュー反映:
      - H1: PnL 向け特徴量抽出 (AS ラベル非依存)
      - H2: run_id フィルタリング
      - M1: absolute_min_score
      - X2: モデル二重ロード解消

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

    # 127# H2: run_id フィルタリング
    run_id_filter: str | list[str] | None = cfg.get("run_id_filter")
    exclude_missing = cfg.get("exclude_missing_run_id", True)
    latest_run_only = cfg.get("latest_run_only", True)

    # Step 1: データロード
    try:
        records = load_fill_records(
            results_dir,
            exclude_missing_run_id=exclude_missing,
        )
    except FileNotFoundError:
        return {**result, "status": "skipped", "reason": "no fill_records found"}

    # 127# H2: latest_run_only — 最新 run_id に絞り込み
    if latest_run_only and "run_id" in records.columns:
        valid_runs = records["run_id"].dropna()
        if len(valid_runs) > 0:
            latest_run = valid_runs.iloc[-1]
            records = records[records["run_id"] == latest_run]
            result["run_id"] = str(latest_run)
            logger.info(f"H2: Filtered to latest run_id={latest_run} ({len(records)} records)")
    elif run_id_filter is not None and "run_id" in records.columns:
        if isinstance(run_id_filter, str):
            run_id_filter = [run_id_filter]
        records = records[records["run_id"].isin(run_id_filter)]

    # 130# Y5: balance_forced_switch=True のレコードを学習対象から除外
    # 残高制約による強制 side 切替はノイズ — PnL/AS 評価を歪める
    if "balance_forced_switch" in records.columns:
        n_before = len(records)
        balance_mask = records["balance_forced_switch"].fillna(False).astype(bool)
        n_forced = int(balance_mask.sum())
        if n_forced > 0:
            records = records[~balance_mask].reset_index(drop=True)
            logger.info(
                f"130# Y5: Excluded {n_forced}/{n_before} balance_forced_switch records "
                f"({n_forced/n_before*100:.1f}%)"
            )

    enriched = enrich_fill_records(
        records,
        trades_fallback_recent_days=cfg.get("trades_fallback_recent_days", 1),
    )

    # 127# H1: PnL 回帰向け特徴量抽出 (AS ラベル非依存)
    # filled かつ spread 有りのみ (AS ラベルは不要)
    pnl_mask = enriched["filled"].astype(bool)
    if "spread_at_order" in enriched.columns:
        pnl_mask = pnl_mask & enriched["spread_at_order"].notna()
    if "spread_offset_ratio" in enriched.columns:
        pnl_mask = pnl_mask & enriched["spread_offset_ratio"].notna()
    pnl_data = enriched.loc[pnl_mask]

    if len(pnl_data) < 10:
        return {**result, "status": "skipped", "reason": f"Insufficient filled samples: {len(pnl_data)}"}

    # 特徴量構築 (build_preorder_as_features の特徴量ロジックを再利用)
    try:
        X_base, _ = build_preorder_as_features(
            enriched.assign(
                # H1: pnl_mask 行のみ使う。AS ラベルを一時的に全行に付与して
                # build_preorder_as_features の filter を通す
                adverse_selected_raw=lambda df: df["adverse_selected_raw"].fillna(0),
            ),
            require_spread=True,
        )
    except ValueError:
        # AS ラベル補完後でもサンプル不足
        return {**result, "status": "skipped", "reason": "Insufficient samples after feature build"}

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

    # H1: ドロップ統計記録
    n_original_filled = int(pnl_mask.sum())
    result["filled_records"] = n_original_filled
    result["dropped_by_feature_build"] = n_original_filled - int(len(X_full))

    # Step 2: 最小サンプルチェック (130# Bootstrap 2段化)
    bootstrap_threshold = cfg.get("bootstrap_threshold", 100)
    is_bootstrap = len(X_valid) < bootstrap_threshold
    if is_bootstrap:
        min_total = cfg.get("bootstrap_min_total_samples", 30)
        result["phase"] = "bootstrap"
        logger.info(
            f"130# Bootstrap phase: {len(X_valid)} < {bootstrap_threshold}, "
            f"using min_total={min_total}"
        )
    else:
        min_total = cfg.get("min_total_samples", 100)
        result["phase"] = "stable"
    if len(X_valid) < min_total:
        return {
            **result,
            "status": "skipped",
            "reason": f"insufficient samples: {len(X_valid)} < {min_total} ({result['phase']})",
        }

    # 127# X2: 前モデルを一度だけロード (n_samples + WF score を取得)
    prev_n_samples = 0
    prev_score = 0.0
    prev_gate_loaded = False
    if model_path.exists():
        try:
            prev_gate = SkipGate.load(model_path)
            prev_n_samples = prev_gate.metadata.get("n_samples", 0)
            prev_wf = prev_gate.metadata.get("wf_results", {})
            prev_score = prev_wf.get("profit_score", 0.0)
            prev_gate_loaded = True
            del prev_gate  # メモリ早期解放
        except Exception:
            pass

    # Step 3: 新規サンプルチェック (130# Bootstrap 2段化)
    if is_bootstrap:
        min_new = cfg.get("bootstrap_min_new_samples", 10)
    else:
        min_new = cfg.get("min_new_samples", 30)
    new_samples = len(X_valid) - prev_n_samples
    result["new_samples"] = int(new_samples)

    if new_samples < min_new:
        return {
            **result,
            "status": "skipped",
            "reason": f"insufficient new samples: {new_samples} < {min_new} ({result['phase']})",
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

        # 127# M1: 前モデル不在時の絶対最低 score チェック
        absolute_min = cfg.get("absolute_min_score", -0.10)
        if not prev_gate_loaded and wf_result["score"] < absolute_min:
            logger.warning(
                f"Quality gate REJECT (no prev model): "
                f"score={wf_result['score']:.4f} < absolute_min={absolute_min}. "
            )
            return {**result, "status": "rejected", "reason": "absolute_min_score"}

        # 品質ゲート: 前モデルの score と比較 (127# X2: prev_score は Step 3 で取得済み)
        min_improvement = cfg.get("min_score_improvement", -0.05)

        improvement = wf_result["score"] - prev_score
        result["score_improvement"] = improvement
        result["prev_score"] = prev_score
        if prev_gate_loaded and improvement < min_improvement:
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

    # SkipGateConfig — 127# C1: mode を設定から取得
    sg_config = SkipGateConfig(
        mode=cfg.get("mode", "pnl"),
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


def run_scheduler(cfg: dict[str, Any], config_path: Path | None = None) -> None:
    """定期再学習ループ.

    130# L2: サイクルごとに YAML を再読み込みし、YAML 変更を再起動なしで反映。
    """
    interval = cfg.get("interval_sec", 3600)
    logger.info(
        f"=== 126# Retrain Scheduler started ===\n"
        f"  interval: {interval}s ({interval / 3600:.1f}h)\n"
        f"  model_path: {cfg['model_path']}\n"
        f"  target: {cfg['target']}\n"
        f"  min_new_samples: {cfg['min_new_samples']}\n"
        f"  quality_gate: {cfg.get('quality_gate_enabled', True)}\n"
        f"  config_hot_reload: {config_path is not None}"
    )

    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)
    history_path = log_dir / "retrain_history.jsonl"

    while True:
        # 130# L2: サイクルごとに YAML からリロード (閾値・ターゲット変更を反映)
        if config_path is not None:
            try:
                new_cfg = load_retrain_config(config_path)
                # interval は起動時のみ有効 (loop 途中変更は次回起動で反映)
                for key in new_cfg:
                    if key != "interval_sec":
                        cfg[key] = new_cfg[key]
                logger.debug("130# Config reloaded from YAML")
            except Exception as e:
                logger.warning(f"130# Config reload failed (using previous): {e}")

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
        "--all-runs", action="store_true",
        help="Y3: 全 run_id のデータを使用 (latest_run_only を無効化)。"
             "過去蓄積データ (1048+) を含むオフライン再訓練用。",
    )
    parser.add_argument(
        "--config", type=str, default="configs/v460/fill_test.yaml",
        help="YAML 設定ファイルパス",
    )
    args = parser.parse_args()

    # 127# X1: logging 初期化を main() 内に移動 (module-level FileHandler レース防止)
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(log_dir / "retrain_scheduler.log", encoding="utf-8"),
        ],
    )

    cfg = load_retrain_config(Path(args.config))

    # Y3: --all-runs — 全 run のデータを学習に使用 (118# Appendix F)
    if args.all_runs:
        cfg["latest_run_only"] = False
        cfg["exclude_missing_run_id"] = False
        # Bootstrap 閾値を無効化 (十分なデータがある前提)
        cfg["min_total_samples"] = 30
        cfg["min_new_samples"] = 0  # 新規サンプル要件を緩和
        # Y3: relative quality gate を無効化 (既存モデルと target が異なる可能性)
        # absolute_min_score チェックは維持 (-0.10 以下は棄却)
        cfg["min_score_improvement"] = -999.0
        logger.info(
            "Y3: --all-runs enabled → latest_run_only=False, "
            "exclude_missing_run_id=False, min_new_samples=0, "
            "relative quality gate bypassed (absolute_min retained)"
        )

    if args.once:
        logger.info("=== One-shot retrain ===")
        result = retrain_model(cfg)
        logger.info(f"Result: {json.dumps(result, indent=2, default=str)}")
    else:
        run_scheduler(cfg, config_path=Path(args.config))


if __name__ == "__main__":
    main()
