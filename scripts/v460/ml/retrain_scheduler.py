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


def _safe_import_ztb_module(dotted_path: str) -> Any:
    """ztb サブモジュールを circular import を回避して読み込む.

    ztb.analysis/__init__.py → ztb.trading 間の循環参照があるため、
    通常の import ではサブモジュール (redundancy, gate_checks, splitter) が
    ロード不可になる場合がある。importlib.util で直接ロードすることで回避する。
    """
    import importlib.util
    import types as _builtin_types

    # まず通常 import を試行 (循環が解消されている場合はこちらが速い)
    try:
        parts = dotted_path.rsplit(".", 1)
        mod = __import__(dotted_path, fromlist=[parts[-1]] if len(parts) > 1 else [])
        return mod
    except ImportError:
        pass

    # フォールバック: importlib.util で直接ファイルロード
    file_path = Path(dotted_path.replace(".", "/") + ".py")
    if not file_path.exists():
        raise ImportError(f"{dotted_path} not found at {file_path}")

    # 親パッケージが必要な場合 (relative import 解決用) sys.modules にスタブ登録
    parts = dotted_path.split(".")
    for i in range(1, len(parts)):
        pkg_name = ".".join(parts[:i])
        if pkg_name not in sys.modules:
            pkg_dir = Path("/".join(parts[:i]))
            pkg = _builtin_types.ModuleType(pkg_name)
            pkg.__path__ = [str(pkg_dir)]
            pkg.__package__ = pkg_name
            sys.modules[pkg_name] = pkg
            # 既にロード済みの子を親に紐付け
            for child_name, child_mod in list(sys.modules.items()):
                if child_name.startswith(pkg_name + "."):
                    short = child_name[len(pkg_name) + 1:].split(".")[0]
                    setattr(pkg, short, child_mod)

    spec = importlib.util.spec_from_file_location(dotted_path, str(file_path))
    mod = importlib.util.module_from_spec(spec)
    mod.__package__ = ".".join(parts[:-1])
    sys.modules[dotted_path] = mod
    spec.loader.exec_module(mod)  # type: ignore[union-attr]
    # 親パッケージにアトリビュート設定
    if len(parts) > 1:
        parent_name = ".".join(parts[:-1])
        if parent_name in sys.modules:
            setattr(sys.modules[parent_name], parts[-1], mod)
    return mod

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
    # E1: warm-start (前モデルの Booster を init_model に使用)
    "warm_start_enabled": True,
    # E2: early stopping (WF val split で過学習を自動停止)
    "early_stopping_rounds": 20,        # N ラウンド改善なし → 停止
    "lgbm_n_estimators_max": 300,        # early stopping 時の上限
    # E3: dead feature pruning (split=0 の特徴量を自動除外)
    "feature_pruning_enabled": True,
    "feature_pruning_min_importance": 0,  # split 回数がこれ以下なら除外
    "feature_pruning_min_trees": 20,      # 131# A.1 #7: WF eval 木数がこれ未満なら pruning 不安定のためスキップ
    "feature_pruning_require_consecutive": True,  # 131# B: 連続 dead のみ prune (振動防止)
    # E4: enriched data cache (I/O 削減)
    "enriched_cache_enabled": True,
    # C1: WF multi-window evaluation (131# WalkForwardSplitter 統合)
    "wf_multi_window_enabled": True,
    "wf_initial_train_pct": 0.50,
    "wf_val_pct": 0.10,
    "wf_test_pct": 0.15,
    "wf_step_pct": 0.20,
    "wf_embargo_rows": 0,            # fill records は日次でない → 行数ベース (splitter min=1)
    "wf_min_window_train": 30,       # window 当たり最低訓練サンプル数
    "wf_min_window_test": 10,        # window 当たり最低テストサンプル数
    # C2: 統計的品質ゲート (131# gate_checks 統合)
    "statistical_gate_enabled": True,
    "statistical_gate_alpha": 0.05,
    "statistical_gate_min_effect": 0.147,   # retrain は少量サンプル → small effect
    "statistical_gate_min_test_samples": 40, # 合計テストサンプル < これなら統計ゲートスキップ
    # C3: 冗長特徴量除去 (131# redundancy 統合)
    "redundancy_pruning_enabled": True,
    "redundancy_correlation_threshold": 0.85,
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
    # 131# A.1 #5: target と model_path の命名不整合警告
    if model_path and target:
        path_has_pnl120 = "pnl120" in model_path
        path_has_pnl30 = "pnl30" in model_path
        if (target == "pnl30" and path_has_pnl120 and not path_has_pnl30):
            logger.warning(
                f"131# A.1 #5: target='{target}' but model_path contains 'pnl120': "
                f"{model_path}. 運用上の誤認に注意。"
                f"metadata.target を確認してください。"
            )
        elif (target == "pnl120" and path_has_pnl30 and not path_has_pnl120):
            logger.warning(
                f"131# A.1 #5: target='{target}' but model_path contains 'pnl30': "
                f"{model_path}. 運用上の誤認に注意。"
            )


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


def _get_enriched_cache_path(results_dir: Path) -> Path:
    """E4: enriched data cache のパスを返す."""
    cache_dir = Path("cache/data")
    cache_dir.mkdir(parents=True, exist_ok=True)
    return cache_dir / f"enriched_{results_dir.name}.pkl"


def _load_enriched_cache(
    cache_path: Path, n_records: int,
    cache_key: str | None = None,
) -> pd.DataFrame | None:
    """E4: enriched cache を読み込み。レコード数不一致 or cache_key 不一致なら invalidate.

    131# A.1 #6: 行数のみの invalidation を廃止。
    cache_key (target + feature_cols + config digest) を併用。
    """
    if not cache_path.exists():
        return None
    try:
        import pickle
        with open(cache_path, "rb") as f:
            payload = pickle.load(f)  # noqa: S301
        # 後方互換: 旧 cache は DataFrame 直接
        if isinstance(payload, pd.DataFrame):
            cached = payload
            stored_key = None
        elif isinstance(payload, dict):
            cached = payload.get("data")
            stored_key = payload.get("cache_key")
        else:
            logger.info("E4: Cache format unrecognized, invalidated")
            return None
        if cached is None or len(cached) != n_records:
            logger.info(
                f"E4: Cache invalidated (records: cached={len(cached) if cached is not None else 0}, current={n_records})"
            )
            return None
        if cache_key is not None and stored_key != cache_key:
            logger.info(
                f"E4: Cache invalidated (key mismatch: stored={stored_key}, current={cache_key})"
            )
            return None
        logger.info(f"E4: Loaded enriched cache ({len(cached)} records) from {cache_path}")
        return cached
    except Exception as e:
        logger.warning(f"E4: Cache load failed: {e}")
        return None


def _save_enriched_cache(
    cache_path: Path,
    enriched: pd.DataFrame,
    cache_key: str | None = None,
) -> None:
    """E4: enriched data を cache に保存 (cache_key 付き)."""
    try:
        import pickle
        payload = {
            "data": enriched,
            "cache_key": cache_key,
            "n_records": len(enriched),
        }
        with open(cache_path, "wb") as f:
            pickle.dump(payload, f)
        logger.info(f"E4: Saved enriched cache ({len(enriched)} records, key={cache_key}) to {cache_path}")
    except Exception as e:
        logger.warning(f"E4: Cache save failed: {e}")


def _build_lgbm_regressor(
    cfg: dict[str, Any],
    n_estimators_override: int | None = None,
) -> "lgb.LGBMRegressor":
    """共通 LGBMRegressor 構築 (DRY)."""
    import lightgbm as lgb

    return lgb.LGBMRegressor(
        n_estimators=n_estimators_override or cfg.get("lgbm_n_estimators", 150),
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


def _evaluate_wf(
    X: pd.DataFrame,
    y: pd.Series,
    enriched: pd.DataFrame,
    cfg: dict[str, Any],
    prev_booster: "Any | None" = None,
) -> dict[str, Any]:
    """Walk-Forward OOS 評価ディスパッチ.

    131# C1: wf_multi_window_enabled=True なら WalkForwardSplitter で
    multi-window 評価を実行。データ不足時は single-window にフォールバック。
    """
    if cfg.get("wf_multi_window_enabled", True):
        try:
            result = _evaluate_wf_multi(X, y, enriched, cfg, prev_booster)
            if result is not None:
                return result
        except Exception as e:
            logger.warning(f"C1: Multi-window WF failed ({e}), falling back to single")
    return _evaluate_wf_single(X, y, enriched, cfg, prev_booster)


def _evaluate_wf_multi(
    X: pd.DataFrame,
    y: pd.Series,
    enriched: pd.DataFrame,
    cfg: dict[str, Any],
    prev_booster: "Any | None" = None,
) -> dict[str, Any] | None:
    """131# C1: Multi-window Walk-Forward 評価 (WalkForwardSplitter 統合).

    複数の WF ウィンドウで独立に train→predict し、per-window PnL を収集。
    g1_judgment / holm_bonferroni_gate 用の fold-level データを返す。

    Returns:
        評価結果 dict。ウィンドウが 2 未満なら None (single-window フォールバック)。
    """
    from sklearn.impute import SimpleImputer
    from sklearn.preprocessing import StandardScaler

    try:
        import lightgbm as lgb
    except ImportError:
        return None

    try:
        _splitter_mod = _safe_import_ztb_module("ztb.evaluation.walk_forward.splitter")
        WalkForwardSplitter = _splitter_mod.WalkForwardSplitter
    except ImportError:
        logger.info("C1: WalkForwardSplitter not available, skipping multi-window")
        return None

    n = len(X)
    # Splitter にはダミー DataFrame を渡す (行数さえ合えばよい)
    dummy_df = pd.DataFrame(index=range(n))
    splitter = WalkForwardSplitter(
        initial_train_pct=cfg.get("wf_initial_train_pct", 0.50),
        val_pct=cfg.get("wf_val_pct", 0.10),
        test_pct=cfg.get("wf_test_pct", 0.15),
        step_pct=cfg.get("wf_step_pct", 0.20),
        embargo_days=cfg.get("wf_embargo_rows", 0),
    )

    try:
        windows = splitter.split(dummy_df)
    except ValueError as e:
        logger.info(f"C1: WalkForwardSplitter could not split (n={n}): {e}")
        return None

    min_train = cfg.get("wf_min_window_train", 30)
    min_test = cfg.get("wf_min_window_test", 10)

    # 有効ウィンドウのみ選択
    valid_windows = [
        w for w in windows
        if (w.train_end - w.train_start) >= min_train
        and (w.test_end - w.test_start) >= min_test
    ]
    if len(valid_windows) < 2:
        logger.info(
            f"C1: Only {len(valid_windows)} valid window(s) from {len(windows)} total "
            f"(min_train={min_train}, min_test={min_test}), falling back to single-window"
        )
        return None

    logger.info(f"C1: Multi-window WF with {len(valid_windows)} windows (n={n})")

    # Per-window 評価
    window_scores: list[float] = []
    window_imp30: list[float] = []
    window_imp120: list[float] = []
    fold_pnl30: list[tuple[list[float], list[float]]] = []
    fold_pnl120: list[tuple[list[float], list[float]]] = []
    all_feat_importance: dict[str, int] = {}
    total_n_trees = 0
    total_n_test = 0
    total_n_train = 0

    early_stop = cfg.get("early_stopping_rounds", 0)
    n_est = cfg.get("lgbm_n_estimators_max", 300) if early_stop > 0 else cfg.get("lgbm_n_estimators", 150)

    for win in valid_windows:
        X_train = X.iloc[win.train_start:win.train_end]
        y_train = y.iloc[win.train_start:win.train_end]
        X_val = X.iloc[win.val_start:win.val_end]
        y_val = y.iloc[win.val_start:win.val_end]
        X_test = X.iloc[win.test_start:win.test_end]
        y_test = y.iloc[win.test_start:win.test_end]

        # 前処理
        imputer = SimpleImputer(strategy="median")
        scaler = StandardScaler()
        X_train_sc = pd.DataFrame(
            scaler.fit_transform(imputer.fit_transform(X_train)),
            columns=X_train.columns, index=X_train.index,
        )
        X_val_sc = pd.DataFrame(
            scaler.transform(imputer.transform(X_val)),
            columns=X_val.columns, index=X_val.index,
        )
        X_test_sc = pd.DataFrame(
            scaler.transform(imputer.transform(X_test)),
            columns=X_test.columns, index=X_test.index,
        )

        lgbm_model = _build_lgbm_regressor(cfg, n_estimators_override=n_est)

        fit_kwargs: dict[str, Any] = {}
        if early_stop > 0 and len(X_val) >= 5:
            fit_kwargs["eval_set"] = [(X_val_sc, y_val)]
            fit_kwargs["callbacks"] = [
                lgb.early_stopping(stopping_rounds=early_stop, verbose=False),
                lgb.log_evaluation(period=0),
            ]

        lgbm_model.fit(X_train_sc, y_train, **fit_kwargs)
        preds_test = lgbm_model.predict(X_test_sc)

        # OOS PnL 参照
        test_idx = X_test.index
        pnl30_col = "post_fill_30s_pnl"
        pnl120_col = "post_fill_120s_pnl"
        pnl30 = (
            enriched.loc[test_idx, pnl30_col].astype(float).values
            if pnl30_col in enriched.columns
            else np.full(len(test_idx), np.nan)
        )
        pnl120 = (
            enriched.loc[test_idx, pnl120_col].astype(float).values
            if pnl120_col in enriched.columns
            else np.full(len(test_idx), np.nan)
        )

        # Skip bottom 20% predicted PnL
        threshold = np.percentile(preds_test, 20)
        keep_mask = preds_test >= threshold

        baseline_30 = float(np.nanmean(pnl30))
        baseline_120 = float(np.nanmean(pnl120))
        kept_30 = float(np.nanmean(pnl30[keep_mask]))
        kept_120 = float(np.nanmean(pnl120[keep_mask]))

        imp_30 = kept_30 - baseline_30
        imp_120 = kept_120 - baseline_120
        score = imp_120 - max(0, -imp_30)

        window_scores.append(score)
        window_imp30.append(imp_30)
        window_imp120.append(imp_120)

        # NaN を除外した fold-level PnL データ (statistical gate 用)
        kept_pnl30_clean = [float(v) for v in pnl30[keep_mask] if not np.isnan(v)]
        all_pnl30_clean = [float(v) for v in pnl30 if not np.isnan(v)]
        kept_pnl120_clean = [float(v) for v in pnl120[keep_mask] if not np.isnan(v)]
        all_pnl120_clean = [float(v) for v in pnl120 if not np.isnan(v)]
        fold_pnl30.append((kept_pnl30_clean, all_pnl30_clean))
        fold_pnl120.append((kept_pnl120_clean, all_pnl120_clean))

        # Feature importance 集計
        if hasattr(lgbm_model, "feature_importances_"):
            for col, imp in zip(X_train.columns, lgbm_model.feature_importances_):
                all_feat_importance[col] = all_feat_importance.get(col, 0) + int(imp)

        n_trees = lgbm_model.booster_.num_trees() if hasattr(lgbm_model, "booster_") else n_est
        total_n_trees += n_trees
        total_n_test += len(X_test)
        total_n_train += len(X_train)

    if not window_scores:
        logger.warning("C1: All windows failed, falling back to single-window")
        return None

    n_windows = len(window_scores)
    avg_score = float(np.mean(window_scores))
    avg_imp30 = float(np.mean(window_imp30))
    avg_imp120 = float(np.mean(window_imp120))

    logger.info(
        f"C1: Multi-window results: {n_windows} windows, "
        f"avg_score={avg_score:.4f}, avg_imp30={avg_imp30:.4f}, avg_imp120={avg_imp120:.4f}"
    )

    return {
        "score": avg_score,
        "pnl30_improvement": avg_imp30,
        "pnl120_improvement": avg_imp120,
        "n_test": total_n_test,
        "n_train": total_n_train,
        "actual_n_trees": total_n_trees // max(n_windows, 1),
        "feature_importance": all_feat_importance,
        # C1/C2: fold-level data for statistical gate
        "n_windows": n_windows,
        "fold_pnl30": fold_pnl30,
        "fold_pnl120": fold_pnl120,
        "window_scores": window_scores,
    }


def _evaluate_wf_single(
    X: pd.DataFrame,
    y: pd.Series,
    enriched: pd.DataFrame,
    cfg: dict[str, Any],
    prev_booster: "Any | None" = None,
) -> dict[str, float]:
    """Walk-Forward OOS 評価で品質スコアを算出.

    直近 test_ratio をテストセットとし、残りで訓練→テスト予測の skip simulation。
    E1: prev_booster があれば warm-start で学習。
    E2: early_stopping_rounds で過学習を自動防止。

    Returns:
        {"score": float, "pnl30_improvement": float, "pnl120_improvement": float, ...}
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

    # E2: early stopping 有効時は上限を引き上げ
    early_stop = cfg.get("early_stopping_rounds", 0)
    if early_stop > 0:
        n_est = cfg.get("lgbm_n_estimators_max", 300)
    else:
        n_est = cfg.get("lgbm_n_estimators", 150)

    lgbm_model = _build_lgbm_regressor(cfg, n_estimators_override=n_est)

    # E2: early stopping 用の前処理 (Pipeline 内で fit するため手動分離)
    imputer = SimpleImputer(strategy="median")
    scaler = StandardScaler()
    X_train_imp = pd.DataFrame(
        imputer.fit_transform(X_train), columns=X_train.columns, index=X_train.index,
    )
    X_train_sc = pd.DataFrame(
        scaler.fit_transform(X_train_imp), columns=X_train.columns, index=X_train.index,
    )
    X_test_imp = pd.DataFrame(
        imputer.transform(X_test), columns=X_test.columns, index=X_test.index,
    )
    X_test_sc = pd.DataFrame(
        scaler.transform(X_test_imp), columns=X_test.columns, index=X_test.index,
    )

    # E1: warm-start — 前モデルの booster を init_model に使用
    fit_kwargs: dict[str, Any] = {}
    if early_stop > 0:
        fit_kwargs["eval_set"] = [(X_test_sc, y_test)]
        # LightGBM 4.x: callbacks で early stopping
        fit_kwargs["callbacks"] = [
            lgb.early_stopping(stopping_rounds=early_stop, verbose=False),
            lgb.log_evaluation(period=0),  # suppress iteration log
        ]
    if prev_booster is not None and cfg.get("warm_start_enabled", True):
        fit_kwargs["init_model"] = prev_booster
        logger.info("E1: Using prev booster as init_model for WF eval")

    lgbm_model.fit(X_train_sc, y_train, **fit_kwargs)
    preds_test = lgbm_model.predict(X_test_sc)

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

    # E2: 実際に使用された木の数を記録
    actual_n_trees = lgbm_model.booster_.num_trees() if hasattr(lgbm_model, "booster_") else n_est

    # E3: feature importance を記録 (pruning 判定に使用)
    feat_importance: dict[str, int] = {}
    if hasattr(lgbm_model, "feature_importances_"):
        for col, imp in zip(X_train.columns, lgbm_model.feature_importances_):
            feat_importance[col] = int(imp)

    # C2: single-window でも fold-level PnL を記録 (holm_bonferroni_gate 用)
    kept_pnl30_clean = [float(v) for v in pnl30[keep_mask] if not np.isnan(v)]
    all_pnl30_clean = [float(v) for v in pnl30 if not np.isnan(v)]
    kept_pnl120_clean = [float(v) for v in pnl120[keep_mask] if not np.isnan(v)]
    all_pnl120_clean = [float(v) for v in pnl120 if not np.isnan(v)]

    return {
        "score": score,
        "pnl30_improvement": imp_30,
        "pnl120_improvement": imp_120,
        "n_test": int(len(X_test)),
        "n_train": int(len(X_train)),
        "actual_n_trees": actual_n_trees,
        "feature_importance": feat_importance,
        # C2: statistical gate 用 per-sample data
        "n_windows": 1,
        "fold_pnl30": [(kept_pnl30_clean, all_pnl30_clean)],
        "fold_pnl120": [(kept_pnl120_clean, all_pnl120_clean)],
    }


def _apply_statistical_gate(
    wf_result: dict[str, Any],
    cfg: dict[str, Any],
) -> dict[str, Any]:
    """131# C2: 統計的品質ゲート.

    WF 評価の per-window / per-sample PnL データに対して
    gate_checks の統計検定を適用する。

    - Multi-window (n_windows >= 2): g1_judgment (fold p-mean → Holm → AND)
    - Single-window: holm_bonferroni_gate (per-sample PnL)
    - データ不足: スキップ (applied=False)

    Returns:
        {"applied": bool, "pass": bool, "method": str, ...}
    """
    try:
        _gc_mod = _safe_import_ztb_module("ztb.metrics.gate_checks")
        g1_judgment = _gc_mod.g1_judgment
        holm_bonferroni_gate = _gc_mod.holm_bonferroni_gate
    except ImportError:
        return {"applied": False, "reason": "gate_checks not importable"}

    alpha = cfg.get("statistical_gate_alpha", 0.05)
    min_effect = cfg.get("statistical_gate_min_effect", 0.147)
    min_test = cfg.get("statistical_gate_min_test_samples", 40)

    fold_pnl30 = wf_result.get("fold_pnl30", [])
    fold_pnl120 = wf_result.get("fold_pnl120", [])
    n_windows = wf_result.get("n_windows", 0)

    # 合計テストサンプル数チェック
    total_test = sum(len(b) for _, b in fold_pnl30) if fold_pnl30 else 0
    if total_test < min_test:
        return {
            "applied": False,
            "reason": f"insufficient_test_samples ({total_test} < {min_test})",
        }

    # fold-level data 構築
    fold_results: dict[str, list[tuple[list[float], list[float]]]] = {}
    if fold_pnl30:
        fold_results["pnl30"] = fold_pnl30
    if fold_pnl120:
        fold_results["pnl120"] = fold_pnl120

    if not fold_results:
        return {"applied": False, "reason": "no_fold_data"}

    if n_windows >= 2:
        # Multi-window: g1_judgment (§5.3 準拠)
        g1 = g1_judgment(fold_results, alpha=alpha, min_effect=min_effect)
        return {
            "applied": True,
            "pass": g1["g1_pass"],
            "method": "g1_judgment",
            "n_windows": n_windows,
            "passed_targets": g1["passed_targets"],
            "details": g1["details"],
            "alpha": alpha,
            "min_effect": min_effect,
        }
    else:
        # Single-window: holm_bonferroni_gate (per-sample)
        hb_input: dict[str, tuple[list[float], list[float]]] = {}
        for key, folds in fold_results.items():
            if folds:
                hb_input[key] = folds[0]  # single window → first fold
        hb = holm_bonferroni_gate(hb_input, alpha=alpha, min_effect=min_effect)
        any_pass = any(v.get("pass", False) for v in hb.values())
        passed_targets = [k for k, v in hb.items() if v.get("pass", False)]
        return {
            "applied": True,
            "pass": any_pass,
            "method": "holm_bonferroni_gate",
            "n_windows": 1,
            "passed_targets": passed_targets,
            "details": hb,
            "alpha": alpha,
            "min_effect": min_effect,
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

    enriched = None
    # E4: enriched data cache — I/O コスト削減
    # 131# A.1 #6: cache_key = target + feature_cols + run_ids で stale cache 防止
    if cfg.get("enriched_cache_enabled", True):
        cache_path = _get_enriched_cache_path(results_dir)
        feature_cols_str = ",".join(sorted(get_gate_feature_cols(use_ob=use_ob)))
        run_ids_str = ""
        if "run_id" in records.columns:
            run_ids_str = ",".join(sorted(records["run_id"].dropna().unique().astype(str)))
        import hashlib as _hl
        cache_key = _hl.md5(
            f"{target}|{feature_cols_str}|{run_ids_str}".encode()
        ).hexdigest()[:16]
        enriched = _load_enriched_cache(cache_path, len(records), cache_key=cache_key)

    if enriched is None:
        enriched = enrich_fill_records(
            records,
            trades_fallback_recent_days=cfg.get("trades_fallback_recent_days", 1),
        )
        if cfg.get("enriched_cache_enabled", True):
            _save_enriched_cache(cache_path, enriched, cache_key=cache_key)

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

    # 127# X2: 前モデルを一度だけロード (n_samples + WF score + E1 booster を取得)
    prev_n_samples = 0
    prev_score = 0.0
    prev_gate_loaded = False
    prev_booster = None  # E1: warm-start 用
    prev_feature_cols: list[str] | None = None  # E3: pruning 参照用
    if model_path.exists():
        try:
            prev_gate = SkipGate.load(model_path)
            prev_n_samples = prev_gate.metadata.get("n_samples", 0)
            prev_wf = prev_gate.metadata.get("wf_results", {})
            prev_score = prev_wf.get("profit_score", 0.0)
            prev_feature_cols = prev_gate.metadata.get("feature_cols")
            # 131# B: 連続 dead pruning 用 — 前回 WF dead features を取得
            result["_prev_wf_dead_features"] = prev_gate.metadata.get(
                "wf_dead_features", [],
            )
            # E1: LightGBM booster を抽出 (warm-start に使用)
            if cfg.get("warm_start_enabled", True):
                if hasattr(prev_gate, "_pipeline") and prev_gate._pipeline is not None:
                    prev_model = prev_gate._pipeline.named_steps.get("model")
                    if hasattr(prev_model, "booster_"):
                        prev_booster = prev_model.booster_
                        logger.info("E1: Extracted prev booster for warm-start")
            prev_gate_loaded = True
            del prev_gate  # メモリ早期解放 (booster は参照保持)
        except Exception as e:
            # 131# A.1 #3: 例外を握り潰さず明示ログ化
            logger.warning(
                f"Prev model load failed: {e}. "
                f"Proceeding without prev model (no warm-start, absolute_min_score gate only)."
            )
            result["prev_model_load_error"] = str(e)

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
        wf_result = _evaluate_wf(X_valid, y_valid, enriched, cfg, prev_booster=prev_booster)
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

        # 131# A.1 #4: --all-runs 時も target PnL improvement >= 0 をハード制約
        if cfg.get("all_runs_require_positive_pnl", False):
            target = cfg.get("target", "pnl120")
            pnl_key = f"{target}_improvement"  # "pnl120_improvement" or "pnl30_improvement"
            pnl_imp = wf_result.get(pnl_key, 0.0)
            if pnl_imp < 0:
                logger.warning(
                    f"Quality gate REJECT (--all-runs positive pnl): "
                    f"{pnl_key}={pnl_imp:.4f} < 0. "
                    f"Negative expected PnL model deployment blocked."
                )
                return {**result, "status": "rejected", "reason": "negative_pnl_improvement"}

        # 131# C2: 統計的品質ゲート (gate_checks 統合)
        if cfg.get("statistical_gate_enabled", True):
            stat_gate_result = _apply_statistical_gate(wf_result, cfg)
            result["statistical_gate"] = stat_gate_result
            if stat_gate_result.get("applied"):
                if not stat_gate_result["pass"]:
                    logger.warning(
                        f"C2: Statistical gate REJECT: {stat_gate_result.get('reason', 'unknown')}. "
                        f"details={stat_gate_result.get('details', {})}"
                    )
                    return {**result, "status": "rejected", "reason": "statistical_gate"}
                logger.info(
                    f"C2: Statistical gate PASS: {stat_gate_result.get('method', 'unknown')}, "
                    f"passed_targets={stat_gate_result.get('passed_targets', [])}"
                )
            else:
                logger.info(
                    f"C2: Statistical gate skipped: {stat_gate_result.get('reason', 'unknown')}"
                )

    # Step 5: 全データで訓練 (E1 warm-start + E2 early stopping + E3 feature pruning)
    feature_cols = list(get_gate_feature_cols(use_ob=use_ob))

    # E3: Dead feature pruning — WF eval の feature_importance から split=0 を除外
    # 131# A.1 #7: WF eval の木数が少なすぎる場合は pruning 不安定のためスキップ
    # 131# B: 連続 dead = 前モデルでも dead だった特徴量のみ prune (振動防止)
    pruned_features: list[str] = []
    wf_dead_features: list[str] = []  # 今回 WF で dead な全特徴量 (metadata 記録用)
    min_trees_for_pruning = cfg.get("feature_pruning_min_trees", 20)
    wf_actual_trees = result.get("wf_eval", {}).get("actual_n_trees", 0)
    if (
        cfg.get("feature_pruning_enabled", True)
        and cfg.get("quality_gate_enabled", True)
        and "wf_eval" in result
        and wf_actual_trees >= min_trees_for_pruning
    ):
        feat_imp = result["wf_eval"].get("feature_importance", {})
        min_imp = cfg.get("feature_pruning_min_importance", 0)
        if feat_imp:
            wf_dead_features = [c for c in feature_cols if feat_imp.get(c, 0) <= min_imp]
            # 131# B: 連続 dead — 前モデルで dead or pruned だった特徴量と交差
            # require_consecutive=True (default) なら prev でも dead だった特徴量のみ prune
            prev_dead: set[str] = set()
            if prev_feature_cols is not None and prev_gate_loaded:
                # 前回 pruning 済み = 前モデル feature_cols に含まれない特徴量
                all_possible = set(get_gate_feature_cols(use_ob=use_ob))
                prev_pruned_set = all_possible - set(prev_feature_cols)
                # 前回 metadata に記録された wf_dead_features
                prev_wf_dead = set()
                if model_path.exists():
                    prev_wf_dead = set(result.get("_prev_wf_dead_features", []))
                prev_dead = prev_pruned_set | prev_wf_dead

            require_consecutive = cfg.get("feature_pruning_require_consecutive", True)
            if require_consecutive and prev_dead:
                pruned = [c for c in wf_dead_features if c in prev_dead]
                if pruned != wf_dead_features:
                    newly_dead = [c for c in wf_dead_features if c not in prev_dead]
                    logger.info(
                        f"E3: {len(newly_dead)} newly-dead features deferred "
                        f"(require consecutive): {newly_dead}"
                    )
            else:
                pruned = list(wf_dead_features)

            if pruned and len(feature_cols) - len(pruned) >= 5:
                # 最低5特徴量は保持 (過剰pruning防止)
                pruned_features = pruned
                feature_cols = [c for c in feature_cols if c not in pruned]
                logger.info(
                    f"E3: Pruned {len(pruned)} dead features: {pruned} "
                    f"→ {len(feature_cols)} features remaining"
                )
                # pruning 後のデータで再構築
                X_valid = X_valid[feature_cols]
            elif pruned:
                logger.info(
                    f"E3: Pruning blocked — would leave only "
                    f"{len(feature_cols) - len(pruned)} features (min=5)"
                )
    elif (
        cfg.get("feature_pruning_enabled", True)
        and wf_actual_trees < min_trees_for_pruning
        and wf_actual_trees > 0
    ):
        logger.info(
            f"E3: Pruning skipped — WF eval used only {wf_actual_trees} trees "
            f"(min={min_trees_for_pruning}). Importance signal too weak."
        )

    # 131# C3: 冗長特徴量除去 (redundancy.find_highly_correlated_features 統合)
    redundancy_pruned: list[str] = []
    if cfg.get("redundancy_pruning_enabled", True) and len(feature_cols) >= 5:
        try:
            _red_mod = _safe_import_ztb_module("ztb.analysis.redundancy")
            calculate_feature_correlations = _red_mod.calculate_feature_correlations
            find_highly_correlated_features = _red_mod.find_highly_correlated_features

            corr_threshold = cfg.get("redundancy_correlation_threshold", 0.85)
            corr_matrix = calculate_feature_correlations(X_valid[feature_cols])
            corr_pairs = find_highly_correlated_features(corr_matrix, corr_threshold)

            if corr_pairs:
                # WF feature_importance を使って、ペアの低 importance 側を除去
                feat_imp = result.get("wf_eval", {}).get("feature_importance", {})
                to_remove: set[str] = set()
                for f1, f2, corr_val in corr_pairs:
                    imp1 = feat_imp.get(f1, 0)
                    imp2 = feat_imp.get(f2, 0)
                    # importance が低い方を除去 (同値なら特徴量名で決定的に)
                    victim = f2 if imp1 >= imp2 else f1
                    if imp1 == imp2:
                        victim = max(f1, f2)  # 名前順で後を除去 (決定的)
                    to_remove.add(victim)

                # 最低5特徴量は保持
                remaining_after = len(feature_cols) - len(to_remove)
                if remaining_after >= 5 and to_remove:
                    redundancy_pruned = sorted(to_remove)
                    feature_cols = [c for c in feature_cols if c not in to_remove]
                    X_valid = X_valid[feature_cols]
                    logger.info(
                        f"C3: Removed {len(redundancy_pruned)} redundant features "
                        f"(corr>{corr_threshold}): {redundancy_pruned} "
                        f"→ {len(feature_cols)} features remaining"
                    )
                elif to_remove:
                    logger.info(
                        f"C3: Redundancy pruning blocked — would leave only "
                        f"{remaining_after} features (min=5)"
                    )
                else:
                    logger.info(f"C3: No redundant features found (threshold={corr_threshold})")
            else:
                logger.info(f"C3: No highly correlated pairs (threshold={corr_threshold})")
        except ImportError:
            logger.info("C3: redundancy module not available, skipping")
        except Exception as e:
            logger.warning(f"C3: Redundancy pruning failed: {e}")

    # 前処理 (Pipeline 内の fit を手動実行 — E1/E2 のため)
    imputer = SimpleImputer(strategy="median")
    scaler = StandardScaler()
    X_imp = pd.DataFrame(
        imputer.fit_transform(X_valid), columns=feature_cols, index=X_valid.index,
    )
    X_sc = pd.DataFrame(
        scaler.fit_transform(X_imp), columns=feature_cols, index=X_valid.index,
    )

    # E2: early stopping 有効時は train/val 分割
    early_stop = cfg.get("early_stopping_rounds", 0)
    if early_stop > 0:
        n_est = cfg.get("lgbm_n_estimators_max", 300)
    else:
        n_est = cfg.get("lgbm_n_estimators", 150)

    lgbm = _build_lgbm_regressor(cfg, n_estimators_override=n_est)

    # E1/E2: fit kwargs
    fit_kwargs: dict[str, Any] = {}
    if early_stop > 0:
        # early stopping 用に内部的に val split (直近20%)
        es_split = int(len(X_sc) * 0.8)
        if es_split > 30 and len(X_sc) - es_split > 10:
            fit_kwargs["eval_set"] = [(X_sc.iloc[es_split:], y_valid.iloc[es_split:])]
            fit_kwargs["callbacks"] = [
                lgb.early_stopping(stopping_rounds=early_stop, verbose=False),
                lgb.log_evaluation(period=0),
            ]

    # E1: warm-start — 前モデルの booster を init_model に使用
    # 注意: feature_cols が変更された場合 (pruning) は warm-start 不可
    if (
        prev_booster is not None
        and cfg.get("warm_start_enabled", True)
        and not pruned_features  # E3 pruning 時は feature 不一致
        and not redundancy_pruned  # C3 redundancy pruning 時も feature 不一致
        and prev_feature_cols == feature_cols  # feature_cols 完全一致が必要
    ):
        fit_kwargs["init_model"] = prev_booster
        logger.info("E1: Using prev booster as init_model for final training")
    elif prev_booster is not None and (pruned_features or redundancy_pruned):
        logger.info("E1: Warm-start skipped (feature set changed by E3/C3 pruning)")

    lgbm.fit(X_sc, y_valid, **fit_kwargs)

    # E2: 実際に使用された木の数を記録
    actual_n_trees = lgbm.booster_.num_trees() if hasattr(lgbm, "booster_") else n_est
    result["actual_n_trees"] = actual_n_trees

    # Pipeline を再構成 (SkipGate.evaluate が pipeline.predict を使うため)
    pipeline = Pipeline([
        ("imputer", imputer),
        ("scaler", scaler),
        ("model", lgbm),
    ])

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
        # E1-E4: 効率化メタデータ
        "warm_start_used": "init_model" in fit_kwargs,
        "early_stopping_used": "eval_set" in fit_kwargs,
        "actual_n_trees": actual_n_trees,
        "pruned_features": pruned_features,
        "wf_dead_features": wf_dead_features,  # 131# B: 次回 consecutive-dead 参照用
        "enriched_cache_used": cfg.get("enriched_cache_enabled", True),
        # 131# C1-C3: ztb asset 統合メタデータ
        "wf_multi_window": wf_result.get("n_windows", 1) if cfg.get("quality_gate_enabled") else 0,
        "redundancy_pruned_features": redundancy_pruned,
        "statistical_gate": result.get("statistical_gate", {}),
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
        # 131# A.1 #1 fix: with_suffix は最終 suffix のみ置換。
        # 旧コードは ".pkl.tmp.sha256" で二重 .pkl が発生していた。
        tmp_hash = tmp_path.with_suffix(tmp_path.suffix + ".sha256")
        real_hash = model_path.with_suffix(model_path.suffix + ".sha256")
        if tmp_hash.exists():
            os.replace(str(tmp_hash), str(real_hash))
            logger.info(f"SHA256 hash atomically moved to {real_hash}")
        else:
            # hash ファイルが見つからない場合は再計算して直接書き込み
            import hashlib
            model_data = model_path.read_bytes()
            digest = hashlib.sha256(model_data).hexdigest()
            real_hash.write_text(digest)
            logger.warning(
                f"SHA256 hash file not found at {tmp_hash}, "
                f"regenerated from deployed model (sha256={digest[:12]}...)"
            )
        logger.info(f"Model atomically deployed to {model_path}")
    except Exception as e:
        # tmp 残留防止
        tmp_hash_cleanup = tmp_path.with_suffix(tmp_path.suffix + ".sha256")
        for p in [tmp_path, tmp_hash_cleanup]:
            if p.exists():
                p.unlink()
        return {**result, "status": "error", "reason": f"atomic_write_failed: {e}"}

    result["status"] = "deployed"
    result["model_path"] = str(model_path)

    # 131# B: Post-deploy 自己検証 — SkipGate.load() で hash 一致を確認
    # A.2 逆行リスク対策: deployed (ファイル保存) と activated (hot-reload) の区別
    try:
        verify_gate = SkipGate.load(model_path)
        verify_n = verify_gate.metadata.get("n_samples", 0)
        del verify_gate
        if verify_n == len(X_valid):
            result["status"] = "deployed_verified"
            logger.info(
                f"Post-deploy verification PASSED: "
                f"SkipGate.load() succeeded, n_samples={verify_n}"
            )
        else:
            logger.warning(
                f"Post-deploy verification WARNING: "
                f"n_samples mismatch (expected={len(X_valid)}, got={verify_n})"
            )
    except Exception as e:
        logger.error(
            f"Post-deploy verification FAILED: {e}. "
            f"Model file may be corrupt — hot-reload will also fail."
        )
        result["deploy_verify_error"] = str(e)

    logger.info(
        f"Retrain complete: {len(X_valid)} samples → {model_path} "
        f"(+{new_samples} new, score={wf_results_meta.get('profit_score', 'N/A')}, "
        f"status={result['status']})"
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
        # Y3: relative quality gate を緩和 (既存モデルと target が異なる可能性)
        # absolute_min_score チェックは維持 (-0.10 以下は棄却)
        # 131# A.1 #4: 加えて pnl_improvement >= 0 のハード制約を追加
        cfg["min_score_improvement"] = -999.0
        cfg["all_runs_require_positive_pnl"] = True  # 131# A.1 #4
        logger.info(
            "Y3: --all-runs enabled → latest_run_only=False, "
            "exclude_missing_run_id=False, min_new_samples=0, "
            "relative quality gate bypassed (absolute_min + positive pnl retained)"
        )

    if args.once:
        logger.info("=== One-shot retrain ===")
        result = retrain_model(cfg)
        logger.info(f"Result: {json.dumps(result, indent=2, default=str)}")
    else:
        run_scheduler(cfg, config_path=Path(args.config))


if __name__ == "__main__":
    main()
