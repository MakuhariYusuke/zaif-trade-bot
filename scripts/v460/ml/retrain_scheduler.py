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
import signal
import sys
import threading
import time
import types
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
from scripts.v460.ml.skip_eval_utils import compute_skip_slice_metrics
from ztb.ml.skip_gate import (
    SkipGate,
    SkipGateConfig,
    _BASE_FEATURE_COLS,
    _OB_FEATURE_COLS,
    get_gate_feature_cols,
)
from ztb.io.jsonl import append_jsonl
from ztb.ml.artifact_paths import atomic_pickle_tmp_path, hash_sidecar_path
from ztb.ml.metadata_utils import current_iso_timestamp
from ztb.utils.safety import ensure_dict, safe_to_bool, safe_to_float, safe_to_int
from ztb.utils.types import ConfigMap

# 127# X1: module-level FileHandler を廃止。main() 内で初期化する。
logger = logging.getLogger(__name__)

FoldPnlSamples = tuple[list[float], list[float]]

# 161# SIGTERM graceful shutdown
_shutdown_event = threading.Event()


def _resolve_phase_thresholds(cfg: ConfigMap, sample_count: int) -> tuple[str, int, int]:
    """130# Bootstrap 2段しきい値を一箇所で解決する.

    Returns:
        (phase, min_total, min_new)
    """
    bootstrap_threshold = safe_to_int(cfg.get("bootstrap_threshold", 100), 100)
    is_bootstrap = sample_count < bootstrap_threshold
    if is_bootstrap:
        return (
            "bootstrap",
            safe_to_int(cfg.get("bootstrap_min_total_samples", 30), 30),
            safe_to_int(cfg.get("bootstrap_min_new_samples", 10), 10),
        )
    return (
        "stable",
        safe_to_int(cfg.get("min_total_samples", 100), 100),
        safe_to_int(cfg.get("min_new_samples", 30), 30),
    )


def _install_signal_handlers() -> None:
    """SIGTERM/SIGINT で graceful 停止するためのハンドラを設定."""
    def _handler(signum: int, _frame: object) -> None:
        name = signal.Signals(signum).name
        logger.warning(f"[161#] Received {name} — scheduling graceful shutdown")
        _shutdown_event.set()

    signal.signal(signal.SIGTERM, _handler)
    signal.signal(signal.SIGINT, _handler)


def _safe_import_ztb_module(dotted_path: str) -> types.ModuleType:
    """ztb サブモジュールを circular import を回避して読み込む.

    ztb.analysis/__init__.py → ztb.trading 間の循環参照があるため、
    通常の import ではサブモジュール (redundancy, gate_checks, splitter) が
    ロード不可になる場合がある。importlib.util で直接ロードすることで回避する。
    """
    import importlib.util

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
            pkg = types.ModuleType(pkg_name)
            pkg.__path__ = [str(pkg_dir)]
            pkg.__package__ = pkg_name
            sys.modules[pkg_name] = pkg
            # 既にロード済みの子を親に紐付け
            for child_name, child_mod in list(sys.modules.items()):
                if child_name.startswith(pkg_name + "."):
                    short = child_name[len(pkg_name) + 1:].split(".")[0]
                    setattr(pkg, short, child_mod)

    spec = importlib.util.spec_from_file_location(dotted_path, str(file_path))
    if spec is None or spec.loader is None:
        raise ImportError(f"Failed to build import spec for {dotted_path}")
    mod = importlib.util.module_from_spec(spec)
    mod.__package__ = ".".join(parts[:-1])
    sys.modules[dotted_path] = mod
    spec.loader.exec_module(mod)
    # 親パッケージにアトリビュート設定
    if len(parts) > 1:
        parent_name = ".".join(parts[:-1])
        if parent_name in sys.modules:
            setattr(sys.modules[parent_name], parts[-1], mod)
    return mod

# デフォルト設定 (127# C1: model_path/mode/use_ob は skip_gate: から継承)
_DEFAULT_CONFIG: ConfigMap = {
    "interval_sec": 3600,           # 再学習間隔 (秒) — 1時間
    "min_new_samples": 30,          # 再学習に必要な最小新規サンプル数
    "min_total_samples": 100,       # 再学習に必要な最小合計サンプル数
    # 130# Bootstrap Phase: run切替直後のスタベーション回避
    "bootstrap_min_total_samples": 30,   # Bootstrap 段階での最小合計
    "bootstrap_min_new_samples": 10,     # Bootstrap 段階での最小新規
    "bootstrap_threshold": 100,          # total < この値なら Bootstrap 段階と判定
    "target": "pnl120",            # 127# M2: "pnl120" or "pnl30"
    "fill_records_max_files": None, # 最新 N ファイルのみ読み込む上限 (None=全件)
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
    # 465# D1/D2: モデル退化ガード (定数出力モデルの deploy 阻止)
    "min_deploy_trees": 3,          # 訓練後の木数がこれ未満なら棄却
    "min_pred_std": 0.01,           # 予測の標準偏差がこれ未満なら棄却 (定数出力検出)
    # C1: WF multi-window evaluation (131# WalkForwardSplitter 統合)
    "wf_multi_window_enabled": True,
    "wf_initial_train_pct": 0.50,
    "wf_val_pct": 0.10,
    "wf_test_pct": 0.15,
    "wf_step_pct": 0.20,
    "wf_max_windows": None,          # 305# 最大評価window数 (None/<=0=全件)
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
    # 130# D.1 Q3: trades I/O fallback
    "trades_fallback_recent_days": 1,
    # 137# §9 #4: retrain trigger 設定
    "trigger_check_trades_health": True,
    "trigger_trades_lookback_days": 3,
    "trigger_trades_stale_threshold_hours": 36.0,
    "trigger_trades_max_missing_days": 1,  # 158# deadlock gap 対応
    "trigger_backoff_multiplier": 2.0,
    "trigger_backoff_max_interval_sec": 14400,
    "trigger_check_feature_freshness": False,
    "trigger_feature_trades_stale_hours": 6.0,
    "trigger_feature_ob_stale_hours": 6.0,
    # SkipGate config
    "adaptive_threshold": True,
    "target_skip_rate_buy": 0.15,
    "target_skip_rate_sell": 0.20,
    # 141# P1-01/02: side 分離モデル + target 二層化
    "side_specific_enabled": False,       # side 別モデル追加学習を有効化
    "target_buy": "pnl30",               # buy 側ターゲット (primary)
    "target_sell": "pnl30",              # sell 側ターゲット (YAML で pnl120 に上書き)
    "side_min_samples": 50,              # side 別学習の最小サンプル数
    # 189# multi-horizon: alt (副 horizon) モデル学習
    "alt_horizon_enabled": False,         # alt horizon モデル追加学習を有効化
    "target_buy_alt": "pnl120",           # buy 側 alt ターゲット (長期)
    "target_sell_alt": "pnl30",           # sell 側 alt ターゲット (短期)
    # 141# P1-12: オンラインパフォーマンスモニター
    "online_monitor_enabled": True,       # P1-12 online monitor を有効化
    "online_monitor_window": 100,         # 評価ウィンドウ (直近 N fill)
    "online_monitor_pnl_column": "post_fill_30s_pnl",
    "online_monitor_degraded_threshold_bps": -0.3,
    # 145# R-2a: レジーム重み付き再学習
    # レジーム別 sample_weight で現レジームに近いサンプルを upweight
    "regime_weighting_enabled": False,     # 安全デフォルト: 無効
    "regime_sample_weights": {             # レジーム → weight マッピング
        "high_vol": 1.0,
        "trending": 1.0,
        "trending_up": 1.0,      # 176# 横展開
        "trending_down": 1.0,    # 176# 横展開
        "ranging": 1.0,
        "unknown": 1.0,
    },
    "regime_current_boost": 1.5,          # 直近レジームに一致するサンプルの追加ブースト倍率
    "regime_current_lookback": 10,        # 直近 N 件から「現在レジーム」を多数決で推定
    "regime_weight_floor": 0.1,           # weight の最低値 (0近接回避)
}


def load_retrain_config(config_path: Path | None = None) -> ConfigMap:
    """YAML retrain + skip_gate セクションから設定を読み込む.

    127# C1: model_path / mode / use_ob_features は skip_gate: から継承。
    retrain: セクションで重複定義せずに single source of truth を保証。
    """
    cfg = dict(_DEFAULT_CONFIG)
    yaml_path = config_path or Path("configs/v460/fill_test.yaml")
    if yaml_path.exists():
        try:
            from scripts.v460.lib.config_loader import load_fill_test_config

            yaml_data = ensure_dict(load_fill_test_config(yaml_path))

            # 127# C1: skip_gate セクションから model_path / mode / use_ob_features を継承
            sg_cfg = ensure_dict(yaml_data.get("skip_gate"))
            cfg["model_path"] = sg_cfg.get("model_path", "models/v460/skip_gate_lgbm_pnl120.pkl")
            cfg["mode"] = sg_cfg.get("mode", "pnl")
            cfg["use_ob_features"] = sg_cfg.get("use_ob_features", True)
            # 141# P1-01: side 別モデルパスを skip_gate セクションから継承
            cfg["model_path_buy"] = sg_cfg.get("model_path_buy", "")
            cfg["model_path_sell"] = sg_cfg.get("model_path_sell", "")
            # 189# multi-horizon: alt モデルパスを skip_gate セクションから継承
            cfg["model_path_buy_long"] = sg_cfg.get("model_path_buy_long", "")
            cfg["model_path_sell_short"] = sg_cfg.get("model_path_sell_short", "")
            # results_dir はトップレベルから継承
            cfg["results_dir"] = yaml_data.get("results_dir", "results/v460/fill_test")

            retrain_cfg = ensure_dict(yaml_data.get("retrain"))
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


def _validate_config(cfg: ConfigMap) -> None:
    """127# C1: 設定の整合性を検証 (fail-fast)."""
    mode = str(cfg.get("mode", "pnl"))
    target = str(cfg.get("target", "pnl120"))

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
    model_path = str(cfg.get("model_path", ""))
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
    # 148# P1: side 別 target/model_path ミスマッチ警告
    _validate_side_target_path_mismatch(cfg, "buy")
    _validate_side_target_path_mismatch(cfg, "sell")


def _validate_side_target_path_mismatch(cfg: ConfigMap, side: str) -> None:
    """148# P1: side 別 target と model_path のミスマッチ警告."""
    target_key = f"target_{side}"
    path_key = f"model_path_{side}"
    target = str(cfg.get(target_key, ""))
    model_path = str(cfg.get(path_key, ""))
    if not target or not model_path:
        return  # どちらかが未設定なら skip_gate は統一モデルにフォールバック
    path_has_pnl120 = "pnl120" in model_path
    path_has_pnl30 = "pnl30" in model_path
    if target == "pnl30" and path_has_pnl120 and not path_has_pnl30:
        logger.warning(
            f"148# P1: {side} target='{target}' but model_path contains 'pnl120': "
            f"{model_path}. side ごとの target/path ミスマッチに注意。"
        )
    elif target == "pnl120" and path_has_pnl30 and not path_has_pnl120:
        logger.warning(
            f"148# P1: {side} target='{target}' but model_path contains 'pnl30': "
            f"{model_path}. side ごとの target/path ミスマッチに注意。"
        )


def _append_jsonl_record(path: Path, payload: object) -> None:
    """JSONL に 1 レコードを追記."""
    append_jsonl(path, [payload], ensure_ascii=False, default=str)


def _extract_numeric_column(df: pd.DataFrame, index: pd.Index, column: str) -> np.ndarray:
    """指定列を float 配列として抽出 (欠損は NaN)."""
    if column not in df.columns:
        return np.full(len(index), np.nan, dtype=np.float64)
    values = pd.to_numeric(df.loc[index, column], errors="coerce")
    return values.to_numpy(dtype=np.float64, copy=False)


def _clean_float_values(values: np.ndarray) -> list[float]:
    if values.size == 0:
        return []
    finite = values[np.isfinite(values)]
    return finite.astype(np.float64, copy=False).tolist()


def _compute_skip_metrics(
    preds: np.ndarray,
    pnl30: np.ndarray,
    pnl120: np.ndarray,
    skip_percentile: float,
) -> tuple[float, float, float, FoldPnlSamples, FoldPnlSamples]:
    """skip 評価スコアと統計ゲート入力用 fold データを返す."""
    stats = compute_skip_slice_metrics(
        preds,
        pnl30,
        pnl120,
        skip_pct=skip_percentile,
        skip_low_scores=True,
    )

    imp_30 = stats.pnl30_improvement
    imp_120 = stats.pnl120_improvement
    score = imp_120 - max(0.0, -imp_30)

    fold_pnl30 = (_clean_float_values(pnl30[stats.keep_mask]), _clean_float_values(pnl30))
    fold_pnl120 = (_clean_float_values(pnl120[stats.keep_mask]), _clean_float_values(pnl120))
    return score, imp_30, imp_120, fold_pnl30, fold_pnl120


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
            payload_map = ensure_dict(payload)
            cached_obj = payload_map.get("data")
            stored_key = payload_map.get("cache_key")
            cached = cached_obj if isinstance(cached_obj, pd.DataFrame) else None
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
    """E4: enriched data を cache に保存 (cache_key 付き, アトミック書き込み)."""
    try:
        import pickle
        payload = {
            "data": enriched,
            "cache_key": cache_key,
            "n_records": len(enriched),
        }
        # 160# fix: アトミック書き込み (中断時の破損 cache 防止)
        tmp_path = atomic_pickle_tmp_path(cache_path)
        with open(tmp_path, "wb") as f:
            pickle.dump(payload, f)
        tmp_path.replace(cache_path)
        logger.info(f"E4: Saved enriched cache ({len(enriched)} records, key={cache_key}) to {cache_path}")
    except Exception as e:
        logger.warning(f"E4: Cache save failed: {e}")
        # tmp ファイル残留を防止
        try:
            tmp_path.unlink(missing_ok=True)  # type: ignore[possibly-undefined]
        except Exception:  # noqa: R-18 cleanup best-effort
            pass

def _resolve_early_stopping(cfg: ConfigMap) -> tuple[int, int]:
    """E2: early_stopping_rounds と n_estimators を一括解決 (DRY).

    Returns:
        (early_stopping_rounds, n_estimators)
    """
    early_stop = safe_to_int(cfg.get("early_stopping_rounds", 0), 0)
    if early_stop > 0:
        n_est = safe_to_int(cfg.get("lgbm_n_estimators_max", 300), 300)
    else:
        n_est = safe_to_int(cfg.get("lgbm_n_estimators", 150), 150)
    return early_stop, n_est


def _build_lgbm_regressor(
    cfg: ConfigMap,
    n_estimators_override: int | None = None,
) -> "lgb.LGBMRegressor":
    """共通 LGBMRegressor 構築 (DRY)."""
    import lightgbm as lgb

    return lgb.LGBMRegressor(
        n_estimators=n_estimators_override or safe_to_int(cfg.get("lgbm_n_estimators", 150), 150),
        max_depth=safe_to_int(cfg.get("lgbm_max_depth", 4), 4),
        learning_rate=safe_to_float(cfg.get("lgbm_learning_rate", 0.05), 0.05),
        num_leaves=safe_to_int(cfg.get("lgbm_num_leaves", 15), 15),
        min_child_samples=safe_to_int(cfg.get("lgbm_min_child_samples", 20), 20),
        # 133# Y1-Y6: YAML 外部化 (旧: ハードコード)
        subsample=safe_to_float(cfg.get("lgbm_subsample", 0.8), 0.8),
        colsample_bytree=safe_to_float(cfg.get("lgbm_colsample_bytree", 0.8), 0.8),
        reg_alpha=safe_to_float(cfg.get("lgbm_reg_alpha", 1.0), 1.0),
        reg_lambda=safe_to_float(cfg.get("lgbm_reg_lambda", 1.0), 1.0),
        random_state=safe_to_int(cfg.get("lgbm_random_state", 42), 42),
        verbose=-1,
        n_jobs=safe_to_int(cfg.get("lgbm_n_jobs", 1), 1),
    )


def _evaluate_wf(
    X: pd.DataFrame,
    y: pd.Series,
    enriched: pd.DataFrame,
    cfg: ConfigMap,
    prev_booster: object | None = None,
    sample_weight: np.ndarray | None = None,
) -> ConfigMap:
    """Walk-Forward OOS 評価ディスパッチ.

    131# C1: wf_multi_window_enabled=True なら WalkForwardSplitter で
    multi-window 評価を実行。データ不足時は single-window にフォールバック。
    145# R-2a: sample_weight 対応 (レジーム重み付き学習)。
    """
    if cfg.get("wf_multi_window_enabled", True):
        try:
            result = _evaluate_wf_multi(X, y, enriched, cfg, prev_booster, sample_weight)
            if result is not None:
                return result
        except Exception as e:
            logger.warning(f"C1: Multi-window WF failed ({e}), falling back to single")
    return _evaluate_wf_single(X, y, enriched, cfg, prev_booster, sample_weight)


def _evaluate_wf_multi(
    X: pd.DataFrame,
    y: pd.Series,
    enriched: pd.DataFrame,
    cfg: ConfigMap,
    prev_booster: object | None = None,
    sample_weight: np.ndarray | None = None,
) -> ConfigMap | None:
    """131# C1: Multi-window Walk-Forward 評価 (WalkForwardSplitter 統合).

    複数の WF ウィンドウで独立に train→predict し、per-window PnL を収集。
    g1_judgment / holm_bonferroni_gate 用の fold-level データを返す。
    145# R-2a: sample_weight 対応。

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
        initial_train_pct=safe_to_float(cfg.get("wf_initial_train_pct", 0.50), 0.50),
        val_pct=safe_to_float(cfg.get("wf_val_pct", 0.10), 0.10),
        test_pct=safe_to_float(cfg.get("wf_test_pct", 0.15), 0.15),
        step_pct=safe_to_float(cfg.get("wf_step_pct", 0.20), 0.20),
        embargo_days=safe_to_int(cfg.get("wf_embargo_rows", 0), 0),
    )

    try:
        windows = splitter.split(dummy_df)
    except ValueError as e:
        logger.info(f"C1: WalkForwardSplitter could not split (n={n}): {e}")
        return None

    min_train = safe_to_int(cfg.get("wf_min_window_train", 30), 30)
    min_test = safe_to_int(cfg.get("wf_min_window_test", 10), 10)

    # 有効ウィンドウのみ選択
    valid_windows = [
        w for w in windows
        if (w.train_end - w.train_start) >= min_train
        and (w.test_end - w.test_start) >= min_test
    ]
    max_windows = safe_to_int(cfg.get("wf_max_windows", 0), 0)
    if max_windows > 0 and len(valid_windows) > max_windows:
        valid_windows = valid_windows[:max_windows]
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

    early_stop, n_est = _resolve_early_stopping(cfg)
    skip_pct = safe_to_float(cfg.get("skip_percentile", 20), 20.0)
    pnl30_all = _extract_numeric_column(enriched, X.index, "post_fill_30s_pnl")
    pnl120_all = _extract_numeric_column(enriched, X.index, "post_fill_120s_pnl")

    for win in valid_windows:
        X_train = X.iloc[win.train_start:win.train_end]
        y_train = y.iloc[win.train_start:win.train_end]
        X_val = X.iloc[win.val_start:win.val_end]
        y_val = y.iloc[win.val_start:win.val_end]
        X_test = X.iloc[win.test_start:win.test_end]

        # 前処理
        imputer = SimpleImputer(strategy="median")
        scaler = StandardScaler()
        X_train_sc = scaler.fit_transform(imputer.fit_transform(X_train))
        X_test_sc = scaler.transform(imputer.transform(X_test))

        lgbm_model = _build_lgbm_regressor(cfg, n_estimators_override=n_est)

        fit_kwargs: dict[str, object] = {}
        if early_stop > 0 and len(X_val) >= 5:
            X_val_sc = scaler.transform(imputer.transform(X_val))
            fit_kwargs["eval_set"] = [(X_val_sc, y_val)]
            fit_kwargs["callbacks"] = [
                lgb.early_stopping(stopping_rounds=early_stop, verbose=False),
                lgb.log_evaluation(period=0),
            ]

        # 145# R-2a: レジーム重み付き学習 — train window 分の weight を切り出し
        if sample_weight is not None:
            fit_kwargs["sample_weight"] = sample_weight[win.train_start:win.train_end]

        lgbm_model.fit(X_train_sc, y_train, **fit_kwargs)
        preds_test = np.asarray(lgbm_model.predict(X_test_sc), dtype=np.float64)

        # OOS PnL 参照
        pnl30 = pnl30_all[win.test_start:win.test_end]
        pnl120 = pnl120_all[win.test_start:win.test_end]
        score, imp_30, imp_120, fold30, fold120 = _compute_skip_metrics(
            preds_test,
            pnl30,
            pnl120,
            skip_pct,
        )

        window_scores.append(score)
        window_imp30.append(imp_30)
        window_imp120.append(imp_120)
        fold_pnl30.append(fold30)
        fold_pnl120.append(fold120)

        # Feature importance 集計
        if hasattr(lgbm_model, "feature_importances_"):
            for col, imp in zip(X_train.columns, lgbm_model.feature_importances_):
                all_feat_importance[col] = all_feat_importance.get(col, 0) + int(imp)

        n_trees = lgbm_model.booster_.num_trees() if hasattr(lgbm_model, "booster_") else n_est
        total_n_trees += n_trees
        total_n_test += len(X_test)
        total_n_train += len(X_train)

        # 466# メモリリーク防止: per-window オブジェクトを即時解放
        del lgbm_model, imputer, scaler, X_train_sc, X_test_sc
        del X_train, y_train, X_val, y_val, X_test

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
    cfg: ConfigMap,
    prev_booster: object | None = None,
    sample_weight: np.ndarray | None = None,
) -> ConfigMap:
    """Walk-Forward OOS 評価で品質スコアを算出.

    直近 test_ratio をテストセットとし、残りで訓練→テスト予測の skip simulation。
    E1: prev_booster があれば warm-start で学習。
    E2: early_stopping_rounds で過学習を自動防止。
    145# R-2a: sample_weight 対応。
    158# P2-1: early stopping に val セットを使用 (テストリーク修正)。

    Returns:
        {"score": float, "pnl30_improvement": float, "pnl120_improvement": float, ...}
    """
    from sklearn.impute import SimpleImputer
    from sklearn.preprocessing import StandardScaler

    try:
        import lightgbm as lgb
    except ImportError:
        raise RuntimeError("LightGBM required")

    test_ratio = safe_to_float(cfg.get("wf_test_ratio", 0.2), 0.2)
    val_ratio = safe_to_float(cfg.get("wf_val_ratio_single", 0.1), 0.1)
    embargo_rows = safe_to_int(cfg.get("wf_embargo_rows", 0), 0)
    n = len(X)
    # 158# P2-1: train / embargo / val / test に4分割 (リーク修正)
    test_size = max(1, int(n * test_ratio))
    val_size = max(1, int(n * val_ratio))
    train_size = n - test_size - val_size - embargo_rows
    if train_size < 50 or test_size < 20:
        logger.warning(
            f"Insufficient data for WF eval: train={train_size}, "
            f"val={val_size}, test={test_size}, embargo={embargo_rows}"
        )
        return {"score": 0.0, "pnl30_improvement": 0.0, "pnl120_improvement": 0.0}

    train_end = train_size
    val_start = train_end + embargo_rows
    val_end = val_start + val_size
    test_start_idx = val_end
    # test_end = n

    X_train = X.iloc[:train_end]
    y_train = y.iloc[:train_end]
    X_val = X.iloc[val_start:val_end]
    y_val = y.iloc[val_start:val_end]
    X_test = X.iloc[test_start_idx:]
    y_test = y.iloc[test_start_idx:]  # noqa: F841

    # E2: early stopping 有効時は上限を引き上げ
    early_stop, n_est = _resolve_early_stopping(cfg)

    lgbm_model = _build_lgbm_regressor(cfg, n_estimators_override=n_est)

    # E2: early stopping 用の前処理 (Pipeline 内で fit するため手動分離)
    imputer = SimpleImputer(strategy="median")
    scaler = StandardScaler()
    X_train_sc = scaler.fit_transform(imputer.fit_transform(X_train))
    X_test_sc = scaler.transform(imputer.transform(X_test))

    # E1: warm-start — 前モデルの booster を init_model に使用
    fit_kwargs: dict[str, object] = {}
    # 158# P2-1: early stopping は val セットで行う (test リーク防止)
    if early_stop > 0 and len(X_val) >= 5:
        X_val_sc = scaler.transform(imputer.transform(X_val))
        fit_kwargs["eval_set"] = [(X_val_sc, y_val)]
        # LightGBM 4.x: callbacks で early stopping
        fit_kwargs["callbacks"] = [
            lgb.early_stopping(stopping_rounds=early_stop, verbose=False),
            lgb.log_evaluation(period=0),  # suppress iteration log
        ]
    if prev_booster is not None and safe_to_bool(cfg.get("warm_start_enabled", True), True):
        fit_kwargs["init_model"] = prev_booster
        logger.info("E1: Using prev booster as init_model for WF eval")

    # 145# R-2a: レジーム重み付き学習 — train 分の weight を切り出し
    if sample_weight is not None:
        fit_kwargs["sample_weight"] = sample_weight[:train_end]

    lgbm_model.fit(X_train_sc, y_train, **fit_kwargs)
    preds_test = np.asarray(lgbm_model.predict(X_test_sc), dtype=np.float64)

    # OOS PnL 参照 (pnl30/pnl120)
    test_idx = X_test.index
    pnl30 = _extract_numeric_column(enriched, test_idx, "post_fill_30s_pnl")
    pnl120 = _extract_numeric_column(enriched, test_idx, "post_fill_120s_pnl")
    score, imp_30, imp_120, fold30, fold120 = _compute_skip_metrics(
        preds_test,
        pnl30,
        pnl120,
        safe_to_float(cfg.get("skip_percentile", 20), 20.0),
    )

    # E2: 実際に使用された木の数を記録
    actual_n_trees = lgbm_model.booster_.num_trees() if hasattr(lgbm_model, "booster_") else n_est

    # E3: feature importance を記録 (pruning 判定に使用)
    feat_importance: dict[str, int] = {}
    if hasattr(lgbm_model, "feature_importances_"):
        for col, imp in zip(X_train.columns, lgbm_model.feature_importances_):
            feat_importance[col] = int(imp)

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
        "fold_pnl30": [fold30],
        "fold_pnl120": [fold120],
    }


def _apply_statistical_gate(
    wf_result: ConfigMap,
    cfg: ConfigMap,
) -> ConfigMap:
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

    alpha = safe_to_float(cfg.get("statistical_gate_alpha", 0.05), 0.05)
    min_effect = safe_to_float(cfg.get("statistical_gate_min_effect", 0.147), 0.147)
    min_test = safe_to_int(cfg.get("statistical_gate_min_test_samples", 40), 40)

    fold_pnl30 = wf_result.get("fold_pnl30", [])
    fold_pnl120 = wf_result.get("fold_pnl120", [])
    n_windows = safe_to_int(wf_result.get("n_windows", 0), 0)

    # 合計テストサンプル数チェック
    total_test = sum(len(b) for _, b in fold_pnl30) if fold_pnl30 else 0
    if total_test < min_test:
        return {
            "applied": False,
            "reason": f"insufficient_test_samples ({total_test} < {min_test})",
        }

    # fold-level data 構築
    def _normalize_fold_pairs(raw_folds: object) -> list[FoldPnlSamples]:
        normalized: list[FoldPnlSamples] = []
        if not isinstance(raw_folds, list):
            return normalized
        for pair in raw_folds:
            if not isinstance(pair, (tuple, list)) or len(pair) != 2:
                continue
            kept_raw, all_raw = pair
            if not isinstance(kept_raw, list) or not isinstance(all_raw, list):
                continue
            kept_vals = [safe_to_float(v, float("nan")) for v in kept_raw]
            all_vals = [safe_to_float(v, float("nan")) for v in all_raw]
            normalized.append((_clean_float_values(np.asarray(kept_vals)), _clean_float_values(np.asarray(all_vals))))
        return normalized

    fold_results: dict[str, list[FoldPnlSamples]] = {}
    norm_30 = _normalize_fold_pairs(fold_pnl30)
    norm_120 = _normalize_fold_pairs(fold_pnl120)
    if norm_30:
        fold_results["pnl30"] = norm_30
    if norm_120:
        fold_results["pnl120"] = norm_120

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
        hb_input: dict[str, FoldPnlSamples] = {}
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


def _compute_regime_sample_weights(
    enriched: pd.DataFrame,
    valid_index: pd.Index,
    cfg: ConfigMap,
) -> tuple[np.ndarray, ConfigMap]:
    """145# R-2a: レジーム別 sample_weight を算出.

    各サンプルの regime 列を参照し、cfg の weight マッピングで重み付け。
    直近 N 件から多数決で「現在レジーム」を推定し、追加ブーストを適用。

    Args:
        enriched: enriched DataFrame (regime 列を含む).
        valid_index: X_valid / y_valid のインデックス (enriched の部分集合).
        cfg: retrain 設定 dict.

    Returns:
        (sample_weight 配列, メタデータ dict)
    """
    regime_weights_raw = ensure_dict(cfg.get("regime_sample_weights"))
    regime_weights_map = {str(k): safe_to_float(v, 1.0) for k, v in regime_weights_raw.items()}
    current_boost = safe_to_float(cfg.get("regime_current_boost", 1.5), 1.5)
    # §11-#2: lookback=0 ガード
    lookback = max(1, safe_to_int(cfg.get("regime_current_lookback", 10), 10))
    weight_floor = safe_to_float(cfg.get("regime_weight_floor", 0.1), 0.1)

    # §11-#1: empty valid_index ガード — 空配列の np.min/np.max で ValueError を回避
    if len(valid_index) == 0:
        logger.info("R-2a: empty valid_index, returning uniform weights")
        return np.ones(0, dtype=np.float64), {
            "regime_weighting": "uniform",
            "reason": "empty_valid_index",
        }

    # regime 列がなければ均一重み
    if "regime" not in enriched.columns:
        logger.info("R-2a: regime column not found, using uniform weights")
        return np.ones(len(valid_index), dtype=np.float64), {
            "regime_weighting": "uniform",
            "reason": "no_regime_column",
        }

    regimes = enriched.loc[valid_index, "regime"].fillna("unknown")

    # Step 1: 基本重み (config マッピング)
    weights = regimes.map(
        lambda r: max(regime_weights_map.get(r, 1.0), weight_floor)
    ).astype(np.float64).values

    # Step 2: 直近 N 件から「現在レジーム」を推定 (多数決)
    current_regime = "unknown"
    if lookback > 0 and len(regimes) >= lookback:
        recent = regimes.iloc[-lookback:]
        regime_counts = recent.value_counts()
        current_regime = regime_counts.index[0]
        # 現在レジーム一致サンプルにブースト適用
        boost_mask = (regimes == current_regime).values
        weights[boost_mask] *= current_boost
        logger.info(
            f"R-2a: current_regime={current_regime} "
            f"(from last {lookback}, confidence={regime_counts.iloc[0]}/{lookback}), "
            f"boost={current_boost}x applied to {int(boost_mask.sum())} samples"
        )
    elif lookback > 0:
        logger.info(
            f"R-2a: Insufficient data for current regime detection "
            f"({len(regimes)} < {lookback}), boost skipped"
        )

    # Step 3: 正規化 (mean=1.0 に保つことで学習率スケールへの影響を抑制)
    mean_w = float(np.mean(weights))
    if mean_w > 0:
        weights = weights / mean_w

    # weight_floor 再適用 (正規化後)
    weights = np.maximum(weights, weight_floor)

    # 統計情報収集
    regime_dist = regimes.value_counts().to_dict()
    weight_stats = {
        "regime_weighting": "applied",
        "current_regime": current_regime,
        "regime_distribution": {str(k): int(v) for k, v in regime_dist.items()},
        "weight_mean": round(float(np.mean(weights)), 4),
        "weight_std": round(float(np.std(weights)), 4),
        "weight_min": round(float(np.min(weights)), 4),
        "weight_max": round(float(np.max(weights)), 4),
        "config_weights": regime_weights_map,
        "current_boost": current_boost,
        "lookback": lookback,
    }

    return weights, weight_stats


def retrain_model(cfg: ConfigMap) -> ConfigMap:
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
    model_path = Path(str(cfg.get("model_path", "")))
    results_dir = Path(str(cfg.get("results_dir", "results/v460/fill_test")))
    target = str(cfg.get("target", "pnl120"))
    use_ob = safe_to_bool(cfg.get("use_ob_features", True), True)

    result: ConfigMap = {
        "timestamp": current_iso_timestamp(utc=True),
        "status": "pending",
    }

    # 127# H2: run_id フィルタリング
    run_id_filter_raw = cfg.get("run_id_filter")
    run_id_filter: str | list[str] | None = (
        run_id_filter_raw if isinstance(run_id_filter_raw, (str, list)) else None
    )
    exclude_missing = safe_to_bool(cfg.get("exclude_missing_run_id", True), True)
    latest_run_only = safe_to_bool(cfg.get("latest_run_only", True), True)
    max_files_raw = cfg.get("fill_records_max_files")
    fill_records_max_files: int | None = None
    if max_files_raw is not None:
        parsed_max_files = safe_to_int(max_files_raw, 0)
        if parsed_max_files > 0:
            fill_records_max_files = parsed_max_files

    # Step 1: データロード
    try:
        records = load_fill_records(
            results_dir,
            exclude_missing_run_id=exclude_missing,
            max_files=fill_records_max_files,
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

    # 141# P1-01: side 別モデル — 指定 side のみにフィルタリング
    side_filter_raw = cfg.get("side_filter")
    side_filter = str(side_filter_raw) if isinstance(side_filter_raw, str) and side_filter_raw else None
    if side_filter and "side" in records.columns:
        n_before_side = len(records)
        records = records[records["side"] == side_filter].reset_index(drop=True)
        logger.info(
            f"141# Side filter: {side_filter} → {len(records)}/{n_before_side} records"
        )
        side_min_samples = safe_to_int(cfg.get("side_min_samples", 50), 50)
        if len(records) < side_min_samples:
            return {
                **result,
                "status": "skipped",
                "reason": f"Insufficient {side_filter} samples: {len(records)} < {side_min_samples}",
                "side_filter": side_filter,
            }
        result["side_filter"] = side_filter

    # raw records 数だけで不可能なケースは enrichment 前に早期スキップ
    # X_valid <= len(records) なので、この下限を下回る場合は以降の重い処理は不要。
    _bootstrap_min_total = safe_to_int(cfg.get("bootstrap_min_total_samples", 30), 30)
    _stable_min_total = safe_to_int(cfg.get("min_total_samples", 100), 100)
    _min_total_lower_bound = min(_bootstrap_min_total, _stable_min_total)
    if len(records) < _min_total_lower_bound:
        phase, min_total, _ = _resolve_phase_thresholds(cfg, len(records))
        return {
            **result,
            "phase": phase,
            "status": "skipped",
            "reason": f"insufficient raw samples: {len(records)} < {min_total} ({phase})",
        }

    enriched = None
    # E4: enriched data cache — I/O コスト削減
    # 131# A.1 #6: cache_key = target + feature_cols + run_ids で stale cache 防止
    if safe_to_bool(cfg.get("enriched_cache_enabled", True), True):
        cache_path = _get_enriched_cache_path(results_dir)
        feature_cols_str = ",".join(sorted(get_gate_feature_cols(use_ob=use_ob)))
        run_ids_str = ""
        if "run_id" in records.columns:
            run_ids_str = ",".join(sorted(records["run_id"].dropna().unique().astype(str)))
        import hashlib as _hl
        cache_key = _hl.md5(
            f"{target}|{feature_cols_str}|{run_ids_str}|{side_filter or 'all'}".encode()
        ).hexdigest()[:16]
        enriched = _load_enriched_cache(cache_path, len(records), cache_key=cache_key)

    if enriched is None:
        enriched = enrich_fill_records(
            records,
            trades_fallback_recent_days=safe_to_int(cfg.get("trades_fallback_recent_days", 1), 1),
        )
        if safe_to_bool(cfg.get("enriched_cache_enabled", True), True):
            _save_enriched_cache(cache_path, enriched, cache_key=cache_key)
    del records

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
    del pnl_data  # 466# メモリリーク防止: 以降未使用

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
    del X_base  # 466# メモリリーク防止: 以降未使用
    valid_mask = y_target.notna()
    X_valid = X_full.loc[valid_mask]
    y_valid = y_target.loc[valid_mask]

    result["total_samples"] = int(len(X_full))
    result["valid_target_samples"] = int(len(X_valid))

    # H1: ドロップ統計記録
    n_original_filled = int(pnl_mask.sum())
    result["filled_records"] = n_original_filled
    result["dropped_by_feature_build"] = n_original_filled - int(len(X_full))
    del X_full  # 466# メモリリーク防止: 以降未使用

    # 145# R-2b: 現在レジーム検出 (R-2a weighting の有無に関わらず常に実行)
    # 直近 N 件の多数決で推定し、result に記録 → run_scheduler で trigger に伝搬
    # §11-#2: lookback=0 ガード (max(1,...) で IndexError 回避)
    _regime_lookback = max(1, safe_to_int(cfg.get("regime_current_lookback", 10), 10))
    if "regime" in enriched.columns and len(X_valid) >= _regime_lookback:
        _recent_regimes = enriched.loc[X_valid.index, "regime"].fillna("unknown").iloc[-_regime_lookback:]
        _regime_counts = _recent_regimes.value_counts()
        if len(_regime_counts) > 0:
            result["current_regime"] = str(_regime_counts.index[0])
        else:
            result["current_regime"] = "unknown"
    elif "regime" in enriched.columns and len(X_valid) > 0:
        _all_regimes = enriched.loc[X_valid.index, "regime"].fillna("unknown")
        _regime_counts = _all_regimes.value_counts()
        if len(_regime_counts) > 0:
            result["current_regime"] = str(_regime_counts.index[0])
        else:
            result["current_regime"] = "unknown"
    else:
        result["current_regime"] = "unknown"

    # 145# R-2a: レジーム重み付きサンプルウェイト計算
    regime_sample_weight: np.ndarray | None = None
    if safe_to_bool(cfg.get("regime_weighting_enabled", False), False):
        regime_sample_weight, weight_stats = _compute_regime_sample_weights(
            enriched, X_valid.index, cfg,
        )
        result["regime_weighting"] = weight_stats
        logger.info(
            f"R-2a: Regime weighting enabled — "
            f"mean={weight_stats['weight_mean']}, "
            f"std={weight_stats['weight_std']}, "
            f"current={weight_stats.get('current_regime', 'N/A')}"
        )
    else:
        result["regime_weighting"] = {"regime_weighting": "disabled"}

    # Step 2: 最小サンプルチェック (130# Bootstrap 2段化)
    phase, min_total, min_new = _resolve_phase_thresholds(cfg, len(X_valid))
    result["phase"] = phase
    if phase == "bootstrap":
        logger.info(
            f"130# Bootstrap phase: {len(X_valid)} < "
            f"{safe_to_int(cfg.get('bootstrap_threshold', 100), 100)}, using min_total={min_total}"
        )
    if len(X_valid) < min_total:
        return {
            **result,
            "status": "skipped",
            "reason": f"insufficient samples: {len(X_valid)} < {min_total} ({result['phase']})",
        }

    # 127# X2: 前モデルを一度だけロード (n_samples + WF score + E1 booster を取得)
    prev_n_samples = 0
    prev_score = 0.0
    prev_source_run_id = ""  # 140# §8.1-#3: 前モデルの source_run_id
    prev_gate_loaded = False
    prev_booster = None  # E1: warm-start 用
    prev_feature_cols: list[str] | None = None  # E3: pruning 参照用
    if model_path.exists():
        try:
            prev_gate = SkipGate.load(model_path)
            prev_n_samples = prev_gate.metadata.get("n_samples", 0)
            prev_source_run_id = prev_gate.metadata.get("source_run_id", "")
            prev_wf = prev_gate.metadata.get("wf_results", {})
            prev_score = prev_wf.get("profit_score", 0.0)
            prev_feature_cols = prev_gate.metadata.get("feature_cols")
            # 131# B: 連続 dead pruning 用 — 前回 WF dead features を取得
            result["_prev_wf_dead_features"] = prev_gate.metadata.get(
                "wf_dead_features", [],
            )
            # E1: LightGBM booster を抽出 (warm-start に使用)
            if safe_to_bool(cfg.get("warm_start_enabled", True), True):
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
    # 140# §8.1-#3: run_id 直接比較で run 切替を検出 (139# のヒューリスティックを補強)
    # latest_run_only=true で現 run の run_id と前モデルの source_run_id を比較。
    # 不一致 or prev_source_run_id が空 (旧モデル) かつ raw < 0 なら run 切替。
    current_run_id = result.get("run_id", "")
    raw_new_samples = len(X_valid) - prev_n_samples
    run_switched = False
    if prev_source_run_id and current_run_id and prev_source_run_id != current_run_id:
        # 明示的な run_id 不一致 → run 切替確定
        run_switched = True
        new_samples = len(X_valid)
        logger.info(
            f"140# Run switch by run_id: prev={prev_source_run_id} != "
            f"current={current_run_id}. Treating all {new_samples} as new. "
            f"(raw_delta={raw_new_samples})"
        )
    elif raw_new_samples < 0:
        # 139# フォールバック: run_id がない旧モデルでも負値なら run 切替と推定
        run_switched = True
        new_samples = len(X_valid)
        logger.info(
            f"139# Run switch detected (fallback): prev_n_samples={prev_n_samples} > "
            f"current={len(X_valid)}. Treating all {new_samples} as new. "
            f"(raw_delta={raw_new_samples})"
        )
    else:
        new_samples = raw_new_samples
    result["run_switched"] = run_switched
    result["new_samples"] = int(new_samples)
    result["raw_new_samples"] = int(raw_new_samples)  # 133# 診断用

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
    if safe_to_bool(cfg.get("quality_gate_enabled", True), True):
        wf_result = _evaluate_wf(
            X_valid, y_valid, enriched, cfg,
            prev_booster=prev_booster,
            sample_weight=regime_sample_weight,
        )
        del enriched  # 466# メモリリーク防止: WF eval 後は未使用
        result["wf_eval"] = wf_result
        logger.info(
            f"WF eval: score={wf_result['score']:.4f}, "
            f"pnl30_imp={wf_result['pnl30_improvement']:.4f}, "
            f"pnl120_imp={wf_result['pnl120_improvement']:.4f}"
        )

        # 127# M1: 前モデル不在時の絶対最低 score チェック
        absolute_min = safe_to_float(cfg.get("absolute_min_score", -0.10), -0.10)
        if not prev_gate_loaded and wf_result["score"] < absolute_min:
            logger.warning(
                f"Quality gate REJECT (no prev model): "
                f"score={wf_result['score']:.4f} < absolute_min={absolute_min}. "
            )
            return {**result, "status": "rejected", "reason": "absolute_min_score"}

        # 品質ゲート: 前モデルの score と比較 (127# X2: prev_score は Step 3 で取得済み)
        min_improvement = safe_to_float(cfg.get("min_score_improvement", -0.05), -0.05)

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
        if safe_to_bool(cfg.get("all_runs_require_positive_pnl", False), False):
            target = str(cfg.get("target", "pnl120"))
            pnl_key = f"{target}_improvement"  # "pnl120_improvement" or "pnl30_improvement"
            pnl_imp = safe_to_float(wf_result.get(pnl_key, 0.0), 0.0)
            if pnl_imp < 0:
                logger.warning(
                    f"Quality gate REJECT (--all-runs positive pnl): "
                    f"{pnl_key}={pnl_imp:.4f} < 0. "
                    f"Negative expected PnL model deployment blocked."
                )
                return {**result, "status": "rejected", "reason": "negative_pnl_improvement"}

        # 131# C2: 統計的品質ゲート (gate_checks 統合)
        # 159# P0-1: 前モデル不在時 (初回訓練) は統計比較が無意味 → skip
        if not prev_gate_loaded:
            logger.info(
                "C2: Statistical gate skipped: no previous model for comparison "
                "(initial training)"
            )
            result["statistical_gate"] = {
                "applied": False,
                "reason": "no_previous_model",
            }
        elif safe_to_bool(cfg.get("statistical_gate_enabled", True), True):
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
    min_trees_for_pruning = safe_to_int(cfg.get("feature_pruning_min_trees", 20), 20)
    wf_actual_trees = result.get("wf_eval", {}).get("actual_n_trees", 0)
    if (
        safe_to_bool(cfg.get("feature_pruning_enabled", True), True)
        and safe_to_bool(cfg.get("quality_gate_enabled", True), True)
        and "wf_eval" in result
        and wf_actual_trees >= min_trees_for_pruning
    ):
        feat_imp = result["wf_eval"].get("feature_importance", {})
        min_imp = safe_to_int(cfg.get("feature_pruning_min_importance", 0), 0)
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

            require_consecutive = safe_to_bool(
                cfg.get("feature_pruning_require_consecutive", True), True
            )
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
        safe_to_bool(cfg.get("feature_pruning_enabled", True), True)
        and wf_actual_trees < min_trees_for_pruning
        and wf_actual_trees > 0
    ):
        logger.info(
            f"E3: Pruning skipped — WF eval used only {wf_actual_trees} trees "
            f"(min={min_trees_for_pruning}). Importance signal too weak."
        )

    # 131# C3: 冗長特徴量除去 (redundancy.find_highly_correlated_features 統合)
    redundancy_pruned: list[str] = []
    if safe_to_bool(cfg.get("redundancy_pruning_enabled", True), True) and len(feature_cols) >= 5:
        try:
            _red_mod = _safe_import_ztb_module("ztb.analysis.redundancy")
            calculate_feature_correlations = _red_mod.calculate_feature_correlations
            find_highly_correlated_features = _red_mod.find_highly_correlated_features

            corr_threshold = safe_to_float(cfg.get("redundancy_correlation_threshold", 0.85), 0.85)
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
    from sklearn.impute import SimpleImputer
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    imputer = SimpleImputer(strategy="median")
    scaler = StandardScaler()
    X_imp = pd.DataFrame(
        imputer.fit_transform(X_valid), columns=feature_cols, index=X_valid.index,
    )
    X_sc = pd.DataFrame(
        scaler.fit_transform(X_imp), columns=feature_cols, index=X_valid.index,
    )

    # E2: early stopping 有効時は train/val 分割
    early_stop, n_est = _resolve_early_stopping(cfg)

    lgbm = _build_lgbm_regressor(cfg, n_estimators_override=n_est)

    # E1/E2: fit kwargs
    fit_kwargs: dict[str, object] = {}
    if early_stop > 0:
        try:
            import lightgbm as lgb
        except ImportError:
            return {"status": "error", "reason": "lightgbm not installed"}
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
        and safe_to_bool(cfg.get("warm_start_enabled", True), True)
        and not pruned_features  # E3 pruning 時は feature 不一致
        and not redundancy_pruned  # C3 redundancy pruning 時も feature 不一致
        and prev_feature_cols == feature_cols  # feature_cols 完全一致が必要
    ):
        fit_kwargs["init_model"] = prev_booster
        logger.info("E1: Using prev booster as init_model for final training")
    elif prev_booster is not None and (pruned_features or redundancy_pruned):
        logger.info("E1: Warm-start skipped (feature set changed by E3/C3 pruning)")

    # 145# R-2a: レジーム重み付き学習 — 全データ fit に sample_weight 注入
    if regime_sample_weight is not None:
        # E3/C3 pruning で X_valid の行数は変わらない (列のみ) → weight はそのまま使用
        fit_kwargs["sample_weight"] = regime_sample_weight

    lgbm.fit(X_sc, y_valid, **fit_kwargs)

    # E2: 実際に使用された木の数を記録
    actual_n_trees = lgbm.booster_.num_trees() if hasattr(lgbm, "booster_") else n_est
    result["actual_n_trees"] = actual_n_trees

    # 465# D1: モデル退化ガード — 1-tree / 定数出力モデルの deploy を阻止
    # 根本原因: 少数サンプル + min_child_samples + early_stopping → 1 tree → 定数出力
    # 定数出力モデルは EV スコアを固定値化し、ceiling / deep-night AS の連鎖劣化を引き起こす
    min_deploy_trees = safe_to_int(cfg.get("min_deploy_trees", 3), 3)
    if actual_n_trees < min_deploy_trees:
        logger.warning(
            f"465# D1: Model degeneration guard REJECT: "
            f"actual_n_trees={actual_n_trees} < min_deploy_trees={min_deploy_trees}. "
            f"Model with {len(X_valid)} samples produced only {actual_n_trees} tree(s) — "
            f"constant-output model would destroy EV discrimination."
        )
        return {
            **result,
            "status": "rejected",
            "reason": f"degenerate_model: {actual_n_trees} trees < {min_deploy_trees}",
        }

    # 465# D2: 予測分散ガード — 定数出力モデルの検出
    preds = lgbm.predict(X_sc)
    pred_std = float(np.std(preds))
    result["pred_std"] = pred_std
    min_pred_std = safe_to_float(cfg.get("min_pred_std", 0.01), 0.01)
    if pred_std < min_pred_std:
        logger.warning(
            f"465# D2: Prediction variance guard REJECT: "
            f"pred_std={pred_std:.6f} < min_pred_std={min_pred_std}. "
            f"Model outputs are effectively constant — no feature discrimination."
        )
        return {
            **result,
            "status": "rejected",
            "reason": f"constant_output: pred_std={pred_std:.6f} < {min_pred_std}",
        }

    # Pipeline を再構成 (SkipGate.evaluate が pipeline.predict を使うため)
    pipeline = Pipeline([
        ("imputer", imputer),
        ("scaler", scaler),
        ("model", lgbm),
    ])
    del X_sc, X_imp  # 466# メモリリーク防止: 訓練データは Pipeline 構築後は不要

    # SkipGateConfig — 127# C1: mode を設定から取得
    sg_config = SkipGateConfig(
        mode=str(cfg.get("mode", "pnl")),
        enabled=True,
        buy_enabled=True,
        sell_enabled=True,
        threshold_bps=0.0,
        use_ob_features=use_ob,
        adaptive_threshold=safe_to_bool(cfg.get("adaptive_threshold", True), True),
        target_skip_rate_buy=safe_to_float(cfg.get("target_skip_rate_buy", 0.15), 0.15),
        target_skip_rate_sell=safe_to_float(cfg.get("target_skip_rate_sell", 0.20), 0.20),
        adaptive_window=50,
        adaptive_min_samples=20,
        adaptive_step=0.05,
        adaptive_floor=0.35,
        adaptive_ceiling=0.80,
    )

    wf_results_meta = {}
    if safe_to_bool(cfg.get("quality_gate_enabled", True), True):
        wf_results_meta = {
            "profit_score": wf_result["score"],
            "skip20_pnl30_improvement_bps": wf_result["pnl30_improvement"],
            "skip20_pnl120_improvement_bps": wf_result["pnl120_improvement"],
        }

    metadata = {
        "version": f"v4_lgbm_{target}_retrained",
        "trained_at": current_iso_timestamp(),
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
        "source_run_id": result.get("run_id", ""),  # 140# §8.1-#3: run_id 直接比較用
        "run_switched": result.get("run_switched", False),  # 140# run 切替フラグ
        "enriched_cache_used": safe_to_bool(cfg.get("enriched_cache_enabled", True), True),
        # 131# C1-C3: ztb asset 統合メタデータ
        "wf_multi_window": wf_result.get("n_windows", 1)
        if safe_to_bool(cfg.get("quality_gate_enabled", True), True)
        else 0,
        "redundancy_pruned_features": redundancy_pruned,
        "statistical_gate": result.get("statistical_gate", {}),
        # 145# R-2a: レジーム重み付き学習メタデータ
        "regime_weighting": result.get("regime_weighting", {}),
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
    tmp_path = atomic_pickle_tmp_path(model_path)
    try:
        gate.save(tmp_path)
        # アトミック rename (Windows: os.replace)
        os.replace(str(tmp_path), str(model_path))
        # SHA256 も更新 (save が tmp に書いたハッシュを本体パスに移動)
        # 131# A.1 #1 fix: with_suffix は最終 suffix のみ置換。
        # 旧コードは ".pkl.tmp.sha256" で二重 .pkl が発生していた。
        tmp_hash = hash_sidecar_path(tmp_path)
        real_hash = hash_sidecar_path(model_path)
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
        tmp_hash_cleanup = hash_sidecar_path(tmp_path)
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


def _run_online_monitor(cfg: ConfigMap) -> ConfigMap | None:
    """141# P1-12: 直近 N fill でオンラインパフォーマンスを評価.

    retrain サイクルの最後に呼び出し、skip gate の判定品質を
    直近 window 件のみで評価する。結果をログ + dict で返却。
    """
    try:
        from ztb.ml.online_monitor import (
            OnlineMonitor,
            OnlineMonitorConfig,
            log_online_monitor_summary,
        )

        window = safe_to_int(cfg.get("online_monitor_window", 100), 100)
        pnl_col = str(cfg.get("online_monitor_pnl_column", "post_fill_30s_pnl"))
        degraded_threshold = safe_to_float(cfg.get("online_monitor_degraded_threshold_bps", -0.3), -0.3)

        if not safe_to_bool(cfg.get("online_monitor_enabled", True), True):
            return None

        results_dir = Path(str(cfg.get("results_dir", "results/v460")))
        try:
            records = load_fill_records(results_dir, exclude_missing_run_id=False)
        except FileNotFoundError:
            return None

        if len(records) == 0:
            return None

        monitor = OnlineMonitor(OnlineMonitorConfig(
            window=window,
            pnl_column=pnl_col,
            degraded_threshold_bps=degraded_threshold,
        ))
        result = monitor.evaluate(records)
        log_online_monitor_summary(result)
        return result.to_dict()
    except Exception as e:
        logger.warning(f"141# P1-12: Online monitor failed (non-fatal): {e}")
        return None


def _retrain_side_specific(
    cfg: ConfigMap,
    history_path: Path,
) -> list[ConfigMap]:
    """141# P1-01/02 + 189# multi-horizon: buy/sell 分離モデルを追加学習.

    統一モデル retrain 後に呼び出される。各 side のデータだけを使い、
    side 固有のターゲット (buy=pnl30, sell=pnl120) で個別学習。
    189# alt_horizon_enabled=True の場合、副 horizon モデルも追加学習。
    十分なサンプルがない side はスキップ。

    Args:
        cfg: retrain 設定 dict.
        history_path: 履歴 JSONL パス.

    Returns:
        side 別 retrain 結果リスト.
    """
    results: list[ConfigMap] = []

    # --- primary horizon (既存) ---
    model_path_map = {
        "buy": cfg.get("model_path_buy", ""),
        "sell": cfg.get("model_path_sell", ""),
    }
    target_map = {
        "buy": cfg.get("target_buy", cfg.get("target", "pnl30")),
        "sell": cfg.get("target_sell", cfg.get("target", "pnl30")),
    }

    # --- 189# alt horizon (副 horizon for ev_weighted) ---
    alt_enabled = bool(cfg.get("alt_horizon_enabled", False))
    alt_model_path_map = {
        "buy": cfg.get("model_path_buy_long", ""),
        "sell": cfg.get("model_path_sell_short", ""),
    }
    alt_target_map = {
        "buy": cfg.get("target_buy_alt", "pnl120"),
        "sell": cfg.get("target_sell_alt", "pnl30"),
    }

    # 訓練対象: [(side, model_path, target, label), ...]
    train_specs: list[tuple[str, str, str, str]] = []
    for side in ("buy", "sell"):
        # primary
        if model_path_map[side]:
            train_specs.append((side, str(model_path_map[side]), str(target_map[side]), "primary"))
        # alt (189#)
        if alt_enabled and alt_model_path_map[side]:
            train_specs.append((side, str(alt_model_path_map[side]), str(alt_target_map[side]), "alt"))

    for side, model_path, target, horizon_label in train_specs:
        if not model_path:
            logger.debug(f"141#/189# Side model {side}/{horizon_label}: no model_path, skipping")
            continue

        side_cfg = {**cfg}
        side_cfg["side_filter"] = side
        side_cfg["target"] = target
        side_cfg["model_path"] = model_path
        # side 学習では warm_start は unified モデルと feature set が異なり得るため無効化
        side_cfg["warm_start_enabled"] = False

        try:
            side_result = retrain_model(side_cfg)
            side_result["side_model"] = side
            side_result["horizon_label"] = horizon_label  # 189#
            logger.info(
                f"141#/189# Side model {side}/{horizon_label}: "
                f"status={side_result['status']}, target={target}, path={model_path}"
            )
            _append_jsonl_record(history_path, side_result)
            results.append(side_result)
        except Exception as e:
            logger.error(f"141#/189# Side model {side}/{horizon_label} failed: {e}", exc_info=True)
            results.append({
                "side_model": side,
                "horizon_label": horizon_label,
                "status": "error",
                "reason": str(e),
            })

    return results


def run_scheduler(cfg: ConfigMap, config_path: Path | None = None) -> None:
    """定期再学習ループ.

    130# L2: サイクルごとに YAML を再読み込みし、YAML 変更を再起動なしで反映。
    136# P1-01: RetainTrigger で事前チェック + 適応的バックオフ。
    """
    from ztb.ml.retrain_trigger import RetrainTrigger, RetrainTriggerConfig

    interval = safe_to_int(cfg.get("interval_sec", 3600), 3600)

    # 136# P1-01 + §9 #4: トリガー設定を YAML から完全外部化
    trigger_cfg = RetrainTriggerConfig(
        base_interval_sec=interval,
        check_trades_health=safe_to_bool(cfg.get("trigger_check_trades_health", True), True),
        trades_lookback_days=safe_to_int(cfg.get("trigger_trades_lookback_days", 3), 3),
        trades_stale_threshold_hours=safe_to_float(
            cfg.get("trigger_trades_stale_threshold_hours", 36.0), 36.0
        ),
        trades_max_missing_days=safe_to_int(
            cfg.get("trigger_trades_max_missing_days", 1), 1
        ),
        backoff_multiplier=safe_to_float(cfg.get("trigger_backoff_multiplier", 2.0), 2.0),
        backoff_max_interval_sec=safe_to_int(cfg.get("trigger_backoff_max_interval_sec", 14400), 14400),
        check_feature_freshness=safe_to_bool(cfg.get("trigger_check_feature_freshness", False), False),
        feature_trades_stale_hours=safe_to_float(cfg.get("trigger_feature_trades_stale_hours", 6.0), 6.0),
        feature_ob_stale_hours=safe_to_float(cfg.get("trigger_feature_ob_stale_hours", 6.0), 6.0),
        # 145# R-2b: レジーム別 interval 倍率
        regime_interval_multipliers={
            str(k): safe_to_float(v, 1.0)
            for k, v in ensure_dict(
                cfg.get(
                    "trigger_regime_interval_multipliers",
                    {
                        "high_vol": 0.5,
                        "trending": 0.75,
                        "trending_up": 0.75,    # 176# 横展開
                        "trending_down": 0.75,  # 176# 横展開
                        "ranging": 1.5,
                        "unknown": 1.0,
                    },
                )
            ).items()
        },
    )
    trigger = RetrainTrigger(
        results_dir=Path(str(cfg.get("results_dir", "results/v460/fill_test"))),
        raw_dir=Path(str(cfg.get("raw_dir", "data/v460/raw"))),
        config=trigger_cfg,
    )

    logger.info(
        f"=== 126# Retrain Scheduler started ===\n"
        f"  interval: {interval}s ({interval / 3600:.1f}h)\n"
        f"  model_path: {cfg.get('model_path', '')}\n"
        f"  target: {cfg.get('target', '')}\n"
        f"  min_new_samples: {cfg.get('min_new_samples', '')}\n"
        f"  quality_gate: {safe_to_bool(cfg.get('quality_gate_enabled', True), True)}\n"
        f"  config_hot_reload: {config_path is not None}\n"
        f"  136# trigger: fill_mtime={trigger_cfg.check_fill_records_mtime}, "
        f"trades_health={trigger_cfg.check_trades_health}, "
        f"backoff_mul={trigger_cfg.backoff_multiplier}"
    )

    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)
    history_path = log_dir / "retrain_history.jsonl"

    _install_signal_handlers()
    logger.info("[161#] Graceful shutdown handlers installed (SIGTERM/SIGINT)")

    while not _shutdown_event.is_set():
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

        # 136# P1-01: 事前トリガーチェック
        should_run, trigger_reason = trigger.should_retrain()
        if not should_run:
            effective_interval = trigger.get_effective_interval()
            logger.info(
                f"[136# P1-01] Trigger skip ({trigger_reason}). "
                f"consecutive_skips={trigger.consecutive_skips}, "
                f"next_check_in={effective_interval}s ({effective_interval / 3600:.1f}h)"
            )
            # 履歴にトリガースキップも記録
            skip_result = {
                "timestamp": current_iso_timestamp(utc=True),
                "status": "skipped_trigger",
                "reason": trigger_reason,
                "consecutive_skips": trigger.consecutive_skips,
            }
            _append_jsonl_record(history_path, skip_result)
            # 161# graceful shutdown: wait with event for responsiveness
            if _shutdown_event.wait(timeout=effective_interval):
                break
            continue

        try:
            result = retrain_model(cfg)
            logger.info(f"Retrain cycle: status={result['status']}")
            # 145# R-2b: current_regime を trigger に伝搬
            trigger.record_result(
                result["status"],
                current_regime=result.get("current_regime", "unknown"),
            )
            # 履歴ファイルに記録
            _append_jsonl_record(history_path, result)

            # 141# P1-01/02: side 別モデル追加学習
            if safe_to_bool(cfg.get("side_specific_enabled", False), False):
                _retrain_side_specific(cfg, history_path)

            # 141# P1-12: オンラインパフォーマンスモニター
            _run_online_monitor(cfg)
        except Exception as e:
            logger.error(f"Retrain cycle failed: {e}", exc_info=True)
            trigger.record_result("error")
        finally:
            clear_ml_data_caches_with_log(
                logger,
                context="retrain_scheduler.cycle",
                collect_garbage=True,
            )

        effective_interval = trigger.get_effective_interval()
        logger.info(f"Next retrain in {effective_interval}s ({effective_interval / 3600:.1f}h)")
        # 161# graceful shutdown: wait with event for responsiveness
        _shutdown_event.wait(timeout=effective_interval)

    logger.info("[161#] Scheduler stopped gracefully")


def main() -> None:
    # 474# 多重起動防止: lockfile で単一プロセスを保証
    lock_path = Path("logs/retrain_scheduler.lock")
    lock_path.parent.mkdir(exist_ok=True)
    try:
        _lock_fd = os.open(str(lock_path), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        os.write(_lock_fd, str(os.getpid()).encode())
        os.close(_lock_fd)
    except FileExistsError:
        # 既存ロックの PID 生存チェック
        try:
            existing_pid = int(lock_path.read_text().strip())
            import psutil  # type: ignore[import-untyped]
            if psutil.pid_exists(existing_pid):
                proc = psutil.Process(existing_pid)
                if proc.is_running() and "retrain" in " ".join(proc.cmdline()):
                    print(
                        f"[474#] retrain_scheduler already running (PID={existing_pid}). Exiting."
                    )
                    sys.exit(0)
        except Exception:
            pass
        # stale lock を回収
        try:
            lock_path.unlink()
        except OSError:
            pass
        _lock_fd = os.open(str(lock_path), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        os.write(_lock_fd, str(os.getpid()).encode())
        os.close(_lock_fd)
    try:
        _run_retrain_scheduler_main()
    finally:
        clear_ml_data_caches_with_log(
            logger,
            context="retrain_scheduler.exit",
            collect_garbage=True,
        )
        try:
            lock_path.unlink()
        except OSError:
            pass


def _run_retrain_scheduler_main() -> None:
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
        # absolute_min_score チェックは維持 (159# P0-1: -0.10 → -0.50 に緩和)
        # 初回 side 別モデル構築で WF 窓が 1 個 (test n≈34) → 高分散のため
        # 131# A.1 #4: 加えて pnl_improvement >= 0 のハード制約を追加
        cfg["min_score_improvement"] = -999.0
        cfg["absolute_min_score"] = -0.50  # 159# P0-1: 初回訓練用に緩和
        cfg["all_runs_require_positive_pnl"] = True  # 131# A.1 #4
        logger.info(
            "Y3: --all-runs enabled → latest_run_only=False, "
            "exclude_missing_run_id=False, min_new_samples=0, "
            "absolute_min_score=-0.50, "
            "relative quality gate bypassed (absolute_min + positive pnl retained)"
        )

    if args.once:
        logger.info("=== One-shot retrain ===")
        result = retrain_model(cfg)
        logger.info(f"Result: {json.dumps(result, indent=2, default=str)}")
        # 133# P0-02: --once でも retrain_history.jsonl に記録
        # (run_scheduler 経由時のみ記録されていた監査ログ欠落を修正)
        log_dir = Path("logs")
        log_dir.mkdir(exist_ok=True)
        history_path = log_dir / "retrain_history.jsonl"
        try:
            _append_jsonl_record(history_path, result)
            logger.info(f"133# P0-02: One-shot result appended to {history_path}")
        except Exception as e:
            logger.warning(f"133# P0-02: Failed to write one-shot history: {e}")

        # 141# P1-01/02: --once でも side 別モデル追加学習
        if safe_to_bool(cfg.get("side_specific_enabled", False), False):
            _retrain_side_specific(cfg, history_path)

        # 141# P1-12: オンラインパフォーマンスモニター
        _run_online_monitor(cfg)
    else:
        run_scheduler(cfg, config_path=Path(args.config))


if __name__ == "__main__":
    main()
