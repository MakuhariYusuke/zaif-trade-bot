"""365# P6: SAC Sidecar 定期再訓練スケジューラ.

SAC sidecar モデルの warm-start incremental training を
定期的に実行し、sidecar_signal.json を更新する。

設計 (365# §5):
  - rolling window OHLCV を使い warm-start + 追加ステップで再訓練
  - OOS validation gate を通過したモデルのみ atomic deploy
  - SkipGate retrain_scheduler と完全独立プロセス (CPU 競合回避)
  - 起動: 独立プロセス or ops スクリプト

Usage:
  python scripts/v460/ml/sac_retrain_scheduler.py \\
    --config configs/v460/experiments/g2_sac_train.yaml

  # ワンショット (1回学習して終了):
  python scripts/v460/ml/sac_retrain_scheduler.py \\
    --config configs/v460/experiments/g2_sac_train.yaml --once
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
# --- PyTorch DLL load fix (import early) ---
import torch
# ----------------------------------------
from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol, cast

logger = logging.getLogger(__name__)

from ztb.io.yaml_io import read_yaml
from ztb.utils.time_utils import current_compact_timestamp, current_iso_timestamp
from scripts.v460.ml.sidecar_scheduler_common import (
    BaseRetrainResult,
    DataFileRetrainTrigger,
    atomic_replace_with_tmp,
    append_history_jsonl,
    best_effort_training_cleanup,
    run_with_timeout,
)

# ── graceful shutdown ──────────────────────────────────────
_shutdown_event = threading.Event()

# 495# 訓練タイムアウト (秒) — model.learn() の無限ハング防止
_TRAINING_TIMEOUT_SEC = 3600  # 1時間

# 495# RSS メモリ警告閾値 (MB) — サイクル間リーク検出
_RSS_WARNING_MB = 2048


def _install_signal_handlers() -> None:
    """SIGTERM/SIGINT で graceful 停止."""

    def _handler(signum: int, _frame: object) -> None:
        name = signal.Signals(signum).name
        logger.warning(f"[365# P6] Received {name} — scheduling graceful shutdown")
        _shutdown_event.set()

    signal.signal(signal.SIGTERM, _handler)
    signal.signal(signal.SIGINT, _handler)


# 600# データウィンドウ重複検出 — warm-start 過学習防止
# 前回 deploy 成功時の val 最終タイムスタンプを記録
# 同一ウィンドウで連続 warm-start すると過学習するため、
# データウィンドウが更新されるまで retrain をスキップする
_last_deployed_val_ts_max: float = 0.0


# ════════════════════════════════════════════════════════════════
# Config
# ════════════════════════════════════════════════════════════════


@dataclass
class SACRetrainConfig:
    """SAC retrain scheduler の設定.

    YAML (g2_sac_train.yaml) + retrain セクションから構築する。
    """

    # ── データ ──
    ohlcv_path: str = "data/btc_jpy_1m_full_registry_features.parquet"
    rolling_window_days: int = 7

    # ── モデル / バッファパス ──
    model_path: Path = field(default_factory=lambda: Path("models/v460/sac_sidecar.zip"))
    buffer_path: Path = field(default_factory=lambda: Path("models/v460/sac_sidecar.buffer.pkl"))
    signal_path: Path = field(default_factory=lambda: Path("cache/sidecar_signal.json"))
    norm_path: Path = field(default_factory=lambda: Path("models/v460/sac_sidecar.norm.json"))

    # ── 訓練 ──
    total_timesteps: int = 50_000  # 初回 (cold-start)
    incremental_timesteps: int = 15_000  # warm-start
    val_ratio: float = 0.2
    seed: int = 42

    # ── SAC ハイパーパラメータ ──
    gamma: float = 0.80
    learning_rate: float = 3e-4
    batch_size: int = 256
    buffer_size: int = 100_000
    learning_starts: int = 100
    tau: float = 0.005
    train_freq: int = 1
    gradient_steps: int = 1
    ent_coef: str = "auto"

    # ── 特徴量 ──
    feature_columns: list[str] = field(default_factory=list)

    # ── 環境 ──
    transaction_cost: float = 0.001
    max_position_size: float = 0.01
    initial_portfolio_value: float = 10_000_000.0
    use_simple_reward: bool = False  # 677# P0-1: True で simple reward (PnL直結)
    reward_scaling: float = 1.0  # 677# reward scaling factor

    # ── OOS Gate ──
    min_gross_roi: float = 0.0  # > 0 で gate 通過
    n_eval_episodes: int = 3
    confidence_roi_full: float = 0.005  # この ROI 以上で confidence=1.0
    min_trade_count: int = 3  # 372# Deploy Gate: OOS 中の最低取引回数
    min_profit_factor: float = 0.0  # 676# Deploy Gate: OOS profit_factor 下限 (0=無効)

    # ── スケジューラ ──
    check_interval_sec: int = 300  # polling 間隔 (5分)
    retrain_interval_sec: int = 7200  # 最短再訓練間隔 (2h)
    retrain_interval_max_sec: int = 14400  # 最長再訓練間隔 (4h)
    min_new_rows: int = 120  # rolling 更新に必要な新規行数 (2h分 = 120行)
    history_path: Path = field(default_factory=lambda: Path("logs/sac_retrain_history.jsonl"))

    # ── 649# データ鮮度チェック (retrain trigger 非依存) ──
    data_freshness_check_interval_sec: int = 3600  # 鮮度チェック間隔 (1h)
    max_data_stale_hours: float = 48.0  # この時間を超えると自動更新

    # ── 600# Conditional neutral fallback ──
    # OOS 失敗時、直近 deploy 済み signal がこの時間(h)以内なら neutral 化しない
    max_signal_staleness_hours: float = 24.0

    @classmethod
    def from_yaml_dict(cls, cfg: dict) -> SACRetrainConfig:
        """g2_sac_train.yaml の dict から SACRetrainConfig を構築."""
        data_cfg: dict = cfg.get("data", {})  # type: ignore[assignment]
        sac_cfg: dict = cfg.get("sac_hyperparameters", {})  # type: ignore[assignment]
        training_cfg: dict = cfg.get("training", {})  # type: ignore[assignment]
        env_cfg: dict = cfg.get("environment", {})  # type: ignore[assignment]
        feat_cfg: dict = cfg.get("features", {})  # type: ignore[assignment]
        output_cfg: dict = cfg.get("output", {})  # type: ignore[assignment]
        retrain_cfg: dict = cfg.get("sac_retrain", {})  # type: ignore[assignment]

        selected = feat_cfg.get("selected", [])
        feature_columns = [str(c) for c in selected] if isinstance(selected, list) else []

        model_dir = Path(str(output_cfg.get("model_dir", "models/v460")))

        return cls(
            ohlcv_path=str(
                data_cfg.get("ohlcv_path", cls.ohlcv_path)  # type: ignore[arg-type]
            ),
            rolling_window_days=int(retrain_cfg.get("rolling_window_days", 7)),
            model_path=model_dir / str(retrain_cfg.get("model_name", "sac_sidecar.zip")),
            buffer_path=model_dir / str(retrain_cfg.get("buffer_name", "sac_sidecar.buffer.pkl")),
            signal_path=Path(str(retrain_cfg.get("signal_path", "cache/sidecar_signal.json"))),
            total_timesteps=int(training_cfg.get("total_timesteps", 50_000)),
            incremental_timesteps=int(
                training_cfg.get("incremental_timesteps", retrain_cfg.get("incremental_timesteps", 15_000))
            ),
            val_ratio=float(training_cfg.get("val_ratio", 0.2)),
            seed=int(cfg.get("seed", 42)),
            gamma=float(sac_cfg.get("gamma", 0.80)),
            learning_rate=float(sac_cfg.get("learning_rate", 3e-4)),
            batch_size=int(sac_cfg.get("batch_size", 256)),
            buffer_size=int(sac_cfg.get("buffer_size", 100_000)),
            learning_starts=int(sac_cfg.get("learning_starts", 100)),
            tau=float(sac_cfg.get("tau", 0.005)),
            train_freq=int(sac_cfg.get("train_freq", 1)),
            gradient_steps=int(sac_cfg.get("gradient_steps", 1)),
            ent_coef=str(sac_cfg.get("ent_coef", "auto")),
            feature_columns=feature_columns,
            transaction_cost=float(env_cfg.get("transaction_cost", 0.001)),
            max_position_size=float(env_cfg.get("max_position_size", 0.01)),
            initial_portfolio_value=float(env_cfg.get("initial_portfolio_value", 10_000_000.0)),
            use_simple_reward=bool(env_cfg.get("use_simple_reward", False)),
            reward_scaling=float(env_cfg.get("reward_scaling", 1.0)),
            min_gross_roi=float(retrain_cfg.get("min_gross_roi", 0.0)),
            n_eval_episodes=int(retrain_cfg.get("n_eval_episodes", cfg.get("evaluation", {}).get("n_episodes", 3))),
            check_interval_sec=int(retrain_cfg.get("check_interval_sec", 300)),
            retrain_interval_sec=int(retrain_cfg.get("retrain_interval_sec", 7200)),
            retrain_interval_max_sec=int(retrain_cfg.get("retrain_interval_max_sec", 14400)),
            min_new_rows=int(retrain_cfg.get("min_new_rows", 120)),
            history_path=Path(str(retrain_cfg.get("history_path", "logs/sac_retrain_history.jsonl"))),
            confidence_roi_full=float(retrain_cfg.get("confidence_roi_full", 0.005)),
            min_trade_count=int(retrain_cfg.get("min_trade_count", 3)),
            min_profit_factor=float(retrain_cfg.get("min_profit_factor", 0.0)),
            max_signal_staleness_hours=float(
                retrain_cfg.get("max_signal_staleness_hours", 24.0)
            ),
            data_freshness_check_interval_sec=int(
                retrain_cfg.get("data_freshness_check_interval_sec", 3600)
            ),
            max_data_stale_hours=float(
                retrain_cfg.get("max_data_stale_hours", 48.0)
            ),
        )

    def __post_init__(self) -> None:
        """373# 値域バリデーション — YAML 誤設定による訓練暴走を早期検出."""
        if self.rolling_window_days < 1:
            raise ValueError(f"rolling_window_days must be >= 1, got {self.rolling_window_days}")
        if self.total_timesteps < 1:
            raise ValueError(f"total_timesteps must be >= 1, got {self.total_timesteps}")
        if self.incremental_timesteps < 1:
            raise ValueError(f"incremental_timesteps must be >= 1, got {self.incremental_timesteps}")
        if not (0.0 < self.val_ratio < 1.0):
            raise ValueError(f"val_ratio must be in (0, 1), got {self.val_ratio}")
        if self.batch_size < 1:
            raise ValueError(f"batch_size must be >= 1, got {self.batch_size}")
        if self.buffer_size < self.batch_size:
            raise ValueError(
                f"buffer_size ({self.buffer_size}) must be >= batch_size ({self.batch_size})"
            )
        if self.learning_rate <= 0:
            raise ValueError(f"learning_rate must be > 0, got {self.learning_rate}")
        if self.check_interval_sec < 1:
            raise ValueError(f"check_interval_sec must be >= 1, got {self.check_interval_sec}")
        if self.retrain_interval_sec < 1:
            raise ValueError(f"retrain_interval_sec must be >= 1, got {self.retrain_interval_sec}")
        if self.retrain_interval_max_sec < self.retrain_interval_sec:
            raise ValueError(
                f"retrain_interval_max_sec ({self.retrain_interval_max_sec}) must be >= "
                f"retrain_interval_sec ({self.retrain_interval_sec})"
            )
        if self.n_eval_episodes < 1:
            raise ValueError(f"n_eval_episodes must be >= 1, got {self.n_eval_episodes}")
        if self.min_trade_count < 0:
            raise ValueError(f"min_trade_count must be >= 0, got {self.min_trade_count}")
        if self.min_profit_factor < 0:
            raise ValueError(f"min_profit_factor must be >= 0, got {self.min_profit_factor}")
        if self.data_freshness_check_interval_sec < 60:
            raise ValueError(
                f"data_freshness_check_interval_sec must be >= 60, "
                f"got {self.data_freshness_check_interval_sec}"
            )
        if self.max_data_stale_hours <= 0:
            raise ValueError(
                f"max_data_stale_hours must be > 0, got {self.max_data_stale_hours}"
            )


def _run_data_freshness_check(
    ohlcv_path: str,
    *,
    max_stale_hours: float,
) -> bool:
    """Run non-fatal data freshness maintenance for scheduler paths."""
    from scripts.v460.ml.update_training_data import ensure_data_fresh

    return bool(ensure_data_fresh(ohlcv_path, max_stale_hours=max_stale_hours))


# ════════════════════════════════════════════════════════════════
# Training Protocol — sac_common から統一定義を import
# ════════════════════════════════════════════════════════════════

from ztb.training.sac import (  # noqa: E402
    SACModelProtocol,
    TrainingEnvProtocol,
    build_post_cycle_memory_status as _build_post_cycle_memory_status,
    build_training_debug_details as _canonical_build_training_debug_details,
    adjust_buffer_size,
    cleanup_envs,
    cleanup_training_resources,
    create_env_from_config,
    create_sac_model,
    evaluate_model_oos,
    extract_roi_from_env,
    train_val_split,
)


# ════════════════════════════════════════════════════════════════
# Core retrain logic
# ════════════════════════════════════════════════════════════════


@dataclass
class RetrainResult(BaseRetrainResult):
    """1 サイクルの再訓練結果."""

    gross_roi: float = 0.0
    trade_count: int = 0

    def to_dict(self) -> dict[str, object]:
        """JSON serializable dict."""
        payload = BaseRetrainResult.to_dict(self)
        payload.update({
            "gross_roi": round(self.gross_roi, 6),
            "trade_count": self.trade_count,
        })
        return payload


class _LatestObservationEnvProtocol(Protocol):
    current_step: int

    def _get_observation(self) -> object:
        ...


def _build_training_debug_details(
    train_df: object,
    val_df: object,
    cfg: SACRetrainConfig,
    *,
    env: TrainingEnvProtocol | None = None,
) -> dict[str, object]:
    """Backward-compatible wrapper around canonical SAC debug helper."""
    return _canonical_build_training_debug_details(
        train_df,
        val_df,
        feature_columns_configured=len(cfg.feature_columns),
        env=env,
    )


def _as_latest_observation_env(env: object) -> _LatestObservationEnvProtocol | None:
    current_step = getattr(env, "current_step", None)
    get_observation = getattr(env, "_get_observation", None)
    if not isinstance(current_step, int):
        return None
    if not callable(get_observation):
        return None
    return cast(_LatestObservationEnvProtocol, env)


def retrain_once(cfg: SACRetrainConfig) -> RetrainResult:
    """1 サイクルの SAC 再訓練を実行.

    365# §5.2 フロー:
      1. データ準備 (rolling window)
      2. Warm-start or cold-start
      3. 訓練
      4. OOS validation gate
      5. Atomic deploy + sidecar signal 更新
    """
    timestamp = current_iso_timestamp(utc=True)
    model_version = f"sac_sidecar_{current_compact_timestamp(utc=True, fmt='%Y%m%d_%H%M')}"

    # 495# finally で参照するローカル変数を事前宣言 — NameError 防止
    train_df: object = None
    val_df: object = None
    env: TrainingEnvProtocol | None = None
    val_env: TrainingEnvProtocol | None = None
    model: SACModelProtocol | None = None
    debug_details: dict[str, object] = {}

    # ── 0. データ鮮度チェック + 自動更新 (552#) ──
    try:
        _run_data_freshness_check(
            cfg.ohlcv_path,
            max_stale_hours=cfg.max_data_stale_hours,
        )
    except Exception as e:
        logger.warning(f"[552#] Data freshness check failed (non-fatal): {e}")

    # ── 1. データ読み込み ──
    try:
        import pandas as pd

        from scripts.v460.lib.data_loader import load_parquet

        df = load_parquet(cfg.ohlcv_path)
    except Exception as e:
        logger.error(f"Data load failed: {e}")
        return RetrainResult(
            status="error", timestamp=timestamp,
            error_message=f"data_load: {e}",
        )

    try:
        # Rolling window: 直近 N 日
        if cfg.rolling_window_days > 0 and len(df) > 0:
            rows_per_day = 1440  # 1-min bars
            max_rows = cfg.rolling_window_days * rows_per_day
            if len(df) > max_rows:
                df = df.iloc[-max_rows:].copy()
                logger.info(
                    f"Rolling window: {cfg.rolling_window_days}d → "
                    f"{len(df)} rows (last {max_rows})"
                )

        # Train/Val split
        train_df, val_df = train_val_split(df, cfg.val_ratio)
    finally:
        del df

    logger.info(f"Train: {len(train_df)} rows, Val: {len(val_df)} rows")

    # ── 600# データウィンドウ重複検出 ──
    # 同一 val ウィンドウで warm-start を繰り返すと過学習するため、
    # 前回 deploy 時と同じウィンドウならスキップ
    global _last_deployed_val_ts_max
    import pandas as pd

    _current_val_ts_max: float = 0.0
    if hasattr(val_df, "columns"):
        _val_df_typed = cast("pd.DataFrame", val_df)
        if "timestamp" in _val_df_typed.columns:
            _ts_val = _val_df_typed["timestamp"].iloc[-1]
            # pandas Timestamp → Unix epoch float
            _current_val_ts_max = float(
                _ts_val.timestamp() if hasattr(_ts_val, "timestamp") else _ts_val
            )
        elif _val_df_typed.index.name == "timestamp":
            _idx_val = _val_df_typed.index[-1]
            _current_val_ts_max = float(
                _idx_val.timestamp() if hasattr(_idx_val, "timestamp") else _idx_val
            )

    if (
        _last_deployed_val_ts_max > 0
        and _current_val_ts_max > 0
        and _current_val_ts_max == _last_deployed_val_ts_max
    ):
        logger.info(
            "[600#] Data window unchanged since last deploy "
            f"(val_ts_max={_current_val_ts_max:.0f}) — skipping warm-start retrain"
        )
        return RetrainResult(
            status="oos_failed",
            timestamp=timestamp,
            model_version=model_version,
            error_message="data_window_unchanged",
        )

    try:
        # 384# import_real_sb3 廃止 — pip版 SB3 を直接 import
        from stable_baselines3 import SAC as SB3_SAC  # noqa: F811

        env = _create_env(train_df, cfg)
        debug_details = _build_training_debug_details(train_df, val_df, cfg, env=env)
        logger.info(
            "SAC retrain debug: %s",
            json.dumps(
                debug_details,
                sort_keys=True,
            ),
        )
        is_warm_start = cfg.model_path.exists()

        if is_warm_start:
            model = SB3_SAC.load(str(cfg.model_path), env=env)
            logger.info(f"Warm-start: loaded model from {cfg.model_path}")

            if cfg.buffer_path.exists():
                model.load_replay_buffer(str(cfg.buffer_path))
                logger.info(f"Warm-start: loaded buffer from {cfg.buffer_path}")

            timesteps = cfg.incremental_timesteps
        else:
            model = create_sac_model(
                env,
                learning_rate=cfg.learning_rate,
                buffer_size=adjust_buffer_size(cfg.buffer_size, cfg.total_timesteps),
                learning_starts=cfg.learning_starts,
                batch_size=cfg.batch_size,
                tau=cfg.tau,
                gamma=cfg.gamma,
                train_freq=cfg.train_freq,
                gradient_steps=cfg.gradient_steps,
                ent_coef=cfg.ent_coef,
                seed=cfg.seed,
            )
            logger.info("Cold-start: new SAC model created")
            timesteps = cfg.total_timesteps

        # Training — 495# タイムアウト付き訓練
        start_time = time.time()
        try:
            run_with_timeout(
                timeout_sec=_TRAINING_TIMEOUT_SEC,
                target=lambda: model.learn(
                    total_timesteps=timesteps,
                    reset_num_timesteps=not is_warm_start,
                ),
                timeout_message=f"model.learn() exceeded {_TRAINING_TIMEOUT_SEC}s timeout",
            )
        except TimeoutError:
            training_time = time.time() - start_time
            logger.error(
                f"[495#] Training TIMEOUT after {training_time:.1f}s "
                f"(limit={_TRAINING_TIMEOUT_SEC}s) — aborting cycle"
            )
            # 495# タイムアウト時のメモリリーク防止:
            # daemon thread が model/env を掴んだまま残るため
            # ローカル参照を切ってから例外を投げ、finally の cleanup に任せる
            model = None
            raise TimeoutError(
                f"model.learn() exceeded {_TRAINING_TIMEOUT_SEC}s timeout"
            )

        training_time = time.time() - start_time
        logger.info(f"Training completed in {training_time:.1f}s ({timesteps} steps)")

        # ── 4. OOS validation ──
        val_env = _create_env(val_df, cfg)
        eval_result = _evaluate_model(model, val_env, cfg)

        if eval_result["gross_roi"] <= cfg.min_gross_roi:
            logger.warning(
                f"OOS validation FAILED: gross_roi={eval_result['gross_roi']:.6f} "
                f"<= {cfg.min_gross_roi:.6f}"
            )
            # 600# Conditional neutral fallback:
            # 直近 deploy 済み signal が fresh なら neutral 化をスキップ
            if _is_signal_fresh_and_active(
                cfg.signal_path, cfg.max_signal_staleness_hours
            ):
                logger.info(
                    "[600#] Keeping existing sidecar signal "
                    f"(last deploy < {cfg.max_signal_staleness_hours:.0f}h)"
                )
            else:
                _push_neutral_fallback(cfg.signal_path)
            return RetrainResult(
                status="oos_failed",
                timestamp=timestamp,
                model_version=model_version,
                training_time_sec=training_time,
                total_timesteps=timesteps,
                warm_start=is_warm_start,
                gross_roi=float(eval_result["gross_roi"]),
                trade_count=int(eval_result.get("trade_count", 0)),
                debug_details=debug_details,
            )

        # 372# Deploy Gate 強化: OOS 中の最低取引回数チェック
        _oos_trade_count = int(eval_result.get("trade_count", 0))
        if cfg.min_trade_count > 0 and _oos_trade_count < cfg.min_trade_count:
            logger.warning(
                f"OOS validation FAILED: trade_count={_oos_trade_count} "
                f"< {cfg.min_trade_count}"
            )
            # 600# Conditional neutral fallback
            if _is_signal_fresh_and_active(
                cfg.signal_path, cfg.max_signal_staleness_hours
            ):
                logger.info(
                    "[600#] Keeping existing sidecar signal "
                    f"(last deploy < {cfg.max_signal_staleness_hours:.0f}h)"
                )
            else:
                _push_neutral_fallback(cfg.signal_path)
            return RetrainResult(
                status="oos_failed",
                timestamp=timestamp,
                model_version=model_version,
                training_time_sec=training_time,
                total_timesteps=timesteps,
                warm_start=is_warm_start,
                gross_roi=float(eval_result["gross_roi"]),
                trade_count=_oos_trade_count,
                debug_details=debug_details,
            )

        # 676# Deploy Gate: profit_factor チェック
        _oos_pf = float(eval_result.get("pf", 0.0))
        if cfg.min_profit_factor > 0 and _oos_pf < cfg.min_profit_factor:
            logger.warning(
                f"OOS validation FAILED: profit_factor={_oos_pf:.3f} "
                f"< {cfg.min_profit_factor:.3f}"
            )
            if _is_signal_fresh_and_active(
                cfg.signal_path, cfg.max_signal_staleness_hours
            ):
                logger.info(
                    "[600#] Keeping existing sidecar signal "
                    f"(last deploy < {cfg.max_signal_staleness_hours:.0f}h)"
                )
            else:
                _push_neutral_fallback(cfg.signal_path)
            return RetrainResult(
                status="oos_failed",
                timestamp=timestamp,
                model_version=model_version,
                training_time_sec=training_time,
                total_timesteps=timesteps,
                warm_start=is_warm_start,
                gross_roi=float(eval_result["gross_roi"]),
                trade_count=_oos_trade_count,
                debug_details=debug_details,
            )

        # ── 5. Atomic deploy ──
        _atomic_deploy_model(model, cfg, model_version)

        # ── 5b. Feature norms (617# §3.1) ──
        _export_feature_norms(train_df, cfg.feature_columns, cfg.norm_path)

        # ── 6. Sidecar signal 更新 (621# NormLoader 統合) ──
        _update_sidecar_signal(
            model, env, cfg, model_version, eval_result,
            train_df=train_df,
        )

        # 600# Deploy 成功 → データウィンドウを記録
        if _current_val_ts_max > 0:
            _last_deployed_val_ts_max = _current_val_ts_max

        logger.info(
            f"✅ Deploy SUCCESS: {model_version} | "
            f"ROI={eval_result['gross_roi']:.4f} | "
            f"trades={eval_result.get('trade_count', 0)}"
        )

        return RetrainResult(
            status="deployed",
            timestamp=timestamp,
            model_version=model_version,
            training_time_sec=training_time,
            total_timesteps=timesteps,
            warm_start=is_warm_start,
            gross_roi=float(eval_result["gross_roi"]),
            trade_count=int(eval_result.get("trade_count", 0)),
            debug_details=debug_details,
        )

    except ImportError as e:
        logger.error(f"SB3 import failed: {e}")
        _push_neutral_fallback(cfg.signal_path)
        return RetrainResult(
            status="error", timestamp=timestamp,
            error_message=f"import: {e}",
            debug_details=debug_details,
        )
    except Exception as e:
        logger.error(f"Retrain failed: {e}", exc_info=True)
        # 491# P0: 訓練例外時も neutral fallback を push し signal stale を防止
        _push_neutral_fallback(cfg.signal_path)
        return RetrainResult(
            status="error", timestamp=timestamp,
            error_message=str(e),
            debug_details=debug_details,
        )
    finally:
        # 487# メモリリーク防止: model/env/DataFrame を包括的に解放 + gc.collect
        cleanup_training_resources(
            models=[model],
            envs=[val_env, env],
            dataframes=[train_df, val_df],
        )
        # 495# ローカル参照も明示的にクリア — 循環参照 GC 支援
        model = None
        env = None
        val_env = None
        train_df = None
        val_df = None


# ════════════════════════════════════════════════════════════════
# Trigger logic
# ════════════════════════════════════════════════════════════════


class SACRetrainTrigger(DataFileRetrainTrigger[SACRetrainConfig]):
    """再訓練トリガー判定.

    365# §5.2: 新規データ蓄積量ベースの判定。
    - OHLCV ファイルの mtime を監視
    - 最短 retrain_interval_sec 経過後にトリガー
    - 連続失敗時は backoff
    """
    def __init__(self, cfg: SACRetrainConfig) -> None:
        super().__init__(cfg=cfg, data_path_getter=lambda current_cfg: current_cfg.ohlcv_path)


# ════════════════════════════════════════════════════════════════
# Memory monitoring
# ════════════════════════════════════════════════════════════════

_last_cycle_rss_mb: float = 0.0


def _post_cycle_memory_check() -> None:
    """495# サイクル後 RSS モニタリング + PyTorch キャッシュ解放.

    - 各サイクル後に gc.collect() + torch cache clear
    - RSS 増加がしきい値超え → 警告ログ
    """
    global _last_cycle_rss_mb

    # PyTorch 内部キャッシュ + GC
    best_effort_training_cleanup()

    memory_details = _build_post_cycle_memory_status(
        _last_cycle_rss_mb,
        rss_warning_mb=_RSS_WARNING_MB,
    )
    current_rss = float(memory_details.get("rss_mb", 0.0))

    if bool(memory_details.get("leak_warning")):
        delta = float(memory_details.get("rss_delta_mb", 0.0))
        logger.warning(
            f"[495#] RSS increased by {delta:.1f}MB "
            f"({_last_cycle_rss_mb:.1f} → {current_rss:.1f}MB) — possible leak"
        )

    if bool(memory_details.get("rss_warning")):
        logger.warning(
            f"[495#] RSS {current_rss:.1f}MB exceeds threshold {_RSS_WARNING_MB}MB"
        )

    _last_cycle_rss_mb = current_rss
    logger.info(
        "[495#] Post-cycle RSS: %.1fMB | cache_total_entries=%s",
        current_rss,
        int(memory_details.get("cache_total_entries", 0.0)),
    )


# ════════════════════════════════════════════════════════════════
# Main scheduler loop
# ════════════════════════════════════════════════════════════════


def run_scheduler(cfg: SACRetrainConfig) -> None:
    """定期再学習メインループ.

    365# §5.2 / §6.2 に準拠。
    既存 retrain_scheduler.py (SkipGate) と同一パターン:
      while not shutdown → trigger check → retrain → wait

    649#: データ鮮度チェックを retrain trigger から分離。
    ensure_data_fresh() を独立した周期で呼び出し、
    chicken-and-egg デッドロック (data_unchanged ループ) を解消。
    """
    # 495# シグナルハンドラは main() で既にインストール済み — 二重登録は安全
    _install_signal_handlers()
    logger.info("[365# P6] Signal handlers installed (SIGTERM/SIGINT)")

    trigger = SACRetrainTrigger(cfg=cfg)

    # 649# データ鮮度チェック用タイマー (retrain trigger とは独立)
    _last_data_freshness_check: float = 0.0

    logger.info(
        f"=== 365# SAC Retrain Scheduler started ===\n"
        f"  model_path: {cfg.model_path}\n"
        f"  signal_path: {cfg.signal_path}\n"
        f"  retrain_interval: {cfg.retrain_interval_sec}s "
        f"({cfg.retrain_interval_sec / 3600:.1f}h)\n"
        f"  incremental_timesteps: {cfg.incremental_timesteps}\n"
        f"  rolling_window_days: {cfg.rolling_window_days}\n"
        f"  ohlcv_path: {cfg.ohlcv_path}\n"
        f"  min_gross_roi: {cfg.min_gross_roi}\n"
        f"  data_freshness_check: every {cfg.data_freshness_check_interval_sec}s, "
        f"stale_threshold={cfg.max_data_stale_hours}h"
    )

    cfg.history_path.parent.mkdir(parents=True, exist_ok=True)

    while not _shutdown_event.is_set():
        # ── 649# 周期的データ鮮度チェック (retrain trigger 非依存) ──
        now = time.time()
        if now - _last_data_freshness_check >= cfg.data_freshness_check_interval_sec:
            _last_data_freshness_check = now
            try:
                updated = _run_data_freshness_check(
                    cfg.ohlcv_path,
                    max_stale_hours=cfg.max_data_stale_hours,
                )
                if updated:
                    logger.info(
                        "[649#] Data refreshed by periodic check — "
                        "next retrain trigger should detect mtime change"
                    )
            except Exception as e:
                logger.warning(f"[649#] Periodic data freshness check failed: {e}")

        # 495# trigger/history 操作を try/except で保護 — ループ死亡を防止
        try:
            should_run, reason = trigger.should_retrain()
        except Exception as e:
            logger.error(f"[495#] trigger.should_retrain() failed: {e}", exc_info=True)
            if _shutdown_event.wait(timeout=cfg.check_interval_sec):
                break
            continue

        if not should_run:
            # 680#: data_unchanged は INFO に昇格 (沈黙バグの早期検出)
            _log_fn = logger.info if reason == "data_unchanged" else logger.debug
            _elapsed = time.time() - trigger._last_retrain_time
            _log_fn(
                f"[365# P6] Trigger skip: {reason} | "
                f"since_last_retrain={_elapsed:.0f}s "
                f"({_elapsed / 3600:.1f}h) | "
                f"next_check_in={cfg.check_interval_sec}s"
            )
            if _shutdown_event.wait(timeout=cfg.check_interval_sec):
                break
            continue

        logger.info(f"[365# P6] Trigger fired: {reason}")
        result = retrain_once(cfg)

        # 495# record_result / _append_history もループ外に漏れないよう保護
        try:
            trigger.record_result(result.status)
        except Exception as e:
            logger.error(f"[495#] trigger.record_result() failed: {e}", exc_info=True)

        try:
            _append_history(cfg.history_path, result)
        except Exception as e:
            logger.error(f"[495#] _append_history() failed: {e}", exc_info=True)

        # 495# サイクル後 RSS モニタリング — リーク早期検出
        _post_cycle_memory_check()

        logger.info(
            f"[365# P6] Cycle complete: status={result.status} | "
            f"next_in={trigger.effective_interval:.0f}s "
            f"({trigger.effective_interval / 3600:.1f}h)"
        )

        # 次の check まで wait
        if _shutdown_event.wait(timeout=cfg.check_interval_sec):
            break

    logger.info("[365# P6] SAC Retrain Scheduler stopped gracefully")


# ════════════════════════════════════════════════════════════════
# Internal helpers
# ════════════════════════════════════════════════════════════════


def _create_env(
    df: object,  # pd.DataFrame
    cfg: SACRetrainConfig,
) -> TrainingEnvProtocol:
    """487# 重複削減: create_env_from_config に委譲."""
    from ztb.trading.environment.utils.config import EnvironmentConfig, RewardSettings

    env_config = EnvironmentConfig(
        transaction_cost=cfg.transaction_cost,
        max_position_size=cfg.max_position_size,
        initial_portfolio_value=cfg.initial_portfolio_value,
        use_continuous_actions=True,
        action_space_type="continuous_1d",
        exchange="coincheck",
        timeframe="1m",
        # 677# P0-1: YAML の use_simple_reward を環境に反映
        reward_settings=RewardSettings(
            use_simple_reward=cfg.use_simple_reward,
            reward_scaling=cfg.reward_scaling,
        ),
    )

    if cfg.feature_columns:
        env_config.feature_names = list(cfg.feature_columns)

    return create_env_from_config(df, env_config)


def _evaluate_model(
    model: SACModelProtocol,
    env: TrainingEnvProtocol,
    cfg: SACRetrainConfig,
) -> dict[str, float | int]:
    """OOS evaluation — 365# §5.2 step 4.

    sac_common.evaluate_model_oos に委譲。
    372# audit fix: 複数エピソードの ROI / trade_count を正しく集約。
    """
    return evaluate_model_oos(model, env, n_episodes=cfg.n_eval_episodes)


def _atomic_deploy_model(
    model: SACModelProtocol,
    cfg: SACRetrainConfig,
    model_version: str,
) -> None:
    """モデル + buffer を atomic deploy (tmp → rename).

    365# §5.2 step 5.
    """
    atomic_replace_with_tmp(
        target_path=cfg.model_path,
        prefix=".sac_model_",
        suffix=".tmp.zip",
        writer=model.save,
    )
    logger.info(f"Model deployed: {cfg.model_path}")

    # Buffer: best-effort (非クリティカル)
    try:
        atomic_replace_with_tmp(
            target_path=cfg.buffer_path,
            prefix=".sac_buffer_",
            suffix=".tmp.pkl",
            writer=model.save_replay_buffer,
        )
        logger.info(f"Buffer deployed: {cfg.buffer_path}")
    except Exception as e:
        logger.warning(f"Buffer save failed (non-critical): {e}")


def _export_feature_norms(
    train_df: object,
    feature_columns: list[str],
    output_path: Path,
) -> None:
    """617# §3.1: 訓練データの特徴量統計を norm.json として出力.

    retrain 成功時にモデルと同時に保存し、推論時の Z-score 変換に使用する。
    """
    import pandas as pd

    df = cast(pd.DataFrame, train_df)
    feature_stats: dict[str, dict[str, float]] = {}
    for col in feature_columns:
        if col not in df.columns:
            continue
        series = df[col].dropna()
        if len(series) == 0:
            continue
        std_val = float(series.std())
        feature_stats[col] = {
            "mean": float(series.mean()),
            "std": std_val if std_val > 1e-10 else 1e-10,
            "min": float(series.min()),
            "max": float(series.max()),
        }

    payload = {
        "feature_stats": feature_stats,
        "metadata": {
            "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "n_features": len(feature_stats),
            "n_rows": len(df),
        },
    }

    try:
        def _write_norm_payload(tmp_path: str) -> None:
            Path(tmp_path).write_text(
                json.dumps(payload, indent=2, ensure_ascii=False),
                encoding="utf-8",
            )

        atomic_replace_with_tmp(
            target_path=output_path,
            prefix=".sac_norm_",
            suffix=".tmp.json",
            writer=_write_norm_payload,
        )
        logger.info(f"Feature norms deployed: {output_path} ({len(feature_stats)} features)")
    except Exception as e:
        logger.warning(f"Feature norms save failed (non-critical): {e}")


def _get_latest_obs(env: TrainingEnvProtocol) -> object:
    """372# F2 fix: 訓練データ末尾の observation を取得.

    env.reset() は current_step を先頭にリワインドしてしまうため、
    代わりに末尾の step に移動して _get_observation() を呼ぶ。
    環境の正規化パイプライン (OnlineScaler, action_masks) がそのまま適用される。

    HeavyTradingEnv / LiteTradingEnv 両対応。
    """
    import numpy as np

    # HeavyTradingEnv: df 属性あり
    df = getattr(env, "df", None)
    if df is not None and hasattr(df, "__len__"):
        last_step = max(0, len(df) - 1)
    # LiteTradingEnv: _feature_matrix 属性あり
    elif hasattr(env, "_feature_matrix"):
        fm = getattr(env, "_feature_matrix")
        last_step = max(0, fm.shape[0] - 1)
    else:
        # フォールバック: reset() を使用 (旧動作)
        obs, _ = env.reset()
        return obs

    latest_env = _as_latest_observation_env(env)
    if latest_env is None:
        obs, _ = env.reset()
        return obs

    # current_step を末尾に設定して observation を取得
    saved_step = latest_env.current_step
    try:
        latest_env.current_step = last_step
        obs = latest_env._get_observation()
    finally:
        latest_env.current_step = saved_step

    return obs


def _is_signal_fresh_and_active(
    signal_path: Path | str,
    max_staleness_hours: float,
) -> bool:
    """600# 既存 sidecar signal が non-neutral かつ fresh かどうか判定.

    直近 deploy 成功した signal がまだ有効なら True を返す。
    OOS 失敗時に neutral 化するかどうかの判定に使用する。
    """
    from datetime import datetime, timezone

    try:
        from scripts.v460.lib.sidecar_signal_io import read_sidecar_signal

        signal = read_sidecar_signal(signal_path, ttl_sec=0)
        if signal is None:
            return False
        if signal.model_version == "neutral" or signal.directional_bias == 0.0:
            return False
        # timestamp パース → 経過時間チェック
        ts = datetime.fromisoformat(signal.timestamp)
        if ts.tzinfo is None:
            ts = ts.replace(tzinfo=timezone.utc)
        age_hours = (datetime.now(timezone.utc) - ts).total_seconds() / 3600.0
        return age_hours < max_staleness_hours
    except Exception as e:
        logger.debug(f"[600#] Signal freshness check failed: {e}")
        return False


def _push_neutral_fallback(
    signal_path: Path | str = "cache/sidecar_signal.json",
) -> bool:
    """379# P3-C: OOS Gate 失敗時の自動フォールバック (Neutral Bias)."""
    from scripts.v460.lib.sidecar_signal_io import (
        create_neutral_signal,
        write_sidecar_signal,
    )
    neutral_signal = create_neutral_signal()
    try:
        write_sidecar_signal(neutral_signal, signal_path)
    except OSError as exc:
        logger.warning(
            "Neutral bias fallback write failed for %s: %s",
            signal_path,
            exc,
        )
        return False
    logger.info("Neutral bias fallback successfully pushed to sidecar: %s", signal_path)
    return True


def _update_sidecar_signal(
    model: SACModelProtocol,
    env: TrainingEnvProtocol,
    cfg: SACRetrainConfig,
    model_version: str,
    eval_result: dict[str, float | int],
    train_df: object | None = None,
) -> None:
    """Sidecar signal ファイルを更新.

    365# §5.2 step 6 / §5.3 フォーマット準拠。
    372# F2 fix: 訓練データ末尾 (最新) の observation で推論。
    621# NormLoader 統合: train_df + norm.json ベースの正規化を優先。
    """
    import numpy as np

    from scripts.v460.lib.sidecar_signal_io import write_sidecar_signal
    from scripts.v460.lib.sidecar_types import SidecarSignal

    features_snapshot: dict[str, float] = {}
    try:
        # 621# NormLoader 推論パス: norm.json ベースの正規化を優先
        obs = None
        if train_df is not None and cfg.norm_path.exists():
            import pandas as pd

            from ztb.features.norm_loader import NormLoader

            _df = cast(pd.DataFrame, train_df)
            _norm = NormLoader(cfg.norm_path)
            if _norm.is_loaded:
                _raw: dict[str, float] = {}
                for col in cfg.feature_columns:
                    if col in _df.columns:
                        _raw[col] = float(_df[col].iloc[-1])
                norm_obs = _norm.normalize(_raw).astype(np.float32)
                # 621# Feature Parity 診断: NormLoader vs OnlineScaler
                env_obs = np.asarray(_get_latest_obs(env), dtype=np.float32)
                if len(norm_obs) == len(env_obs):
                    obs = norm_obs
                    _dot = float(np.dot(norm_obs, env_obs))
                    _norms = float(
                        np.linalg.norm(norm_obs) * np.linalg.norm(env_obs)
                    )
                    _cos = _dot / (_norms + 1e-10)
                    _maxd = float(np.max(np.abs(norm_obs - env_obs)))
                    logger.info(
                        f"[621#] NormLoader parity: cos_sim={_cos:.6f} "
                        f"max_diff={_maxd:.4f} dim={len(norm_obs)}"
                    )
                else:
                    logger.warning(
                        f"[621#] NormLoader dim mismatch: "
                        f"norm={len(norm_obs)} vs env={len(env_obs)}"
                    )
        if obs is None:
            obs = np.asarray(_get_latest_obs(env), dtype=np.float32)

        action, _ = model.predict(obs, deterministic=True)
        # SB3 SAC continuous → action[0] が [-1, +1]
        raw_bias = float(action[0]) if hasattr(action, "__getitem__") else float(action)
        bias = max(-1.0, min(1.0, raw_bias))
        # features_snapshot: 診断用に obs の最初の数値を保存
        feature_names = getattr(cfg, "feature_columns", None) or []
        for i, name in enumerate(feature_names):
            if i < len(obs):
                features_snapshot[name] = float(obs[i])
    except Exception as e:
        logger.warning(f"Sidecar inference failed, using neutral: {e}")
        bias = 0.0

    # 372# confidence 動的計算: OOS gross_roi の gate margin から導出
    # min_gross_roi を僅かに超えた程度 → 低 confidence (慎重な offset)
    # confidence_roi_full 以上 → confidence=1.0 (full boost)
    _oos_roi = float(eval_result.get("gross_roi", 0.0))
    _gate_threshold = cfg.min_gross_roi
    _full_roi = cfg.confidence_roi_full
    if _full_roi <= _gate_threshold:
        # misconfigured → フォールバック 1.0
        _confidence = 1.0
    elif _oos_roi <= _gate_threshold:
        _confidence = 0.0
    else:
        _confidence = min(1.0, (_oos_roi - _gate_threshold) / (_full_roi - _gate_threshold))

    signal_obj = SidecarSignal(
        timestamp=current_iso_timestamp(utc=True),
        directional_bias=bias,
        model_version=model_version,
        confidence=_confidence,
        regime_hint="",
        features_snapshot=features_snapshot,
        training_metrics={
            "gross_roi": float(eval_result.get("gross_roi", 0.0)),
            "trade_count": float(eval_result.get("trade_count", 0)),
        },
    )

    write_sidecar_signal(signal_obj, cfg.signal_path)
    logger.info(
        f"Sidecar signal updated: bias={bias:+.4f} "
        f"confidence={_confidence:.3f} (roi={_oos_roi:.4f}) | {cfg.signal_path}"
    )


def _append_history(path: Path, result: RetrainResult) -> None:
    """再訓練履歴を JSONL に追記."""
    try:
        append_history_jsonl(path, result.to_dict())
    except OSError as e:
        logger.warning(f"History append failed: {e}")


# ════════════════════════════════════════════════════════════════
# YAML Config loader
# ════════════════════════════════════════════════════════════════


def load_config(config_path: str | Path) -> SACRetrainConfig:
    """YAML ファイルから SACRetrainConfig を構築."""
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Config not found: {path}")

    raw = read_yaml(path)
    if raw is None:
        raw = {}
    if not isinstance(raw, dict):
        raise TypeError(f"Expected YAML mapping in {path}")

    return SACRetrainConfig.from_yaml_dict(raw)


# ════════════════════════════════════════════════════════════════
# CLI Entry Point
# ════════════════════════════════════════════════════════════════


# 495# 最大連続リスタート回数 (無限ループ防止)
_MAX_AUTO_RESTARTS = 5
_RESTART_BACKOFF_SEC = 60


def main() -> None:
    """CLI エントリポイント.

    495# 改修:
      - シグナルハンドラを起動直後にインストール (load_config 前)
      - main() 全体を try/except で囲み致命エラーをログ出力
      - logging.shutdown + stream flush で Windows バッファ消失を防止
      - run_scheduler() 異常終了時の自動リスタート (上限付き)
    """
    # 495# シグナルハンドラを最初にインストール — 起動中の SIGTERM を捕捉
    _install_signal_handlers()

    parser = argparse.ArgumentParser(
        description="365# P6: SAC Sidecar retrain scheduler",
    )
    parser.add_argument(
        "--config",
        required=True,
        help="g2_sac_train.yaml へのパス",
    )
    parser.add_argument(
        "--once",
        action="store_true",
        help="ワンショット実行 (ループなし)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="シードの上書き (デフォルト: YAML の値)",
    )
    args = parser.parse_args()

    # Logging setup
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    try:
        cfg = load_config(args.config)
        if args.seed is not None:
            cfg.seed = args.seed

        if args.once:
            logger.info("[365# P6] One-shot mode")
            result = retrain_once(cfg)
            # 378# --once でも履歴を記録する (run_scheduler のみだった)
            _append_history(cfg.history_path, result)
            logger.info(f"Result: {json.dumps(result.to_dict(), indent=2)}")
        else:
            # 495# 自動リスタート: run_scheduler が予期せず死んだ場合にリトライ
            restart_count = 0
            while not _shutdown_event.is_set():
                try:
                    run_scheduler(cfg)
                    break  # graceful shutdown で正常終了
                except Exception as e:
                    restart_count += 1
                    if restart_count > _MAX_AUTO_RESTARTS:
                        logger.critical(
                            f"[495#] Auto-restart limit reached "
                            f"({_MAX_AUTO_RESTARTS}), giving up: {e}",
                            exc_info=True,
                        )
                        break
                    backoff = _RESTART_BACKOFF_SEC * restart_count
                    logger.error(
                        f"[495#] run_scheduler crashed ({restart_count}/{_MAX_AUTO_RESTARTS}), "
                        f"restarting in {backoff}s: {e}",
                        exc_info=True,
                    )
                    if _shutdown_event.wait(timeout=backoff):
                        break  # shutdown 要求中 → リスタートしない
    except KeyboardInterrupt:
        logger.info("[495#] KeyboardInterrupt — shutting down")
    except Exception as e:
        logger.critical(f"[495#] Fatal error in main: {e}", exc_info=True)
    finally:
        # 495# Windows バッファ消失防止: 明示的にログ + ストリームをフラッシュ
        logger.info("[495#] main() exiting — flushing logs")
        logging.shutdown()
        try:
            sys.stdout.flush()
        except Exception:
            pass
        try:
            sys.stderr.flush()
        except Exception:
            pass


if __name__ == "__main__":
    main()
