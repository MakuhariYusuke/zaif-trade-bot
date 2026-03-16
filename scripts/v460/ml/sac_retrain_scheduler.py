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
from datetime import datetime, timezone
from pathlib import Path
from typing import Protocol, cast

logger = logging.getLogger(__name__)

from ztb.io.yaml_io import read_yaml

# ── graceful shutdown ──────────────────────────────────────
_shutdown_event = threading.Event()


def _install_signal_handlers() -> None:
    """SIGTERM/SIGINT で graceful 停止."""

    def _handler(signum: int, _frame: object) -> None:
        name = signal.Signals(signum).name
        logger.warning(f"[365# P6] Received {name} — scheduling graceful shutdown")
        _shutdown_event.set()

    signal.signal(signal.SIGTERM, _handler)
    signal.signal(signal.SIGINT, _handler)


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

    # ── OOS Gate ──
    min_gross_roi: float = 0.0  # > 0 で gate 通過
    n_eval_episodes: int = 3
    confidence_roi_full: float = 0.005  # この ROI 以上で confidence=1.0
    min_trade_count: int = 3  # 372# Deploy Gate: OOS 中の最低取引回数

    # ── スケジューラ ──
    check_interval_sec: int = 300  # polling 間隔 (5分)
    retrain_interval_sec: int = 7200  # 最短再訓練間隔 (2h)
    retrain_interval_max_sec: int = 14400  # 最長再訓練間隔 (4h)
    min_new_rows: int = 120  # rolling 更新に必要な新規行数 (2h分 = 120行)
    history_path: Path = field(default_factory=lambda: Path("logs/sac_retrain_history.jsonl"))

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
            min_gross_roi=float(retrain_cfg.get("min_gross_roi", 0.0)),
            n_eval_episodes=int(retrain_cfg.get("n_eval_episodes", cfg.get("evaluation", {}).get("n_episodes", 3))),
            check_interval_sec=int(retrain_cfg.get("check_interval_sec", 300)),
            retrain_interval_sec=int(retrain_cfg.get("retrain_interval_sec", 7200)),
            retrain_interval_max_sec=int(retrain_cfg.get("retrain_interval_max_sec", 14400)),
            min_new_rows=int(retrain_cfg.get("min_new_rows", 120)),
            history_path=Path(str(retrain_cfg.get("history_path", "logs/sac_retrain_history.jsonl"))),
            confidence_roi_full=float(retrain_cfg.get("confidence_roi_full", 0.005)),
            min_trade_count=int(retrain_cfg.get("min_trade_count", 3)),
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


# ════════════════════════════════════════════════════════════════
# Training Protocol — sac_common から統一定義を import
# ════════════════════════════════════════════════════════════════

from scripts.v460.lib.sac_common import (  # noqa: E402
    SACModelProtocol,
    TrainingEnvProtocol,
    adjust_buffer_size,
    cleanup_envs,
    evaluate_model_oos,
    extract_roi_from_env,
    train_val_split,
)


# ════════════════════════════════════════════════════════════════
# Core retrain logic
# ════════════════════════════════════════════════════════════════


@dataclass
class RetrainResult:
    """1 サイクルの再訓練結果."""

    status: str  # "deployed" | "oos_failed" | "error" | "skipped"
    timestamp: str = ""
    model_version: str = ""
    training_time_sec: float = 0.0
    total_timesteps: int = 0
    warm_start: bool = False
    gross_roi: float = 0.0
    trade_count: int = 0
    error_message: str = ""

    def to_dict(self) -> dict:
        """JSON serializable dict."""
        return {
            "status": self.status,
            "timestamp": self.timestamp,
            "model_version": self.model_version,
            "training_time_sec": round(self.training_time_sec, 1),
            "total_timesteps": self.total_timesteps,
            "warm_start": self.warm_start,
            "gross_roi": round(self.gross_roi, 6),
            "trade_count": self.trade_count,
            "error_message": self.error_message,
        }


class _LatestObservationEnvProtocol(Protocol):
    current_step: int

    def _get_observation(self) -> object:
        ...


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
    timestamp = datetime.now(timezone.utc).isoformat()
    model_version = f"sac_sidecar_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M')}"

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
    del df
    logger.info(f"Train: {len(train_df)} rows, Val: {len(val_df)} rows")

    # ── 2-3. Model creation + training ──
    env: TrainingEnvProtocol | None = None
    val_env: TrainingEnvProtocol | None = None

    try:
        # 384# import_real_sb3 廃止 — pip版 SB3 を直接 import
        from stable_baselines3 import SAC as SB3_SAC

        env = _create_env(train_df, cfg)
        is_warm_start = cfg.model_path.exists()

        if is_warm_start:
            model = SB3_SAC.load(str(cfg.model_path), env=env)
            logger.info(f"Warm-start: loaded model from {cfg.model_path}")

            if cfg.buffer_path.exists():
                model.load_replay_buffer(str(cfg.buffer_path))
                logger.info(f"Warm-start: loaded buffer from {cfg.buffer_path}")

            timesteps = cfg.incremental_timesteps
        else:
            model = SB3_SAC(
                "MlpPolicy",
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
                verbose=0,
                seed=cfg.seed,
            )
            logger.info("Cold-start: new SAC model created")
            timesteps = cfg.total_timesteps

        # Training
        start_time = time.time()
        model.learn(total_timesteps=timesteps, reset_num_timesteps=not is_warm_start)
        training_time = time.time() - start_time
        logger.info(f"Training completed in {training_time:.1f}s ({timesteps} steps)")

        # ── 4. OOS validation ──
        val_env = _create_env(val_df, cfg)
        eval_result = _evaluate_model(model, val_env, cfg)

        if eval_result["gross_roi"] <= cfg.min_gross_roi:
            logger.warning(
                f"OOS validation FAILED: gross_roi={eval_result['gross_roi']:.6f} "
                f"<= {cfg.min_gross_roi:.6f} — enforcing neutral bias fallback"
            )
            # 379# P3-C: Neutral Bias Fallback
            _push_neutral_fallback()
            return RetrainResult(
                status="oos_failed",
                timestamp=timestamp,
                model_version=model_version,
                training_time_sec=training_time,
                total_timesteps=timesteps,
                warm_start=is_warm_start,
                gross_roi=float(eval_result["gross_roi"]),
                trade_count=int(eval_result.get("trade_count", 0)),
            )

        # 372# Deploy Gate 強化: OOS 中の最低取引回数チェック
        _oos_trade_count = int(eval_result.get("trade_count", 0))
        if cfg.min_trade_count > 0 and _oos_trade_count < cfg.min_trade_count:
            logger.warning(
                f"OOS validation FAILED: trade_count={_oos_trade_count} "
                f"< {cfg.min_trade_count} — enforcing neutral bias fallback"
            )
            # 379# P3-C: Neutral Bias Fallback
            _push_neutral_fallback()
            return RetrainResult(
                status="oos_failed",
                timestamp=timestamp,
                model_version=model_version,
                training_time_sec=training_time,
                total_timesteps=timesteps,
                warm_start=is_warm_start,
                gross_roi=float(eval_result["gross_roi"]),
                trade_count=_oos_trade_count,
            )

        # ── 5. Atomic deploy ──
        _atomic_deploy_model(model, cfg, model_version)

        # ── 6. Sidecar signal 更新 ──
        _update_sidecar_signal(
            model, env, cfg, model_version, eval_result,
        )

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
        )

    except ImportError as e:
        logger.error(f"SB3 import failed: {e}")
        return RetrainResult(
            status="error", timestamp=timestamp,
            error_message=f"import: {e}",
        )
    except Exception as e:
        logger.error(f"Retrain failed: {e}", exc_info=True)
        return RetrainResult(
            status="error", timestamp=timestamp,
            error_message=str(e),
        )
    finally:
        cleanup_envs(val_env, env)
        del train_df, val_df


# ════════════════════════════════════════════════════════════════
# Trigger logic
# ════════════════════════════════════════════════════════════════


@dataclass
class SACRetrainTrigger:
    """再訓練トリガー判定.

    365# §5.2: 新規データ蓄積量ベースの判定。
    - OHLCV ファイルの mtime を監視
    - 最短 retrain_interval_sec 経過後にトリガー
    - 連続失敗時は backoff
    """

    cfg: SACRetrainConfig
    _last_retrain_time: float = 0.0
    _last_data_mtime: float = 0.0
    _consecutive_failures: int = 0

    def should_retrain(self) -> tuple[bool, str]:
        """再訓練すべきか判定.

        Returns:
            (should_retrain, reason)
        """
        now = time.time()

        # 最短間隔チェック
        elapsed = now - self._last_retrain_time
        effective_interval = self._get_effective_interval()
        if elapsed < effective_interval:
            remaining = effective_interval - elapsed
            return False, f"interval_wait ({remaining:.0f}s remaining)"

        # データファイル更新チェック
        data_path = Path(self.cfg.ohlcv_path)
        if not data_path.exists():
            return False, f"data_not_found: {data_path}"

        try:
            current_mtime = data_path.stat().st_mtime
        except OSError as e:
            return False, f"stat_failed: {e}"

        if self._last_data_mtime > 0 and current_mtime <= self._last_data_mtime:
            return False, "data_unchanged"

        return True, "data_updated"

    def record_result(self, status: str) -> None:
        """結果を記録しトリガー状態を更新."""
        self._last_retrain_time = time.time()
        if status == "deployed":
            self._consecutive_failures = 0
        elif status in ("oos_failed", "error"):
            self._consecutive_failures += 1
        # data mtime 更新
        data_path = Path(self.cfg.ohlcv_path)
        if data_path.exists():
            try:
                self._last_data_mtime = data_path.stat().st_mtime
            except OSError:
                pass

    def _get_effective_interval(self) -> float:
        """連続失敗時の backoff を考慮した実効間隔."""
        base = self.cfg.retrain_interval_sec
        if self._consecutive_failures > 0:
            # exponential backoff (capped at max)
            backoff_mult = 2.0 ** min(self._consecutive_failures, 4)
            return min(base * backoff_mult, self.cfg.retrain_interval_max_sec)
        return float(base)

    @property
    def effective_interval(self) -> float:
        """現在の実効間隔 (外部参照用)."""
        return self._get_effective_interval()


# ════════════════════════════════════════════════════════════════
# Main scheduler loop
# ════════════════════════════════════════════════════════════════


def run_scheduler(cfg: SACRetrainConfig) -> None:
    """定期再学習メインループ.

    365# §5.2 / §6.2 に準拠。
    既存 retrain_scheduler.py (SkipGate) と同一パターン:
      while not shutdown → trigger check → retrain → wait
    """
    _install_signal_handlers()
    logger.info("[365# P6] Signal handlers installed (SIGTERM/SIGINT)")

    trigger = SACRetrainTrigger(cfg=cfg)

    logger.info(
        f"=== 365# SAC Retrain Scheduler started ===\n"
        f"  model_path: {cfg.model_path}\n"
        f"  signal_path: {cfg.signal_path}\n"
        f"  retrain_interval: {cfg.retrain_interval_sec}s "
        f"({cfg.retrain_interval_sec / 3600:.1f}h)\n"
        f"  incremental_timesteps: {cfg.incremental_timesteps}\n"
        f"  rolling_window_days: {cfg.rolling_window_days}\n"
        f"  ohlcv_path: {cfg.ohlcv_path}\n"
        f"  min_gross_roi: {cfg.min_gross_roi}"
    )

    cfg.history_path.parent.mkdir(parents=True, exist_ok=True)

    while not _shutdown_event.is_set():
        should_run, reason = trigger.should_retrain()

        if not should_run:
            logger.debug(
                f"[365# P6] Trigger skip: {reason} | "
                f"next_check_in={cfg.check_interval_sec}s"
            )
            if _shutdown_event.wait(timeout=cfg.check_interval_sec):
                break
            continue

        logger.info(f"[365# P6] Trigger fired: {reason}")
        result = retrain_once(cfg)
        trigger.record_result(result.status)

        # 履歴記録
        _append_history(cfg.history_path, result)

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
    """訓練環境を作成 (sac_train.py の _create_training_env を簡略化)."""
    from ztb.trading.environment.heavy_env.core import HeavyTradingEnv
    from ztb.trading.environment.utils.config import EnvironmentConfig

    env_config = EnvironmentConfig(
        transaction_cost=cfg.transaction_cost,
        max_position_size=cfg.max_position_size,
        initial_portfolio_value=cfg.initial_portfolio_value,
        use_continuous_actions=True,
        action_space_type="continuous_1d",
        exchange="coincheck",
        timeframe="1m",
    )

    if cfg.feature_columns:
        env_config.feature_names = list(cfg.feature_columns)

    env = HeavyTradingEnv(df=df, config=env_config)
    return cast(TrainingEnvProtocol, env)


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
    import tempfile

    cfg.model_path.parent.mkdir(parents=True, exist_ok=True)

    # Model: tmp → atomic rename
    fd_m, tmp_model = tempfile.mkstemp(
        dir=str(cfg.model_path.parent),
        prefix=".sac_model_",
        suffix=".tmp.zip",
    )
    os.close(fd_m)
    try:
        model.save(tmp_model)
        os.replace(tmp_model, str(cfg.model_path))
        logger.info(f"Model deployed: {cfg.model_path}")
    except Exception:
        try:
            os.unlink(tmp_model)
        except OSError:
            pass
        raise

    # Buffer: best-effort (非クリティカル)
    try:
        fd_b, tmp_buffer = tempfile.mkstemp(
            dir=str(cfg.buffer_path.parent),
            prefix=".sac_buffer_",
            suffix=".tmp.pkl",
        )
        os.close(fd_b)
        model.save_replay_buffer(tmp_buffer)
        os.replace(tmp_buffer, str(cfg.buffer_path))
        logger.info(f"Buffer deployed: {cfg.buffer_path}")
    except Exception as e:
        logger.warning(f"Buffer save failed (non-critical): {e}")
        try:
            os.unlink(tmp_buffer)
        except (OSError, NameError):
            pass


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


def _push_neutral_fallback() -> None:
    """379# P3-C: OOS Gate 失敗時の自動フォールバック (Neutral Bias)."""
    from scripts.v460.lib.sidecar_signal_io import (
        create_neutral_signal,
        write_sidecar_signal,
    )
    neutral_signal = create_neutral_signal()
    write_sidecar_signal(neutral_signal)
    logger.info("Neutral bias fallback successfully pushed to sidecar.")


def _update_sidecar_signal(
    model: SACModelProtocol,
    env: TrainingEnvProtocol,
    cfg: SACRetrainConfig,
    model_version: str,
    eval_result: dict[str, float | int],
) -> None:
    """Sidecar signal ファイルを更新.

    365# §5.2 step 6 / §5.3 フォーマット準拠。
    372# F2 fix: 訓練データ末尾 (最新) の observation で推論。
    env.reset() は訓練データ先頭にリワインドするため使用しない。
    """
    from scripts.v460.lib.sidecar_signal_io import write_sidecar_signal
    from scripts.v460.lib.sidecar_types import SidecarSignal

    # 372# F2 fix: 訓練データ末尾 (= 最新市場状態) で推論
    features_snapshot: dict[str, float] = {}
    try:
        obs = _get_latest_obs(env)
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
        timestamp=datetime.now(timezone.utc).isoformat(),
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
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with open(path, "a", encoding="utf-8") as f:
            f.write(json.dumps(result.to_dict(), ensure_ascii=False) + "\n")
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


def main() -> None:
    """CLI エントリポイント."""
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
        run_scheduler(cfg)


if __name__ == "__main__":
    main()
