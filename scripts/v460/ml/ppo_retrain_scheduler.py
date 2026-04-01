"""PPO sidecar retrain scheduler.

675#/676#/678# で復旧した PPO foundation の上に、current runtime で使える
最小 scheduler を載せる。SAC scheduler と同じ運用骨格を持つが、
現段階では安全側に寄せて以下の方針を採る。

- signal が無い時は従来動作のまま
- 学習失敗時は neutral PPO signal を push
- live merge は confidence/action_margin gate により cycle 側で再判定
- 既存 model がある場合は warm-start を試し、
  失敗時だけ cold start にフォールバックする
"""

from __future__ import annotations

import argparse
import logging
import signal
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol, cast

import numpy as np
import torch
from numpy.typing import NDArray

from scripts.v460.lib.sidecar_signal_io import (
    create_neutral_ppo_signal,
    write_ppo_sidecar_signal,
)
from scripts.v460.lib.sidecar_types import PPOSidecarSignal
from scripts.v460.ml.ppo_sidecar_config import PPOSidecarConfig
from scripts.v460.ml.sidecar_scheduler_common import (
    BaseRetrainResult,
    DataFileRetrainTrigger,
    atomic_replace_with_tmp,
    append_history_jsonl,
    best_effort_training_cleanup,
    run_with_timeout,
)
from ztb.io.data_loader import DataLoader
from ztb.io.yaml_io import read_yaml
from ztb.trading.environment.environment import HeavyTradingEnv
from ztb.training.config.trainer_params import SELLMitigationParams
from ztb.training.core.ppo_trainer import wrap_env_with_action_masker
from ztb.training.experiments.sell_mitigation_ppo_trainer import (
    SELLBiasMitigationPPOTrainer,
)
from ztb.utils.logging_utils import get_logger
from ztb.utils.time_utils import current_compact_timestamp, current_iso_timestamp

logger = get_logger(__name__)

_shutdown_event = threading.Event()
_TRAINING_TIMEOUT_SEC = 3600


def _install_signal_handlers() -> None:
    """SIGTERM/SIGINT で graceful 停止."""

    def _handler(signum: int, _frame: object) -> None:
        name = signal.Signals(signum).name
        logger.warning("[675#] Received %s — scheduling graceful shutdown", name)
        _shutdown_event.set()

    signal.signal(signal.SIGTERM, _handler)
    signal.signal(signal.SIGINT, _handler)


@dataclass(slots=True)
class PPORetrainResult(BaseRetrainResult):
    """1 サイクルの PPO retrain 結果."""

    action: str = "skip"
    confidence: float = 0.0
    action_margin: float = 0.0

    def to_dict(self) -> dict[str, object]:
        payload: dict[str, object] = BaseRetrainResult.to_dict(self)
        payload.update(
            {
                "action": self.action,
                "confidence": round(self.confidence, 6),
                "action_margin": round(self.action_margin, 6),
            }
        )
        return payload


class PPORetrainTrigger(DataFileRetrainTrigger[PPOSidecarConfig]):
    """PPO sidecar の再訓練トリガー."""

    def __init__(self, cfg: PPOSidecarConfig) -> None:
        super().__init__(cfg=cfg, data_path_getter=lambda current_cfg: current_cfg.data_path)


class _PPOModelProtocol(Protocol):
    @property
    def policy(self) -> object: ...

    def save(self, path: str) -> None: ...

    def predict(
        self,
        observation: object,
        deterministic: bool = True,
    ) -> tuple[object, object | None]: ...


def _coerce_action_index(action: object) -> int:
    if isinstance(action, np.ndarray):
        if action.size == 0:
            return 0
        return int(action.reshape(-1)[0])
    if isinstance(action, (int, np.integer)):
        return int(action)
    return 0


def _one_hot_ppo_probabilities(action_index: int) -> dict[str, float]:
    clamped_action = 0 if action_index < 0 or action_index > 2 else action_index
    return {
        "skip": 1.0 if clamped_action == 0 else 0.0,
        "buy": 1.0 if clamped_action == 1 else 0.0,
        "sell": 1.0 if clamped_action == 2 else 0.0,
    }


def _extract_action_probabilities(
    model: _PPOModelProtocol,
    observation: object,
    *,
    action_masks: NDArray[np.bool_] | None = None,
) -> dict[str, float]:
    """current PPO policy から buy/sell/skip probability を抽出する."""

    policy = getattr(model, "policy", None)
    if policy is None or not hasattr(policy, "obs_to_tensor") or not hasattr(
        policy, "get_distribution"
    ):
        action, _ = model.predict(observation, deterministic=True)
        return _one_hot_ppo_probabilities(_coerce_action_index(action))

    obs_to_tensor = cast(Any, getattr(policy, "obs_to_tensor"))
    get_distribution = cast(Any, getattr(policy, "get_distribution"))
    obs_tensor, _ = obs_to_tensor(observation)
    distribution = get_distribution(obs_tensor)
    raw_distribution = getattr(distribution, "distribution", distribution)

    probs_like = getattr(raw_distribution, "probs", None)
    if probs_like is None:
        logits = getattr(raw_distribution, "logits", None)
        if logits is not None:
            probs_like = torch.softmax(logits, dim=-1)

    if probs_like is None:
        action, _ = model.predict(observation, deterministic=True)
        return _one_hot_ppo_probabilities(_coerce_action_index(action))

    if hasattr(probs_like, "detach"):
        probabilities = np.asarray(probs_like.detach().cpu().numpy(), dtype=float)
    else:
        probabilities = np.asarray(probs_like, dtype=float)
    flat_probabilities = probabilities.reshape(-1)
    if flat_probabilities.size < 3:
        action, _ = model.predict(observation, deterministic=True)
        return _one_hot_ppo_probabilities(_coerce_action_index(action))

    clipped = flat_probabilities[:3]
    if action_masks is not None and action_masks.shape[0] >= 3:
        masked = clipped * action_masks[:3].astype(float)
        if masked.sum() > 0.0:
            clipped = masked / masked.sum()

    return {
        "skip": float(clipped[0]),
        "buy": float(clipped[1]),
        "sell": float(clipped[2]),
    }


def _build_inference_env(cfg: PPOSidecarConfig) -> HeavyTradingEnv:
    """signal 抽出用の lightweight discrete PPO env を作る."""
    df = DataLoader.load_csv_strict(cfg.data_path)
    if len(df) > 512:
        df = df.iloc[-512:].copy()

    env = HeavyTradingEnv(
        df=df,
        config=cfg.build_trainer_config(total_timesteps=cfg.incremental_timesteps),
    )
    return env


def _atomic_deploy_model(model: _PPOModelProtocol, model_path: Path) -> None:
    """PPO model を tmp -> rename で deploy."""
    atomic_replace_with_tmp(
        target_path=model_path,
        prefix=".ppo_model_",
        suffix=".tmp.zip",
        writer=model.save,
    )


def _push_neutral_fallback(signal_path: Path | str) -> bool:
    """訓練失敗時に neutral PPO signal を push する."""
    try:
        write_ppo_sidecar_signal(create_neutral_ppo_signal(), signal_path)
    except OSError as exc:
        logger.warning("Neutral PPO fallback write failed for %s: %s", signal_path, exc)
        return False
    logger.info("Neutral PPO fallback successfully pushed to sidecar: %s", signal_path)
    return True


def _update_ppo_sidecar_signal(
    model: _PPOModelProtocol,
    cfg: PPOSidecarConfig,
    model_version: str,
) -> PPOSidecarSignal:
    """最新 discrete PPO model から sidecar signal を更新する."""

    inference_env = _build_inference_env(cfg)
    wrapped_env = wrap_env_with_action_masker(inference_env)
    try:
        observation, _ = wrapped_env.reset()
        action_masks = np.asarray(wrapped_env.get_action_masks(), dtype=np.bool_)
        action_probabilities = _extract_action_probabilities(
            model,
            observation,
            action_masks=action_masks,
        )
        signal_obj = PPOSidecarSignal.from_probabilities(
            timestamp=current_iso_timestamp(utc=True),
            action_probabilities=action_probabilities,
            model_version=model_version,
            training_metrics={
                "min_override_confidence": cfg.min_override_confidence,
                "min_action_probability_gap": cfg.min_action_probability_gap,
            },
        )
        write_ppo_sidecar_signal(signal_obj, cfg.signal_path)
        logger.info(
            "PPO sidecar signal updated: action=%s confidence=%.3f margin=%.3f | %s",
            signal_obj.action,
            signal_obj.confidence,
            signal_obj.action_margin,
            cfg.signal_path,
        )
        return signal_obj
    finally:
        wrapped_env.close()


def _build_trainer_params(
    cfg: PPOSidecarConfig,
    *,
    total_timesteps: int,
) -> SELLMitigationParams:
    trainer_config = cfg.build_trainer_config(total_timesteps=total_timesteps)
    return SELLMitigationParams(
        data_path=cfg.data_path,
        config=cast(Any, trainer_config),
        checkpoint_dir=str(cfg.checkpoint_dir),
        enable_lagrange=False,
        enable_probes=False,
        enable_weights=False,
        enable_pan=cfg.enable_pan,
        enable_target_entropy=cfg.enable_target_entropy,
        enable_stratified_sampling=cfg.enable_stratified_sampling,
        allow_reverse=cfg.allow_reverse,
    )


def _train_with_timeout(
    trainer: SELLBiasMitigationPPOTrainer,
    *,
    session_id: str,
) -> _PPOModelProtocol:
    """PPO training を timeout 保護付きで実行する."""
    trained_model = run_with_timeout(
        timeout_sec=_TRAINING_TIMEOUT_SEC,
        target=lambda: trainer.train(session_id=session_id),
        timeout_message=f"PPO trainer exceeded {_TRAINING_TIMEOUT_SEC}s timeout",
    )
    return cast(_PPOModelProtocol, trained_model)


def _cleanup_training_cycle() -> None:
    """PPO retrain cycle 後の best-effort cleanup."""
    best_effort_training_cleanup()


def retrain_once(cfg: PPOSidecarConfig) -> PPORetrainResult:
    """1 サイクルの PPO sidecar retrain を実行する."""

    timestamp = current_iso_timestamp(utc=True)
    model_version = f"ppo_sidecar_{current_compact_timestamp(utc=True, fmt='%Y%m%d_%H%M')}"
    is_warm_start = cfg.model_path.exists()
    timesteps = cfg.incremental_timesteps if is_warm_start else cfg.total_timesteps
    trainer = SELLBiasMitigationPPOTrainer(
        _build_trainer_params(cfg, total_timesteps=timesteps)
    )

    try:
        start_time = time.time()
        if is_warm_start and hasattr(trainer, "load_and_continue"):
            model = run_with_timeout(
                timeout_sec=_TRAINING_TIMEOUT_SEC,
                target=lambda: cast(
                    _PPOModelProtocol,
                    trainer.load_and_continue(
                        cfg.model_path,
                        timesteps,
                        session_id=model_version,
                    ),
                ),
                timeout_message=(
                    f"PPO trainer exceeded {_TRAINING_TIMEOUT_SEC}s timeout"
                ),
            )
        else:
            model = _train_with_timeout(trainer, session_id=model_version)
        if model is None:
            raise RuntimeError("PPO trainer returned no model")
        _atomic_deploy_model(model, cfg.model_path)
        signal_obj = _update_ppo_sidecar_signal(
            model,
            cfg,
            model_version,
        )
        training_time = time.time() - start_time
        return PPORetrainResult(
            status="deployed",
            timestamp=timestamp,
            model_version=model_version,
            training_time_sec=training_time,
            total_timesteps=timesteps,
            warm_start=is_warm_start,
            action=signal_obj.action,
            confidence=signal_obj.confidence,
            action_margin=signal_obj.action_margin,
            debug_details={
                "data_path": cfg.data_path,
                "checkpoint_dir": str(cfg.checkpoint_dir),
                "trainer_mode": "warm_start_resume" if is_warm_start else "cold_start",
            },
        )
    except Exception as exc:
        logger.error("PPO retrain failed: %s", exc, exc_info=True)
        _push_neutral_fallback(cfg.signal_path)
        return PPORetrainResult(
            status="error",
            timestamp=timestamp,
            model_version=model_version,
            total_timesteps=timesteps,
            warm_start=is_warm_start,
            error_message=str(exc),
            debug_details={"data_path": cfg.data_path},
        )
    finally:
        _cleanup_training_cycle()


def run_scheduler(cfg: PPOSidecarConfig) -> None:
    """定期 PPO retrain メインループ."""
    _install_signal_handlers()
    trigger = PPORetrainTrigger(cfg)

    logger.info(
        "=== PPO sidecar retrain scheduler started ===\n"
        "  model_path: %s\n"
        "  signal_path: %s\n"
        "  retrain_interval: %ss (%.1fh)\n"
        "  total_timesteps: %s\n"
        "  incremental_timesteps: %s\n"
        "  data_path: %s",
        cfg.model_path,
        cfg.signal_path,
        cfg.retrain_interval_sec,
        cfg.retrain_interval_sec / 3600.0,
        cfg.total_timesteps,
        cfg.incremental_timesteps,
        cfg.data_path,
    )

    while not _shutdown_event.is_set():
        should_run, reason = trigger.should_retrain()
        if not should_run:
            logger.debug(
                "[675#] PPO trigger skip: %s | next_check_in=%ss",
                reason,
                cfg.check_interval_sec,
            )
            if _shutdown_event.wait(timeout=cfg.check_interval_sec):
                break
            continue

        logger.info("[675#] PPO trigger fired: %s", reason)
        result = retrain_once(cfg)
        trigger.record_result(result.status)
        try:
            append_history_jsonl(cfg.history_path, result.to_dict())
        except OSError as exc:
            logger.warning("PPO history append failed: %s", exc)

        logger.info(
            "[675#] PPO cycle complete: status=%s action=%s next_in=%.0fs",
            result.status,
            result.action,
            trigger.effective_interval,
        )
        if _shutdown_event.wait(timeout=cfg.check_interval_sec):
            break

    logger.info("[675#] PPO Retrain Scheduler stopped gracefully")


def load_config(config_path: str | Path) -> PPOSidecarConfig:
    """YAML ファイルから PPO sidecar config を構築する."""
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Config not found: {path}")

    raw = read_yaml(path)
    if raw is None:
        raw = {}
    if not isinstance(raw, dict):
        raise TypeError(f"Expected YAML mapping in {path}")

    return PPOSidecarConfig.from_yaml_dict(raw)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="675# PPO sidecar retrain scheduler",
    )
    parser.add_argument(
        "--config",
        required=True,
        help="PPO sidecar config を含む YAML へのパス",
    )
    parser.add_argument(
        "--once",
        action="store_true",
        help="ワンショット実行 (ループなし)",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    _install_signal_handlers()
    cfg = load_config(args.config)

    if args.once:
        result = retrain_once(cfg)
        print(result.to_dict())
        return

    run_scheduler(cfg)


if __name__ == "__main__":
    main()
