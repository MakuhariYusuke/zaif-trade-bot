#!/usr/bin/env python3
"""
Phase C training entrypoint for SAC v454 (trend-focused).
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from ztb.analysis.market_regime_classifier import RegimeType
from ztb.training.v4xx_unified_trainer import V4XXUnifiedTrainer
from ztb.utils.logging_utils import get_logger

logger = get_logger(__name__)

ConfigDict = dict[str, object]


def _ensure_dict(parent: ConfigDict, key: str) -> ConfigDict:
    value = parent.get(key)
    if isinstance(value, dict):
        return value
    value = {}
    parent[key] = value
    return value


def _ensure_regime_filter(config: ConfigDict) -> ConfigDict:
    env_cfg = _ensure_dict(config, "environment")
    hybrid_cfg = _ensure_dict(env_cfg, "hybrid_config")
    regime_filter = _ensure_dict(hybrid_cfg, "regime_filter")

    regime_filter["enabled"] = True
    if "mode" not in regime_filter:
        regime_filter["mode"] = "soft"
    if "force_exit" not in regime_filter:
        regime_filter["force_exit"] = True
    if "excluded_regimes" not in regime_filter:
        regime_filter["excluded_regimes"] = []
    if "regime_constraints" not in regime_filter:
        regime_filter["regime_constraints"] = {}
    return regime_filter


def _apply_trend_only_filter(config: ConfigDict) -> None:
    trend_regimes = {
        "strong_bull_trend",
        "moderate_bull_trend",
        "weak_bull_trend",
        "strong_bear_trend",
        "moderate_bear_trend",
        "weak_bear_trend",
    }
    all_regimes = {r.value for r in RegimeType}
    excluded = sorted(all_regimes - trend_regimes)

    regime_filter = _ensure_regime_filter(config)
    regime_filter["excluded_regimes"] = excluded
    logger.info("Trend-only filter active: excluding %s regimes", len(excluded))


def _configure_training_overrides(
    trainer: V4XXUnifiedTrainer,
    *,
    total_timesteps: int | None,
    episodes: int | None,
    reset_num_timesteps: bool,
    trend_only: bool,
) -> None:
    if not isinstance(trainer.config, dict):
        raise TypeError("Unified trainer config is not a dict")

    if trend_only:
        _apply_trend_only_filter(trainer.config)

    training_cfg = trainer.config.setdefault("training", {})
    if not isinstance(training_cfg, dict):
        trainer.config["training"] = {}
        training_cfg = trainer.config["training"]

    if reset_num_timesteps:
        for key in (
            "resume_from",
            "init_model_path",
            "initial_model_path",
            "pretrained_model_path",
        ):
            training_cfg.pop(key, None)
        training_cfg["reset_num_timesteps"] = True

    if episodes is not None:
        from ztb.training.reward_function_optimizer.constants import (
            DEFAULT_MAX_EPISODE_LENGTH,
        )

        max_ep = None
        env_section = training_cfg.get("environment", {})
        if isinstance(env_section, dict):
            inner_cfg = env_section.get("config", env_section)
            if isinstance(inner_cfg, dict):
                max_ep = inner_cfg.get("max_episode_length") or inner_cfg.get(
                    "max_episode_steps"
                )

        if max_ep is None:
            max_ep = training_cfg.get("max_episode_length")

        if max_ep is None:
            max_ep = DEFAULT_MAX_EPISODE_LENGTH

        training_cfg["episodes"] = int(episodes)
        training_cfg["total_timesteps"] = int(episodes) * int(max_ep)
        logger.info(
            "Overriding config: episodes=%s, computed total_timesteps=%s (max_episode_length=%s)",
            episodes,
            training_cfg["total_timesteps"],
            max_ep,
        )
    elif total_timesteps is not None:
        training_cfg["total_timesteps"] = int(total_timesteps)
        logger.info(
            "Overriding config: total_timesteps=%s", training_cfg["total_timesteps"]
        )


def main() -> int:
    parser = argparse.ArgumentParser(description="SAC v454 Phase C (trend-focused)")
    parser.add_argument(
        "--config",
        type=str,
        default="config/v454/sac_v454_phaseC_config.json",
        help="Path to configuration file",
    )
    parser.add_argument(
        "--version", type=str, default="v454", help="Override version detection"
    )
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--episodes", type=int, help="Number of episodes to run")
    group.add_argument(
        "--total-timesteps",
        type=int,
        help="Total timesteps to train for (overrides config)",
    )
    parser.add_argument(
        "--reset-num-timesteps",
        action="store_true",
        help="Force a fresh run (ignore resume/init model paths) and reset timestep counter",
    )
    parser.add_argument(
        "--allow-non-trend",
        action="store_true",
        help="Allow non-trend regimes during training (default: trend-only)",
    )

    args = parser.parse_args()

    try:
        logger.info("Phase C training starting")
        logger.info("Config: %s", args.config)

        trainer = V4XXUnifiedTrainer(config_path=args.config, version=args.version)
        _configure_training_overrides(
            trainer,
            total_timesteps=args.total_timesteps,
            episodes=args.episodes,
            reset_num_timesteps=args.reset_num_timesteps,
            trend_only=not args.allow_non_trend,
        )

        if not trainer.validate_config():
            logger.error("Configuration validation failed")
            return 1

        success = trainer.train()

        from ztb.utils.training_utils import display_training_complete

        training_time = getattr(trainer, "training_time", 0.0)
        final_metrics = {}
        if hasattr(trainer, "training_report") and trainer.training_report:
            report_stats = trainer.training_report.get("training_stats", {})
            if isinstance(report_stats, dict):
                final_metrics = report_stats
                training_time = report_stats.get("training_time", training_time)

        display_training_complete(final_metrics if success else {}, training_time)
        return 0 if success else 1

    except Exception as exc:
        logger.error("Phase C training failed: %s", exc)
        import traceback

        traceback.print_exc()
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
