#!/usr/bin/env python3
"""
Canonical 1M Training Runner for Zaif Trade Bot.

Runs a 1 million timestep PPO training session with resume capability,
periodic evaluation, and proper artifact management.
"""

import argparse
import copy
import json
import logging
import os
import sys
from pathlib import Path
from typing import Any, Dict, Optional

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from ztb.training.core.ppo_trainer import PPOTrainer
from ztb.training.config.ppo_config import get_ppo_config
from ztb.utils import DiscordNotifier
def _create_streaming_pipeline(
    enable_streaming: bool, stream_batch_size: int, logger: logging.Logger
):
    """Instantiate streaming pipeline when requested."""
    if not enable_streaming:
        return None

    logger.info("Enabling streaming pipeline")
    from ztb.data.streaming_pipeline import create_streaming_pipeline as _factory

    pipeline = _factory(buffer_capacity=max(stream_batch_size * 2000, 1_000_000))
    logger.debug(
        "Streaming buffer capacity=%s rows", getattr(pipeline.buffer, "capacity", "unknown")
    )
    return pipeline



def setup_logging(verbose: bool) -> logging.Logger:
    """Setup logging configuration."""
    log_level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=log_level, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    return logging.getLogger(__name__)


def validate_training_setup(data_path: str, checkpoint_dir: str, correlation_id: str) -> bool:
    """Validate training setup before starting."""
    logger = logging.getLogger(__name__)
    
    # Validate data path
    data_path_obj = Path(data_path)
    if not data_path_obj.exists():
        logger.error(f"Data file not found: {data_path_obj}")
        return False

    # Check for duplication (resume invariants)
    session_dir = Path(checkpoint_dir) / correlation_id
    if session_dir.exists():
        logger.warning(f"Session {correlation_id} already exists at {session_dir}")
        logger.warning("Use resume functionality or choose a different correlation-id")
        return False
    
    return True


def _deep_merge(base: Dict[str, Any], overrides: Dict[str, Any]) -> Dict[str, Any]:
    """Recursively merge two dictionaries without mutating inputs."""

    def _merge(a: Dict[str, Any], b: Dict[str, Any]) -> Dict[str, Any]:
        result: Dict[str, Any] = copy.deepcopy(a)
        for key, value in b.items():
            if (
                key in result
                and isinstance(result[key], dict)
                and isinstance(value, dict)
            ):
                result[key] = _merge(result[key], value)
            else:
                result[key] = copy.deepcopy(value)
        return result

    return _merge(base, overrides)


def _load_unified_overrides() -> Dict[str, Any]:
    """Load serialized unified configuration overrides from the environment."""
    raw = os.environ.get("ZTB_UNIFIED_ITERATIVE_CONFIG")
    if not raw:
        return {}

    try:
        overrides = json.loads(raw)
        return overrides if isinstance(overrides, dict) else {}
    except json.JSONDecodeError:
        logging.getLogger(__name__).warning(
            "Failed to decode unified config overrides; falling back to CLI defaults",
            exc_info=True,
        )
        return {}


def _apply_cli_overrides(config: Dict[str, Any], args: argparse.Namespace) -> Dict[str, Any]:
    """Apply CLI-sourced overrides on top of the merged configuration."""
    cfg = copy.deepcopy(config)

    ppo_cfg = cfg.setdefault("ppo", {})
    memory_cfg = cfg.setdefault("memory_optimization", {})
    env_cfg = cfg.setdefault("environment", {})
    feature_cfg = cfg.setdefault("features", {})

    # Core bookkeeping
    cfg["total_timesteps"] = args.total_timesteps
    ppo_cfg["total_timesteps"] = args.total_timesteps
    cfg["log_dir"] = args.log_dir
    cfg["model_dir"] = args.model_dir
    cfg["tensorboard_log"] = args.log_dir
    cfg["checkpoint_dir"] = args.checkpoint_dir
    cfg["checkpoint_interval"] = args.checkpoint_interval
    cfg["data_path"] = args.data_path
    cfg["enable_streaming"] = bool(args.enable_streaming)
    cfg["stream_batch_size"] = args.stream_batch_size
    cfg["offline_mode"] = bool(args.offline_mode)
    cfg["verbose"] = 1 if args.verbose else 0
    ppo_cfg["verbose"] = cfg["verbose"]

    # Trading environment settings
    cfg["transaction_cost"] = args.transaction_cost
    ppo_cfg["transaction_cost"] = args.transaction_cost
    env_cfg.setdefault("transaction_cost", args.transaction_cost)

    cfg["max_position_size"] = args.max_position_size
    ppo_cfg["max_position_size"] = args.max_position_size
    env_cfg.setdefault("max_position_size", args.max_position_size)

    cfg["timeframe"] = args.timeframe
    feature_cfg.setdefault("feature_set", args.feature_set)
    cfg["feature_set"] = feature_cfg["feature_set"]

    if args.data_rows_limit is not None:
        cfg["data_rows_limit"] = args.data_rows_limit
        memory_cfg["data_rows_limit"] = args.data_rows_limit

    if args.max_features is not None:
        cfg["max_features"] = args.max_features
        memory_cfg["max_features"] = args.max_features

    # Reward shaping parameters
    cfg["reward_trade_frequency_penalty"] = args.reward_trade_frequency_penalty
    cfg["reward_trade_frequency_halflife"] = args.reward_trade_frequency_halflife
    cfg["reward_trade_cooldown_steps"] = args.reward_trade_cooldown_steps
    cfg["reward_trade_cooldown_penalty"] = args.reward_trade_cooldown_penalty
    cfg["reward_max_consecutive_trades"] = args.reward_max_consecutive_trades
    cfg["reward_consecutive_trade_penalty"] = args.reward_consecutive_trade_penalty

    # Evaluation defaults (can be overridden via unified config)
    cfg.setdefault("eval_freq", 10000)
    cfg.setdefault("n_eval_episodes", 5)

    return cfg


def build_training_config(
    args: argparse.Namespace, config_overrides: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """Build training configuration from command line arguments and overrides."""

    # Baseline PPO configuration (keeps legacy defaults for standalone usage)
    base_config: Dict[str, Any] = {
        "ppo": dict(get_ppo_config()),
        "memory_optimization": {},
        "features": {
            "feature_set": args.feature_set,
        },
        "environment": {},
        "seed": 42,
        "checkpoint": {
            "keep_last": 5,
            "compress": "zstd",
            "async_save": True,
            "include_optimizer": True,
            "include_replay_buffer": False,
            "include_rng_state": True,
        },
    }

    merged = _deep_merge(base_config, config_overrides or {})
    return _apply_cli_overrides(merged, args)


def main() -> int:
    parser = argparse.ArgumentParser(description="Run canonical 1M timestep training")
    parser.add_argument(
        "--correlation-id",
        required=True,
        help="Correlation ID for this training session",
    )
    parser.add_argument(
        "--data-path",
        default="ml-dataset.csv",
        help="Path to training data (default: ml-dataset.csv)",
    )
    parser.add_argument(
        "--total-timesteps",
        type=int,
        default=1_000_000,
        help="Total training timesteps (default: 1,000,000)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Dry run mode - validate setup without training",
    )
    parser.add_argument(
        "--checkpoint-dir",
        default="checkpoints",
        help="Checkpoint directory (default: checkpoints)",
    )
    parser.add_argument(
        "--checkpoint-interval",
        type=int,
        default=10000,
        help="Steps between checkpoints (default: 10000)",
    )
    parser.add_argument(
        "--log-dir", default="logs", help="Log directory (default: logs)"
    )
    parser.add_argument(
        "--model-dir", default="models", help="Model directory (default: models)"
    )
    parser.add_argument("--verbose", action="store_true", help="Enable verbose logging")
    parser.add_argument(
        "--offline-mode",
        action="store_true",
        help="Offline mode - disable Discord notifications and internet-dependent features",
    )
    parser.add_argument(
        "--enable-streaming",
        action="store_true",
        help="Enable streaming pipeline (default: disabled)",
    )
    parser.add_argument(
        "--stream-batch-size",
        type=int,
        default=256,
        help="Streaming batch size (default: 256)",
    )
    parser.add_argument(
        "--feature-set",
        default="full",
        help="Feature set name or preset (default: full)",
    )
    parser.add_argument(
        "--timeframe",
        default="1m",
        help="Timeframe label for training dataset (default: 1m)",
    )
    parser.add_argument(
        "--reward-trade-frequency-penalty",
        type=float,
        default=0.3,
        help="Penalty for frequent trading (default: 0.3)",
    )
    parser.add_argument(
        "--reward-trade-frequency-halflife",
        type=float,
        default=12.0,
        help="Halflife for trade frequency penalty decay (default: 12.0)",
    )
    parser.add_argument(
        "--reward-trade-cooldown-steps",
        type=int,
        default=3,
        help="Cooldown steps between trades (default: 3)",
    )
    parser.add_argument(
        "--reward-trade-cooldown-penalty",
        type=float,
        default=0.5,
        help="Penalty for trading during cooldown (default: 0.5)",
    )
    parser.add_argument(
        "--reward-max-consecutive-trades",
        type=int,
        default=3,
        help="Maximum consecutive trades allowed (default: 3)",
    )
    parser.add_argument(
        "--reward-consecutive-trade-penalty",
        type=float,
        default=0.4,
        help="Penalty for exceeding consecutive trade limit (default: 0.4)",
    )
    parser.add_argument(
        "--transaction-cost",
        type=float,
        default=0.001,
        help="Transaction cost per trade (default: 0.001)",
    )
    parser.add_argument(
        "--max-position-size",
        type=float,
        default=1.0,
        help="Maximum position size (default: 1.0)",
    )
    parser.add_argument(
        "--data-rows-limit",
        type=int,
        default=None,
        help="Optional cap on number of data rows loaded into memory",
    )
    parser.add_argument(
        "--max-features",
        type=int,
        default=None,
        help="Optional cap on number of features to retain",
    )

    args = parser.parse_args()

    # Setup logging
    setup_logging(args.verbose)

    logger = logging.getLogger(__name__)

    # Load any unified configuration overrides and consume the env var early
    config_overrides = _load_unified_overrides()
    os.environ.pop("ZTB_UNIFIED_ITERATIVE_CONFIG", None)

    # Initialize Discord notifier (disabled in offline mode)
    if args.offline_mode:
        logger.info("Offline mode enabled - Discord notifications disabled")
        DiscordNotifier(webhook_url=None)
    else:
        DiscordNotifier()

    # Validate training setup
    if not validate_training_setup(args.data_path, args.checkpoint_dir, args.correlation_id):
        return 1

    if args.dry_run:
        logger.info(f"Dry run: would train with correlation_id {args.correlation_id}")
        logger.info(f"Data path: {args.data_path}")
        logger.info(f"Total timesteps: {args.total_timesteps}")
        logger.info("Setup validation complete")
        return 0

    try:
        # Build training configuration
        config = build_training_config(args, config_overrides=config_overrides)

        if config_overrides:
            logger.info("Applying unified configuration overrides for iterative training")

        logger.info(f"Starting 1M training session: {args.correlation_id}")
        logger.info(f"Data: {args.data_path}")
        logger.info(f"Timesteps: {args.total_timesteps}")

        # Create streaming pipeline if enabled
        streaming_pipeline = _create_streaming_pipeline(
            args.enable_streaming, args.stream_batch_size, logger
        )
        if streaming_pipeline is not None:
            config.setdefault("streaming", {})["enabled"] = True

        # Create trainer
        trainer = PPOTrainer(
            data_path=str(args.data_path) if not args.enable_streaming else None,
            config=config,
            checkpoint_interval=args.checkpoint_interval,
            checkpoint_dir=args.checkpoint_dir,
        )

        # Run training
        trainer.train(session_id=args.correlation_id)

        logger.info(f"Training completed successfully: {args.correlation_id}")
        return 0

    except Exception as e:
        logger.error(f"Training failed: {e}", exc_info=True)
        return 1


if __name__ == "__main__":
    sys.exit(main())