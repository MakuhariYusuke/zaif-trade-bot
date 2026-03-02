#!/usr/bin/env python3
"""
CLI utilities for Unified Training.

Handles command-line argument parsing and main entry point.
"""

import argparse
import logging
from typing import Any, cast

from ztb.training.unified_trainer import UnifiedTrainer
from ztb.utils.config_loader import ConfigLoader
from ztb.utils.errors import safe_operation
from ztb.utils.logging_utils import get_logger

logger = get_logger(__name__)

# Check for Stable-Baselines3 availability
import importlib.util  # noqa: E402
STABLE_BASELINES3_AVAILABLE = importlib.util.find_spec("stable_baselines3") is not None

def configure_progress_bar(
    config: dict[str, Any],
    cli_override: bool | None = None,
    log: logging.Logger | None = None,
) -> bool:
    """
    Normalize progress bar settings and coordinate Stable-Baselines3 verbosity.

    Args:
        config: Configuration dict
        cli_override: CLI override for progress bar setting
        log: Logger instance

    Returns:
        Whether progress bar should be enabled
    """
    if log is None:
        log = logger

    # Determine desired progress bar setting
    desired_progress_bar = cli_override
    if desired_progress_bar is None:
        # Check config for progress bar preference
        desired_progress_bar = config.get("progress_bar", True)

    # Apply progress bar setting to config
    config["progress_bar"] = desired_progress_bar

    # Coordinate with Stable-Baselines3 verbosity
    if STABLE_BASELINES3_AVAILABLE:
        # set Stable-Baselines3 verbosity based on progress bar preference
        # When progress bar is enabled, reduce SB3 verbosity to avoid conflicts
        sb3_verbose = 0 if desired_progress_bar else 1
        config.setdefault("sb3_verbose", sb3_verbose)

    return desired_progress_bar

def load_config(config_path: str) -> dict[str, Any] | None:
    """
    Load configuration from file.

    Args:
        config_path: Path to config file

    Returns:
        Configuration dict or None if loading failed
    """
    try:
        loader = ConfigLoader()
        return cast(dict[str, Any] | None, loader.load_config(config_path))
    except Exception as e:
        logger.error(f"Failed to load config from {config_path}: {e}")
        return None

def create_argument_parser() -> argparse.ArgumentParser:
    """
    Create and configure the argument parser.

    Returns:
        Configured argument parser
    """
    parser = argparse.ArgumentParser(
        description="Unified Training CLI for ZTB Trading Bot",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        "config",
        type=str,
        help="Path to configuration file",
    )
    parser.add_argument(
        "--algorithm",
        type=str,
        choices=["ppo", "iterative", "base_ml", "ensemble", "curriculum"],
        help="Override algorithm from config file",
    )
    parser.add_argument(
        "--data-path", type=str, help="Override data path from config file"
    )
    parser.add_argument(
        "--total-timesteps",
        type=int,
        help="Override total timesteps from config file",
    )
    parser.add_argument(
        "--session-id", type=str, help="Override session ID from config file"
    )
    parser.add_argument("--verbose", action="store_true", help="Enable verbose logging")
    parser.add_argument(
        "--log-level",
        type=str,
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        default="INFO",
        help="set logging level (default: INFO). Overrides --verbose flag.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force execution without long-running operation confirmation",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Dry run mode - validate setup without training",
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
        "--max-features",
        type=int,
        default=None,
        help="Maximum number of features to use (default: all features)",
    )
    parser.add_argument(
        "--progress-bar",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Enable or disable the training progress bar (defaults to config or PPO verbose setting)",
    )

    return parser

def main() -> int:
    """
    Main entry point for unified training.
    """
    return cast(
        int,
        safe_operation(
            _main_impl,
            logger=logger,
            context="Unified training execution",
            fallback=1,
        ),
    )

def _main_impl() -> int:
    """
    Implementation of main function.
    """
    parser = create_argument_parser()
    args = parser.parse_args()

    # Setup logging
    # --log-level takes precedence over --verbose
    if hasattr(args, "log_level") and args.log_level:
        log_level = getattr(logging, args.log_level)
    else:
        log_level = logging.DEBUG if args.verbose else logging.INFO

    logging.basicConfig(
        level=log_level, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )

    # Also set the root logger level to suppress third-party DEBUG logs
    logging.getLogger().setLevel(log_level)

    logger = get_logger(__name__)

    # Load configuration
    config = load_config(args.config)
    if config is None:
        raise FileNotFoundError(f"Could not load config from {args.config}")
    logger.info(f"Loaded config from {args.config}")

    # Override config with command line arguments
    if args.algorithm:
        config["algorithm"] = args.algorithm
    if args.data_path:
        config["data_path"] = args.data_path
    if args.total_timesteps:
        config["total_timesteps"] = args.total_timesteps
    if args.session_id:
        config["session_id"] = args.session_id

    cli_progress_preference = getattr(args, "progress_bar", None)
    progress_enabled = configure_progress_bar(
        config, cli_override=cli_progress_preference, log=logger
    )
    logger.info(
        "Progress bar %s (Stable-Baselines3 %s)",
        "enabled" if progress_enabled else "disabled",
        "available" if STABLE_BASELINES3_AVAILABLE else "unavailable",
    )

    logger.info(f"Using algorithm: {config.get('algorithm', 'ppo')}")
    logger.info(f"Session ID: {config.get('session_id', 'default')}")

    # Create and run trainer
    trainer = UnifiedTrainer(
        config,
        args.force,
        args.dry_run,
        args.enable_streaming,
        args.stream_batch_size,
        args.max_features,
    )
    result = trainer.train()

    if result is None:
        logger.warning("Training returned None - may have been cancelled or failed")
    else:
        logger.info("Training completed successfully")
    return 0

if __name__ == "__main__":
    import sys

    sys.exit(main())
