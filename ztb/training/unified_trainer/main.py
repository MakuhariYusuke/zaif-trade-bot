#!/usr/bin/env python3
"""
Main entry point for Unified Trainer.
"""

import argparse
import sys
import logging
import warnings

from ztb.training.unified_trainer.trainer import UnifiedTrainer
from ztb.utils.logging_utils import get_logger, setup_logging

# Suppress common warnings
warnings.filterwarnings("ignore", category=RuntimeWarning, module="pandas")
warnings.filterwarnings("ignore", category=RuntimeWarning, module="numpy")
warnings.filterwarnings("ignore", category=UserWarning, module="pandas")
warnings.filterwarnings("ignore", category=UserWarning, module="numpy")


def main() -> None:
    """Main entry point for unified training."""
    # Parse command line arguments first to get log level
    parser = argparse.ArgumentParser(
        description="Unified Training Runner for Zaif Trade Bot"
    )
    parser.add_argument(
        "--config",
        "-c",
        type=str,
        required=True,
        help="Path to configuration JSON file",
    )
    parser.add_argument(
        "--force",
        "-f",
        action="store_true",
        help="Force execution without prompts",
    )
    parser.add_argument(
        "--dry-run",
        "-d",
        action="store_true",
        help="Validate configuration without executing training",
    )
    parser.add_argument(
        "--enable-streaming",
        action="store_true",
        help="Enable streaming data pipeline",
    )
    parser.add_argument(
        "--stream-batch-size",
        "-b",
        type=int,
        default=256,
        help="Batch size for streaming (default: 256)",
    )
    parser.add_argument(
        "--max-features",
        type=int,
        help="Maximum number of features to use",
    )
    parser.add_argument(
        "--total-timesteps",
        "--timesteps",
        "-s",
        type=int,
        help="Override total_timesteps from config",
    )
    parser.add_argument(
        "--resume",
        "-r",
        action="store_true",
        help="Resume training from the latest checkpoint",
    )
    parser.add_argument(
        "--log-level",
        "-l",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Override logging level from config (DEBUG, INFO, WARNING, ERROR)",
    )

    args = parser.parse_args()

    # Set up logging based on command line argument
    log_level_map = {
        "DEBUG": logging.DEBUG,
        "INFO": logging.INFO,
        "WARNING": logging.WARNING,
        "ERROR": logging.ERROR,
    }
    log_level = log_level_map.get(args.log_level, logging.INFO)
    setup_logging(log_level)
    logger = get_logger(__name__)
    logger.setLevel(log_level)

    # Load and validate configuration
    try:
        if args.config:
            # Load JSON config directly for training configs
            import json

            with open(args.config, "r") as f:
                raw_config = json.load(f)

            # Process config using ConfigManager to build unified config
            from ztb.training.core.config_manager import ConfigManager

            # Determine timesteps override (prefer --total-timesteps over --timesteps)
            timesteps_override = args.total_timesteps

            config_manager = ConfigManager(raw_config)
            config = config_manager.build_unified_config(
                enable_streaming=args.enable_streaming,
                stream_batch_size=args.stream_batch_size,
                total_timesteps_override=timesteps_override,
            )
        else:
            print("DEBUG: Using GlobalConfigManager")
            # Use ConfigManager for default config loading
            from ztb.config.manager import ConfigManager as GlobalConfigManager

            config_manager = GlobalConfigManager.get_instance()
            config = config_manager.load_config()
            timesteps_override = args.total_timesteps

    except Exception as e:
        print(f"❌ Failed to load configuration: {e}")
        sys.exit(1)

    print("✅ Configuration loaded successfully")

    # Create and run trainer
    trainer = UnifiedTrainer(
        config=config,
        force=args.force,
        dry_run=args.dry_run,
        enable_streaming=args.enable_streaming,
        stream_batch_size=args.stream_batch_size,
        max_features=args.max_features,
        total_timesteps=timesteps_override,
        resume=args.resume,
    )

    success = trainer.run()
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
