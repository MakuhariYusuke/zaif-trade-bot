#!/usr/bin/env python3
"""
Main entry point for Unified Trainer.
"""

import argparse
import sys

from ztb.training.unified_trainer.trainer import UnifiedTrainer
from ztb.utils.logging_utils import get_logger


def main() -> None:
    """Main entry point for unified training."""
    logger = get_logger(__name__)

    parser = argparse.ArgumentParser(
        description="Unified Training Runner for Zaif Trade Bot"
    )
    parser.add_argument(
        "--config",
        required=True,
        help="Path to configuration JSON file",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force execution without prompts",
    )
    parser.add_argument(
        "--dry-run",
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
        type=int,
        help="Override total_timesteps from config",
    )

    args = parser.parse_args()

    # Load and validate configuration
    try:
        if args.config:
            # Load JSON config directly for training configs
            import json

            with open(args.config, "r") as f:
                raw_config = json.load(f)

            # Process config using ConfigManager to build unified config
            from ztb.training.core.config_manager import ConfigManager

            config_manager = ConfigManager(raw_config)
            config = config_manager.build_unified_config(
                enable_streaming=args.enable_streaming,
                stream_batch_size=args.stream_batch_size,
                total_timesteps_override=args.total_timesteps,
            )
            print(f"DEBUG: Unified config environment: {config.get('environment', {})}")
            print(f"DEBUG: Unified config keys: {list(config.keys())}")
        else:
            # Use ConfigManager for default config loading
            from ztb.config.manager import ConfigManager as GlobalConfigManager

            config_manager = GlobalConfigManager.get_instance()
            config = config_manager.load_config()

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
        total_timesteps=args.total_timesteps,
    )

    success = trainer.run()
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
