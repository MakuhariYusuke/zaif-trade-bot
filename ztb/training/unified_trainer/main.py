#!/usr/bin/env python3
"""
Main entry point for Unified Trainer.
"""

import argparse
import sys
from pathlib import Path
from typing import Any, Dict, Optional

from ztb.training.unified_trainer.config import load_config
from ztb.training.unified_trainer.trainer import UnifiedTrainer
from ztb.utils.errors import safe_operation


def main() -> None:
    """Main entry point for unified training."""
    safe_operation(
        _main_impl,
        logger=None,  # Will be configured inside
        context="Unified training execution"
    )


def _main_impl(logger) -> None:
    """Implementation of main function."""
    parser = argparse.ArgumentParser(description="Unified Training Runner for Zaif Trade Bot")
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

    # Load configuration
    config = load_config(args.config)
    if config is None:
        logger.error(f"Failed to load configuration from {args.config}")
        sys.exit(1)

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