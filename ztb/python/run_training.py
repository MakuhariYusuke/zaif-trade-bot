#!/usr/bin/env python3
"""
Unified Training Runner - Single entry point for all training operations.

This script provides a clean interface to the UnifiedTrainer, supporting:
- PPO training with optimized hyperparameters
- SELL bias mitigation with Lagrange constraints
- Curriculum learning
- Ensemble training
- Easy configuration through JSON files

Usage:
    python run_training.py --config configs/training/ppo_100k_optimized.json
    python run_training.py --config configs/training/curriculum_learning.json
    python run_training.py --config configs/training/ensemble.json --algorithm ensemble

Examples:
    # Quick 100k training with optimized parameters
    python run_training.py --config configs/training/ppo_100k_optimized.json

    # Full 1M training
    python run_training.py --config configs/training/ppo_1m_optimized.json

    # Quick validation run (override timesteps)
    python run_training.py --config configs/training/ppo_balanced_mem_optimized.json --timesteps 10000

    # Curriculum learning for extreme bias scenarios
    python run_training.py --config configs/training/curriculum.json --algorithm curriculum

    # Dry run to validate configuration
    python run_training.py --config configs/training/ppo_100k_optimized.json --dry-run
"""

import argparse
import importlib.util
import logging
import sys
from pathlib import Path
from typing import Any, Dict, Optional

import yaml

from ztb.utils.logging_utils import setup_logging

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from ztb.training.unified_trainer import UnifiedTrainer
from ztb.utils.logging_utils import get_logger

logger = get_logger(__name__)
STABLE_BASELINES3_AVAILABLE = importlib.util.find_spec("stable_baselines3") is not None


def load_config(config_path: str) -> Dict[str, Any]:
    """Load configuration from JSON file."""
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")

    with open(path, "r") as f:
        config = yaml.safe_load(f)

    logger.info(f"Loaded configuration from {config_path}")
    logger.info(f"  Algorithm: {config.get('algorithm', 'ppo')}")
    logger.info(f"  Session ID: {config.get('session_id', 'N/A')}")

    # Format timesteps with proper handling of None/N/A
    total_timesteps = config.get("total_timesteps")
    if total_timesteps is not None:
        logger.info(f"  Total timesteps: {total_timesteps:,}")
    else:
        logger.info("  Total timesteps: N/A")

    # Log key optimizations if present
    if config.get("enable_sell_mitigation"):
        logger.info("  SELL bias mitigation: ENABLED")
        if config.get("enable_lagrange"):
            logger.info(
                f"    Lagrange r_target: {config.get('lagrange_r_target', 'N/A')}"
            )
            logger.info(
                f"    Lagrange tolerance: {config.get('lagrange_tolerance', 'N/A')}"
            )
            logger.info(f"    Lagrange eta: {config.get('lagrange_eta', 'N/A')}")

    return config


def setup_progress_bar(
    config: Dict[str, Any], cli_override: Optional[bool] = None
) -> bool:
    """
    Configure progress bar usage based on Stable-Baselines3 availability.

    Args:
        config: Loaded training configuration dictionary (modified in-place).
        cli_override: Optional override from CLI flags (True/False).

    Returns:
        bool: True if any progress display should be enabled, False otherwise.
    """
    if config.get("_progress_configured"):
        return bool(config.get("progress_bar", False))

    progress_preference: Optional[bool] = cli_override

    # Preserve backward compatibility with legacy config keys
    legacy_top_level = config.pop("progress_bar", None)
    training_section = config.get("training")
    legacy_training = None
    if isinstance(training_section, dict):
        legacy_training = training_section.pop("progress_bar", None)

    if progress_preference is None and legacy_top_level is not None:
        progress_preference = bool(legacy_top_level)
    if progress_preference is None and legacy_training is not None:
        progress_preference = bool(legacy_training)

    ppo_config = config.setdefault("ppo", {})
    if not isinstance(ppo_config, dict):
        logger.warning(
            "PPO configuration expected to be a dict, but received %s. "
            "Skipping progress bar configuration.",
            type(ppo_config),
        )
        return False

    if progress_preference is None:
        progress_preference = bool(ppo_config.get("verbose", 0))

    use_progress_bar = bool(progress_preference)

    if STABLE_BASELINES3_AVAILABLE:
        desired_verbose = 1 if use_progress_bar else 0
        current_verbose = ppo_config.get("verbose")
        if current_verbose != desired_verbose:
            logger.info(
                "Stable-Baselines3 detected; adjusting PPO verbose to %s for progress control.",
                desired_verbose,
            )
        ppo_config["verbose"] = desired_verbose
        config["progress_bar"] = use_progress_bar
        logger.debug(
            "Configured Stable-Baselines3 progress: enabled=%s, verbose=%s",
            use_progress_bar,
            desired_verbose,
        )
    else:
        logger.info(
            "Stable-Baselines3 not available; %s fallback progress bar.",
            "enabling" if use_progress_bar else "disabling",
        )
        config["progress_bar"] = use_progress_bar
        if not use_progress_bar:
            ppo_config["verbose"] = 0

    config["_progress_configured"] = True
    return use_progress_bar


def main():
    parser = argparse.ArgumentParser(
        description="Unified Training Runner for Zaif Trade Bot",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    parser.add_argument(
        "--config", required=True, help="Path to configuration JSON file"
    )

    parser.add_argument(
        "--algorithm",
        choices=["ppo", "base_ml", "iterative", "ensemble", "curriculum"],
        help="Override algorithm from config file",
    )

    parser.add_argument(
        "--timesteps",
        type=int,
        help="Override total_timesteps from config (useful for quick validation runs)",
    )

    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate configuration without running training",
    )

    parser.add_argument(
        "--force",
        action="store_true",
        help="Force execution without confirmation prompts",
    )

    parser.add_argument("--verbose", action="store_true", help="Enable verbose logging")

    parser.add_argument(
        "--progress-bar",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Enable or disable the training progress bar (defaults to config or PPO verbose setting)",
    )

    args = parser.parse_args()

    # Setup logging with rotation (Bug #40 fix)
    log_level = logging.DEBUG if args.verbose else logging.INFO
    log_dir = Path(args.log_dir) if hasattr(args, "log_dir") else Path("logs")
    log_dir.mkdir(exist_ok=True)

    setup_logging(
        level=log_level,
        log_file=str(log_dir / "training_log.txt"),
        max_bytes=10 * 1024 * 1024,  # 10MB
        backup_count=5,
    )

    try:
        # Load configuration
        config = load_config(args.config)

        # Override algorithm if specified
        if args.algorithm:
            config["algorithm"] = args.algorithm
            logger.info(f"Algorithm overridden to: {args.algorithm}")

        # Normalize progress bar configuration before trainer initialization
        cli_progress = getattr(args, "progress_bar", None)
        setup_progress_bar(config, cli_override=cli_progress)

        # Dry run mode - delegate to UnifiedTrainer
        if args.dry_run:
            trainer = UnifiedTrainer(
                config=config,
                force=args.force,
                dry_run=True,
                total_timesteps=args.timesteps,
            )
            logger.info("=" * 80)
            logger.info("DRY RUN MODE - Configuration validated successfully")
            logger.info("=" * 80)
            return 0

        # Initialize trainer (all validation logic is in UnifiedTrainer)
        logger.info("Initializing Unified Trainer...")
        trainer = UnifiedTrainer(
            config=config,
            force=args.force,
            dry_run=False,
            total_timesteps=args.timesteps,
        )

        # Execute training
        logger.info("=" * 80)
        logger.info("STARTING TRAINING")
        logger.info("=" * 80)

        model = trainer.train()

        logger.info("=" * 80)
        logger.info("TRAINING COMPLETED SUCCESSFULLY")
        logger.info("=" * 80)

        if model is not None:
            session_id = config.get("session_id", "model")
            model_dir = config.get("model_dir", "models")
            checkpoint_dir = config.get("checkpoint_dir", "checkpoints")
            logger.info(f"Model saved to: {model_dir}/{session_id}.zip")
            logger.info(f"Checkpoints saved to: {checkpoint_dir}")
            logger.info(f"TensorBoard logs: {checkpoint_dir}")

        return 0

    except KeyboardInterrupt:
        logger.warning("\nTraining interrupted by user (Ctrl+C)")
        return 130

    except Exception as e:
        logger.error(f"Training failed with error: {e}", exc_info=True)
        return 1


if __name__ == "__main__":
    sys.exit(main())
