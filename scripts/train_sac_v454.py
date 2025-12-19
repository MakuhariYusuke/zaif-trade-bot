#!/usr/bin/env python3
"""
SAC v454 Training Script - Inverse Confidence Paradox Resolution

Enhanced SAC training with v454 features (Noise Index, Trend Deviation, Volatility EMA).
"""

import argparse
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from ztb.training.v4xx_unified_trainer import V4XXUnifiedTrainer
from ztb.utils.logging_utils import get_logger

logger = get_logger(__name__)


def main() -> bool:
    """Main training function using unified trainer."""
    parser = argparse.ArgumentParser(
        description="SAC v454 Training - Inverse Confidence Paradox Resolution"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="config/v454/sac_v454_config.json",
        help="Path to configuration file",
    )
    parser.add_argument(
        "--version", type=str, default="v454", help="Override version detection"
    )
    group = parser.add_mutually_exclusive_group()
    group.add_argument(
        "--episodes",
        type=int,
        help=(
            "Number of episodes to run (overrides config). "
            "If provided, total_timesteps = episodes * max_episode_length"
        ),
    )
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

    args = parser.parse_args()

    try:
        print("🚀 SAC v454 Training - Inverse Confidence Paradox Resolution")
        print(f"Configuration: {args.config}")

        # Initialize unified trainer
        trainer = V4XXUnifiedTrainer(config_path=args.config, version=args.version)

        # Optional CLI overrides (match `ztb/training/v4xx_unified_trainer.py` behavior)
        if not isinstance(trainer.config, dict):
            raise TypeError("Unified trainer config is not a dict")

        training_cfg = trainer.config.setdefault("training", {})
        if not isinstance(training_cfg, dict):
            trainer.config["training"] = {}
            training_cfg = trainer.config["training"]

        if args.reset_num_timesteps:
            # Ensure we don't accidentally resume or warm-start when the user wants a clean run.
            for key in (
                "resume_from",
                "init_model_path",
                "initial_model_path",
                "pretrained_model_path",
            ):
                training_cfg.pop(key, None)
            training_cfg["reset_num_timesteps"] = True

        if args.episodes is not None:
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

            training_cfg["episodes"] = int(args.episodes)
            training_cfg["total_timesteps"] = int(args.episodes) * int(max_ep)
            logger.info(
                "Overriding config: episodes=%s, computed total_timesteps=%s (max_episode_length=%s)",
                args.episodes,
                training_cfg["total_timesteps"],
                max_ep,
            )
        elif args.total_timesteps is not None:
            training_cfg["total_timesteps"] = int(args.total_timesteps)
            logger.info(
                "Overriding config: total_timesteps=%s", training_cfg["total_timesteps"]
            )

        # Validate configuration
        if not trainer.validate_config():
            logger.error("Configuration validation failed")
            return False

        # Start training
        success = trainer.train()
        if success:
            print("✅ SAC v454 training completed successfully!")
            return True
        else:
            print("❌ SAC v454 training failed!")
            return False

    except Exception as e:
        logger.error(f"Training failed with error: {e}")
        print(f"❌ Training failed: {e}")
        # Print stack trace for debugging
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
