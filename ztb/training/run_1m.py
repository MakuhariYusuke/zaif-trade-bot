#!/usr/bin/env python3
"""
Canonical 1M Training Runner for Zaif Trade Bot.

Runs a 1 million timestep PPO training session with resume capability,
periodic evaluation, and proper artifact management.
"""

import argparse
import logging
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from ztb.trading.ppo_trainer import PPOTrainer


def main():
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
        "--log-dir", default="logs", help="Log directory (default: logs)"
    )
    parser.add_argument(
        "--model-dir", default="models", help="Model directory (default: models)"
    )
    parser.add_argument("--verbose", action="store_true", help="Enable verbose logging")
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

    args = parser.parse_args()

    # Setup logging
    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=log_level, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )

    logger = logging.getLogger(__name__)

    # Validate data path
    data_path = Path(args.data_path)
    if not data_path.exists():
        logger.error(f"Data file not found: {data_path}")
        return 1

    # Check for duplication (resume invariants)
    session_dir = Path(args.checkpoint_dir) / args.correlation_id
    if session_dir.exists():
        logger.warning(f"Session {args.correlation_id} already exists at {session_dir}")
        logger.warning("Use resume functionality or choose a different correlation-id")
        return 1

    if args.dry_run:
        logger.info(f"Dry run: would train with correlation_id {args.correlation_id}")
        logger.info(f"Data path: {data_path}")
        logger.info(f"Total timesteps: {args.total_timesteps}")
        logger.info("Setup validation complete")
        return 0

    try:
        # Training configuration
        config = {
            "total_timesteps": args.total_timesteps,
            "log_dir": args.log_dir,
            "model_dir": args.model_dir,
            "tensorboard_log": args.log_dir,
            "verbose": 1 if args.verbose else 0,
            "seed": 42,
            # PPO hyperparameters (conservative defaults)
            "learning_rate": 3e-4,
            "n_steps": 2048,
            "batch_size": 64,
            "n_epochs": 10,
            "gamma": 0.99,
            "gae_lambda": 0.95,
            "clip_range": 0.2,
            "ent_coef": 0.0,
            "vf_coef": 0.5,
            "max_grad_norm": 0.5,
            # Evaluation settings
            "eval_freq": 10000,
            "n_eval_episodes": 5,
            # Checkpoint settings
            "checkpoint": {
                "keep_last": 5,
                "compress": "zstd",
                "async_save": True,
                "include_optimizer": True,
                "include_replay_buffer": False,
                "include_rng_state": True,
            },
        }

        logger.info(f"Starting 1M training session: {args.correlation_id}")
        logger.info(f"Data: {data_path}")
        logger.info(f"Timesteps: {args.total_timesteps}")

        # Create streaming pipeline if enabled
        streaming_pipeline = None
        if args.enable_streaming:
            logger.info("Enabling streaming pipeline")
            from ztb.data.streaming_pipeline import StreamingPipeline

            streaming_pipeline = StreamingPipeline(
                batch_size=args.stream_batch_size,
                # Add other config as needed
            )

        # Create trainer
        trainer = PPOTrainer(
            data_path=str(data_path) if not args.enable_streaming else None,
            config=config,
            checkpoint_interval=10000,
            checkpoint_dir=args.checkpoint_dir,
            streaming_pipeline=streaming_pipeline,
            stream_batch_size=args.stream_batch_size,
        )

        # Run training
        model = trainer.train(session_id=args.correlation_id)

        logger.info(f"Training completed successfully: {args.correlation_id}")
        return 0

    except Exception as e:
        logger.error(f"Training failed: {e}", exc_info=True)
        return 1


if __name__ == "__main__":
    sys.exit(main())
