#!/usr/bin/env python3
"""Train multiple PPO ensemble models with predefined hyperparameters."""

from __future__ import annotations

import argparse
import logging
import sys
from copy import deepcopy
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

# Ensure project root is on sys.path
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ztb.trading.ppo_trainer import (  # noqa: E402  pylint: disable=wrong-import-position
    PPOTrainer,
)

LOGGER = logging.getLogger(__name__)

MODEL_SPECS = [
    # Randomly sampled ent_coef with diversity
    {
        "session_id": "ensemble_model_1",
        "ent_coef": 0.35,  # Random sample from [0.1, 0.9]
        "reward_profit_bonus_multipliers": [1.1, 1.15, 0.8],
    },
    {
        "session_id": "ensemble_model_2",
        "ent_coef": 0.67,  # Random sample from [0.1, 0.9]
        "reward_profit_bonus_multipliers": [0.9, 1.1, 1.0],
    },
    {
        "session_id": "ensemble_model_3",
        "ent_coef": 0.82,  # Random sample from [0.1, 0.9]
        "reward_profit_bonus_multipliers": [1.0, 1.0, 1.0],
    },
    {
        "session_id": "ensemble_model_4",
        "ent_coef": 0.23,  # Random sample from [0.1, 0.9]
        "reward_profit_bonus_multipliers": [1.2, 0.95, 0.85],
    },
    {
        "session_id": "ensemble_model_5",
        "ent_coef": 0.91,  # Random sample from [0.1, 0.9]
        "reward_profit_bonus_multipliers": [0.85, 1.05, 1.1],
    },
]


def build_base_config(args: argparse.Namespace) -> Dict[str, Any]:
    """Return a baseline PPO configuration shared by ensemble members."""
    return {
        "algorithm": "ppo",
        "data_path": str(args.data_path),
        "total_timesteps": args.total_timesteps,
        "learning_rate": args.learning_rate,
        "n_steps": args.n_steps,
        "batch_size": args.batch_size,
        "gamma": args.gamma,
        "gae_lambda": args.gae_lambda,
        "clip_range": args.clip_range,
        "vf_coef": args.vf_coef,
        "max_grad_norm": args.max_grad_norm,
        "tensorboard_log": str(args.logs_dir),
        "model_dir": str(args.models_dir),
        "checkpoint_dir": str(args.checkpoints_dir),
        "log_dir": str(args.logs_dir),
        "offline_mode": True,
        "feature_set": args.feature_set,
        "timeframe": args.timeframe,
        "reward_scaling": args.reward_scaling,
        "transaction_cost": args.transaction_cost,
        "max_position_size": args.max_position_size,
        "seed": args.seed,
    }


def train_model(
    base_config: Dict[str, Any], spec: Dict[str, Any], args: argparse.Namespace
) -> Path:
    """Train a single model according to *spec* and save it to disk."""
    config = deepcopy(base_config)
    config.update(
        {
            "session_id": spec["session_id"],
            "correlation_id": spec["session_id"],
            "ent_coef": spec["ent_coef"],
            "reward_profit_bonus_multipliers": spec["reward_profit_bonus_multipliers"],
            "tensorboard_log": str(args.logs_dir / spec["session_id"]),
        }
    )

    args.logs_dir.mkdir(parents=True, exist_ok=True)
    args.models_dir.mkdir(parents=True, exist_ok=True)
    args.checkpoints_dir.mkdir(parents=True, exist_ok=True)

    LOGGER.info("Starting training for %s", spec["session_id"])
    start_time = datetime.now()

    trainer = PPOTrainer(
        data_path=str(args.data_path),
        config=config,
        checkpoint_dir=str(args.checkpoints_dir),
        max_features=args.max_features,
    )
    model = trainer.train(session_id=spec["session_id"])

    duration = datetime.now() - start_time
    LOGGER.info("Finished training %s in %s", spec["session_id"], duration)

    target_path = args.models_dir / f"{spec['session_id']}.zip"
    if model is not None:
        model.save(str(target_path))
        LOGGER.info("Saved model to %s", target_path)
    else:
        LOGGER.warning(
            "Trainer returned None for %s; no model saved", spec["session_id"]
        )

    return target_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train PPO ensemble models")
    parser.add_argument(
        "--data-path",
        type=Path,
        default=Path("ml-dataset-enhanced-balanced.csv"),
        help="Path to the balanced training dataset",
    )
    parser.add_argument(
        "--models-dir",
        type=Path,
        default=Path("models"),
        help="Directory to store trained ensemble models",
    )
    parser.add_argument(
        "--logs-dir",
        type=Path,
        default=Path("logs/ensemble"),
        help="Directory for training logs and tensorboard data",
    )
    parser.add_argument(
        "--checkpoints-dir",
        type=Path,
        default=Path("checkpoints/ensemble"),
        help="Directory for intermediate checkpoints",
    )
    parser.add_argument(
        "--total-timesteps",
        type=int,
        default=200_000,
        help="Total timesteps to train each ensemble member",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=3e-4,
        help="PPO learning rate",
    )
    parser.add_argument(
        "--n-steps",
        type=int,
        default=2048,
        help="Number of steps to run for each environment update",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=256,
        help="Mini-batch size for PPO updates",
    )
    parser.add_argument(
        "--gamma",
        type=float,
        default=0.99,
        help="Discount factor",
    )
    parser.add_argument(
        "--gae-lambda",
        type=float,
        default=0.95,
        help="GAE lambda",
    )
    parser.add_argument(
        "--clip-range",
        type=float,
        default=0.2,
        help="Clipping range for PPO",
    )
    parser.add_argument(
        "--vf-coef",
        type=float,
        default=0.5,
        help="Value function coefficient",
    )
    parser.add_argument(
        "--max-grad-norm",
        type=float,
        default=0.5,
        help="Maximum gradient norm",
    )
    parser.add_argument(
        "--feature-set",
        default="full",
        help="Feature set to use for the environment",
    )
    parser.add_argument(
        "--timeframe",
        default="1m",
        help="Timeframe for the trading environment",
    )
    parser.add_argument(
        "--reward-scaling",
        type=float,
        default=1.0,
        help="Reward scaling factor",
    )
    parser.add_argument(
        "--transaction-cost",
        type=float,
        default=0.0,
        help="Transaction cost used during training",
    )
    parser.add_argument(
        "--max-position-size",
        type=float,
        default=1.0,
        help="Maximum position size",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility",
    )
    parser.add_argument(
        "--max-features",
        type=int,
        default=None,
        help="Optional cap on the number of features",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logging",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    )

    base_config = build_base_config(args)

    trained_models: List[Path] = []
    for spec in MODEL_SPECS:
        trained_models.append(train_model(base_config, spec, args))

    LOGGER.info("Training completed for %d ensemble members", len(trained_models))
    LOGGER.info("Models saved: %s", ", ".join(str(p) for p in trained_models))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
