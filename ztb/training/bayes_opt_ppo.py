#!/usr/bin/env python3
"""Bayesian optimization for PPO hyperparameters."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
from skopt import gp_minimize
from skopt.space import Integer, Real
from skopt.utils import use_named_args

from ztb.utils.logging_utils import get_logger

# Ensure project root is on sys.path
PROJECT_ROOT = Path(__file__).resolve().parents[2]
import sys

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ztb.training.ppo_trainer import PPOTrainer  # noqa: E402

LOGGER = get_logger(__name__)

# Define search space
SEARCH_SPACE = [
    Real(1e-5, 1e-3, name="learning_rate", prior="log-uniform"),
    Integer(128, 1024, name="batch_size"),
    Real(0.1, 0.3, name="clip_range"),
    Real(0.9, 0.99, name="gae_lambda"),
]


def objective_function(**params: Any) -> float:
    """Objective function for Bayesian optimization."""
    try:
        # Fixed parameters
        config = {
            "algorithm": "ppo",
            "data_path": "data/ml-dataset-enhanced-balanced.csv",
            "total_timesteps": 25000,  # Short training for optimization
            "n_steps": 2048,
            "gamma": 0.99,
            "vf_coef": 0.5,
            "max_grad_norm": 0.5,
            "ent_coef": 0.5,
            "tensorboard_log": "logs/bayes_opt",
            "model_dir": "models/bayes_opt",
            "checkpoint_dir": "checkpoints/bayes_opt",
            "log_dir": "logs/bayes_opt",
            "offline_mode": True,
            "feature_set": "full",
            "timeframe": "1m",
            "reward_scaling": 1.0,
            "transaction_cost": 0.0,
            "max_position_size": 1.0,
            "seed": 42,
            **params,  # Override with optimized parameters
        }

        trainer = PPOTrainer(
            data_path=config["data_path"],
            config=config,
            checkpoint_dir=config["checkpoint_dir"],
        )

        model = trainer.train(session_id=f"bayes_opt_{hash(str(params))}")

        # Evaluate performance (use validation reward as objective)
        eval_reward = trainer.get_reward_stats().get("mean_reward", -1000)

        # We want to maximize reward, so return negative for minimization
        return -eval_reward

    except Exception as e:
        LOGGER.error(f"Training failed with params {params}: {e}")
        return 1000  # Large penalty for failed runs


@use_named_args(SEARCH_SPACE)
def wrapped_objective(**params: Any) -> float:
    """Wrapped objective function for skopt."""
    return objective_function(**params)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Bayesian optimization for PPO hyperparameters"
    )
    parser.add_argument(
        "--n-calls", type=int, default=20, help="Number of optimization calls"
    )
    parser.add_argument(
        "--n-random-starts", type=int, default=5, help="Number of random starts"
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/bayes_opt"),
        help="Output directory",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)

    LOGGER.info("Starting Bayesian optimization...")
    LOGGER.info(f"Search space: {[dim.name for dim in SEARCH_SPACE]}")
    LOGGER.info(f"Number of calls: {args.n_calls}")

    # Run optimization
    result = gp_minimize(
        func=wrapped_objective,
        dimensions=SEARCH_SPACE,
        n_calls=args.n_calls,
        n_random_starts=args.n_random_starts,
        random_state=42,
        verbose=True,
    )

    # Save results
    best_params = {dim.name: value for dim, value in zip(SEARCH_SPACE, result.x)}

    results = {
        "best_params": best_params,
        "best_score": -result.fun,  # Convert back to reward
        "all_scores": [-score for score in result.func_vals],
        "all_params": [
            {dim.name: value for dim, value in zip(SEARCH_SPACE, params)}
            for params in result.x_iters
        ],
    }

    with open(args.output_dir / "bayes_opt_results.json", "w") as f:
        json.dump(results, f, indent=2)

    LOGGER.info(f"Optimization completed!")
    LOGGER.info(f"Best parameters: {best_params}")
    LOGGER.info(f"Best score: {results['best_score']}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
