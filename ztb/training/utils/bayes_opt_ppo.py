#!/usr/bin/env python3
"""Bayesian optimization for PPO hyperparameters."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Any, cast, Dict

from skopt import gp_minimize
from skopt.space import Integer, Real
from skopt.utils import use_named_args

from ztb.training.config.ppo_config import get_ppo_config
from ztb.utils.file_utils import safe_json_dump
from ztb.utils.logging_utils import get_logger
from ztb.utils.path_utils import ensure_dir
from ztb.training.utils.training_utils import setup_project_path
from ztb.utils.config import ZTBConfig

# Ensure project root is on sys.path
setup_project_path()

from ztb.training.core.ppo_trainer import PPOTrainer  # noqa: E402  # type: ignore[attr-defined]
from ztb.features.curated_features import FeatureSet  # noqa: E402
)

LOGGER = get_logger(__name__)

# Define search space
SEARCH_SPACE = [
    Real(1e-5, 1e-3, name="learning_rate", prior="log-uniform"),
    Integer(128, 1024, name="batch_size"),
    Real(0.1, 0.3, name="clip_range"),
    Real(0.9, 0.99, name="gae_lambda"),
]


def objective_function(
    learning_rate: float,
    batch_size: int,
    clip_range: float,
    gae_lambda: float,
) -> float:
    """Objective function for Bayesian optimization."""
    params = {
        "learning_rate": learning_rate,
        "batch_size": batch_size,
        "clip_range": clip_range,
        "gae_lambda": gae_lambda,
    }
    try:
        # Get base config from common configuration
        base_config = get_ppo_config({
            "total_timesteps": 25000,  # Short training for optimization
            "ent_coef": 0.5,  # Override for optimization
            "tensorboard_log": "logs/bayes_opt",
            "model_dir": ZTBConfig().get_model_path("bayes_opt"),
            "checkpoint_dir": "checkpoints/bayes_opt",
            "log_dir": "logs/bayes_opt",
            "offline_mode": True,
            "feature_set": "full",
            "timeframe": "1m",
            "reward_scaling": 1.0,
            "transaction_cost": 0.0,
            "max_position_size": 1.0,
            "seed": 42,
        })

        # Override with optimized parameters
        config_dict = cast(Dict[str, Any], base_config)
        config_dict.update(params)
        config_dict.update({
            "algorithm": "PPO",
            "data_path": "data/ml-dataset-enhanced-balanced.csv",
            "feature_set": FeatureSet.FULL,
            "timeframe": "M1",
        })
        config = config_dict

        trainer = PPOTrainer(
            data_path=config["data_path"],
            config=config,
            checkpoint_dir=config["checkpoint_dir"],
        )

        model = trainer.train(session_id=f"bayes_opt_{hash(str(params))}")

        # Evaluate performance (use validation reward as objective)
        eval_reward = float(trainer.get_reward_stats().get("mean_reward", -1000))

        # We want to maximize reward, so return negative for minimization
        return -eval_reward

    except Exception as e:
        LOGGER.error(f"Training failed with params {params}: {e}")
        return 1000  # Large penalty for failed runs


@use_named_args(SEARCH_SPACE)  # type: ignore[misc]
def wrapped_objective(**params: Any) -> float:  # type: ignore[misc]
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

    ensure_dir(args.output_dir)

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

    safe_json_dump(results, args.output_dir / "bayes_opt_results.json", indent=2)

    LOGGER.info(f"Optimization completed!")
    LOGGER.info(f"Best parameters: {best_params}")
    LOGGER.info(f"Best score: {results['best_score']}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
