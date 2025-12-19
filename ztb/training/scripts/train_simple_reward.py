#!/usr/bin/env python3
# ruff: noqa: E402
"""
Simple reward function training test
"""

import logging
import time
from typing import Dict, List

from ztb.utils.logging_utils import setup_logging
from ztb.utils.training_utils import display_training_complete

setup_logging()
logger = logging.getLogger(__name__)

# Type alias for configuration
ConfigType = Dict[str, str | float]

from ztb.trading.env_config import get_trading_env_config
from ztb.trading.environment import HeavyTradingEnv
from ztb.training.callbacks.callbacks import SimpleTrainingCallback
from ztb.training.utils.training_utils import (
    create_ppo_model,
    load_training_data,
    print_training_results,
    print_training_start,
    save_model_with_path,
    display_training_complete,
    setup_project_path,
)


def train_simple_reward(
    config_name: str = "default",
    reward_scaling: float = 1.0,
    entropy_coef: float = 0.01,
    learning_rate: float = 3e-4,
) -> str:
    """Train with simple portfolio reward for 100k steps with configurable parameters"""

    # Setup project path
    setup_project_path()

    # Load data
    df = load_training_data()

    # Create environment with simple reward
    env_config = dict(get_trading_env_config())
    env_config.update(
        {
            "reward_scaling": reward_scaling,
            "curriculum_stage": "simple_portfolio",  # Use simple reward
        }
    )

    env = HeavyTradingEnv(
        df=df,
        config=env_config,
        streaming_pipeline=None,
        stream_batch_size=1000,
        max_features=68,
    )

    # Create PPO model with custom config
    model_config = {
        "learning_rate": learning_rate,
        "ent_coef": entropy_coef,
    }
    model = create_ppo_model(env, model_config)

    # Create callback
    callback = SimpleTrainingCallback()

    print_training_start(config_name, reward_scaling, entropy_coef, learning_rate)

    # Train for 100k steps
    start_time = time.time()
    model.learn(total_timesteps=100000, callback=callback)
    training_time = time.time() - start_time

    print_training_results(callback.episode_rewards)

    # Save model with config name
    model_path = save_model_with_path(model, f"aggressive_{config_name}")
    print(f"\nModel saved to: {model_path}")

    env.close()
    return model_path


if __name__ == "__main__":
    # Define fine-tuned reward scaling configurations
    configs: List[ConfigType] = [
        # Fine-tune reward_scaling around optimal value (7.5)
        {
            "name": "reward_scale_6_0",
            "reward_scaling": 6.0,
            "entropy_coef": 0.03,
            "learning_rate": 1e-3,
        },
        {
            "name": "reward_scale_6_5",
            "reward_scaling": 6.5,
            "entropy_coef": 0.03,
            "learning_rate": 1e-3,
        },
        {
            "name": "reward_scale_7_0",
            "reward_scaling": 7.0,
            "entropy_coef": 0.03,
            "learning_rate": 1e-3,
        },
        {
            "name": "reward_scale_7_5",
            "reward_scaling": 7.5,
            "entropy_coef": 0.03,
            "learning_rate": 1e-3,
        },
        {
            "name": "reward_scale_8_0",
            "reward_scaling": 8.0,
            "entropy_coef": 0.03,
            "learning_rate": 1e-3,
        },
        {
            "name": "reward_scale_8_5",
            "reward_scaling": 8.5,
            "entropy_coef": 0.03,
            "learning_rate": 1e-3,
        },
        {
            "name": "reward_scale_9_0",
            "reward_scaling": 9.0,
            "entropy_coef": 0.03,
            "learning_rate": 1e-3,
        },
    ]

    trained_models = []
    start_time = time.time()

    for config in configs:
        print(f"\n{'='*60}")
        print(f"Training configuration: {config['name']}")
        print(f"{'='*60}")

        try:
            model_path = train_simple_reward(
                config_name=config["name"],  # type: ignore
                reward_scaling=config["reward_scaling"],  # type: ignore
                entropy_coef=config["entropy_coef"],  # type: ignore
                learning_rate=config["learning_rate"],  # type: ignore
            )
            trained_models.append((config["name"], model_path))
            logger.info(f"✅ Successfully trained: {config['name']}")

        except Exception as e:
            logger.error(f"❌ Failed to train {config['name']}: {e}")
            continue

    training_time = time.time() - start_time
    final_metrics = {
        "models_trained": len(trained_models),
        "successful_configs": [name for name, _ in trained_models],
        "failed_configs": [config["name"] for config in configs if config["name"] not in [name for name, _ in trained_models]]
    }
    display_training_complete(final_metrics, training_time)
