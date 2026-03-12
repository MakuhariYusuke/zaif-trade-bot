#!/usr/bin/env python3
"""
SAC Training Script with Continuous Actions

Trains SAC model with continuous action space to eliminate SELL bias.
Continuous actions range from -1 (SELL) to +1 (BUY), converted to discrete actions.
"""

import json
import logging
import sys
import time
from pathlib import Path
from typing import Any, Dict

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

import numpy as np
import pandas as pd
from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.monitor import Monitor

from ztb.trading.environment.heavy_env.core import HeavyTradingEnv
from ztb.trading.environment.utils.config import EnvironmentConfig
from ztb.utils.logging_utils import get_logger




class TrainingProgressCallback(BaseCallback):
    """Callback for monitoring training progress and action distribution."""

    def __init__(self, check_freq: int = 1000, verbose: int = 1):
        super().__init__(verbose)
        self.check_freq = check_freq
        self.continuous_actions = []  # Store continuous actions
        self.discrete_actions = []  # Store converted discrete actions
        self.reward_history = []
        self.episode_rewards = []
        self._logger = get_logger(__name__)

    def _on_step(self) -> bool:
        # Record continuous action taken
        if hasattr(self.locals, "actions"):
            continuous_action = self.locals["actions"][0]
            if isinstance(continuous_action, np.ndarray):
                continuous_action = continuous_action.item()
            self.continuous_actions.append(continuous_action)


    # Override curriculum stage to forced_balance for initial training
    env_config_dict["curriculum_stage"] = "forced_balance"
    print("DEBUG: EnvironmentConfig created")

    env = HeavyTradingEnv(df=pd.read_csv("btc_jpy_real_dataset.csv"), config=env_config)
    print("DEBUG: HeavyTradingEnv created, action_space:", type(env.action_space))

    # Wrap with monitor for logging
    env = Monitor(env)

    return env
        gradient_steps=1,
        ent_coef="auto",  # Auto-tune entropy
        target_update_interval=1,
        target_entropy="auto",
        verbose=1,
    )

    # Create callback for monitoring
    callback = TrainingProgressCallback(check_freq=log_interval)

    logger.info("Starting SAC training with corrected reward function")
    logger.info(f"Total timesteps: {total_timesteps}")

    start_time = time.time()

    # Train the model
    model.learn(total_timesteps=total_timesteps, callback=callback, progress_bar=True)

    training_time = time.time() - start_time
    logger.info(".2f")

    # Save the trained model
    model.save(model_save_path)
            episode_reward += reward
            continuous_actions.append(action_continuous)
            discrete_actions.append(discrete_action)

            done = terminated or truncated
            step += 1

        episode_rewards.append(episode_reward)
        logger.info(f"Validation Episode {episode + 1}: Reward = {episode_reward:.2f}")

    # Calculate action distribution from discrete actions
    total_actions = len(discrete_actions)
        # Create environment
        env = create_environment(config_path)

        # Train model
        model = train_sac_model(
            env=env, total_timesteps=total_timesteps, model_save_path=model_save_path
        )

        # Validate training
        validation_results = validate_training(model, env, n_episodes=5)

        # Save validation results
        with open(
            "results/sac_continuous_training_validation.json", "w", encoding="utf-8"
        ) as f:
            json.dump(validation_results, f, indent=2, ensure_ascii=False)

        print("\n" + "=" * 60)
        print("SAC CONTINUOUS ACTION TRAINING COMPLETED")
        print("=" * 60)
        print(f"Model saved: {model_save_path}")
        print(".2f")
        print(".1%")
        print(".1%")
        print(".1%")
        print(".3f")
        print("=" * 60)

    except Exception as e:
        logging.error(f"Training failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
