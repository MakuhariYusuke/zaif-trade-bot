#!/usr/bin/env python3
"""
Common training utilities for reducing code duplication across training scripts
"""

import sys
from pathlib import Path
import numpy as np
from typing import Dict, Any, Optional, Tuple
import pandas as pd
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv

from ztb.trading.environment.environment import HeavyTradingEnv
from ztb.training.config.ppo_config import get_ppo_config, PPOConfig


def setup_project_path() -> Path:
    """Add project root to Python path and return project root path"""
    project_root = Path(__file__).parent.parent.parent  # ztb/training -> ztb -> project_root
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))
    return project_root


def create_trading_env(
    df: pd.DataFrame,
    config: Dict[str, Any],
    vec_env: bool = True
) -> HeavyTradingEnv | DummyVecEnv:
    """Create trading environment with common configuration"""
    env = HeavyTradingEnv(df=df, config=config)
    if vec_env:
        env = DummyVecEnv([lambda: env])  # type: ignore[assignment]
    return env


def create_ppo_model(
    env: Any,
    config_override: Optional[Dict[str, Any]] = None,
    tensorboard_log: str = "./tensorboard"
) -> PPO:
    """Create PPO model with common configuration"""
    ppo_config: PPOConfig = get_ppo_config()

    # Apply overrides if provided
    if config_override:
        for key, value in config_override.items():
            ppo_config[key] = value  # type: ignore[literal-required]

    model = PPO(
        "MlpPolicy",
        env,
        learning_rate=ppo_config.get("learning_rate", 3e-4),
        n_steps=ppo_config.get("n_steps", 2048),
        batch_size=ppo_config.get("batch_size", 64),
        n_epochs=ppo_config.get("n_epochs", 10),
        gamma=ppo_config.get("gamma", 0.99),
        gae_lambda=ppo_config.get("gae_lambda", 0.95),
        clip_range=ppo_config.get("clip_range", 0.2),
        ent_coef=ppo_config.get("ent_coef", 0.01),
        vf_coef=ppo_config.get("vf_coef", 0.5),
        max_grad_norm=ppo_config.get("max_grad_norm", 0.5),
        verbose=int(ppo_config.get("verbose", 1) or 1),
        tensorboard_log=tensorboard_log,
    )

    return model


def save_model_with_path(model: PPO, model_name: str, base_dir: str = "models") -> str:
    """Save model to a standardized path and return the path"""
    from pathlib import Path
    model_path = Path(base_dir) / f"{model_name}.zip"
    model_path.parent.mkdir(exist_ok=True)
    model.save(str(model_path))
    return str(model_path)


def evaluate_model(
    model: PPO,
    env: Any,
    max_steps: int = 1000,
    deterministic: bool = True
) -> Tuple[float, int]:
    """Evaluate a trained model and return episode reward and step count"""
    obs = env.reset()
    episode_reward = 0
    step_count = 0
    done = False

    while not done and step_count < max_steps:
        action, _ = model.predict(obs, deterministic=deterministic)
        obs, reward, done, _ = env.step(action)
        episode_reward += reward[0] if hasattr(reward, "__len__") else reward
        step_count += 1

        if done:
            break

    return episode_reward, step_count


def print_training_results(
    episode_rewards: list[float],
    title: str = "Training Results"
) -> None:
    """Print standardized training results"""
    print(f"\n=== {title} ===")
    print(f"Total episodes: {len(episode_rewards)}")
    print(f"Average episode reward: {np.mean(episode_rewards):.6f}")
    print(f"Reward std: {np.std(episode_rewards):.6f}")
    print(f"Best episode reward: {np.max(episode_rewards):.6f}")
    print(f"Worst episode reward: {np.min(episode_rewards):.6f}")


def print_training_start(
    config_name: str,
    reward_scaling: float,
    entropy_coef: float,
    learning_rate: float,
    total_steps: int = 100000
) -> None:
    """Print standardized training start message"""
    print(f"Starting training with config: {config_name}")
    print(
        f"Reward scaling: {reward_scaling}, Entropy coef: {entropy_coef}, Learning rate: {learning_rate}"
    )
    print(f"Training for {total_steps:,} steps...")


def load_training_data(csv_path: str = "ml-dataset-enhanced.csv") -> pd.DataFrame:
    """Load and preprocess training data"""
    df = pd.read_csv(csv_path)
    df = df.sort_values("timestamp").reset_index(drop=True)
    return df