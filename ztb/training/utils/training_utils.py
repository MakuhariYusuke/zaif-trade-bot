#!/usr/bin/env python3
"""
Common training utilities for reducing code duplication across training scripts
"""

import sys
from pathlib import Path
from typing import Any, Callable, Dict, Optional, Tuple, Union

import numpy as np
import pandas as pd
from stable_baselines3 import PPO
from stable_baselines3.common.type_aliases import GymEnv
from stable_baselines3.common.vec_env import DummyVecEnv

from ztb.trading.environment.constants import PPO_DEFAULT_N_STEPS
from ztb.trading.environment.environment import HeavyTradingEnv
from ztb.training.config.ppo_config import PPOConfig, get_ppo_config
from ztb.training.constants import DEFAULT_BATCH_SIZE_PPO, DEFAULT_GAMMA, DEFAULT_CLIP_RANGE, DEFAULT_ENT_COEF_PPO, DEFAULT_N_EPOCHS_PPO, DEFAULT_GAE_LAMBDA, DEFAULT_VF_COEF, DEFAULT_MAX_GRAD_NORM
from ztb.training.utils.parallel_utils import DataLoaderParallelizer, default_processor
from ztb.cache.memory_cache import default_memory_manager


def setup_project_path() -> Path:
    """Add project root to Python path and return project root path"""
    project_root = Path(
        __file__
    ).parent.parent.parent  # ztb/training -> ztb -> project_root
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))
    return project_root


def create_trading_env(
    df: pd.DataFrame, config: Dict[str, Any], vec_env: bool = True
) -> HeavyTradingEnv | DummyVecEnv:
    """Create trading environment with common configuration"""
    env = HeavyTradingEnv(df=df, config=config)
    if vec_env:
        env = DummyVecEnv([lambda: env])  # type: ignore[assignment]
    return env


def create_ppo_model(
    env: GymEnv,
    config_override: Optional[Dict[str, Any]] = None,
    tensorboard_log: str = "./tensorboard",
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
        n_steps=ppo_config.get("n_steps", PPO_DEFAULT_N_STEPS),
        batch_size=ppo_config.get("batch_size", DEFAULT_BATCH_SIZE_PPO),
        n_epochs=ppo_config.get("n_epochs", DEFAULT_N_EPOCHS_PPO),
        gamma=ppo_config.get("gamma", DEFAULT_GAMMA),
        gae_lambda=ppo_config.get("gae_lambda", DEFAULT_GAE_LAMBDA),
        clip_range=ppo_config.get("clip_range", DEFAULT_CLIP_RANGE),
        ent_coef=ppo_config.get("ent_coef", DEFAULT_ENT_COEF_PPO),
        vf_coef=ppo_config.get("vf_coef", DEFAULT_VF_COEF),
        max_grad_norm=ppo_config.get("max_grad_norm", DEFAULT_MAX_GRAD_NORM),
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
    model: PPO, env: GymEnv, max_steps: int = 1000, deterministic: bool = True
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
    episode_rewards: list[float], title: str = "Training Results"
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
    total_steps: int = 100000,
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


def load_training_data_parallel(csv_paths: list[str], combine: bool = True,
                              preprocess_func: Optional[Callable[[pd.DataFrame], pd.DataFrame]] = None,
                              enable_memory_cache: bool = False) -> Union[pd.DataFrame, list[pd.DataFrame]]:
    """
    Load multiple training data files in parallel with memory caching.

    Args:
        csv_paths: List of CSV file paths to load
        combine: Whether to combine all DataFrames into one
        preprocess_func: Optional preprocessing function to apply to each DataFrame
        enable_memory_cache: Whether to use memory caching

    Returns:
        Combined DataFrame if combine=True, otherwise list of DataFrames
    """
    # Create cache key from file paths
    cache_key = f"training_data_{'_'.join(sorted(csv_paths))}_{combine}"

    # Check memory cache first
    if enable_memory_cache:
        cached_data = default_memory_manager.get_cached_training_data(cache_key)
        if cached_data is not None:
            return cached_data

    data_loader = DataLoaderParallelizer()

    # Load CSV files in parallel
    dataframes = data_loader.parallel_csv_loading(csv_paths)

    # Apply preprocessing if provided
    if preprocess_func is not None:
        dataframes = data_loader.parallel_data_preprocessing(dataframes, preprocess_func)

    if combine:
        # Combine all DataFrames
        combined_df = pd.concat(dataframes, ignore_index=True)
        # Sort by timestamp and reset index
        if 'timestamp' in combined_df.columns:
            combined_df = combined_df.sort_values("timestamp").reset_index(drop=True)

        # Cache the result
        if enable_memory_cache:
            default_memory_manager.cache_training_data(cache_key, combined_df)

        return combined_df
    else:
        # Cache individual dataframes if not combining
        if enable_memory_cache:
            for i, df in enumerate(dataframes):
                df_cache_key = f"training_data_{csv_paths[i]}"
                default_memory_manager.cache_training_data(df_cache_key, df)

        return dataframes


def parallel_data_preprocessing(df: pd.DataFrame, chunk_size: int = 10000,
                              preprocess_func: Optional[Callable[[pd.DataFrame], pd.DataFrame]] = None,
                              enable_memory_cache: bool = False) -> pd.DataFrame:
    """
    Apply preprocessing to DataFrame in parallel chunks with memory caching.

    Args:
        df: Input DataFrame
        chunk_size: Size of each processing chunk
        preprocess_func: Preprocessing function to apply
        enable_memory_cache: Whether to use memory caching

    Returns:
        Preprocessed DataFrame
    """
    if preprocess_func is None:
        return df

    # Create cache key for preprocessing result
    cache_key = f"preprocessed_data_{hash(str(df.values.tobytes()) if hasattr(df, 'values') else str(df))}_{chunk_size}"

    # Check memory cache first
    if enable_memory_cache:
        cached_data = default_memory_manager.get_cached_training_data(cache_key)
        if cached_data is not None:
            return cached_data

    if len(df) <= chunk_size:
        # Small dataset
        result = preprocess_func(df)
        if enable_memory_cache:
            default_memory_manager.cache_training_data(cache_key, result)
        return result

    # Split DataFrame into chunks
    chunks = []
    for i in range(0, len(df), chunk_size):
        chunk = df.iloc[i:i + chunk_size].copy()
        chunks.append(chunk)

    # Process chunks in parallel
    def process_chunk(chunk: pd.DataFrame) -> pd.DataFrame:
        return preprocess_func(chunk)

    processed_chunks = default_processor.parallel_map(process_chunk, chunks)

    # Combine processed chunks
    result_df = pd.concat(processed_chunks, ignore_index=True)

    # Cache the result
    if enable_memory_cache:
        default_memory_manager.cache_training_data(cache_key, result_df)

    return result_df
