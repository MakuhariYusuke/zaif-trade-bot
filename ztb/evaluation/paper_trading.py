"""
Paper trading evaluation helpers.

UnifiedEvaluator uses these helpers for EvaluationType.PAPER_TRADING.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

import numpy as np
import pandas as pd
from stable_baselines3 import SAC

from ztb.io.data_loader import DataLoader
from ztb.io.json_io import write_json
from ztb.trading.constants import ACTION_BUY, ACTION_HOLD, ACTION_SELL
from ztb.trading.environment.heavy_env import HeavyTradingEnv
from ztb.trading.environment.utils.config import EnvironmentConfig
from ztb.utils.logging_utils import get_logger

logger = get_logger(__name__)


def _load_model(model_path: str) -> SAC:
    logger.info("Loading model from %s", model_path)
    model = SAC.load(model_path)
    logger.info("Model loaded successfully")
    return model


def _load_data(data_path: str) -> pd.DataFrame:
    logger.info("Loading data from %s", data_path)
    df = DataLoader.load_csv_strict(data_path)
    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        df = df.sort_values("timestamp").reset_index(drop=True)
    logger.info("Loaded %s data points", len(df))
    return df


def _create_environment(
    df: pd.DataFrame, config: Optional[EnvironmentConfig]
) -> HeavyTradingEnv:
    env_config = config or EnvironmentConfig()
    env = HeavyTradingEnv(df=df, config=env_config)
    return env


def run_paper_trading(
    model: SAC,
    env: HeavyTradingEnv,
    num_episodes: int = 10,
    max_steps_per_episode: Optional[int] = None,
) -> Dict[str, Any]:
    """Run paper trading simulation and return summary stats."""
    logger.info("Starting paper trading with %s episodes", num_episodes)

    all_episode_results = []
    total_rewards = []
    total_portfolio_values = []
    action_counts = {"HOLD": 0, "BUY": 0, "SELL": 0}

    for episode in range(num_episodes):
        logger.info("Episode %s/%s", episode + 1, num_episodes)
        obs, info = env.reset()
        episode_reward = 0
        episode_portfolio_values = []
        episode_actions = []

        done = False
        step = 0
        while not done:
            if max_steps_per_episode and step >= max_steps_per_episode:
                break

            action, _ = model.predict(obs, deterministic=True)

            continuous_action = action[0] if isinstance(action, np.ndarray) else action
            if continuous_action < -0.1:
                discrete_action = ACTION_SELL
                action_counts["SELL"] += 1
            elif continuous_action > 0.1:
                discrete_action = ACTION_BUY
                action_counts["BUY"] += 1
            else:
                discrete_action = ACTION_HOLD
                action_counts["HOLD"] += 1

            episode_actions.append(discrete_action)
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

            episode_reward += reward
            episode_portfolio_values.append(info.get("portfolio_value", 0))
            step += 1

        episode_result = {
            "episode": episode + 1,
            "total_reward": episode_reward,
            "final_portfolio_value": episode_portfolio_values[-1]
            if episode_portfolio_values
            else 0,
            "initial_portfolio_value": episode_portfolio_values[0]
            if episode_portfolio_values
            else 0,
            "steps": step,
            "action_distribution": {
                "HOLD": episode_actions.count(ACTION_HOLD) / len(episode_actions)
                if episode_actions
                else 0,
                "BUY": episode_actions.count(ACTION_BUY) / len(episode_actions)
                if episode_actions
                else 0,
                "SELL": episode_actions.count(ACTION_SELL) / len(episode_actions)
                if episode_actions
                else 0,
            },
        }

        all_episode_results.append(episode_result)
        total_rewards.append(episode_reward)
        total_portfolio_values.append(episode_result["final_portfolio_value"])

        logger.info(
            "Episode %s completed: Reward=%.2f, Portfolio=%.2f",
            episode + 1,
            episode_reward,
            episode_result["final_portfolio_value"],
        )

    summary = {
        "num_episodes": num_episodes,
        "avg_reward": float(np.mean(total_rewards)) if total_rewards else 0.0,
        "std_reward": float(np.std(total_rewards)) if total_rewards else 0.0,
        "avg_portfolio_value": float(np.mean(total_portfolio_values))
        if total_portfolio_values
        else 0.0,
        "std_portfolio_value": float(np.std(total_portfolio_values))
        if total_portfolio_values
        else 0.0,
        "total_action_counts": action_counts,
        "action_distribution_percent": {
            "HOLD": action_counts["HOLD"] / sum(action_counts.values()) * 100
            if sum(action_counts.values()) > 0
            else 0,
            "BUY": action_counts["BUY"] / sum(action_counts.values()) * 100
            if sum(action_counts.values()) > 0
            else 0,
            "SELL": action_counts["SELL"] / sum(action_counts.values()) * 100
            if sum(action_counts.values()) > 0
            else 0,
        },
        "episode_results": all_episode_results,
    }

    logger.info("Paper trading completed")
    return summary


def evaluate_paper_trading(
    model_path: str,
    data_path: str,
    num_episodes: int = 10,
    env_config: Optional[EnvironmentConfig] = None,
    max_steps_per_episode: Optional[int] = None,
    output_path: Optional[str] = None,
) -> Dict[str, Any]:
    """Complete paper trading evaluation pipeline."""
    model = _load_model(model_path)
    df = _load_data(data_path)
    env = _create_environment(df, env_config)
    results = run_paper_trading(
        model, env, num_episodes=num_episodes, max_steps_per_episode=max_steps_per_episode
    )
    if output_path:
        write_json(output_path, results, indent=2, default=str)
        logger.info("Results saved to %s", output_path)
    return results
