#!/usr/bin/env python3
"""Paper trading evaluation script for ensemble PPO models."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
import torch
from stable_baselines3 import PPO

# Annual trading days used for Sharpe ratio approximation
TRADING_DAYS_PER_YEAR = 252

PROJECT_ROOT = Path(__file__).parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ztb.trading.environment import HeavyTradingEnv  # noqa: E402  pylint: disable=wrong-import-position

LOGGER = logging.getLogger(__name__)


def load_models(model_paths: List[Path]) -> List[PPO]:
    models: List[PPO] = []
    for path in model_paths:
        LOGGER.info("Loading model %s", path)
        models.append(PPO.load(str(path)))
    return models


def compute_mean_action(models: List[PPO], observation: np.ndarray) -> Dict[str, np.ndarray]:
    obs_tensor = torch.as_tensor(observation[None, :], dtype=torch.float32)
    probs: List[np.ndarray] = []
    for model in models:
        obs_on_device = obs_tensor.to(model.device)
        with torch.no_grad():
            distribution = model.policy.get_distribution(obs_on_device)
            dist = distribution.distribution  # type: ignore[attr-defined]
            probs.append(dist.probs.detach().cpu().numpy()[0])
    mean_probs = np.mean(probs, axis=0)
    action = int(np.argmax(mean_probs))
    return {"action": action, "probs": mean_probs}


def run_episode(models: List[PPO], df: pd.DataFrame, args: argparse.Namespace) -> Dict[str, float]:
    env_config = {
        "reward_scaling": 1.0,
        "transaction_cost": 0.0,
        "max_position_size": 1.0,
        "feature_set": args.feature_set,
        "timeframe": args.timeframe,
    }
    env = HeavyTradingEnv(df=df, config=env_config, random_start=True)
    obs, _ = env.reset()

    episode_reward = 0.0
    action_counts = {"BUY": 0, "SELL": 0, "HOLD": 0}
    current_balance = args.initial_balance

    for step in range(args.max_steps):
        ensemble_result = compute_mean_action(models, obs)
        action = ensemble_result["action"]
        probs = ensemble_result["probs"]
        LOGGER.debug("Step %d - action=%d probs=%s", step, action, probs)

        obs, reward, terminated, truncated, info = env.step(action)
        episode_reward += reward

        if action == 0:
            action_counts["HOLD"] += 1
        elif action == 1:
            action_counts["BUY"] += 1
        elif action == 2:
            action_counts["SELL"] += 1

        current_balance += info.get("pnl", 0.0)

        if terminated or truncated:
            break

    total_return = (current_balance - args.initial_balance) / args.initial_balance
    env.close()

    return {
        "reward": episode_reward,
        "return": total_return,
        "buy": action_counts["BUY"],
        "sell": action_counts["SELL"],
        "hold": action_counts["HOLD"],
    }


def summarize(results: List[Dict[str, float]]) -> Dict[str, float]:
    rewards = [r["reward"] for r in results]
    returns = [r["return"] for r in results]
    buys = sum(r["buy"] for r in results)
    sells = sum(r["sell"] for r in results)
    holds = sum(r["hold"] for r in results)

    sharpe = 0.0
    if len(returns) > 1 and np.std(returns) > 0:
        sharpe = (
            np.mean(returns) / np.std(returns) * np.sqrt(TRADING_DAYS_PER_YEAR)
        )

    win_rate = sum(1 for r in returns if r > 0) / len(returns) if returns else 0.0
    avg_return = float(np.mean(returns)) if returns else 0.0

    return {
        "avg_reward": float(np.mean(rewards)) if rewards else 0.0,
        "avg_return": avg_return,
        "total_return": float(np.prod([1 + r for r in returns]) - 1) if returns else 0.0,
        "win_rate": win_rate,
        "sharpe_ratio": sharpe,
        "buy_actions": buys,
        "sell_actions": sells,
        "hold_actions": holds,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate ensemble PPO models via paper trading")
    parser.add_argument(
        "--model-paths",
        nargs="+",
        default=[
            "models/ensemble_model_1.zip",
            "models/ensemble_model_2.zip",
            "models/ensemble_model_3.zip",
        ],
        help="Paths to ensemble models",
    )
    parser.add_argument(
        "--data-path",
        default="ml-dataset-enhanced-balanced.csv",
        help="Dataset used for paper trading",
    )
    parser.add_argument(
        "--episodes",
        type=int,
        default=10,
        help="Number of episodes to evaluate",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=1000,
        help="Maximum steps per episode",
    )
    parser.add_argument(
        "--initial-balance",
        type=float,
        default=100000.0,
        help="Initial portfolio value",
    )
    parser.add_argument(
        "--feature-set",
        default="full",
        help="Feature set to use inside the environment",
    )
    parser.add_argument(
        "--timeframe",
        default="1m",
        help="Timeframe for HeavyTradingEnv",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable debug logging",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    )

    model_paths = [Path(p) for p in args.model_paths]
    models = load_models(model_paths)

    LOGGER.info("Loading dataset %s", args.data_path)
    df = pd.read_csv(args.data_path)

    results: List[Dict[str, float]] = []
    for episode in range(args.episodes):
        LOGGER.info("Starting episode %d/%d", episode + 1, args.episodes)
        results.append(run_episode(models, df, args))

    summary = summarize(results)
    LOGGER.info("Ensemble evaluation summary: %s", summary)
    print("\n=== Ensemble Evaluation Summary ===")
    for key, value in summary.items():
        if "actions" in key:
            print(f"{key}: {int(value)}")
        else:
            print(f"{key}: {value:.4f}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
