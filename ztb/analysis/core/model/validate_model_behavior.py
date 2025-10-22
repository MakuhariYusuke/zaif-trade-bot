#!/usr/bin/env python3
"""
Validate Model Behavior

Quick validation of trained model's action distribution and basic metrics.
"""

import argparse
import sys
from collections import Counter
from pathlib import Path

import pandas as pd
from sb3_contrib import MaskablePPO
from stable_baselines3 import PPO

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from ztb.trading.environment.environment import HeavyTradingEnv
from ztb.utils.config import TypedConfig


def validate_model(model_path: str, data_path: str, num_episodes: int = 3) -> None:
    """Validate model behavior on sample data."""
    print(f"Loading model from {model_path}")

    # Try loading as MaskablePPO first
    try:
        model = MaskablePPO.load(model_path)
        model_type = "MaskablePPO"
    except Exception:
        model = PPO.load(model_path)
        model_type = "PPO"

    print(f"Model type: {model_type}")

    # Load data
    print(f"Loading data from {data_path}")
    df = pd.read_csv(data_path)
    print(f"Data: {len(df)} rows, {len(df.columns)} columns")

    # Create environment
    config = {
        "transaction_cost": 0.0005,
        "max_position_size": 0.5,
        "curriculum_stage": "forced_balance",  # Match training configuration
    }

    env = HeavyTradingEnv(df=df, config=config, random_start=False)

    # Collect actions over multiple episodes
    all_actions = []
    episode_rewards = []

    for episode in range(num_episodes):
        obs, _ = env.reset()
        done = False
        actions = []
        total_reward = 0
        step = 0

        while not done and step < 200:  # Limit to 200 steps per episode
            # Get action masks if using MaskablePPO
            if model_type == "MaskablePPO":
                action_masks = env.get_action_masks()
                action, _ = model.predict(
                    obs, action_masks=action_masks, deterministic=False
                )
            else:
                action, _ = model.predict(obs, deterministic=False)

            action_int = int(action.item() if hasattr(action, "item") else action)
            actions.append(action_int)

            obs, reward, terminated, truncated, _ = env.step(action_int)
            total_reward += reward
            done = terminated or truncated
            step += 1

        all_actions.extend(actions)
        episode_rewards.append(total_reward)

        action_counts = Counter(actions)
        print(f"\nEpisode {episode + 1}:")
        print(f"  Steps: {len(actions)}")
        print(f"  Total reward: {total_reward:.4f}")
        print(
            f"  Actions: HOLD={action_counts.get(0, 0)}, BUY={action_counts.get(1, 0)}, SELL={action_counts.get(2, 0)}"
        )
        print(
            f"  Action %: HOLD={action_counts.get(0, 0)/len(actions)*100:.1f}%, BUY={action_counts.get(1, 0)/len(actions)*100:.1f}%, SELL={action_counts.get(2, 0)/len(actions)*100:.1f}%"
        )

    # Summary statistics
    total_actions = Counter(all_actions)
    total_steps = len(all_actions)

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Total steps: {total_steps}")
    print(
        f"Average reward per episode: {sum(episode_rewards) / len(episode_rewards):.4f}"
    )
    print("\nAction Distribution:")
    print(
        f"  HOLD:  {total_actions.get(0, 0):4d} ({total_actions.get(0, 0)/total_steps*100:5.1f}%)"
    )
    print(
        f"  BUY:   {total_actions.get(1, 0):4d} ({total_actions.get(1, 0)/total_steps*100:5.1f}%)"
    )
    print(
        f"  SELL:  {total_actions.get(2, 0):4d} ({total_actions.get(2, 0)/total_steps*100:5.1f}%)"
    )

    # Diagnosis
    print("\n" + "=" * 60)
    print("DIAGNOSIS")
    print("=" * 60)

    sell_ratio = total_actions.get(2, 0) / total_steps
    buy_ratio = total_actions.get(1, 0) / total_steps
    hold_ratio = total_actions.get(0, 0) / total_steps

    if sell_ratio > 0.8:
        print("⚠️  SEVERE SELL BIAS DETECTED (>80%)")
        print("   Model is heavily biased toward SELL actions.")
        print("   This will cause losses in uptrends.")
    elif sell_ratio > 0.5:
        print("⚠️  Moderate sell bias detected (>50%)")

    if buy_ratio < 0.1:
        print("⚠️  Very low BUY action frequency (<10%)")
        print("   Model may not be taking advantage of uptrends.")

    if hold_ratio > 0.6:
        print("ℹ️  Model prefers HOLD actions (>60%)")
        print("   This is conservative but may miss trading opportunities.")

    balance_score = min(sell_ratio, buy_ratio, hold_ratio) / max(
        sell_ratio, buy_ratio, hold_ratio
    )
    print(f"\nBalance score: {balance_score:.3f} (0=imbalanced, 1=perfect balance)")

    if balance_score < 0.2:
        print("⚠️  Very imbalanced action distribution")
        print("   Consider retraining with:")


def validate_model_behavior(
    model_path: str, data_path: Optional[str] = None, num_episodes: int = 3
) -> dict:
    """Validate model behavior on sample data.

    Args:
        model_path: Path to the model file
        data_path: Path to the data file
        num_episodes: Number of episodes to run for validation

    Returns:
        Dictionary with validation results
    """
    return validate_model(model_path, data_path, num_episodes)


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate model behavior")

    config = TypedConfig()
    parser.add_argument(
        "--model-path",
        default=config.get_model_path("ppo_100k_optimized.zip"),
        help="Path to model",
    )
    parser.add_argument(
        "--data-path",
        default="ml-dataset-enhanced.csv",
        help="Path to data",
    )
    parser.add_argument(
        "--episodes",
        type=int,
        default=3,
        help="Number of episodes to run",
    )

    args = parser.parse_args()
    validate_model(args.model_path, args.data_path, args.episodes)


if __name__ == "__main__":
    main()
