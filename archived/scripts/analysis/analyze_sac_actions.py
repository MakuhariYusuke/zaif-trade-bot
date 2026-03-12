#!/usr/bin/env python3
"""Analyze SAC v414 action distribution with continuous win rate bonus."""

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from stable_baselines3 import SAC

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from ztb.trading.environment.constants import continuous_to_discrete_action
from ztb.trading.environment.environment import HeavyTradingEnv
from ztb.utils.logging_utils import get_logger

logger = get_logger(__name__)


def analyze_sac_actions(model_path: str, config_path: str, num_samples: int = 5000):
    """Analyze action distribution of trained SAC model."""

    # Load config
    from ztb.utils.config_loader import ConfigLoader

    config = ConfigLoader.load(config_path)

    # Create environment like in training scripts
    import pandas as pd

    from ztb.trading.environment.utils.config import EnvironmentConfig

    # Get environment config section
    env_config_dict = config.get("environment", {})

    # Merge reward_settings into environment config
    if "reward_settings" in config:
        env_config_dict["reward_settings"] = config["reward_settings"]

    env_config = EnvironmentConfig.from_dict(env_config_dict)

    # Load data
    df = pd.read_csv("btc_jpy_real_dataset.csv")

    # Create environment
    env = HeavyTradingEnv(df=df, config=env_config)

    # Load model
    model = SAC.load(model_path)

    # Sample actions
    actions_continuous = []
    actions_discrete = []
    rewards = []

    obs, _ = env.reset()
    for _ in range(num_samples):
        # Get continuous action from model
        action_continuous, _ = model.predict(obs, deterministic=True)
        actions_continuous.append(action_continuous[0])

        # Convert to discrete action
        action_discrete = continuous_to_discrete_action(
            action_continuous[0], threshold=0.1
        )
        actions_discrete.append(action_discrete)

        # Step environment
        obs, reward, terminated, truncated, _ = env.step(action_discrete)
        rewards.append(reward)

        if terminated or truncated:
            obs, _ = env.reset()

    env.close()

    # Analyze distribution
    actions_continuous = np.array(actions_continuous)
    actions_discrete = np.array(actions_discrete)
    rewards = np.array(rewards)

    # Count discrete actions
    unique, counts = np.unique(actions_discrete, return_counts=True)
    action_counts = dict(zip(unique, counts))

    # Calculate percentages
    total_actions = len(actions_discrete)
    action_percentages = {
        k: (v / total_actions) * 100 for k, v in action_counts.items()
    }

    # Print results
    print("=" * 80)
    print("SAC v414 Action Distribution Analysis")
    print("=" * 80)
    print(f"Model: {model_path}")
    print(f"Samples: {num_samples}")
    print("Continuous→Discrete Threshold: 0.1")
    print()

    print("Action Distribution:")
    action_names = {0: "HOLD", 1: "BUY", 2: "SELL"}
    for action_id in [0, 1, 2]:
        count = action_counts.get(action_id, 0)
        percentage = action_percentages.get(action_id, 0)
        print("6d")

    print()
    print("Target Distribution: HOLD 10%, BUY 45%, SELL 45%")
    print()

    # Check if within tolerance
    target_hold = 10.0
    target_buy = 45.0
    target_sell = 45.0
    tolerance = 5.0  # 5% tolerance

    hold_pct = action_percentages.get(0, 0)
    buy_pct = action_percentages.get(1, 0)
    sell_pct = action_percentages.get(2, 0)

    print("Balance Check:")
    print(".1f")
    print(".1f")
    print(".1f")

    hold_ok = abs(hold_pct - target_hold) <= tolerance
    buy_ok = abs(buy_pct - target_buy) <= tolerance
    sell_ok = abs(sell_pct - target_sell) <= tolerance

    print()
    print("Status:")
    print(f"HOLD within tolerance (±{tolerance}%): {'✅' if hold_ok else '❌'}")
    print(f"BUY within tolerance (±{tolerance}%): {'✅' if buy_ok else '❌'}")
    print(f"SELL within tolerance (±{tolerance}%): {'✅' if sell_ok else '❌'}")

    if all([hold_ok, buy_ok, sell_ok]):
        print("\n🎉 SUCCESS: All actions within target distribution!")
    else:
        print("\n⚠️  WARNING: Some actions outside target distribution")

    # Plot continuous action distribution
    plt.figure(figsize=(12, 4))

    plt.subplot(1, 3, 1)
    plt.hist(actions_continuous, bins=50, alpha=0.7, color="blue", edgecolor="black")
    plt.axvline(x=0.1, color="red", linestyle="--", label="BUY threshold (0.1)")
    plt.axvline(x=-0.1, color="green", linestyle="--", label="SELL threshold (-0.1)")
    plt.xlabel("Continuous Action Value")
    plt.ylabel("Frequency")
    plt.title("Continuous Action Distribution")
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.subplot(1, 3, 2)
    action_labels = ["HOLD", "BUY", "SELL"]
    action_values = [action_counts.get(i, 0) for i in range(3)]
    bars = plt.bar(
        action_labels, action_values, color=["orange", "green", "red"], alpha=0.7
    )
    plt.ylabel("Count")
    plt.title("Discrete Action Distribution")

    # Add percentage labels on bars
    for bar, pct in zip(bars, [action_percentages.get(i, 0) for i in range(3)]):
        height = bar.get_height()
        plt.text(
            bar.get_x() + bar.get_width() / 2.0,
            height + 50,
            ".1f",
            ha="center",
            va="bottom",
            fontweight="bold",
        )

    plt.grid(True, alpha=0.3)

    plt.subplot(1, 3, 3)
    plt.hist(rewards, bins=50, alpha=0.7, color="purple", edgecolor="black")
    plt.xlabel("Reward")
    plt.ylabel("Frequency")
    plt.title("Reward Distribution")
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("sac_v414_action_analysis.png", dpi=150, bbox_inches="tight")
    print("\n📊 Plot saved to: sac_v414_action_analysis.png")

    return {
        "action_counts": {int(k): int(v) for k, v in action_counts.items()},
        "action_percentages": {int(k): float(v) for k, v in action_percentages.items()},
        "continuous_stats": {
            "mean": float(np.mean(actions_continuous)),
            "std": float(np.std(actions_continuous)),
            "min": float(np.min(actions_continuous)),
            "max": float(np.max(actions_continuous)),
        },
        "reward_stats": {
            "mean": float(np.mean(rewards)),
            "std": float(np.std(rewards)),
            "min": float(np.min(rewards)),
            "max": float(np.max(rewards)),
        },
    }


def main():
    """Main analysis function."""
    model_path = "checkpoints/sac_session/sac_v414_balanced_trading_final.zip"
    config_path = "config/sac_v414_balanced_trading_config.json"

    if not Path(model_path).exists():
        print(f"Model not found: {model_path}")
        sys.exit(1)

    if not Path(config_path).exists():
        print(f"Config not found: {config_path}")
        sys.exit(1)

    results = analyze_sac_actions(model_path, config_path)

    # Save results
    import json

    output_file = "sac_v414_analysis_results.json"
    with open(output_file, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\n📝 Results saved to: {output_file}")


if __name__ == "__main__":
    main()
