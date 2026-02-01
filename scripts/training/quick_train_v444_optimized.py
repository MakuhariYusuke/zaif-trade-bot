#!/usr/bin/env python3
"""
Quick Train Script for SAC v444.2 - Optimized Configuration
段階的な改善を検証するためのスクリプト
"""

import json
import os
import sys
from datetime import datetime
from pathlib import Path

# Add project to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np

from ztb.trading.environment.heavy_env import HeavyEnvironment
from ztb.trading.environment.components.rewards.utils import RewardUtils
from ztb.training.base_trainer import TrainerConfig, TrainingSession




def create_training_session(config: dict) -> TrainingSession:
    """Create a training session with the given configuration."""
    trainer_config = TrainerConfig(
        algorithm="sac",
        model_name=config.get("model_name", "sac_v444_2"),
        total_timesteps=config["training"]["total_timesteps"],
        data_config=config["training"]["data_config"],
        sac_hyperparameters=config["training"]["sac_hyperparameters"],
        environment_config=config,
    )

    return TrainingSession(trainer_config)


def analyze_action_distribution(
    session: TrainingSession, num_samples: int = 200
) -> dict:
    """Analyze action distribution from the trained model."""
    env = HeavyEnvironment(config=session.config)
    obs, _ = env.reset()

    actions = []
    discrete_actions = []

    for _ in range(num_samples):
        # Get action from policy
        action, _ = session.model.predict(obs, deterministic=True)

        # Extract discrete action
        if isinstance(action, np.ndarray):
            if len(action.shape) > 1:
                discrete_action = np.argmax(action[0])
            else:
                discrete_action = np.argmax(action)
        else:
            discrete_action = int(action)

        actions.append(action)
        discrete_actions.append(discrete_action)

        # Take step
        obs, _, terminated, truncated, _ = env.step(action)
        if terminated or truncated:
            obs, _ = env.reset()

    # Analyze distributions
    discrete_actions = np.array(discrete_actions)
    actions = np.array(actions)

    action_counts = np.bincount(discrete_actions, minlength=3)
    action_ratios = action_counts / len(discrete_actions)

    analysis = {
        "num_samples": num_samples,
        "discrete_action_counts": {
            "HOLD": int(action_counts[0]),
            "BUY": int(action_counts[1]),
            "SELL": int(action_counts[2]),
        },
        "discrete_action_ratios": {
            "HOLD": float(action_ratios[0]),
            "BUY": float(action_ratios[1]),
            "SELL": float(action_ratios[2]),
        },
        "continuous_action_stats": {
            "mean": float(np.mean(actions)),
            "std": float(np.std(actions)),
            "min": float(np.min(actions)),
            "max": float(np.max(actions)),
            "median": float(np.median(actions)),
            "q25": float(np.percentile(actions, 25)),
            "q75": float(np.percentile(actions, 75)),
        },
        "balance_metrics": {
            "buy_sell_diff": RewardUtils.calculate_buy_sell_diff(
                action_ratios[1], action_ratios[2]
            ),
            "buy_sell_ratio": action_ratios[1] / max(action_ratios[2], 1e-6),
        },

    }

    env.close()
    return analysis


def print_summary(config: dict, analysis: dict) -> None:
    """Print training summary."""
    print("\n" + "=" * 80)
    print("SAC v444.2 Training Summary")
    print("=" * 80)

    print(f"\nModel: {config.get('model_name')}")
    print(f"Total Timesteps: {config['training']['total_timesteps']}")
    print(f"Training Timestamp: {datetime.now().isoformat()}")

    print("\n[Behavior Optimization Settings]")
    behavior_opt = config["environment"]["behavior_optimization"]
    print(f"  Balance Penalty Scale: {behavior_opt.get('balance_penalty', 1000.0)}")
    print(
        f"  Entropy Regularization: {behavior_opt.get('entropy_regularization', 0.01)}"
    )
    print(
        f"  Action Balance Target: {behavior_opt.get('action_balance_target', 0.333):.3f}"
    )
    print(
        f"  Redundant Trade Penalty: {behavior_opt.get('redundant_trade_penalty', 10.0)}"
    )

    print("\n[Action Bonuses]")
    action_bonuses = config["environment"]["action_bonuses"]
    print(f"  Buy Bonus: {action_bonuses.get('buy_action_bonus', 0.0)}")
    print(f"  Sell Bonus: {action_bonuses.get('sell_action_bonus', 0.0)}")
    print(f"  Hold Bonus: {action_bonuses.get('hold_action_bonus', 0.0)}")

    if analysis:
        print("\n[Action Distribution Analysis]")
        discrete = analysis["discrete_action_ratios"]
        print(
            f"  HOLD: {discrete['HOLD']:.2%} ({analysis['discrete_action_counts']['HOLD']} actions)"
        )
        print(
            f"  BUY:  {discrete['BUY']:.2%} ({analysis['discrete_action_counts']['BUY']} actions)"
        )
        print(
            f"  SELL: {discrete['SELL']:.2%} ({analysis['discrete_action_counts']['SELL']} actions)"
        )

        balance = analysis["balance_metrics"]
        print(f"\n  BUY/SELL Difference: {balance['buy_sell_diff']:.4f}")
        print(f"  BUY/SELL Ratio: {balance['buy_sell_ratio']:.4f}")

        print("\n[Continuous Action Statistics]")
        cont = analysis["continuous_action_stats"]
        print(f"  Mean: {cont['mean']:.4f}, Std: {cont['std']:.4f}")
        print(f"  Min: {cont['min']:.4f}, Max: {cont['max']:.4f}")
        print(f"  Median: {cont['median']:.4f}")
        print(f"  Q25: {cont['q25']:.4f}, Q75: {cont['q75']:.4f}")

    print("\n" + "=" * 80 + "\n")


def main():
    """Main training function."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Train SAC v444.2 with optimized configuration"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="config/sac_v444_2_optimized_balance_reward.json",
        help="Path to config file",
    )
    parser.add_argument(
        "--steps", type=int, default=5000, help="Number of training timesteps"
    )
    parser.add_argument(
        "--analyze",
        action="store_true",
        help="Analyze action distribution after training",
    )

    args = parser.parse_args()

    # Load configuration
    if not os.path.exists(args.config):
        print(f"Error: Config file not found: {args.config}")
        sys.exit(1)

    config = load_config(args.config)
    config["training"]["total_timesteps"] = args.steps

    print(f"Loading configuration from: {args.config}")
    print(f"Training for {args.steps} timesteps...\n")

    try:
        # Create training session
        session = create_training_session(config)

        # Train the model
        print("Starting training session...")
        session.train()

        # Analyze results if requested
        analysis = None
        if args.analyze:
            print("Analyzing action distribution...")
            analysis = analyze_action_distribution(session)

        # Print summary
        print_summary(config, analysis)

        print("✓ Training completed successfully!")

        # Save analysis results
        if analysis:
            results_file = f"results/sac_v444_2_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            os.makedirs("results", exist_ok=True)
            with open(results_file, "w") as f:
                json.dump(analysis, f, indent=2)
            print(f"✓ Analysis results saved to: {results_file}")

    except Exception as e:
        print(f"Error during training: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
