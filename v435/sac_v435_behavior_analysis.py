#!/usr/bin/env python3
"""
SAC v435 Model Behavior Analysis
SAC v435 モデルの行動パターン分析

Analyzes the behavior patterns of SAC v435 models to understand why trading frequency is low.
"""

import sys
from pathlib import Path
from typing import Any, Dict

import numpy as np
import pandas as pd
from stable_baselines3 import SAC

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from ztb.trading.environment.environment import HeavyTradingEnv


def analyze_model_behavior(
    model_path: str, data_path: str, max_steps: int = 100
) -> Dict[str, Any]:
    """
    Analyze model behavior patterns to understand trading decisions

    Args:
        model_path: Path to the trained model
        data_path: Path to the data file
        max_steps: Maximum steps to analyze

    Returns:
        Analysis results
    """
    print(f"Loading model from {model_path}")
    model = SAC.load(model_path)

    print(f"Loading data from {data_path}")
    df = pd.read_csv(data_path)
    if len(df) > max_steps:
        df = df.head(max_steps)

    print(f"Analyzing {len(df)} steps of data")

    # Create environment with minimal config
    env_config = {
        "transaction_cost": 0.0005,
        "enable_correlation_reduction": True,
        "correlation_threshold": 0.95,
        "max_position_size": 0.5,
        "curriculum_stage": "forced_balance",
        "reward_trade_frequency_penalty": 0.01,
        "reward_trade_frequency_halflife": 1.0,
        "reward_trade_cooldown_steps": 0,
        "reward_trade_cooldown_penalty": 0.01,
        "reward_max_consecutive_trades": 20,
        "reward_consecutive_trade_penalty": 0.01,
        "reward_position_penalty_scale": 0.1,
        "reward_position_penalty_exponent": 2.0,
        "reward_inventory_penalty_scale": 0.01,
        "reward_volatility_penalty_scale": 0.01,
    }

    env = HeavyTradingEnv(df=df, config=env_config, random_start=False)

    # Analyze model behavior
    obs, _ = env.reset()
    actions = []
    action_values = []
    positions = []
    rewards = []
    dones = []

    print("Running behavior analysis...")

    for step in range(max_steps):
        # Get action from model
        action, _ = model.predict(obs, deterministic=True)
        action_values.append(float(action))

        # Convert to discrete action
        if action > 0.1:  # SAC_CONTINUOUS_THRESHOLD
            discrete_action = 1  # BUY
        elif action < -0.1:  # SAC_CONTINUOUS_THRESHOLD_NEG
            discrete_action = 2  # SELL
        else:
            discrete_action = 0  # HOLD

        actions.append(discrete_action)

        # Step environment
        obs, reward, terminated, truncated, info = env.step(discrete_action)
        done = terminated or truncated

        positions.append(env.position)
        rewards.append(float(reward))
        dones.append(done)

        if done:
            break

    # Analyze results
    action_counts = pd.Series(actions).value_counts()
    position_changes = np.diff(positions)

    analysis = {
        "total_steps": len(actions),
        "action_distribution": {
            "HOLD": int(action_counts.get(0, 0)),
            "BUY": int(action_counts.get(1, 0)),
            "SELL": int(action_counts.get(2, 0)),
        },
        "action_values": {
            "mean": float(np.mean(action_values)),
            "std": float(np.std(action_values)),
            "min": float(np.min(action_values)),
            "max": float(np.max(action_values)),
        },
        "position_analysis": {
            "final_position": float(positions[-1]),
            "max_position": float(np.max(np.abs(positions))),
            "position_changes": int(np.sum(np.abs(position_changes) > 0.001)),
            "position_series": [float(p) for p in positions],
        },
        "reward_analysis": {
            "total_reward": float(np.sum(rewards)),
            "mean_reward": float(np.mean(rewards)),
            "reward_std": float(np.std(rewards)),
        },
        "trading_behavior": {
            "trades_opened": int(np.sum(np.abs(position_changes) > 0.001)),
            "hold_percentage": float(action_counts.get(0, 0) / len(actions) * 100),
            "avg_position_size": float(np.mean(np.abs(positions))),
        },
    }

    return analysis


def analyze_multiple_models() -> Dict[str, Any]:
    """Analyze all SAC v435 variants"""
    base_path = Path("checkpoints")
    data_path = "ml-dataset-enhanced.csv"

    variants = {
        "v435": "sac_v435_test_1000_steps.zip",
        "v435_1": "sac_v435_1_test_1000_steps.zip",  # Assuming similar naming
        "v435_2": "sac_v435_2_test_1000_steps.zip",  # Assuming similar naming
    }

    results = {}

    for variant, model_file in variants.items():
        model_path = base_path / model_file
        if model_path.exists():
            print(f"\n=== Analyzing {variant} ===")
            try:
                analysis = analyze_model_behavior(
                    str(model_path), data_path, max_steps=100
                )
                results[variant] = analysis
                print(f"✓ {variant} analysis completed")
            except Exception as e:
                print(f"✗ {variant} analysis failed: {e}")
                results[variant] = {"error": str(e)}
        else:
            print(f"✗ {variant} model not found: {model_path}")
            results[variant] = {"error": "model file not found"}

    return results


def main():
    """Main analysis function"""
    print("=== SAC v435 Model Behavior Analysis ===")

    try:
        results = analyze_multiple_models()

        # Save results
        import json

        output_file = "sac_v435_behavior_analysis.json"
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)

        print(f"\nAnalysis results saved to: {output_file}")

        # Print summary
        print("\n=== Summary ===")
        for variant, analysis in results.items():
            if "error" not in analysis:
                print(f"\n{variant}:")
                print(f"  Total steps: {analysis['total_steps']}")
                print(f"  Action distribution: {analysis['action_distribution']}")
                print(
                    f"  Trades opened: {analysis['trading_behavior']['trades_opened']}"
                )
                print(
                    f"  Hold percentage: {analysis['trading_behavior']['hold_percentage']:.1f}%"
                )
                print(
                    f"  Final position: {analysis['position_analysis']['final_position']:.3f}"
                )
            else:
                print(f"\n{variant}: Error - {analysis['error']}")

    except Exception as e:
        print(f"Analysis failed: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    main()
