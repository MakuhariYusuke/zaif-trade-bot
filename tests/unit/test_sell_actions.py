#!/usr/bin/env python3
"""
Test SAC v445.3 model for SELL action verification
"""

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path.cwd()))

from stable_baselines3 import PPO

from ztb.trading.environment.heavy_env.core import HeavyTradingEnv


def main():
    print("Testing SAC v445.3 model for SELL actions...")

    # Load the trained model
    model_path = "models/sac_v445.3_strong_selling_optimized_final.zip"
    if not Path(model_path).exists():
        print(f"Model not found: {model_path}")
        return

    model = PPO.load(model_path)
    print("✅ Model loaded successfully")

    # Load config and create environment the same way as training
    config_path = "config/v445/sac_v445.3_strong_selling_optimized.json"
    import json

    with open(config_path, "r", encoding="utf-8") as f:
        config = json.load(f)

    # Load data
    data_path = config["training"]["data_config"]["data_path"]
    if not Path(data_path).exists():
        print(f"Data not found: {data_path}")
        return

    df = pd.read_csv(data_path)
    print(f"✅ Data loaded: {len(df)} rows")

    # Create environment config the same way as training
    from ztb.trading.environment.utils.config import EnvironmentConfig

    env_config_dict = config["training"]["environment"].copy()
    # Ensure continuous actions
    env_config_dict["use_continuous_actions"] = True

    # Extract reward_scaling from reward_settings if nested
    if "reward_settings" in env_config_dict and isinstance(
        env_config_dict["reward_settings"], dict
    ):
        if "reward_scaling" in env_config_dict["reward_settings"]:
            env_config_dict["reward_scaling"] = float(
                env_config_dict["reward_settings"]["reward_scaling"]
            )

    # Remove reward_settings to avoid conflicts
    if "reward_settings" in env_config_dict:
        del env_config_dict["reward_settings"]

    # Convert initial_balance to initial_portfolio_value if needed
    if "initial_balance" in env_config_dict:
        env_config_dict["initial_portfolio_value"] = env_config_dict.pop(
            "initial_balance"
        )

    # Remove fields that don't exist in EnvironmentConfig
    fields_to_remove = [
        "feature_engineering",
        "market_regime_detection",
        "risk_management",
        "multi_timeframe_integration",
        "behavior_optimization",
    ]
    for field in fields_to_remove:
        env_config_dict.pop(field, None)

    env_config = EnvironmentConfig(**env_config_dict)
    env = HeavyTradingEnv(
        df=df.head(1000), config=env_config, use_continuous_actions=True
    )
    print("✅ Environment created")

    # Test actions
    obs, _ = env.reset()
    actions = []
    rewards = []

    print("Running 100 test steps...")
    for i in range(100):
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        actions.append(action[0])  # continuous action
        rewards.append(reward)

        if terminated or truncated:
            print(f"Episode ended at step {i}")
            break

    # Analyze actions
    sell_actions = sum(1 for a in actions if a < -0.3)  # threshold for SELL
    buy_actions = sum(1 for a in actions if a > 0.3)  # threshold for BUY
    hold_actions = len(actions) - sell_actions - buy_actions

    print("\n📊 Action Analysis Results:")
    print(f"Total steps tested: {len(actions)}")
    print(f"SELL actions: {sell_actions} ({sell_actions/len(actions)*100:.1f}%)")
    print(f"BUY actions: {buy_actions} ({buy_actions/len(actions)*100:.1f}%)")
    print(f"HOLD actions: {hold_actions} ({hold_actions/len(actions)*100:.1f}%)")
    print(f"Average reward: {sum(rewards)/len(rewards):.2f}")
    print(f"Total reward: {sum(rewards):.2f}")

    if sell_actions > 0:
        print("✅ SUCCESS: SELL actions are occurring!")
    else:
        print("❌ WARNING: No SELL actions detected")


if __name__ == "__main__":
    main()
