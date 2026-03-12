#!/usr/bin/env python3
"""
Check environment features and observation space
"""

import json
import sys
from pathlib import Path

import pandas as pd

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from ztb.trading.environment.heavy_env.core import HeavyTradingEnv
from ztb.trading.environment.utils.config import EnvironmentConfig


def main():
    # Load config
    with open("config/sac_v435_unified_config.json", "r") as f:
        config = json.load(f)

    print(
        "Config training environment feature_names:",
        config["training"]["environment"].get("feature_names"),
    )
    print(
        "Config training features feature_names:",
        config["training"]["features"].get("feature_names"),
    )

    # Load data
    data_path = "data/btc_jpy_featured_dataset.csv"
    df = pd.read_csv(data_path)
    print(f"Data columns: {len(df.columns)}")
    print(f"Data columns: {list(df.columns)}")

    # Create environment config
    env_config = EnvironmentConfig(
        max_position_size=config["training"]["environment"].get(
            "max_position_size", 1.0
        ),
        transaction_cost=config["training"]["environment"].get("transaction_cost", 0.0),
        reward_scaling=config["training"]["environment"].get("reward_scaling", 1.0),
        feature_names=config["training"]["features"].get("feature_names"),
    )

    # Create environment
    try:
        env = HeavyTradingEnv(df=df, config=env_config)
        print(f"Environment observation space: {env.observation_space}")
        print(f"Observation shape: {env.observation_space.shape}")
        print(f"Number of features: {len(env.features)}")
        print(f"Features: {env.features}")
    except Exception as e:
        print(f"Error creating environment: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    main()
