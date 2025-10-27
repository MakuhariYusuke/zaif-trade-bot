#!/usr/bin/env python3
"""
Backtest script for SAC v435.3 scalping model
"""

import json
import os
import sys
from pathlib import Path

import pandas as pd

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from ztb.analysis.sac_backtester import SACBacktester
from ztb.trading.environment.schema_env_factory import create_env_from_schema


def load_config_from_file(config_path: str) -> dict:
    """Load configuration from JSON file."""
    with open(config_path, "r") as f:
        return json.load(f)


def main():
    # Define paths
    model_path = "models/sac_v435.3.zip"
    data_path = "data/btc_jpy_real_dataset.csv"
    config_dir = "backtest_experiments/v435.3"
    output_path = "results/backtest_v435_3_results.json"

    # Check if model exists
    if not os.path.exists(model_path):
        print(f"❌ Model not found: {model_path}")
        return

    # Check if data exists
    if not os.path.exists(data_path):
        print(f"❌ Data not found: {data_path}")
        return

    # Load configurations
    try:
        sac_config_data = load_config_from_file(f"{config_dir}/sac_v435_config.json")
        env_config_data = load_config_from_file(
            f"{config_dir}/sac_v435_environment_config.json"
        )
        reward_config_data = load_config_from_file(
            f"{config_dir}/sac_v435_reward_config.json"
        )

        print("✅ Configurations loaded successfully")
        print(f"   SAC Config keys: {list(sac_config_data.keys())}")
        print(f"   Environment Config keys: {list(env_config_data.keys())}")
        print(f"   Reward Config keys: {list(reward_config_data.keys())}")

    except Exception as e:
        print(f"❌ Failed to load configurations: {e}")
        return

    # Create backtester with model path and config
    try:
        backtester = SACBacktester(model_path=model_path, config_path=None)
        # Set config manually
        backtester.config = {**sac_config_data, **env_config_data, **reward_config_data}

        # Create environment using schema factory to match training setup
        df = pd.read_csv(data_path)
        env = create_env_from_schema("sac_v435.3", df, config=backtester.config)
        backtester.env = env

        print("✅ SAC Backtester and environment created successfully")

    except Exception as e:
        print(f"❌ Failed to create backtester: {e}")
        return

    # Run backtest
    try:
        print("🚀 Starting backtest for SAC v435.3...")
        result = backtester.run_backtest(data_path, num_episodes=10, deterministic=True)

        # Save results
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        result_dict = {
            "performance_metrics": result.performance_metrics,
            "trade_log": result.trade_log,
            "regime_analysis": result.regime_analysis,
        }
        with open(output_path, "w") as f:
            json.dump(result_dict, f, indent=2, ensure_ascii=False)

        print("✅ Backtest completed successfully!")
        print(f"   Results saved to: {output_path}")

        # Print summary
        backtester.print_report(result)

    except Exception as e:
        print(f"❌ Backtest failed: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    main()
