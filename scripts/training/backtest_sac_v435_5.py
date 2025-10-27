#!/usr/bin/env python3
"""
Backtest script for SAC v435.5 micro frequency penalty model
"""

import json
import os
import sys
from pathlib import Path


# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from ztb.analysis.sac_backtester import SACBacktester


def load_config_from_file(config_path: str) -> dict:
    """Load configuration from JSON file."""
    with open(config_path, "r") as f:
        return json.load(f)


def main():
    # Define paths
    model_path = "models/sac_v435.5.zip"
    data_path = "data/btc_jpy_real_dataset.csv"
    config_dir = "backtest_experiments/v435.5"
    output_path = "results/backtest_v435_5_results.json"

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
        print(f"Model: {sac_config_data['model_name']}")
        print(f"Frequency penalty: {reward_config_data['action_frequency_penalty']}")

    except Exception as e:
        print(f"❌ Failed to load configurations: {e}")
        return

    # Create backtester
    try:
        backtester = SACBacktester(
            model_path=model_path,
            data_path=data_path,
            config={
                "sac": sac_config_data,
                "environment": env_config_data,
                "reward": reward_config_data,
            },
        )

        print("✅ Backtester created successfully")

        # Run backtest
        print("🚀 Running backtest...")
        results = backtester.run_backtest()

        # Save results
        with open(output_path, "w") as f:
            json.dump(results, f, indent=2, default=str)

        print(f"✅ Backtest completed! Results saved to {output_path}")

        # Print summary
        print("\n📊 Backtest Summary:")
        print(f"Total Return: {results.get('total_return', 'N/A')}")
        print(f"Total Trades: {results.get('total_trades', 'N/A')}")
        print(f"Win Rate: {results.get('win_rate', 'N/A')}")
        print(f"Max Drawdown: {results.get('max_drawdown', 'N/A')}")

    except Exception as e:
        print(f"❌ Backtest failed: {e}")


if __name__ == "__main__":
    main()
