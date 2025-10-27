#!/usr/bin/env python3
"""
Backtest script for SAC v436 signal guidance variants
"""

import argparse
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
    parser = argparse.ArgumentParser(description="Backtest SAC v436 models")
    parser.add_argument("--config", type=str, required=True, help="Path to config file")
    parser.add_argument("--model", type=str, required=True, help="Path to model file")
    parser.add_argument(
        "--output",
        type=str,
        default="results/backtest_v436_results.json",
        help="Output path",
    )

    args = parser.parse_args()

    # Check if model exists
    if not os.path.exists(args.model):
        print(f"❌ Model not found: {args.model}")
        return

    # Load configuration
    try:
        config_data = load_config_from_file(args.config)
        print("✅ Configuration loaded successfully")
    except Exception as e:
        print(f"❌ Failed to load config: {e}")
        return

    # Extract configurations
    training_config = config_data.get("training", {})
    env_config = training_config.get("environment", {})
    data_config = training_config.get("data_config", {})

    # Load data
    data_path = data_config.get("data_path", "data/btc_jpy_real_dataset.csv")
    if not os.path.exists(data_path):
        print(f"❌ Data not found: {data_path}")
        return

    try:
        df = pd.read_csv(data_path)
        print("✅ Data loaded successfully")
    except Exception as e:
        print(f"❌ Failed to load data: {e}")
        return

    # Create environment
    try:
        env = create_env_from_schema(
            model_name=training_config.get("model_name", "sac_v436"),
            df=df,
            config=env_config,
        )
        print("✅ Environment created successfully")
    except Exception as e:
        print(f"❌ Failed to create environment: {e}")
        return

    # Create backtester
    try:
        backtester = SACBacktester(model_path=args.model, config_path=args.config)
        # Override environment
        backtester.env = env
        print("✅ Backtester created successfully")
    except Exception as e:
        print(f"❌ Failed to create backtester: {e}")
        return

    # Run backtest
    try:
        print("🚀 Running backtest...")
        results = backtester.run_backtest(data_path=data_path, deterministic=False)

        # Save results
        with open(args.output, "w") as f:
            json.dump(results, f, indent=2, default=str)

        print(f"✅ Backtest completed. Results saved to {args.output}")

        # Print summary
        print("\n📊 Backtest Summary:")
        print(
            f"Total Return: {getattr(results, 'performance_metrics', {}).get('total_return', 'N/A')}"
        )
        print(
            f"Sharpe Ratio: {getattr(results, 'performance_metrics', {}).get('sharpe_ratio', 'N/A')}"
        )
        print(
            f"Max Drawdown: {getattr(results, 'performance_metrics', {}).get('max_drawdown', 'N/A')}"
        )
        print(
            f"Win Rate: {getattr(results, 'performance_metrics', {}).get('win_rate', 'N/A')}"
        )
        print(f"Total Trades: {len(getattr(results, 'trade_log', []))}")

        # Action distribution from trade log
        trade_log = getattr(results, "trade_log", [])
        action_counts = {}
        for trade in trade_log:
            action = trade.get("action", "unknown")
            action_counts[action] = action_counts.get(action, 0) + 1

        if action_counts:
            print("\n🎯 Action Distribution:")
            for action, count in action_counts.items():
                print(f"{action}: {count}")

    except Exception as e:
        print(f"❌ Backtest failed: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    main()
