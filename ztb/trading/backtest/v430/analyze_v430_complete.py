#!/usr/bin/env python3
"""
SAC v430 Backtest & Validation - Complete Analysis
"""

import json
import os
import sys
from pathlib import Path

from ztb.utils.file_utils import get_project_root

# Add project root to path
project_root = get_project_root()
sys.path.insert(0, str(project_root))

# Direct imports to avoid complex dependencies
try:
    import pandas as pd
    import torch
    from stable_baselines3 import SAC
    from stable_baselines3.common.monitor import Monitor
    from stable_baselines3.common.vec_env import DummyVecEnv

    # Import environment and data handling
    from ztb.trading.environment import HeavyTradingEnv
    from ztb.trading.environment.utils.config import EnvironmentConfig
    from ztb.utils.logging_utils import get_logger

    logger = get_logger(__name__)

except ImportError as e:
    print(f"Import error: {e}")
    print("Required packages not available. Please install dependencies.")
    sys.exit(1)


def load_optimized_config():
    """Load the optimized SAC v430 configuration."""

    config_path = "configs/v430/sac_v430_optimized.json"
    with open(config_path, "r", encoding="utf-8") as f:
        config = json.load(f)

    return config


def run_backtest():
    """Run backtest with SAC v430 model."""

    print("🔬 SAC v430 Backtest & Validation")
    print("=" * 60)

    # Load optimized config
    config = load_optimized_config()
    training_config = config["training"]
    reward_config = config["reward_function"]
    sac_params = training_config

    print("📋 Model Configuration:")
    print("   Model: models/sac_v430_full/final_model.zip")
    print(f"   Reward scale: {reward_config['reward_scale']:.1f}")
    print()

    try:
        # Load model
        print("🤖 Loading trained model...")
        model_path = "models/sac_v430_full/final_model.zip"
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model not found: {model_path}")

        model = SAC.load(model_path)
        print("✅ Model loaded successfully")

        # Load data
        print("📊 Loading backtest data...")
        data_path = "data/btc_jpy_real_dataset.csv"
        df = pd.read_csv(data_path)
        print(f"Loaded {len(df):,} rows of data")

        # Create environment config
        print("⚙️  Creating environment configuration...")
        env_config_obj = EnvironmentConfig(
            reward_scaling=reward_config["reward_scale"],
            transaction_cost=0.0005,  # Default transaction cost
            max_position_size=0.01,  # Default max position size
            reward_position_penalty_scale=reward_config["trading_bonus"],
            use_continuous_actions=True,  # Enable continuous actions for SAC
        )

        # Create environment
        print("🚀 Creating backtest environment...")
        env = HeavyTradingEnv(
            df=df, config=env_config_obj, random_start=False
        )  # Use sequential data for backtest

        # Wrap environment
        env = Monitor(env)
        env = DummyVecEnv([lambda: env])

        # Run backtest
        print("🎯 Running backtest...")
        print(f"Total steps: {len(df)}")

        obs = env.reset()
        total_reward = 0
        portfolio_history = []
        actions_history = []
        timestamps = []

        step_count = 0
        done = False

        while not done and step_count < len(df):
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, info = env.step(action)

            total_reward += reward
            portfolio_history.append(env.envs[0].env.portfolio_value)
            actions_history.append(float(action[0]))
            timestamps.append(step_count)

            step_count += 1

            if step_count % 1000 == 0:
                print(
                    f"Step {step_count}: Portfolio = {env.envs[0].env.portfolio_value:.2f}"
                )

        # Create backtest results
        backtest_results = {
            "model_version": "sac_v430_optimized",
            "total_steps": step_count,
            "initial_portfolio": portfolio_history[0]
            if portfolio_history
            else 200000.0,
            "final_portfolio": portfolio_history[-1] if portfolio_history else 200000.0,
            "total_reward": total_reward,
            "portfolio_history": portfolio_history,
            "actions_history": actions_history,
            "timestamps": timestamps,
            "config": {
                "reward_scale": reward_config["reward_scale"],
                "transaction_cost": 0.0005,
                "max_position_size": 0.01,
            },
            "performance_metrics": {
                "total_return": (portfolio_history[-1] - portfolio_history[0])
                / portfolio_history[0]
                if portfolio_history
                else 0,
                "total_reward": total_reward,
                "avg_reward_per_step": total_reward / step_count
                if step_count > 0
                else 0,
            },
        }

        # Save backtest results
        results_path = "results/sac_v430_backtest_results.json"
        os.makedirs(os.path.dirname(results_path), exist_ok=True)

        with open(results_path, "w", encoding="utf-8") as f:
            json.dump(backtest_results, f, indent=2, ensure_ascii=False)

        print()
        print("=" * 60)
        print("✅ Backtest completed successfully!")
        print(f"⏱️  Steps executed: {step_count}")
        print(f"💰 Final balance: {portfolio_history[-1]:.2f}")
        print(f"📈 Total reward: {total_reward:.2f}")
        print(f"📊 Results saved to: {results_path}")

        return results_path

    except Exception as e:
        print()
        print("=" * 60)
        print("❌ Backtest failed!")
        print(f"Error: {e}")
        import traceback

        traceback.print_exc()
        print("=" * 60)
        return None


def run_validation(results_path):
    """Run validation analysis on backtest results."""

    print()
    print("🔍 Running Validation Analysis")
    print("=" * 60)

    try:
        # Import unified analyzer
        import argparse

        from ztb.analysis.unified_analyze import UnifiedAnalysisSuite

        # Create args for analyze_backtest
        args = argparse.Namespace()
        args.results = results_path
        args.training_report = None  # No training report for backtest-only analysis
        args.output = "results/sac_v430_validation_report.txt"

        # Run comparative analysis
        suite = UnifiedAnalysisSuite()
        result = suite.run(args)

        if result == 0:
            print("✅ Validation analysis completed successfully!")
            print(f"📄 Report saved to: {args.output}")

            # Display key results
            if os.path.exists(args.output):
                print()
                print("📊 Key Validation Results:")
                print("-" * 40)
                with open(args.output, "r", encoding="utf-8") as f:
                    content = f.read()
                    # Extract key metrics (simplified)
                    lines = content.split("\n")
                    for line in lines[:20]:  # Show first 20 lines
                        if any(
                            keyword in line.lower()
                            for keyword in [
                                "return",
                                "sharpe",
                                "sortino",
                                "max drawdown",
                                "win rate",
                            ]
                        ):
                            print(line.strip())

        else:
            print("❌ Validation analysis failed!")

        return result == 0

    except Exception as e:
        print(f"❌ Validation failed: {e}")
        import traceback

        traceback.print_exc()
        return False


def main():
    """Main function."""
    try:
        # Run backtest
        results_path = run_backtest()

        if results_path:
            # Run validation
            validation_success = run_validation(results_path)

            if validation_success:
                print()
                print("🎉 SAC v430 Complete Analysis Finished!")
                print("✅ Backtest + Validation completed successfully")
                return 0
            else:
                print()
                print("⚠️  Backtest completed but validation failed")
                return 1
        else:
            print()
            print("❌ Backtest failed")
            return 1

    except Exception as e:
        logger.error(f"Analysis failed with error: {e}")
        import traceback

        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
