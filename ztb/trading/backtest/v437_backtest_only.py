#!/usr/bin/env python3
"""
SAC v437 Backtest Only - Generate Results for Analysis
"""

import json
import os
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).resolve().parent.parent.parent.parent
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
    """Load the optimized SAC v437 configuration."""

    config_path = "config/v437/sac_v437_enhanced_config.json"
    with open(config_path, "r", encoding="utf-8") as f:
        config = json.load(f)

    return config


def run_backtest():
    """Run backtest with SAC v437 model."""

    print("🔬 SAC v437 Backtest Execution")
    print("=" * 60)

    # Load optimized config
    config = load_optimized_config()
    training_config = config.get("training", {})
    environment_config = training_config.get("environment", {})
    reward_config = training_config.get("reward_function", {})
    sac_params = training_config.get("sac_hyperparameters", {})

    print("📋 Model Configuration:")
    print("   Model: models/sac_v437_enhanced_features.zip")
    print(f"   Reward scale: {environment_config.get('reward_scaling', 0.1)}")
    print()

    try:
        # Load model
        print("🤖 Loading trained model...")
        model_path = "models/sac_v437_enhanced_features.zip"
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
            reward_scaling=environment_config.get("reward_scaling", 0.1),
            transaction_cost=environment_config.get("transaction_cost", 0.0005),
            max_position_size=environment_config.get("max_position_size", 1.0),
            use_continuous_actions=environment_config.get(
                "use_continuous_actions", True
            ),
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
            portfolio_history.append(float(env.envs[0].env.portfolio_value))
            actions_history.append(
                float(action[0]) if hasattr(action, "__len__") else float(action)
            )
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
            "initial_portfolio": float(portfolio_history[0])
            if portfolio_history
            else 200000.0,
            "final_portfolio": float(portfolio_history[-1])
            if portfolio_history
            else 200000.0,
            "total_reward": float(total_reward),
            "portfolio_history": [float(x) for x in portfolio_history],
            "actions_history": [float(x) for x in actions_history],
            "timestamps": [int(x) for x in timestamps],
            "config": {
                "reward_scale": environment_config.get("reward_scaling", 0.1),
                "transaction_cost": 0.0005,
                "max_position_size": 0.01,
            },
            "performance_metrics": {
                "total_return": float(
                    (portfolio_history[-1] - portfolio_history[0])
                    / portfolio_history[0]
                )
                if portfolio_history
                else 0,
                "total_reward": float(total_reward),
                "avg_reward_per_step": float(total_reward / step_count)
                if step_count > 0
                else 0,
            },
        }

        # Save backtest results
        results_path = "backtest_results/sac_v437_backtest_results.json"
        os.makedirs(os.path.dirname(results_path), exist_ok=True)

        with open(results_path, "w", encoding="utf-8") as f:
            json.dump(backtest_results, f, indent=2, ensure_ascii=False)

        print()
        print("=" * 60)
        print("✅ Backtest completed successfully!")
        print(f"⏱️  Steps executed: {step_count}")
        print(f"💰 Final portfolio: {portfolio_history[-1]:.2f}")
        print(f"📈 Total reward: {float(total_reward):.2f}")
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


def main():
    """Main function."""
    try:
        results_path = run_backtest()
        if results_path:
            print()
            print("🎯 Next step: Run validation analysis")
            print(
                f"python ztb/analysis/unified_analyze.py comparative analyze_backtest --results {results_path}"
            )
            return 0
        else:
            return 1

    except Exception as e:
        logger.error(f"Backtest failed with error: {e}")
        import traceback

        traceback.print_exc()
        return 1


if __name__ == "__main__":
    print("🎯 SAC v437 Backtest Analysis")
    print("=" * 60)
    sys.exit(main())
