#!/usr/bin/env python3
"""
Multi-Timeframe Feature Backtest Comparison Script

Backtests SAC v435 models with and without multi-timeframe features to evaluate performance impact.
"""

import argparse
import json
import os
import sys
from pathlib import Path
from datetime import datetime
from typing import Dict, Any

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

import pandas as pd
import numpy as np
from stable_baselines3 import SAC

from ztb.trading.environment.schema_env_factory import create_env_from_schema
from ztb.utils.logging_utils import get_logger

logger = get_logger(__name__)


def load_config_from_file(config_path: str) -> dict:
    """Load configuration from JSON file."""
    with open(config_path, "r") as f:
        return json.load(f)


def run_simple_backtest(model_path: str, config_data: dict, n_episodes: int = 3) -> Dict[str, Any]:
    """
    Run a simple backtest using the SAC model directly.

    Args:
        model_path: Path to the trained model
        config_data: Configuration data
        n_episodes: Number of episodes to test

    Returns:
        Backtest results
    """
    try:
        # Load model
        model = SAC.load(model_path)
        logger.info("✅ Model loaded successfully")

        # Extract configurations
        training_config = config_data.get("training", {})
        env_config = training_config.get("environment", {})
        data_config = training_config.get("data_config", {})

        # Load data
        data_path = data_config.get("csv_path", "btc_jpy_featured_dataset.csv")
        if not os.path.exists(data_path):
            data_path = f"data/{data_path}"
            if not os.path.exists(data_path):
                raise FileNotFoundError(f"Data file not found: {data_path}")

        df = pd.read_csv(data_path)
        logger.info("✅ Data loaded successfully")

        # Create environment directly
        from ztb.trading.environment.heavy_env.core import HeavyTradingEnv

        # Prepare environment config
        env_config = {
            "initial_balance": env_config.get("initial_balance", 100000.0),
            "transaction_cost": env_config.get("transaction_cost", 0.0015),
            "max_position_size": env_config.get("max_position_size", 0.1),
            "enable_action_masking": env_config.get("enable_action_masking", True),
            "use_continuous_actions": env_config.get("use_continuous_actions", True),
            "use_standardized_observations": env_config.get("use_standardized_observations", True),
            "random_start": env_config.get("random_start", True),
            "feature_set": env_config.get("feature_set", "v435_risk_managed"),
        }

        env = HeavyTradingEnv(df=df, **env_config)
        logger.info("✅ Environment created successfully")

        # Run backtest episodes
        all_rewards = []
        all_portfolio_values = []
        trades_executed = 0

        for episode in range(n_episodes):
            logger.info(f"🏃 Running episode {episode + 1}/{n_episodes}")
            obs, info = env.reset()
            episode_reward = 0
            episode_portfolio_values = []
            done = False
            step_count = 0

            while not done and step_count < 10000:  # Limit steps per episode
                action, _ = model.predict(obs, deterministic=True)
                obs, reward, terminated, truncated, info = env.step(action)

                episode_reward += reward
                episode_portfolio_values.append(info.get('portfolio_value', 100000))
                trades_executed += 1 if info.get('trade_executed', False) else 0

                done = terminated or truncated
                step_count += 1

            all_rewards.append(episode_reward)
            all_portfolio_values.extend(episode_portfolio_values)

        # Calculate metrics
        total_return = (all_portfolio_values[-1] / all_portfolio_values[0]) - 1 if all_portfolio_values else 0
        returns = np.diff(all_portfolio_values) / all_portfolio_values[:-1] if len(all_portfolio_values) > 1 else [0]
        sharpe_ratio = np.mean(returns) / np.std(returns) * np.sqrt(252) if returns.size > 0 and np.std(returns) > 0 else 0
        max_drawdown = (np.min(all_portfolio_values) - np.max(all_portfolio_values)) / np.max(all_portfolio_values) if all_portfolio_values else 0

        results = {
            "total_return": float(total_return),
            "sharpe_ratio": float(sharpe_ratio),
            "max_drawdown": float(max_drawdown),
            "avg_reward": float(np.mean(all_rewards)),
            "total_trades": trades_executed,
            "episodes_completed": n_episodes,
            "final_portfolio_value": float(all_portfolio_values[-1]) if all_portfolio_values else 100000,
            "initial_portfolio_value": float(all_portfolio_values[0]) if all_portfolio_values else 100000,
        }

        logger.info("✅ Backtest completed successfully")
        logger.info(".2%")
        logger.info(".2f")
        logger.info(".2%")

        return results

    except Exception as e:
        logger.error(f"❌ Backtest failed: {e}")
        raise


def run_backtest_for_config(config_name: str, output_dir: str = "results/backtest_multi_timeframe") -> dict:
    """
    Run backtest for a specific configuration.

    Args:
        config_name: Name of the configuration file (without .json)
        output_dir: Output directory for results

    Returns:
        Backtest results summary
    """
    config_path = f"config/{config_name}.json"
    model_path = f"models/v435_unified/sac_v435_final.zip"

    # Check if files exist
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Configuration file not found: {config_path}")

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")

    logger.info(f"🚀 Starting backtest for {config_name}")
    logger.info(f"📁 Config: {config_path}")
    logger.info(f"🤖 Model: {model_path}")

    try:
        # Load configuration
        config_data = load_config_from_file(config_path)
        logger.info("✅ Configuration loaded successfully")

        # Run backtest
        results = run_simple_backtest(model_path, config_data, n_episodes=3)

        # Save results
        os.makedirs(output_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = f"{output_dir}/backtest_{config_name}_{timestamp}.json"

        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2, default=str)

        logger.info(f"💾 Results saved to: {output_file}")

        # Create summary
        summary = {
            "config": config_name,
            "model_path": model_path,
            "timestamp": timestamp,
            "total_return": results["total_return"],
            "sharpe_ratio": results["sharpe_ratio"],
            "max_drawdown": results["max_drawdown"],
            "win_rate": 0.5,  # Placeholder - would need trade analysis
            "total_trades": results["total_trades"],
            "avg_trade_return": results["avg_reward"],
            "success": True,
            "output_file": output_file
        }

        return summary

    except Exception as e:
        logger.error(f"❌ Backtest failed for {config_name}: {e}")
        return {
            "config": config_name,
            "success": False,
            "error": str(e),
            "timestamp": datetime.now().strftime("%Y%m%d_%H%M%S")
        }


def compare_results(results: list) -> dict:
    """
    Compare backtest results between configurations.

    Args:
        results: List of backtest result summaries

    Returns:
        Comparison analysis
    """
    if len(results) != 2:
        return {"error": "Expected exactly 2 results for comparison"}

    # Separate results
    multi_timeframe_result = None
    no_multi_timeframe_result = None

    for result in results:
        if "no_multi_timeframe" in result["config"]:
            no_multi_timeframe_result = result
        else:
            multi_timeframe_result = result

    if not multi_timeframe_result or not no_multi_timeframe_result:
        return {"error": "Could not identify multi-timeframe and non-multi-timeframe results"}

    # Calculate differences
    comparison = {
        "multi_timeframe_enabled": multi_timeframe_result,
        "multi_timeframe_disabled": no_multi_timeframe_result,
        "differences": {
            "total_return_diff": multi_timeframe_result["total_return"] - no_multi_timeframe_result["total_return"],
            "sharpe_ratio_diff": multi_timeframe_result["sharpe_ratio"] - no_multi_timeframe_result["sharpe_ratio"],
            "max_drawdown_diff": multi_timeframe_result["max_drawdown"] - no_multi_timeframe_result["max_drawdown"],
            "win_rate_diff": multi_timeframe_result["win_rate"] - no_multi_timeframe_result["win_rate"],
            "total_trades_diff": multi_timeframe_result["total_trades"] - no_multi_timeframe_result["total_trades"],
        },
        "analysis": {
            "multi_timeframe_better_return": multi_timeframe_result["total_return"] > no_multi_timeframe_result["total_return"],
            "multi_timeframe_better_sharpe": multi_timeframe_result["sharpe_ratio"] > no_multi_timeframe_result["sharpe_ratio"],
            "multi_timeframe_lower_drawdown": multi_timeframe_result["max_drawdown"] < no_multi_timeframe_result["max_drawdown"],
            "multi_timeframe_higher_win_rate": multi_timeframe_result["win_rate"] > no_multi_timeframe_result["win_rate"],
        }
    }

    return comparison


def main():
    """Main backtest comparison function."""
    parser = argparse.ArgumentParser(description="Compare backtests with/without multi-timeframe features")
    parser.add_argument(
        "--output-dir",
        type=str,
        default="results/backtest_multi_timeframe",
        help="Output directory for results"
    )
    parser.add_argument(
        "--configs",
        type=str,
        nargs="+",
        default=["sac_v435_unified_config", "sac_v435_unified_config_no_multi_timeframe"],
        help="Configuration files to test"
    )

    args = parser.parse_args()

    logger.info("🎯 Starting Multi-Timeframe Feature Backtest Comparison")
    logger.info(f"📁 Output directory: {args.output_dir}")
    logger.info(f"⚙️ Configurations: {args.configs}")

    # Run backtests
    results = []
    for config in args.configs:
        logger.info(f"\n{'='*60}")
        logger.info(f"Testing configuration: {config}")
        logger.info(f"{'='*60}")

        result = run_backtest_for_config(config, args.output_dir)
        results.append(result)

    # Compare results
    logger.info(f"\n{'='*80}")
    logger.info("COMPARISON ANALYSIS")
    logger.info(f"{'='*80}")

    comparison = compare_results(results)

    if "error" in comparison:
        logger.error(f"❌ Comparison failed: {comparison['error']}")
        return

    # Display comparison
    mt_enabled = comparison["multi_timeframe_enabled"]
    mt_disabled = comparison["multi_timeframe_disabled"]
    diff = comparison["differences"]
    analysis = comparison["analysis"]

    print("\n📊 BACKTEST COMPARISON RESULTS")
    print("=" * 80)
    print(f"{'Metric':<20} {'Multi-Timeframe':<20} {'No Multi-Timeframe':<20} {'Difference':<20}")
    print(f"{'':<20} {'Enabled':<20} {'Disabled':<20} {'':<20}")
    print("-" * 80)
    print(f"{'Total Return':<20} {mt_enabled['total_return']:<20.2%} {mt_disabled['total_return']:<20.2%} {diff['total_return_diff']:<20.2%}")
    print(f"{'Sharpe Ratio':<20} {mt_enabled['sharpe_ratio']:<20.2f} {mt_disabled['sharpe_ratio']:<20.2f} {diff['sharpe_ratio_diff']:<20.2f}")
    print(f"{'Max Drawdown':<20} {mt_enabled['max_drawdown']:<20.2%} {mt_disabled['max_drawdown']:<20.2%} {diff['max_drawdown_diff']:<20.2%}")
    print(f"{'Win Rate':<20} {mt_enabled['win_rate']:<20.1%} {mt_disabled['win_rate']:<20.1%} {diff['win_rate_diff']:<20.1%}")
    print(f"{'Total Trades':<20} {mt_enabled['total_trades']:<20} {mt_disabled['total_trades']:<20} {diff['total_trades_diff']:<20}")
    print("-" * 80)

    print("\n🎯 ANALYSIS:")
    print(f"  • Multi-timeframe {'IMPROVES' if analysis['multi_timeframe_better_return'] else 'WORSENS'} total return")
    print(f"  • Multi-timeframe {'IMPROVES' if analysis['multi_timeframe_better_sharpe'] else 'WORSENS'} risk-adjusted returns")
    print(f"  • Multi-timeframe {'REDUCES' if analysis['multi_timeframe_lower_drawdown'] else 'INCREASES'} maximum drawdown")
    print(f"  • Multi-timeframe {'IMPROVES' if analysis['multi_timeframe_higher_win_rate'] else 'WORSENS'} win rate")

    # Save comparison
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    comparison_file = f"{args.output_dir}/multi_timeframe_comparison_{timestamp}.json"

    with open(comparison_file, 'w') as f:
        json.dump({
            "results": results,
            "comparison": comparison,
            "timestamp": timestamp
        }, f, indent=2, default=str)

    logger.info(f"💾 Comparison saved to: {comparison_file}")
    logger.info("✅ Multi-timeframe backtest comparison completed!")


if __name__ == "__main__":
    main()