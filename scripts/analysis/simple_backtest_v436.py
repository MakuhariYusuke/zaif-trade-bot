#!/usr/bin/env python3
"""
Simple backtest script for v436 models - avoiding schema metadata issues
"""

import json
import os
import sys
from pathlib import Path


# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from ztb.analysis.sac_backtester import SACBacktester


def main():
    # Use v435 model and config as they exist
    model_path = "checkpoints/sac_v435_test_1000_steps.zip"
    config_path = "config/sac_v435_unified_config.json"
    data_path = "data/btc_jpy_featured_dataset.csv"
    output_path = "backtest_experiments/v436.1/backtest_v435_simple_results.json"

    print("🚀 Running simple backtest for v435 model...")

    # Check if files exist
    if not os.path.exists(model_path):
        print(f"❌ Model not found: {model_path}")
        return

    if not os.path.exists(config_path):
        print(f"❌ Config not found: {config_path}")
        return

    if not os.path.exists(data_path):
        print(f"❌ Data not found: {data_path}")
        return

    # Create backtester
    try:
        backtester = SACBacktester(model_path=model_path, config_path=config_path)
        # Override config to include feature_names for backtesting
        with open(config_path, "r") as f:
            config = json.load(f)
        backtester.config = config["training"]["features"]
        print("✅ Backtester created successfully")
    except Exception as e:
        print(f"❌ Failed to create backtester: {e}")
        return

    # Run backtest
    try:
        print("📊 Running backtest with 3 episodes...")
        results = backtester.run_backtest(
            data_path=data_path, num_episodes=3, deterministic=False
        )

        # Save results
        with open(output_path, "w") as f:
            json.dump(
                {
                    "performance_metrics": results.performance_metrics,
                    "trade_count": len(results.trade_log),
                    "total_reward": results.performance_metrics.get("total_reward", 0),
                    "portfolio_return_pct": results.performance_metrics.get(
                        "portfolio_return_pct", 0
                    ),
                    "sharpe_ratio": results.performance_metrics.get("sharpe_ratio", 0),
                    "max_drawdown": results.performance_metrics.get("max_drawdown", 0),
                    "win_rate": results.performance_metrics.get("win_rate", 0),
                    "all_trades": results.trade_log,  # Save all trades instead of sample
                    "sample_trades": results.trade_log[:5] if results.trade_log else [],
                },
                f,
                indent=2,
                default=str,
            )

        print(f"✅ Backtest completed. Results saved to {output_path}")

        # Print summary
        print("\n📊 Backtest Summary:")
        metrics = results.performance_metrics
        print(f"Total Reward: {metrics.get('total_reward', 'N/A')}")
        port_return = metrics.get("portfolio_return_pct", "N/A")
        if isinstance(port_return, (int, float)):
            print(f"Portfolio Return: {port_return:.2f}%")
        else:
            print(f"Portfolio Return: {port_return}")
        sharpe = metrics.get("sharpe_ratio", "N/A")
        if isinstance(sharpe, (int, float)):
            print(f"Sharpe Ratio: {sharpe:.3f}")
        else:
            print(f"Sharpe Ratio: {sharpe}")
        max_dd = metrics.get("max_drawdown", "N/A")
        if isinstance(max_dd, (int, float)):
            print(f"Max Drawdown: {max_dd:.2f}%")
        else:
            print(f"Max Drawdown: {max_dd}")
        win_rate = metrics.get("win_rate", "N/A")
        if isinstance(win_rate, (int, float)):
            print(f"Win Rate: {win_rate:.2f}")
        else:
            print(f"Win Rate: {win_rate}")
        print(f"Total Trades: {len(results.trade_log)}")

        # Analyze action distribution
        if results.trade_log:
            actions = [trade.get("action", 0) for trade in results.trade_log]
            action_counts = {}
            for action in actions:
                if isinstance(action, (int, float)):
                    bucket = round(action * 10) / 10  # Round to nearest 0.1
                    action_counts[bucket] = action_counts.get(bucket, 0) + 1

            print("\n🎯 Action Distribution (bucketed by 0.1):")
            for action_val in sorted(action_counts.keys()):
                print(f"{action_val:.1f}: {action_counts[action_val]}")
        else:
            print("\n🎯 No trades recorded - checking if model produces actions...")

            # Quick test: run a few steps manually to see actions
            env = backtester.env
            if env:
                obs = env.reset()
                if isinstance(obs, tuple):
                    obs = obs[0]  # Handle Gym API
                print("Testing model actions on first few observations:")
                for i in range(min(10, len(env.df))):
                    action, _ = backtester.model.predict(obs, deterministic=False)
                    print(f"Step {i}: obs_shape={obs.shape}, action={action}")
                    if i < len(env.df) - 1:
                        step_result = env.step(action)
                        if isinstance(step_result, tuple) and len(step_result) >= 2:
                            obs = step_result[0]
                            done = step_result[1] if len(step_result) >= 3 else False
                        else:
                            break
                        if done:
                            break

    except Exception as e:
        print(f"❌ Backtest failed: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    main()
