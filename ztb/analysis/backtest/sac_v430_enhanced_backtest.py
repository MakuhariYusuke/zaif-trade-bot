#!/usr/bin/env python3
"""
SAC v430 Enhanced Backtest with Reward Function Testing
Test different reward function configurations to identify trading issues
"""

import sys
from pathlib import Path

import numpy as np

from ztb.io.data_loader import DataLoader
from ztb.io.json_io import read_json, write_json
from ztb.trading.constants import ACTION_BUY, ACTION_HOLD, ACTION_SELL
from ztb.trading.environment.constants import continuous_to_discrete_action

# Add project root to path
project_root = Path(__file__).resolve().parent
sys.path.insert(0, str(project_root))

def load_config(config_path):
    """Load configuration from file."""
    return read_json(config_path)

def create_enhanced_backtest_config(reward_config_name):
    """Create enhanced backtest configuration with detailed logging."""

    # Map config names to files
    config_map = {
        "original": "configs/v430/sac_v430_optimized.json",
        "fixed_incentives": "configs/v430/sac_v430_fixed_incentives_test.json",
        "profit_focused": "configs/v430/sac_v430_profit_focused_test.json",
        "balanced": "configs/v430/sac_v430_balanced_test.json",
    }

    if reward_config_name not in config_map:
        raise ValueError(f"Unknown config: {reward_config_name}")

    return load_config(config_map[reward_config_name])

def run_enhanced_backtest(config_name="original"):
    """Run enhanced backtest with detailed trade tracking."""

    print(f"🔬 SAC v430 Enhanced Backtest - {config_name}")
    print("=" * 60)

    try:
        # Load configuration
        config = create_enhanced_backtest_config(config_name)
        training_config = config["training"]
        reward_config = config["reward_function"]

        print("📋 Configuration:")
        print(f"   Reward config: {config_name}")
        print(f"   Reward scale: {reward_config['reward_scale']:.1f}")
        print(f"   Trading bonus: {reward_config['trading_bonus']:.6f}")
        print(f"   Sell penalty: {reward_config['sell_penalty']:.6f}")
        print(f"   Buy bonus: {reward_config['buy_bonus']:.6f}")
        print(f"   Hold penalty: {reward_config['hold_penalty']:.6f}")
        print()

        # Import here to avoid numpy issues at module level
        from ztb.trading.environment import HeavyTradingEnv
        from ztb.trading.environment.utils.config import EnvironmentConfig

        # Load data
        print("📊 Loading data...")
        data_path = "data/btc_jpy_real_dataset.csv"
        df = DataLoader.load_csv_strict(data_path)
        print(f"Loaded {len(df):,} rows")

        # Create environment config
        print("⚙️  Creating environment...")
        env_config_obj = EnvironmentConfig(
            reward_scaling=reward_config["reward_scale"],
            transaction_cost=0.0005,
            max_position_size=0.01,
            reward_position_penalty_scale=reward_config["trading_bonus"],
            use_continuous_actions=True,
        )

        env = HeavyTradingEnv(df=df, config=env_config_obj, random_start=False)

        # Initialize tracking variables
        obs = env.reset()
        total_reward = 0
        portfolio_history = [env.portfolio_value]
        actions_history = []
        rewards_history = []
        position_history = [env.position_manager.position]
        pnl_history = []

        # Track trades
        trades_executed = []
        step_count = 0

        print("🎯 Running backtest with enhanced tracking...")

        while step_count < len(df):
            # Get action from environment (random for testing reward function)
            # In real scenario, this would come from the trained model
            action = env.action_space.sample()  # Random action for testing

            # Execute action
            step_result = env.step(action)
            if len(step_result) == 5:
                obs, reward, terminated, truncated, info = step_result
                done = terminated or truncated
            else:
                obs, reward, done, info = step_result

            # Track data
            total_reward += reward
            portfolio_history.append(env.portfolio_value)
            actions_history.append(
                float(action[0]) if isinstance(action, np.ndarray) else float(action)
            )
            rewards_history.append(float(reward))
            position_history.append(env.position_manager.position)

            # Check for trades
            if "pnl" in info and info["pnl"] != 0:
                trades_executed.append(
                    {
                        "step": step_count,
                        "action": action,
                        "pnl": info["pnl"],
                        "position": env.position_manager.position,
                        "portfolio": env.portfolio_value,
                    }
                )

            step_count += 1

            if step_count % 1000 == 0:
                print(
                    f"Step {step_count}: Portfolio = {env.portfolio_value:.2f}, Trades = {len(trades_executed)}"
                )

            if done:
                break

        # Analyze results
        print()
        print("📊 Analysis Results:")
        print(f"   Total steps: {step_count}")
        print(f"   Total reward: {total_reward:.2f}")
        print(f"   Final portfolio: {portfolio_history[-1]:.2f}")
        print(f"   Trades executed: {len(trades_executed)}")
        print(
            f"   Portfolio change: {((portfolio_history[-1] - portfolio_history[0]) / portfolio_history[0] * 100):.3f}%"
        )

        # Action distribution
        actions = np.array(actions_history)
        discrete_actions = [continuous_to_discrete_action(a) for a in actions]
        buy_actions = discrete_actions.count(ACTION_BUY)
        sell_actions = discrete_actions.count(ACTION_SELL)
        hold_actions = discrete_actions.count(ACTION_HOLD)

        print(f"   BUY actions: {buy_actions} ({buy_actions/len(actions)*100:.1f}%)")
        print(f"   HOLD actions: {hold_actions} ({hold_actions/len(actions)*100:.1f}%)")
        print(f"   SELL actions: {sell_actions} ({sell_actions/len(actions)*100:.1f}%)")

        # Reward analysis
        rewards = np.array(rewards_history)
        print(f"   Average reward: {rewards.mean():.6f}")
        print(f"   Reward std: {rewards.std():.6f}")
        print(
            f"   Positive rewards: {np.sum(rewards > 0)} ({np.sum(rewards > 0)/len(rewards)*100:.1f}%)"
        )

        # Save detailed results
        results = {
            "config_name": config_name,
            "total_steps": step_count,
            "total_reward": float(total_reward),
            "initial_portfolio": float(portfolio_history[0]),
            "final_portfolio": float(portfolio_history[-1]),
            "trades_executed": len(trades_executed),
            "action_distribution": {
                "buy": int(buy_actions),
                "hold": int(hold_actions),
                "sell": int(sell_actions),
            },
            "reward_stats": {
                "mean": float(rewards.mean()),
                "std": float(rewards.std()),
                "positive_count": int(np.sum(rewards > 0)),
            },
            "trade_details": [
                dict(t) for t in trades_executed[:10]
            ],  # Convert to dict and take first 10
            "reward_config": reward_config,
        }

        results_path = f"results/sac_v430_backtest_{config_name}_enhanced.json"
        write_json(results_path, results, indent=2, ensure_ascii=False)

        print(f"\n✅ Results saved to: {results_path}")

        return results

    except Exception as e:
        print(f"❌ Enhanced backtest failed: {e}")
        import traceback

        traceback.print_exc()
        return None

def compare_configs():
    """Compare different reward function configurations."""

    print("\n🔄 Comparing Reward Function Configurations")
    print("=" * 60)

    configs_to_test = ["original", "fixed_incentives", "profit_focused", "balanced"]
    results = {}

    for config_name in configs_to_test:
        try:
            result = run_enhanced_backtest(config_name)
            if result:
                results[config_name] = result
        except Exception as e:
            print(f"❌ Failed to test {config_name}: {e}")

    # Summary comparison
    print("\n" + "=" * 60)
    print("📋 Configuration Comparison:")
    print("=" * 60)
    print("<15")
    print("-" * 60)

    for config_name, result in results.items():
        trades = result["trades_executed"]
        reward = result["total_reward"]
        portfolio_change = (
            (result["final_portfolio"] - result["initial_portfolio"])
            / result["initial_portfolio"]
            * 100
        )

        print("<15")

    # Find best configuration
    if results:
        best_config = max(results.items(), key=lambda x: x[1]["trades_executed"])
        print(f"\n🏆 Best configuration for trading activity: {best_config[0]}")
        print(f"   Trades executed: {best_config[1]['trades_executed']}")

def main():
    """Main function."""
    import argparse

    parser = argparse.ArgumentParser(
        description="SAC v430 Enhanced Backtest with Reward Function Testing"
    )
    parser.add_argument(
        "--config",
        default="original",
        choices=["original", "fixed_incentives", "profit_focused", "balanced"],
        help="Reward function configuration to test",
    )
    parser.add_argument(
        "--compare", action="store_true", help="Compare all configurations"
    )

    args = parser.parse_args()

    if args.compare:
        compare_configs()
    else:
        run_enhanced_backtest(args.config)

if __name__ == "__main__":
    main()
