#!/usr/bin/env python3
"""
SAC v430 Reward Function Fix Experiment
Test different reward function configurations to fix the trading issue
"""

import os
import sys
from pathlib import Path

import numpy as np

from ztb.trading.constants import ACTION_BUY, ACTION_HOLD, ACTION_SELL
from ztb.trading.environment.constants import continuous_to_discrete_action
from ztb.io.data_loader import DataLoader
from ztb.io.json_io import read_json, write_json

# Add project root to path
project_root = Path(__file__).resolve().parent
sys.path.insert(0, str(project_root))

def create_fixed_reward_configs():
    """Create reward function configurations that encourage trading."""

    configs = {}

    # Config 1: Remove penalties, add incentives
    configs["fixed_incentives"] = {
        "reward_scale": 140.26367385248548,
        "trading_bonus": 0.01,  # Increased trading bonus
        "sell_penalty": 0.0,  # Remove sell penalty
        "buy_bonus": 0.0,  # Remove buy penalty (make it neutral)
        "action_balance_weight": 0.1,  # Reduce action balance weight
        "hold_penalty": 0.0,  # Remove hold penalty
        "profit_focus": False,
        "risk_penalty": 0.0642814422601983,
    }

    # Config 2: Profit-focused with positive incentives
    configs["profit_focused"] = {
        "reward_scale": 140.26367385248548,
        "trading_bonus": 0.005,
        "sell_penalty": 0.001,  # Small positive incentive for selling
        "buy_bonus": 0.001,  # Small positive incentive for buying
        "action_balance_weight": 0.05,
        "hold_penalty": -0.001,  # Small penalty for not trading
        "profit_focus": True,  # Enable profit focus
        "risk_penalty": 0.03,  # Reduce risk penalty
    }

    # Config 3: Balanced approach
    configs["balanced"] = {
        "reward_scale": 140.26367385248548,
        "trading_bonus": 0.002,
        "sell_penalty": -0.001,  # Small penalty for excessive selling
        "buy_bonus": -0.001,  # Small penalty for excessive buying
        "action_balance_weight": 0.2,
        "hold_penalty": 0.001,  # Small incentive for balanced actions
        "profit_focus": False,
        "risk_penalty": 0.05,
    }

    return configs

def create_test_backtest_config(reward_config, name):
    """Create a backtest configuration with modified reward function."""

    # Load base config
    base_config = read_json("configs/v430/sac_v430_optimized.json")

    # Modify reward function
    base_config["reward_function"] = reward_config
    base_config["description"] = f"SAC v430 with {name} reward fix"

    # Save modified config
    config_path = f"configs/v430/sac_v430_{name}_test.json"
    os.makedirs(os.path.dirname(config_path), exist_ok=True)

    write_json(config_path, base_config, indent=2, ensure_ascii=False)

    return config_path

def run_quick_backtest_test(config_path, name):
    """Run a quick backtest test with modified config."""

    print(f"\n🧪 Testing {name} configuration...")
    print("-" * 40)

    try:
        # Import required modules
        from stable_baselines3 import SAC

        # Load config
        config = read_json(config_path)

        reward_config = config["reward_function"]

        # Load model
        model_path = "models/sac_v430_full/final_model.zip"
        if not os.path.exists(model_path):
            print(f"❌ Model not found: {model_path}")
            return None

        model = SAC.load(model_path)

        # Load data (small subset for quick test)
        data_path = "data/btc_jpy_real_dataset.csv"
        df = DataLoader.load_csv_strict(data_path)
        df = df.head(1000)  # Use only first 1000 rows for quick test

        # Create environment
        from ztb.trading.environment import HeavyTradingEnv
        from ztb.trading.environment.utils.config import EnvironmentConfig

        env_config_obj = EnvironmentConfig(
            reward_scaling=reward_config["reward_scale"],
            transaction_cost=0.0005,
            max_position_size=0.01,
            reward_position_penalty_scale=reward_config["trading_bonus"],
            use_continuous_actions=True,
        )

        env = HeavyTradingEnv(df=df, config=env_config_obj, random_start=False)

        # Run quick backtest
        obs = env.reset()
        total_reward = 0
        actions_taken = []
        portfolio_values = []

        for step in range(len(df)):
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, info = env.step(action)

            total_reward += reward
            actions_taken.append(float(action[0]))
            portfolio_values.append(env.portfolio_value)

            if done:
                break

        # Analyze results
        actions = np.array(actions_taken)

        # Count discrete actions using the standard function
        discrete_actions = [continuous_to_discrete_action(a) for a in actions]
        buy_actions = discrete_actions.count(ACTION_BUY)
        sell_actions = discrete_actions.count(ACTION_SELL)
        hold_actions = discrete_actions.count(ACTION_HOLD)

        # Check portfolio changes (indicating trades)
        portfolio_changes = np.diff(portfolio_values)
        significant_changes = np.sum(np.abs(portfolio_changes) > 1.0)

        print(f"📊 Results for {name}:")
        print(f"   Steps: {len(actions)}")
        print(f"   Total reward: {total_reward:.2f}")
        print(f"   Portfolio start: {portfolio_values[0]:.2f}")
        print(f"   Portfolio end: {portfolio_values[-1]:.2f}")
        print(f"   BUY actions: {buy_actions} ({buy_actions/len(actions)*100:.1f}%)")
        print(f"   HOLD actions: {hold_actions} ({hold_actions/len(actions)*100:.1f}%)")
        print(f"   SELL actions: {sell_actions} ({sell_actions/len(actions)*100:.1f}%)")
        print(f"   Significant portfolio changes: {significant_changes}")

        if significant_changes > 10:  # More than 10 meaningful trades
            print("   ✅ Trading activity detected!")
            return True
        else:
            print("   ❌ Minimal trading activity")
            return False

    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback

        traceback.print_exc()
        return None

def main():
    """Main experiment function."""

    print("🔧 SAC v430 Reward Function Fix Experiment")
    print("=" * 60)

    # Create fixed reward configurations
    reward_configs = create_fixed_reward_configs()

    results = {}

    for name, reward_config in reward_configs.items():
        # Create test config
        config_path = create_test_backtest_config(reward_config, name)

        # Run test
        success = run_quick_backtest_test(config_path, name)

        results[name] = {
            "config_path": config_path,
            "success": success,
            "reward_config": reward_config,
        }

    # Summary
    print("\n" + "=" * 60)
    print("📋 Experiment Summary:")
    print("=" * 60)

    successful_configs = [name for name, result in results.items() if result["success"]]

    if successful_configs:
        print(f"✅ Successful configurations: {', '.join(successful_configs)}")
        print("\n🎯 Recommended next steps:")
        for name in successful_configs:
            config_path = results[name]["config_path"]
            print(
                f"   - Test full backtest with: python backtest_v430_only.py (modify to use {config_path})"
            )
            print("   - Or run: python ztb/trading/backtest/v430/backtest_v430_only.py")
    else:
        print("❌ No configurations showed significant trading activity")
        print("\n🔍 Further investigation needed:")
        print("   - Try more extreme reward modifications")
        print("   - Check if model is properly trained")
        print("   - Verify environment setup")

    print("\n📁 Generated config files:")
    for name, result in results.items():
        print(f"   {result['config_path']}")

if __name__ == "__main__":
    main()
