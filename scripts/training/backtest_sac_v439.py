#!/usr/bin/env python3
"""
Backtest script for SAC v439 aggressive scalping model
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

from ztb.training.environments.heavy_trading_env import HeavyTradingEnv
from ztb.training.environments.environment_config import EnvironmentConfig
from ztb.analysis.sac_backtester import SACBacktester


def load_config_from_file(config_path: str) -> dict:
    """Load configuration from JSON file."""
    with open(config_path, "r") as f:
        return json.load(f)


def main():
    parser = argparse.ArgumentParser(description="Backtest SAC v439 aggressive scalping model")
    parser.add_argument("--config", type=str, default="config/v439/sac_v439_scalping_config.json", help="Path to config file")
    parser.add_argument("--model", type=str, default="models/v439/sac_v439_scalping_final.zip", help="Path to model file")
    parser.add_argument("--data", type=str, default="data/btc_jpy_featured_dataset.csv", help="Path to market data CSV")
    parser.add_argument("--episodes", type=int, default=10, help="Number of backtest episodes")
    parser.add_argument(
        "--output",
        type=str,
        default="results/backtest_v439_results.json",
        help="Output path",
    )

    args = parser.parse_args()

    # Check if model exists
    if not os.path.exists(args.model):
        print(f"❌ Model not found: {args.model}")
        return

    # Load data
    try:
        df = pd.read_csv(args.data)
        print(f"✅ Data loaded successfully: {len(df)} rows")
    except Exception as e:
        print(f"❌ Failed to load data: {e}")
        return

    # Load configuration
    try:
        config_data = load_config_from_file(args.config)
        print("✅ Configuration loaded successfully")
    except Exception as e:
        print(f"❌ Failed to load configuration: {e}")
        return

    # Create environment from config
    try:
        # Create environment similar to training script
        environment_settings = config_data["environment"]
        scalping_settings = environment_settings.get("scalping_optimization", {})
        signal_guidance_settings = environment_settings.get("signal_guidance", {})

        reward_config = config_data.get("reward_function", {})
        reward_settings = {
            "base_profit_bonus_atr_coeff": reward_config.get("base_profit_bonus_atr_coeff", 5.0),
            "base_profit_bonus_portfolio_coeff": reward_config.get("base_profit_bonus_portfolio_coeff", 10.0),
            "base_action_penalty": reward_config.get("base_action_penalty", 0.01),
            "loss_penalty_coeff": reward_config.get("loss_penalty_coeff", -1.0),
            "action_frequency_penalty": reward_config.get("action_frequency_penalty", 0.002),
            "target_action_rate": reward_config.get("target_action_rate", 0.55),
            "low_activity_penalty_scale": reward_config.get("low_activity_penalty_scale", 0.05),
            "overtrade_threshold": reward_config.get("overtrade_threshold", 0.95),
            "overtrade_penalty_scale": reward_config.get("overtrade_penalty_scale", 0.01),
            "hold_penalty_multiplier": reward_config.get("hold_penalty_multiplier", 1.2),
            "long_short_asymmetry": reward_config.get("long_short_asymmetry", True),
            "risk_adjusted_bonus": reward_config.get("risk_adjusted_bonus", True),
            "market_regime_penalty": reward_config.get("market_regime_penalty", True),
            "scalping_mode": reward_config.get("scalping_mode", True),
            "signal_guidance_integration": reward_config.get("signal_guidance_integration", True),
            "use_simple_reward": reward_config.get("use_simple_reward", False),
        }

        # Feature columns for scalping
        feature_columns = []
        if "features" in config_data:
            for category in ["technical_indicators", "price_features", "volatility_features"]:
                if category in config_data["features"]:
                    feature_columns.extend(config_data["features"][category])
        feature_columns = list(dict.fromkeys(feature_columns))

        env_config = EnvironmentConfig(
            initial_balance=environment_settings["initial_balance"],
            max_steps=environment_settings["max_steps"],
            commission=environment_settings["commission"],
            slippage=environment_settings["slippage"],
            max_position_size=environment_settings["max_position_size"],
            min_trade_size=environment_settings.get("min_trade_size", 1e-5),
            min_position_change=scalping_settings.get("min_position_change", 1e-5),
            reward_scaling=environment_settings["reward_scaling"],
            feature_names=feature_columns + ["balance_norm", "position", "unrealized_norm"],
            curriculum_stage=environment_settings.get("curriculum_stage", "pnl_focused"),
            continuous_to_discrete_threshold=scalping_settings.get("action_threshold", 0.01),
            continuous_to_discrete_threshold_neg=scalping_settings.get("negative_action_threshold"),
            signal_guidance_enabled=signal_guidance_settings.get("enabled", True),
            signal_guidance=signal_guidance_settings,
            scalping_optimization=scalping_settings,
        )

        env = HeavyTradingEnv(
            data=df,
            config=env_config,
            feature_columns=feature_columns,
            reward_settings=reward_settings,
        )
        print("✅ Environment created successfully")
    except Exception as e:
        print(f"❌ Failed to create environment: {e}")
        return

    # Load model
    try:
        from stable_baselines3 import SAC
        model = SAC.load(args.model)
        print("✅ Model loaded successfully")
    except Exception as e:
        print(f"❌ Failed to load model: {e}")
        return

    # Run backtest manually
    print(f"🚀 Starting backtest with {args.episodes} episodes...")
    
    total_rewards = []
    total_trades = []
    total_returns = []
    
    for episode in range(args.episodes):
        obs, info = env.reset()
        done = False
        episode_reward = 0
        episode_trades = 0
        initial_balance = env.balance
        
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            episode_reward += reward
            if info.get('trades_count', 0) > episode_trades:
                episode_trades = info['trades_count']
            done = terminated or truncated
        
        final_balance = env.balance
        total_return = (final_balance - initial_balance) / initial_balance
        
        total_rewards.append(episode_reward)
        total_trades.append(episode_trades)
        total_returns.append(total_return)
        
        print(f"Episode {episode + 1}: Reward={episode_reward:.2f}, Trades={episode_trades}, Return={total_return:.2%}")
    
    # Calculate summary
    avg_reward = sum(total_rewards) / len(total_rewards)
    avg_trades = sum(total_trades) / len(total_trades)
    avg_return = sum(total_returns) / len(total_returns)
    win_rate = sum(1 for r in total_returns if r > 0) / len(total_returns)
    
    results = {
        "total_episodes": args.episodes,
        "avg_reward": avg_reward,
        "avg_trades": avg_trades,
        "avg_return": avg_return,
        "win_rate": win_rate,
        "total_return": sum(total_returns),
        "sharpe_ratio": avg_return / (sum((r - avg_return)**2 for r in total_returns) / len(total_returns))**0.5 if len(total_returns) > 1 else 0,
        "max_drawdown": min(total_returns) if total_returns else 0,
        "episode_results": [
            {"reward": r, "trades": t, "return": ret}
            for r, t, ret in zip(total_rewards, total_trades, total_returns)
        ]
    }

    # Save results
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(results, f, indent=2, default=str)

    print(f"✅ Backtest completed! Results saved to {args.output}")

    # Print summary
    print("\n📊 Backtest Summary:")
    print(f"Total Episodes: {results['total_episodes']}")
    print(f"Average Reward: {results['avg_reward']:.2f}")
    print(f"Average Trades: {results['avg_trades']:.1f}")
    print(f"Win Rate: {results['win_rate']:.1%}")
    print(f"Total Return: {results['total_return']:.2%}")
    print(f"Sharpe Ratio: {results['sharpe_ratio']:.2f}")
    print(f"Max Drawdown: {results['max_drawdown']:.2%}")


if __name__ == "__main__":
    main()