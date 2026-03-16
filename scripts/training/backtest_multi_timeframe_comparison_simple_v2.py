#!/usr/bin/env python3
"""
Simplified backtest script for multi-timeframe comparison
"""

import json
import os
import sys
from pathlib import Path

import pandas as pd
import numpy as np
from stable_baselines3 import SAC

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from ztb.training.environments.heavy_trading_env import HeavyTradingEnv
from ztb.training.environments.environment_config import EnvironmentConfig


def load_data():
    """Load market data with features."""
    data_path = "data/btc_jpy_featured_dataset.csv"
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Data file not found: {data_path}")

    df = pd.read_csv(data_path)
    print(f"✅ Loaded {len(df)} rows of market data")
    return df


def create_environment(df, feature_columns=None):
    """Create trading environment."""
    if feature_columns is None:
        # Use common technical indicators
        feature_columns = [
            'returns', 'sma_5', 'sma_10', 'sma_20', 'rsi_14',
            'macd', 'macd_signal', 'macd_hist', 'bb_upper', 'bb_middle', 'bb_lower',
            'volatility_10', 'volume_sma_5'
        ]

    # Filter to available columns
    available_features = [col for col in feature_columns if col in df.columns]
    print(f"Using features: {available_features}")

    config = EnvironmentConfig(
        initial_balance=10000.0,
        max_steps=1000,
        commission=0.001,
        max_position_size=1.0
    )

    env = HeavyTradingEnv(
        data=df,
        config=config,
        feature_columns=available_features
    )

    return env


def load_model(model_path):
    """Load SAC model."""
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found: {model_path}")

    model = SAC.load(model_path)
    print(f"✅ Loaded model: {model_path}")
    return model


def run_simple_backtest(model, env, num_episodes=10):
    """Run simple backtest evaluation."""
    results = []

    for episode in range(num_episodes):
        obs, info = env.reset()
        done = False
        total_reward = 0
        steps = 0

        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            total_reward += reward
            steps += 1

        final_balance = env.portfolio_value
        results.append({
            'episode': episode,
            'total_reward': total_reward,
            'final_balance': final_balance,
            'steps': steps
        })

        print(f"Episode {episode}: Reward={total_reward:.2f}, Balance={final_balance:.2f}")

    return results


def calculate_metrics(results):
    """Calculate performance metrics."""
    if not results:
        return {}

    balances = [r['final_balance'] for r in results]
    rewards = [r['total_reward'] for r in results]

    initial_balance = 10000.0

    total_return_pct = ((np.mean(balances) - initial_balance) / initial_balance) * 100
    win_rate_pct = (sum(1 for b in balances if b > initial_balance) / len(balances)) * 100

    return {
        'total_return_pct': total_return_pct,
        'win_rate_pct': win_rate_pct,
        'avg_final_balance': np.mean(balances),
        'std_final_balance': np.std(balances),
        'avg_reward': np.mean(rewards),
        'total_trades': len(results)  # Simplified
    }


def main():
    # Model paths for comparison
    models = {
        'multi_timeframe_enabled': 'models/sac_v435.7c.zip',
        'multi_timeframe_disabled': 'models/sac_v435.6.zip'
    }

    # Load data
    df = load_data()

    # Create environment
    env = create_environment(df)

    comparison_results = {}

    for config_name, model_path in models.items():
        print(f"\n🔍 Testing {config_name}...")

        try:
            # Load model
            model = load_model(model_path)

            # Run backtest
            results = run_simple_backtest(model, env, num_episodes=5)

            # Calculate metrics
            metrics = calculate_metrics(results)

            comparison_results[config_name] = {
                'metrics': metrics,
                'episode_results': results
            }

            print(f"✅ {config_name} completed: {metrics}")

        except Exception as e:
            print(f"❌ Failed {config_name}: {e}")
            comparison_results[config_name] = {'error': str(e)}

    # Save results
    output_path = "multi_timeframe_backtest_comparison_results.json"
    with open(output_path, 'w') as f:
        json.dump(comparison_results, f, indent=2, default=str)

    print(f"\n📊 Results saved to {output_path}")

    # Print summary
    print("\n📈 Comparison Summary:")
    for config_name, data in comparison_results.items():
        if 'metrics' in data:
            metrics = data['metrics']
            print(f"{config_name}: Return={metrics['total_return_pct']:.2f}%, Win Rate={metrics['win_rate_pct']:.2f}%")


if __name__ == "__main__":
    main()