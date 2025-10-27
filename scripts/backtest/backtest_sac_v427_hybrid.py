#!/usr/bin/env python3
"""
Quick backtest for SAC v427 hybrid model with 109 quality-filtered features.
"""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from stable_baselines3 import SAC
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent))

from ztb.trading.environment.constants import continuous_to_discrete_action
from ztb.utils.logging_utils import get_logger
from ztb.features.sac_v427_feature_engineering import generate_v427_quality_filtered_features
from ztb.trading.environment.heavy_env.core import HeavyTradingEnv

logger = get_logger(__name__)


def run_backtest(
    model_path: str,
    config_path: str,
    output_dir: str = "backtest_results",
    n_episodes: int = 3,
    deterministic: bool = True,
) -> Optional[dict]:
    """Run backtest for SAC v427 hybrid model."""

    logger.info("🔍 Running SAC v427 hybrid backtest (109 quality-filtered features)")

    if not Path(model_path).exists():
        logger.error(f"❌ Model not found: {model_path}")
        logger.info("💡 Please run training first")
        return None

    # Load config
    with open(config_path, 'r') as f:
        config = json.load(f)

    # Load data
    data_path = config.get("data_path", "data/btc_jpy_real_dataset.csv")
    if not Path(data_path).exists():
        logger.error(f"❌ Data file not found: {data_path}")
        return None

    logger.info(f"📊 Loading data from {data_path}")
    df = pd.read_csv(data_path)
    logger.info(f"📈 Data loaded: {len(df)} rows, {len(df.columns)} columns")

    # Generate 109-dimensional quality-filtered features
    logger.info("🔧 Generating 109 quality-filtered v427 features...")
    features_df = generate_v427_quality_filtered_features(df, feature_set="full")
    logger.info(f"✅ Generated {len(features_df.columns)} features")

    # Create environment
    logger.info("🏭 Creating trading environment...")
    env = HeavyTradingEnv(
        df=features_df,
        config=config,
    )

    # Load model
    logger.info(f"🤖 Loading model from {model_path}")
    model = SAC.load(model_path)

    # Run backtest
    logger.info(f"🚀 Running backtest with {n_episodes} episodes...")

    episode_rewards = []
    episode_lengths = []
    all_actions = []
    all_rewards = []
    trades_history = []

    for episode in range(n_episodes):
        logger.info(f"📊 Episode {episode + 1}/{n_episodes}")

        obs, info = env.reset()
        episode_reward = 0
        episode_length = 0
        episode_actions = []
        episode_rewards_list = []

        done = False
        while not done:
            action, _ = model.predict(obs, deterministic=deterministic)
            
            # Convert continuous action to discrete action for HeavyTradingEnv
            if hasattr(action, 'shape') and len(action.shape) > 0:
                # Continuous action (Box space)
                discrete_action = continuous_to_discrete_action(action[0])
            else:
                # Already discrete
                discrete_action = int(action)
            
            episode_actions.append(float(action[0]) if hasattr(action, 'shape') else float(action))

            obs, reward, terminated, truncated, info = env.step(discrete_action)
            episode_reward += reward
            episode_length += 1
            episode_rewards_list.append(reward)

            done = terminated or truncated

            # Record trade if any
            if hasattr(env, 'last_trade') and env.last_trade:
                trades_history.append({
                    'episode': episode,
                    'step': episode_length,
                    'action': float(action[0]),
                    'reward': reward,
                    'portfolio_value': info.get('portfolio_value', 0),
                    'trade_type': env.last_trade.get('type', ''),
                    'trade_size': env.last_trade.get('size', 0),
                })

        episode_rewards.append(episode_reward)
        episode_lengths.append(episode_length)
        all_actions.extend(episode_actions)
        all_rewards.extend(episode_rewards_list)

        logger.info(f"  💰 Episode reward: {episode_reward:.2f}")
        logger.info(f"  📏 Episode length: {episode_length}")
        logger.info(f"  📊 Final portfolio value: {info.get('portfolio_value', 0):.2f}")

    # Calculate statistics
    total_reward = sum(episode_rewards)
    avg_episode_reward = np.mean(episode_rewards)
    total_trades = len(trades_history)

    # Calculate win rate and other metrics
    if trades_history:
        winning_trades = [t for t in trades_history if t['reward'] > 0]
        win_rate = len(winning_trades) / len(trades_history) if trades_history else 0

        # Calculate portfolio metrics
        final_portfolio_values = [t['portfolio_value'] for t in trades_history[-n_episodes:]]
        if final_portfolio_values:
            final_portfolio_value = final_portfolio_values[-1]
            portfolio_return_pct = ((final_portfolio_value - 10000) / 10000) * 100  # Assuming 10k starting capital
        else:
            final_portfolio_value = 10000
            portfolio_return_pct = 0

        # Calculate Sharpe ratio (simplified)
        if len(all_rewards) > 1:
            returns = pd.Series(all_rewards)
            sharpe_ratio = returns.mean() / returns.std() * np.sqrt(252) if returns.std() > 0 else 0
        else:
            sharpe_ratio = 0

        # Calculate max drawdown (simplified)
        if trades_history:
            portfolio_values = [t['portfolio_value'] for t in trades_history]
            if portfolio_values:
                peak = portfolio_values[0]
                max_drawdown = 0
                for value in portfolio_values:
                    if value > peak:
                        peak = value
                    drawdown = (peak - value) / peak
                    max_drawdown = max(max_drawdown, drawdown)
                max_drawdown *= 100  # Convert to percentage
            else:
                max_drawdown = 0
        else:
            max_drawdown = 0
    else:
        win_rate = 0
        final_portfolio_value = 10000
        portfolio_return_pct = 0
        sharpe_ratio = 0
        max_drawdown = 0

    backtest_results = {
        "model": "sac_v427_hybrid_109d",
        "timestamp": datetime.now().isoformat(),
        "total_reward": total_reward,
        "avg_episode_reward": avg_episode_reward,
        "total_trades": total_trades,
        "final_portfolio_value": final_portfolio_value,
        "portfolio_return_pct": portfolio_return_pct,
        "sharpe_ratio": sharpe_ratio,
        "max_drawdown": max_drawdown,
        "win_rate": win_rate,
        "evaluation_episodes": n_episodes,
        "feature_count": len(features_df.columns),
        "action_stats": {
            "mean": float(np.mean(all_actions)),
            "std": float(np.std(all_actions)),
            "min": float(np.min(all_actions)),
            "max": float(np.max(all_actions))
        },
        "reward_stats": {
            "mean": float(np.mean(all_rewards)),
            "std": float(np.std(all_rewards)),
            "min": float(np.min(all_rewards)),
            "max": float(np.max(all_rewards))
        }
    }

    # Save results
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    result_file = Path(output_dir) / f"backtest_results_sac_v427_hybrid_{timestamp}.json"

    with open(result_file, 'w') as f:
        json.dump(backtest_results, f, indent=2, default=str)

    # Save trades history
    trades_file = Path(output_dir) / f"trades_history_sac_v427_hybrid_{timestamp}.csv"
    if trades_history:
        trades_df = pd.DataFrame(trades_history)
        trades_df.to_csv(trades_file, index=False)

    logger.info(f"✅ Backtest completed!")
    logger.info(f"📄 Results saved to {result_file}")
    if trades_history:
        logger.info(f"📄 Trades history saved to {trades_file}")

    return backtest_results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Quick backtest for SAC v427 hybrid model")
    parser.add_argument("--model_path", required=True, help="Path to trained model")
    parser.add_argument("--config", default="config/sac_v427_default_config.json", help="Path to config file")
    parser.add_argument("--output_dir", default="backtest_results", help="Output directory")
    parser.add_argument("--episodes", type=int, default=3, help="Number of episodes to run")
    parser.add_argument("--deterministic", action="store_true", default=True, help="Use deterministic actions")

    args = parser.parse_args()

    run_backtest(
        model_path=args.model_path,
        config_path=args.config,
        output_dir=args.output_dir,
        n_episodes=args.episodes,
        deterministic=args.deterministic,
    )