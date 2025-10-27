#!/usr/bin/env python3
"""
Quick backtest for SAC v437.1 model with balanced features.
"""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Optional

import pandas as pd
from stable_baselines3 import SAC

sys.path.insert(0, str(Path(__file__).parent))

from ztb.features.sac_v427_feature_engineering import SACv427FeatureEngineer
from ztb.trading.environment.heavy_env.core import HeavyTradingEnv
from ztb.utils.logging_utils import get_logger

logger = get_logger(__name__)


def run_backtest(
    model_path: str,
    config_path: str,
    output_dir: str = "backtest_results",
    n_episodes: int = 3,
    deterministic: bool = True,
) -> Optional[dict]:
    """Run backtest for SAC model."""

    logger.info("🔍 Running SAC v437.1 backtest")

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

    # Create environment config
    env_config = {
        "initial_balance": config.get("environment", {}).get("initial_balance", 200000),
        "transaction_cost": config.get("environment", {}).get("transaction_cost", 0.001),
        "max_position_size": config.get("environment", {}).get("max_position_size", 1.0),
        "use_continuous_actions": config.get("environment", {}).get("use_continuous_actions", True),
        "use_standardized_observations": config.get("environment", {}).get("use_standardized_observations", True),
        "random_start": config.get("environment", {}).get("random_start", True),
        "curriculum_stage": config.get("environment", {}).get("curriculum_stage", "profit_optimized"),
        "continuous_to_discrete_threshold": config.get("environment", {}).get("continuous_to_discrete_threshold", 0.1),
        "feature_set": config.get("environment", {}).get("feature_set", "v427_full"),
        "reward_settings": config.get("environment", {}).get("reward_settings", {}),
    }

    # Create environment (let it generate features internally like during training)
    env = HeavyTradingEnv(df=df, config=env_config)

    # Load model
    logger.info(f"🤖 Loading model from {model_path}")
    model = SAC.load(model_path, env=env)

    # Run backtest
    results = []
    total_return = 0
    total_trades = 0

    for episode in range(n_episodes):
        logger.info(f"🏃 Running episode {episode + 1}/{n_episodes}")

        obs, info = env.reset()
        done = False
        episode_return = 0
        episode_trades = 0
        step = 0

        while not done:
            action, _ = model.predict(obs, deterministic=deterministic)
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            episode_return += reward
            step += 1

            # Count trades (position changes)
            if hasattr(env, 'position') and hasattr(env, '_last_position'):
                if env.position != getattr(env, '_last_position', 0):
                    episode_trades += 1
                env._last_position = env.position

        results.append({
            'episode': episode + 1,
            'return': episode_return,
            'trades': episode_trades,
            'steps': step
        })

        total_return += episode_return
        total_trades += episode_trades

        logger.info(f"Episode {episode + 1}: Return={episode_return:.2f}, Trades={episode_trades}")

    # Calculate metrics
    avg_return = total_return / n_episodes
    avg_trades = total_trades / n_episodes

    backtest_results = {
        'model': Path(model_path).name,
        'config': Path(config_path).name,
        'episodes': n_episodes,
        'average_return': avg_return,
        'average_trades': avg_trades,
        'total_return': total_return,
        'total_trades': total_trades,
        'results': results
    }

    # Save results
    os.makedirs(output_dir, exist_ok=True)
    timestamp = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
    result_file = Path(output_dir) / f"backtest_results_sac_v437_1_{timestamp}.json"

    with open(result_file, 'w') as f:
        json.dump(backtest_results, f, indent=2, default=str)

    logger.info(f"✅ Backtest completed!")
    logger.info(f"📊 Average Return: {avg_return:.2f}")
    logger.info(f"📊 Average Trades: {avg_trades:.1f}")
    logger.info(f"💾 Results saved to: {result_file}")

    return backtest_results


def main():
    parser = argparse.ArgumentParser(description="Quick backtest for SAC v437.1")
    parser.add_argument("--model_path", required=True, help="Path to trained model")
    parser.add_argument("--config", required=True, help="Path to config file")
    parser.add_argument("--output_dir", default="backtest_results", help="Output directory")
    parser.add_argument("--n_episodes", type=int, default=3, help="Number of episodes")
    parser.add_argument("--deterministic", action="store_true", help="Use deterministic actions")

    args = parser.parse_args()

    run_backtest(
        model_path=args.model_path,
        config_path=args.config,
        output_dir=args.output_dir,
        n_episodes=args.n_episodes,
        deterministic=args.deterministic,
    )


if __name__ == "__main__":
    main()