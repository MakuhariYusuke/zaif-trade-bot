#!/usr/bin/env python3
"""
SAC v433 Backtest & Analysis Script

Comprehensive backtesting and performance analysis for SAC v433 production model.
"""

import os
import sys
import json
from pathlib import Path
from typing import Dict, Any, Optional, List
from datetime import datetime

import numpy as np
import pandas as pd
from stable_baselines3 import SAC

TRADING_DAYS_PER_YEAR = 252

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from ztb.utils.logging_utils import get_logger
from ztb.trading.environment import HeavyTradingEnv
from ztb.trading.environment.utils.config import EnvironmentConfig

logger = get_logger(__name__)


class SACv433Backtester:
    """Comprehensive backtester for SAC v433 production model."""

    def __init__(self, model_path: str, initial_capital: float = 200000.0):
        self.model_path = model_path
        self.initial_capital = initial_capital
        self.model = None
        self.env = None
        self.results = {}
        self.load_model()

    def load_model(self) -> None:
        """Load the trained SAC v433 model."""
        try:
            if not os.path.exists(self.model_path):
                raise FileNotFoundError(f"Model file not found: {self.model_path}")

            self.model = SAC.load(self.model_path)
            logger.info(f"✅ Loaded SAC v433 model from {self.model_path}")
            logger.info(f"   Observation space: {self.model.observation_space}")
            logger.info(f"   Action space: {self.model.action_space}")
            logger.info(f"   Policy network: {self.model.policy.net_arch}")

        except Exception as e:
            logger.error(f"❌ Failed to load model: {e}")
            raise

    def create_environment(self, data_path: str) -> None:
        """Create trading environment for backtesting."""
        try:
            # Load data
            if not os.path.exists(data_path):
                raise FileNotFoundError(f"Data file not found: {data_path}")

            df = pd.read_csv(data_path)
            logger.info(f"✅ Loaded data: {len(df)} rows, {len(df.columns)} columns")

            # Environment configuration
            env_config = EnvironmentConfig(
                reward_scaling=1.0,
                transaction_cost=0.0015,
                max_position_size=1.0,
                reward_position_penalty_scale=0.1,
                use_continuous_actions=True,
            )

            # Create environment
            self.env = HeavyTradingEnv(df=df, config=env_config, random_start=False)
            logger.info("✅ Trading environment created successfully")

        except Exception as e:
            logger.error(f"❌ Failed to create environment: {e}")
            raise

    def run_backtest(self, num_episodes: int = 1) -> Dict[str, Any]:
        """Run comprehensive backtest."""
        logger.info(f"🚀 Starting SAC v433 backtest ({num_episodes} episodes)")

        all_results = []

        for episode in range(num_episodes):
            logger.info(f"Episode {episode + 1}/{num_episodes}")

            # Reset environment
            obs, info = self.env.reset()
            done = False
            episode_reward = 0
            episode_length = 0
            trades = []

            # Track portfolio value over time
            portfolio_values = [self.initial_capital]
            actions_taken = []
            rewards_received = []

            while not done:
                # Get action from model
                action, _ = self.model.predict(obs, deterministic=True)

                # Execute action in environment
                obs, reward, done, truncated, info = self.env.step(action)

                # Record data
                episode_reward += reward
                episode_length += 1

                # Track portfolio value
                if hasattr(self.env, 'portfolio_value'):
                    portfolio_values.append(self.env.portfolio_value)
                else:
                    # Estimate portfolio value from reward (approximate)
                    portfolio_values.append(portfolio_values[-1] + reward)

                actions_taken.append(action[0] if isinstance(action, np.ndarray) else action)
                rewards_received.append(reward)

                # Record trades
                if hasattr(self.env, 'last_trade') and self.env.last_trade:
                    trades.append(self.env.last_trade)

            # Calculate episode metrics
            episode_results = {
                'episode': episode + 1,
                'total_reward': episode_reward,
                'episode_length': episode_length,
                'final_portfolio_value': portfolio_values[-1],
                'total_return_pct': ((portfolio_values[-1] - self.initial_capital) / self.initial_capital) * 100,
                'num_trades': len(trades),
                'portfolio_values': portfolio_values,
                'actions': actions_taken,
                'rewards': rewards_received,
                'trades': trades
            }

            all_results.append(episode_results)
            logger.info(f"   Episode {episode + 1}: Return {episode_results['total_return_pct']:.2f}%, "
                       f"Trades: {episode_results['num_trades']}")

        # Aggregate results
        self.results = self._aggregate_results(all_results)
        return self.results

    def _aggregate_results(self, all_results: List[Dict]) -> Dict[str, Any]:
        """Aggregate results from multiple episodes."""
        if not all_results:
            return {}

        # Basic aggregation
        total_rewards = [r['total_reward'] for r in all_results]
        total_returns = [r['total_return_pct'] for r in all_results]
        num_trades = [r['num_trades'] for r in all_results]

        # Calculate metrics
        avg_return = np.mean(total_returns)
        std_return = np.std(total_returns)
        max_return = np.max(total_returns)
        min_return = np.min(total_returns)

        # Sharpe ratio (assuming daily returns, risk-free rate = 0)
        if std_return > 0:
            sharpe_ratio = avg_return / std_return
        else:
            sharpe_ratio = 0

        # Win rate (positive returns)
        win_rate = (np.array(total_returns) > 0).mean() * 100

        # Maximum drawdown calculation
        portfolio_values = []
        for result in all_results:
            portfolio_values.extend(result['portfolio_values'])

        if portfolio_values:
            peak = portfolio_values[0]
            max_drawdown = 0
            for value in portfolio_values:
                if value > peak:
                    peak = value
                drawdown = (peak - value) / peak * 100
                max_drawdown = max(max_drawdown, drawdown)
        else:
            max_drawdown = 0

        return {
            'summary': {
                'num_episodes': len(all_results),
                'avg_return_pct': avg_return,
                'std_return_pct': std_return,
                'max_return_pct': max_return,
                'min_return_pct': min_return,
                'sharpe_ratio': sharpe_ratio,
                'win_rate_pct': win_rate,
                'max_drawdown_pct': max_drawdown,
                'avg_trades_per_episode': np.mean(num_trades),
                'total_trades': sum(num_trades)
            },
            'episodes': all_results,
            'timestamp': datetime.now().isoformat()
        }

    def print_results(self) -> None:
        """Print comprehensive results summary."""
        if not self.results:
            logger.warning("No results to display")
            return

        summary = self.results['summary']

        print("\n" + "="*80)
        print("📊 SAC v433 BACKTEST RESULTS")
        print("="*80)
        print(f"Model: {os.path.basename(self.model_path)}")
        print(f"Episodes: {summary['num_episodes']}")
        print(f"Timestamp: {self.results['timestamp']}")
        print()

        print("🎯 PERFORMANCE METRICS:")
        print("-" * 40)
        print(f"Average Return: {summary['avg_return_pct']:.2f}%")
        print(f"Return Std Dev: {summary['std_return_pct']:.2f}%")
        print(f"Max Return: {summary['max_return_pct']:.2f}%")
        print(f"Min Return: {summary['min_return_pct']:.2f}%")
        print(f"Sharpe Ratio: {summary['sharpe_ratio']:.2f}")
        print(f"Win Rate: {summary['win_rate_pct']:.2f}%")
        print(f"Max Drawdown: {summary['max_drawdown_pct']:.2f}%")
        print(f"Total Trades: {summary['total_trades']}")
        print(f"Average Trades/Episode: {summary['avg_trades_per_episode']:.1f}")
        print()

        print("📈 DETAILED EPISODE RESULTS:")
        print("-" * 40)
        for episode in self.results['episodes']:
            print(f"Episode {episode['episode']:2d}: "
                  ".2f"
                  f"Trades: {episode['num_trades']:3d}")

        print("="*80)

    def save_results(self, output_path: str) -> None:
        """Save results to JSON file."""
        try:
            os.makedirs(os.path.dirname(output_path), exist_ok=True)

            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(self.results, f, indent=2, default=str)

            logger.info(f"✅ Results saved to {output_path}")

        except Exception as e:
            logger.error(f"❌ Failed to save results: {e}")


def main():
    """Main backtest function."""
    try:
        logger.info("🤖 SAC v433 Backtest & Analysis")
        logger.info("=" * 60)

        # Configuration
        model_path = "checkpoints/sac_v433_production_migration.zip"
        data_path = "data/btc_jpy_real_dataset.csv"
        output_path = "results/backtest_v433_production_results.json"
        num_episodes = 5  # Multiple episodes for statistical significance

        # Create backtester
        backtester = SACv433Backtester(model_path=model_path)

        # Create environment
        backtester.create_environment(data_path)

        # Run backtest
        results = backtester.run_backtest(num_episodes=num_episodes)

        # Print results
        backtester.print_results()

        # Save results
        backtester.save_results(output_path)

        logger.info("✅ SAC v433 backtest completed successfully!")
        print(f"\n📄 Detailed results saved to: {output_path}")

    except Exception as e:
        logger.error(f"❌ Backtest failed: {e}")
        raise


if __name__ == "__main__":
    main()