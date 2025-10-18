#!/usr/bin/env python3
"""
SAC v423b Backtest Script

Runs backtest for SAC v423b model using the trained policy.
"""

import os
import sys
from pathlib import Path
from typing import Dict, Any, Optional

import numpy as np
import pandas as pd
from stable_baselines3 import SAC

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from ztb.utils.logging_utils import get_logger
from ztb.trading.environment.heavy_env.core import HeavyTradingEnv
from ztb.trading.env_config import TradingEnvConfig

logger = get_logger(__name__)


class SACv423bBacktester:
    """Backtester for SAC v423b model."""

    def __init__(self, model_path: str, initial_capital: float = 200000.0):
        self.model_path = model_path
        self.initial_capital = initial_capital
        self.model = None
        self.env = None
        self.load_model()

    def load_model(self) -> None:
        """Load the trained SAC model."""
        try:
            self.model = SAC.load(self.model_path)
            logger.info(f"Loaded SAC model from {self.model_path}")
            logger.info(f"Model observation space: {self.model.observation_space}")
            logger.info(f"Model action space: {self.model.action_space}")
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise

    def create_environment(self, data_path: str) -> HeavyTradingEnv:
        """Create HeavyTradingEnv with same config as training."""
        # Load data first
        df = pd.read_csv(data_path)
        if 'timestamp' in df.columns:
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            df.set_index('timestamp', inplace=True)

        config = {
            "initial_balance": self.initial_capital,
            "transaction_cost": 1e-05,
            "max_position_size": 1.0,
            "enable_action_masking": False,
            "use_continuous_actions": True,
            "use_standardized_observations": True,
            "random_start": False,  # Disable for backtesting
            "curriculum_stage": "profit_optimized",
            "continuous_to_discrete_threshold": 0.1,
            # Reward settings matching training
            "reward_scale": 500.0,
            "reward_clip_min": -200.0,
            "reward_clip_max": 200.0,
            "profit_bonuses": {
                "base_profit_atr_coefficient": 1.5,
                "base_profit_portfolio_coefficient": 1.2,
                "profit_multipliers": [2.0, 0.6, 0.4],
                "trading_bonus": 0.01,
                "trading_bonus_multiplier": 4.0
            },
            "action_bonuses": {
                "hold_penalty": -0.001,
                "transaction_penalty": -0.01,
                "diversity_bonus": 0.01
            },
            "risk_management": {
                "max_drawdown_penalty": -0.1,
                "volatility_penalty": -0.05,
                "sharpe_ratio_bonus": 0.02
            }
        }

        env = HeavyTradingEnv(df=df, config=config)
        return env

    def run_backtest(self, data_path: str, slippage_bps: float = 5.0) -> Dict[str, Any]:
        """Run backtest simulation using HeavyTradingEnv."""
        logger.info("Starting backtest simulation...")

        # Create environment
        env = self.create_environment(data_path)
        self.env = env

        # Reset environment
        obs, info = env.reset()
        logger.info(f"Environment observation space: {env.observation_space}")
        logger.info(f"Environment action space: {env.action_space}")

        capital = self.initial_capital
        trades = []
        equity_curve = [capital]
        positions = []
        actions_taken = []

        done = False
        total_steps = 0

        while not done:
            try:
                # Get model prediction
                action, _ = self.model.predict(obs, deterministic=True)
                actions_taken.append(float(action[0]))

                # Step environment
                obs, reward, terminated, truncated, info = env.step(action)
                done = terminated or truncated

                # Record position and equity
                positions.append(env.position)
                equity_curve.append(env.portfolio_value)

                # Record trades (simplified - just when position changes)
                if len(positions) > 1 and positions[-1] != positions[-2]:
                    trade_type = "buy" if positions[-1] > positions[-2] else "sell"
                    trades.append({
                        'step': total_steps,
                        'type': trade_type,
                        'position_before': positions[-2],
                        'position_after': positions[-1],
                        'portfolio_value': env.portfolio_value,
                        'timestamp': getattr(env, 'current_timestamp', None)
                    })

                total_steps += 1

                # Safety check to prevent infinite loops
                if total_steps > 100000:
                    logger.warning("Backtest exceeded maximum steps, terminating")
                    break

            except Exception as e:
                logger.error(f"Error during backtest at step {total_steps}: {e}")
                break

        # Calculate performance metrics
        returns = pd.Series(equity_curve).pct_change().dropna()
        if len(returns) > 0:
            total_return = (equity_curve[-1] - equity_curve[0]) / equity_curve[0]
            max_drawdown = self.calculate_max_drawdown(equity_curve)
            sharpe_ratio = self.calculate_sharpe_ratio(returns)
        else:
            total_return = 0.0
            max_drawdown = 0.0
            sharpe_ratio = 0.0

        results = {
            'total_steps': total_steps,
            'initial_portfolio': self.initial_capital,
            'final_portfolio': equity_curve[-1] if equity_curve else self.initial_capital,
            'portfolio_history': equity_curve,
            'timestamps': list(range(len(equity_curve))),  # Simple step-based timestamps
            'total_return': total_return,
            'max_drawdown': max_drawdown,
            'sharpe_ratio': sharpe_ratio,
            'total_trades': len(trades),
            'win_rate': 0.0,  # Simplified
            'positions': positions,
            'actions': actions_taken,
            'trades': trades
        }

        env.close()
        return results

    def calculate_max_drawdown(self, equity_curve: list) -> float:
        """Calculate maximum drawdown from equity curve."""
        if not equity_curve:
            return 0.0

        peak = equity_curve[0]
        max_drawdown = 0.0

        for value in equity_curve:
            if value > peak:
                peak = value
            drawdown = (peak - value) / peak
            max_drawdown = max(max_drawdown, drawdown)

        return max_drawdown

    def calculate_sharpe_ratio(self, returns: pd.Series, risk_free_rate: float = 0.0) -> float:
        """Calculate Sharpe ratio."""
        if len(returns) == 0 or returns.std() == 0:
            return 0.0

        excess_returns = returns - risk_free_rate / 252  # Daily risk-free rate
        return excess_returns.mean() / returns.std() * np.sqrt(252)  # Annualized


def main():
    """Main backtest function."""
    print("🚀 SAC v423b Backtest")
    print("=" * 40)

    # Configuration
    model_path = "models/sac_v423b_step_test.zip"
    data_path = "data/btc_jpy_real_dataset.csv"
    initial_capital = 200000.0

    # Check if files exist
    if not os.path.exists(model_path):
        print(f"❌ Model file not found: {model_path}")
        return

    if not os.path.exists(data_path):
        print(f"❌ Data file not found: {data_path}")
        return

    try:
        # Create backtester
        backtester = SACv423bBacktester(model_path, initial_capital)

        # Run backtest
        results = backtester.run_backtest(data_path)

        # Print results
        print("\n📊 Backtest Results:")
        print(f"   Initial Capital: ¥{results['initial_portfolio']:,.0f}")
        print(f"   Final Equity: ¥{results['final_portfolio']:,.0f}")
        print(f"   Total Return: {results['total_return']:.2%}")
        print(f"   Max Drawdown: {results['max_drawdown']:.2%}")
        print(f"   Sharpe Ratio: {results['sharpe_ratio']:.2f}")
        print(f"   Total Trades: {results['total_trades']}")
        print(f"   Win Rate: {results['win_rate']:.1%}")
        print(f"   Total Steps: {results['total_steps']}")

        print("\n📈 Trade Summary:")
        for trade in results['trades'][:5]:  # Show first 5 trades
            print(f"   Step {trade['step']}: {trade['type']} "
                  f"(pos: {trade['position_before']:.2f} → {trade['position_after']:.2f})")

        if len(results['trades']) > 5:
            print(f"   ... and {len(results['trades']) - 5} more trades")

        print("\n✅ Backtest completed!")
        print(f"Model: {model_path}")
        print(f"Data: {data_path}")

        # Save results to JSON file
        import json
        results_file = "results/sac_v423b_backtest_results.json"
        os.makedirs("results", exist_ok=True)

        with open(results_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)

        print(f"Results saved to: {results_file}")

    except Exception as e:
        print(f"❌ Backtest failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()