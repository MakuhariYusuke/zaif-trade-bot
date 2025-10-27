#!/usr/bin/env python3
"""
SAC v424 Cost-Aware Backtester

Integrated backtester for SAC v424 models from archived script.
Provides comprehensive backtesting capabilities with cost-aware trading simulation.
"""

import sys
from pathlib import Path
from typing import Any, Dict

import numpy as np
import pandas as pd
from stable_baselines3 import SAC

# Add project root to path
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

# 年間取引日数
from ztb.trading.constants import TRADING_DAYS_PER_YEAR  # = 252
from ztb.trading.environment.heavy_env.core import HeavyTradingEnv
from ztb.utils.logging_utils import get_logger

logger = get_logger(__name__)


class SACv424Backtester:
    """Backtester for SAC v424 cost-aware model."""

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
        """Create HeavyTradingEnv with cost-aware config matching v424 training."""
        # Load data first
        df = pd.read_csv(data_path)
        if "timestamp" in df.columns:
            df["timestamp"] = pd.to_datetime(df["timestamp"])
            df.set_index("timestamp", inplace=True)

        # v424 cost-aware configuration
        config = {
            "initial_balance": self.initial_capital,
            "transaction_cost": 1e-05,
            "max_position_size": 1.0,
            "enable_action_masking": False,
            "use_continuous_actions": True,
            "use_standardized_observations": True,
            "random_start": False,  # Disable for backtesting
            "curriculum_stage": "cost_aware_trading",
            "continuous_to_discrete_threshold": 0.1,
            # Reward settings matching v424 training
            "reward_scale": 500.0,
            "reward_clip_min": -200.0,
            "reward_clip_max": 200.0,
            "profit_bonuses": {
                "base_profit_atr_coefficient": 1.5,
                "base_profit_portfolio_coefficient": 1.2,
                "profit_multipliers": [2.0, 0.6, 0.4],
                "trading_bonus": 0.0,  # Removed trading bonus
                "trading_bonus_multiplier": 1.0,
            },
            "action_bonuses": {
                "hold_penalty": 0.005,  # Encourage holding
                "transaction_penalty": -0.05,  # Stronger transaction penalty
                "diversity_bonus": 0.005,  # Encourage diverse actions
            },
            "risk_management": {
                "max_drawdown_penalty": -0.1,
                "volatility_penalty": -0.05,
                "sharpe_ratio_bonus": 0.02,
            },
        }

        env = HeavyTradingEnv(df=df, config=config)
        return env

    def run_backtest(self, data_path: str, slippage_bps: float = 5.0) -> Dict[str, Any]:
        """Run backtest simulation using HeavyTradingEnv."""
        logger.info("Starting v424 backtest simulation...")

        # Create environment
        env = self.create_environment(data_path)
        self.env = env

        # Initialize tracking variables
        obs, info = env.reset()
        done = False
        total_steps = 0
        portfolio_history = [self.initial_capital]
        action_history = []
        reward_history = []
        price_history = []
        timestamp_history = []

        logger.info("Running backtest simulation...")

        while not done:
            # Get action from model
            action, _ = self.model.predict(obs, deterministic=True)

            # Step environment
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

            # Record data
            total_steps += 1
            portfolio_history.append(info["portfolio_value"])
            action_history.append(
                float(action) if np.isscalar(action) else float(action[0])
            )
            reward_history.append(float(reward))

            # Get current price from environment
            current_price = (
                self.env._resolve_price()
                if hasattr(self.env, "_resolve_price")
                else 0.0
            )
            price_history.append(current_price)

            if hasattr(info, "timestamp") and info.get("timestamp"):
                timestamp_history.append(info["timestamp"])
            else:
                timestamp_history.append(total_steps)

            # Progress logging
            if total_steps % 1000 == 0:
                current_value = info["portfolio_value"]
                pnl_pct = (
                    (current_value - self.initial_capital) / self.initial_capital
                ) * 100
                logger.info(
                    f"Step {total_steps}: Portfolio ¥{current_value:,.0f} ({pnl_pct:+.2f}%)"
                )

        # Calculate final results
        final_portfolio = portfolio_history[-1]
        total_return_pct = (
            (final_portfolio - self.initial_capital) / self.initial_capital
        ) * 100

        # Calculate trading statistics
        actions = np.array(action_history)
        hold_threshold = 0.1

        hold_count = np.sum(np.abs(actions) < hold_threshold)
        buy_count = np.sum(actions > hold_threshold)
        sell_count = np.sum(actions < -hold_threshold)
        total_actions = len(actions)

        # Calculate returns and risk metrics
        portfolio_returns = np.diff(portfolio_history) / portfolio_history[:-1]
        if len(portfolio_returns) > 0:
            volatility = np.std(portfolio_returns) * np.sqrt(
                TRADING_DAYS_PER_YEAR
            )  # Annualized
            sharpe_ratio = (
                np.mean(portfolio_returns)
                / np.std(portfolio_returns)
                * np.sqrt(TRADING_DAYS_PER_YEAR)
                if np.std(portfolio_returns) > 0
                else 0
            )
            max_drawdown = np.min(portfolio_history) / np.max(portfolio_history) - 1
        else:
            volatility = 0
            sharpe_ratio = 0
            max_drawdown = 0

        results = {
            "total_steps": total_steps,
            "initial_portfolio": self.initial_capital,
            "final_portfolio": final_portfolio,
            "total_return_pct": total_return_pct,
            "total_trades": buy_count + sell_count,
            "win_rate": 0.0,  # Would need trade-by-trade analysis
            "sharpe_ratio": sharpe_ratio,
            "max_drawdown": max_drawdown,
            "volatility": volatility,
            "action_distribution": {
                "HOLD": hold_count / total_actions if total_actions > 0 else 0,
                "BUY": buy_count / total_actions if total_actions > 0 else 0,
                "SELL": sell_count / total_actions if total_actions > 0 else 0,
            },
            "portfolio_history": portfolio_history,
            "action_history": action_history,
            "reward_history": reward_history,
            "price_history": price_history,
            "timestamp_history": timestamp_history,
        }

        logger.info("Backtest simulation completed!")
        logger.info(
            f"Final Portfolio: ¥{final_portfolio:,.0f} ({total_return_pct:+.2f}%)"
        )
        logger.info(f"Total Trades: {results['total_trades']}")
        logger.info(
            f"Action Distribution: HOLD {results['action_distribution']['HOLD']:.1%}, "
            f"BUY {results['action_distribution']['BUY']:.1%}, "
            f"SELL {results['action_distribution']['SELL']:.1%}"
        )

        return results

    def print_results(self, results: Dict[str, Any]) -> None:
        """Print backtest results in a formatted way."""
        print("\n" + "=" * 60)
        print("SAC v424 COST-AWARE BACKTEST RESULTS")
        print("=" * 60)
        print(f"Initial Capital: ¥{results['initial_portfolio']:,.0f}")
        print(f"Final Portfolio: ¥{results['final_portfolio']:,.0f}")
        print(f"Total Return: {results['total_return_pct']:+.2f}%")
        print(f"Total Trades: {results['total_trades']}")
        print(f"Sharpe Ratio: {results['sharpe_ratio']:.3f}")
        print(f"Max Drawdown: {results['max_drawdown']:.2%}")
        print(f"Volatility: {results['volatility']:.2%}")
        print("\nAction Distribution:")
        for action, pct in results["action_distribution"].items():
            print(f"  {action}: {pct:.1%}")
        print("=" * 60)

    def save_results(self, results: Dict[str, Any], output_path: str) -> None:
        """Save backtest results to JSON file."""
        import json

        # Convert numpy types to native Python types for JSON serialization
        serializable_results = {}
        for key, value in results.items():
            if isinstance(value, np.ndarray):
                serializable_results[key] = value.tolist()
            elif isinstance(value, (np.float64, np.float32)):
                serializable_results[key] = float(value)
            elif isinstance(value, (np.int64, np.int32)):
                serializable_results[key] = int(value)
            else:
                serializable_results[key] = value

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(serializable_results, f, indent=2, ensure_ascii=False)

        logger.info(f"Results saved to {output_path}")
