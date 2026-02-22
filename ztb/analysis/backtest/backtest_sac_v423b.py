#!/usr/bin/env python3
"""
SAC v423b Backtest Script

Runs backtest for SAC v423b model using the trained policy.
"""

import os
import sys
from pathlib import Path
from typing import Any, Dict

import pandas as pd
from stable_baselines3 import SAC

from ztb.metrics.metrics import max_drawdown, sharpe_ratio

TRADING_DAYS_PER_YEAR = 252

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from ztb.trading.environment.heavy_env.core import HeavyTradingEnv
from ztb.utils.logging_utils import get_logger

logger = get_logger(__name__)


class SACv423bBacktester:
    """Backtester for SAC v423b model."""


    def load_model(self) -> None:
        """Load the trained SAC model."""
        try:
            self.model = SAC.load(self.model_path)
            logger.info(f"Loaded SAC model from {self.model_path}")
            logger.info(f"Model observation space: {self.model.observation_space}")
            df["timestamp"] = pd.to_datetime(df["timestamp"])
            df.set_index("timestamp", inplace=True)

        config = {
            "initial_balance": self.initial_capital,
            "transaction_cost": 1e-05,
            "max_position_size": 1.0,
            "enable_action_masking": False,
            "use_continuous_actions": True,
            "use_standardized_observations": True,
            "random_start": False,  # Disable for backtesting
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
                    trades.append(
                        {
                            "step": total_steps,
                            "type": trade_type,
                            "position_before": positions[-2],
                            "position_after": positions[-1],
                            "portfolio_value": env.portfolio_value,
                            "timestamp": getattr(env, "current_timestamp", None),
                        }
                    )

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
            max_drawdown = max_drawdown(pd.Series(equity_curve))
            sharpe_ratio = self.calculate_sharpe_ratio(returns)
        else:
            total_return = 0.0
            max_drawdown = 0.0
            sharpe_ratio = 0.0

        results = {
            "total_steps": total_steps,
            "initial_portfolio": self.initial_capital,
            "final_portfolio": equity_curve[-1]
            if equity_curve
            else self.initial_capital,
            "portfolio_history": equity_curve,
            "timestamps": list(
                range(len(equity_curve))
            ),  # Simple step-based timestamps
            "total_return": total_return,
            "max_drawdown": max_drawdown,
            "sharpe_ratio": sharpe_ratio,
            "total_trades": len(trades),
            "win_rate": 0.0,  # Simplified
            "positions": positions,
            "actions": actions_taken,
            "trades": trades,
        }

        env.close()
        return results

    def calculate_sharpe_ratio(
        self, returns: pd.Series, risk_free_rate: float = 0.0
    ) -> float:
        """Calculate Sharpe ratio."""
        return sharpe_ratio(
            returns, rf=risk_free_rate, period_per_year=TRADING_DAYS_PER_YEAR
        )




if __name__ == "__main__":
    main()
