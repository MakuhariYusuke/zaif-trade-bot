#!/usr/bin/env python3
"""
Live Trading Bot for BTC/JPY using SAC v396 Model.

This script performs live trading on Zaif exchange using the trained SAC v396 model.
Implements continuous action to discrete action conversion for trading decisions.
"""

import argparse
import logging
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from stable_baselines3 import SAC

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from ztb.trading.environment.constants import continuous_to_discrete_action
from ztb.trading.environment.environment import HeavyTradingEnv


class SACLiveTrader:
    """SAC v396 Live Trading Bot."""

    def __init__(
        self,
        model_path: str,
        config_path: str,
        dry_run: bool = True,
        max_position_size: float = 0.01,
        transaction_cost: float = 0.0005,
        max_daily_loss: float = 0.05,  # 5% max daily loss
        max_trades_per_hour: int = 10,
        emergency_stop_balance: float = 9000.0,
    ):
        """Initialize the trader."""
        self.model_path = model_path
        self.config_path = config_path
        self.dry_run = dry_run
        self.max_position_size = max_position_size
        self.transaction_cost = transaction_cost
        self.max_daily_loss = max_daily_loss
        self.max_trades_per_hour = max_trades_per_hour
        self.emergency_stop_balance = emergency_stop_balance

        self.model = None
        self.env = None
        self.current_position = 0.0
        self.initial_balance = 10000.0
        self.balance = self.initial_balance

        self.trade_count_hour = 0
        self.hour_start_time = time.time()
        self.daily_start_balance = self.balance

        self.logger = logging.getLogger(__name__)
        self._setup_logging()

        self._load_model()
        self._create_environment()

    def _setup_logging(self):
        """Setup logging configuration."""
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )

    def _load_model(self):
        """Load the trained SAC model."""
        self.logger.info(f"Loading SAC model from {self.model_path}")
        try:
            self.model = SAC.load(self.model_path)
            self.logger.info("Successfully loaded SAC model")
        except Exception as e:
            self.logger.error(f"Failed to load model: {e}")
            raise

    def _create_environment(self):
        """Create trading environment for observation generation."""
        # Load config
        import json

        with open(self.config_path, "r") as f:
            config_data = json.load(f)

        # Use environment config from training
        env_config = config_data.get("environment", {})
        env_config.update(
            {
                "initial_balance": self.balance,
                "transaction_cost": self.transaction_cost,
                "max_position_size": self.max_position_size,
                "enable_action_masking": False,
                "use_continuous_actions": True,
                "use_standardized_observations": True,
                "random_start": False,
            }
        )

        # Create dummy data for environment (will be updated with live data)
        dummy_data = pd.DataFrame(
            {
                "timestamp": pd.date_range("2023-01-01", periods=100, freq="1min"),
                "open": 5000000,
                "high": 5000000,
                "low": 5000000,
                "close": 5000000,
                "volume": 100,
            }
        )

        self.env = HeavyTradingEnv(
            df=dummy_data,
            config=env_config,
            random_start=False,
        )

        self.logger.info("Created trading environment")

    def get_live_data(self) -> pd.DataFrame:
        """Get live market data from Zaif."""
        # TODO: Implement Zaif API data fetching
        # For now, return dummy data
        current_time = pd.Timestamp.now()
        dummy_price = 5000000  # TODO: Get real price

        data = pd.DataFrame(
            {
                "timestamp": [current_time],
                "open": [dummy_price],
                "high": [dummy_price],
                "low": [dummy_price],
                "close": [dummy_price],
                "volume": [100],
            }
        )

        return data

    def update_environment_data(self, live_data: pd.DataFrame):
        """Update environment with latest live data."""
        # Update the environment's dataframe
        self.env.df = live_data.reset_index(drop=True)
        self.env.current_step = 0

    def get_action(self, observation) -> int:
        """Get trading action from SAC model."""
        action, _ = self.model.predict(observation, deterministic=True)

        # SAC outputs continuous action, convert to discrete
        if isinstance(action, np.ndarray):
            continuous_action = action.item()
        else:
            continuous_action = action

        discrete_action = continuous_to_discrete_action(continuous_action)

        return discrete_action, continuous_action

    def execute_trade(self, action: int, price: float):
        """Execute trade based on action."""
        if action == 0:  # HOLD
            self.logger.info("HOLD - No action taken")
            return

        position_size = self.max_position_size

        if action == 1:  # BUY
            if self.current_position >= 0:  # No position or long, increase long
                self.current_position += position_size
                cost = position_size * price * (1 + self.transaction_cost)
                self.balance -= cost
                self.logger.info(
                    f"BUY - Position: {self.current_position:.4f}, Cost: {cost:.2f}"
                )
            else:  # Short position, close and go long
                # Close short
                pnl = -self.current_position * price * (1 - self.transaction_cost)
                self.balance += pnl
                self.logger.info(f"Close SHORT - PnL: {pnl:.2f}")

                # Go long
                self.current_position = position_size
                cost = position_size * price * (1 + self.transaction_cost)
                self.balance -= cost
                self.logger.info(
                    f"BUY - Position: {self.current_position:.4f}, Cost: {cost:.2f}"
                )

        elif action == 2:  # SELL
            if self.current_position <= 0:  # No position or short, increase short
                self.current_position -= position_size
                revenue = position_size * price * (1 - self.transaction_cost)
                self.balance += revenue
                self.logger.info(
                    f"SELL - Position: {self.current_position:.4f}, Revenue: {revenue:.2f}"
                )
            else:  # Long position, close and go short
                # Close long
                pnl = self.current_position * price * (1 - self.transaction_cost)
                self.balance += pnl
                self.logger.info(f"Close LONG - PnL: {pnl:.2f}")

                # Go short
                self.current_position = -position_size
                revenue = position_size * price * (1 - self.transaction_cost)
                self.balance += revenue
                self.logger.info(
                    f"SELL - Position: {self.current_position:.4f}, Revenue: {revenue:.2f}"
                )

    def check_risk_limits(self) -> bool:
        """Check if trading should continue based on risk limits."""
        current_time = time.time()

        # Reset hourly trade count
        if current_time - self.hour_start_time >= 3600:  # 1 hour
            self.trade_count_hour = 0
            self.hour_start_time = current_time

        # Check daily loss limit
        daily_loss = (
            self.daily_start_balance - self.balance
        ) / self.daily_start_balance
        if daily_loss > self.max_daily_loss:
            self.logger.error(
                f"Daily loss limit exceeded: {daily_loss:.1%} > {self.max_daily_loss:.1%}"
            )
            return False

        # Check emergency stop
        if self.balance < self.emergency_stop_balance:
            self.logger.error(
                f"Emergency stop triggered: balance {self.balance:.2f} < {self.emergency_stop_balance:.2f}"
            )
            return False

        # Check hourly trade limit
        if self.trade_count_hour >= self.max_trades_per_hour:
            self.logger.warning(
                f"Hourly trade limit reached: {self.trade_count_hour}/{self.max_trades_per_hour}"
            )
            return False

        return True

    def run(self, duration_minutes: int = 60):
        """Run live trading for specified duration."""
        self.logger.info(f"Starting SAC v396 live trading (dry_run={self.dry_run})")
        self.logger.info(f"Duration: {duration_minutes} minutes")
        self.logger.info(f"Initial balance: {self.balance:.2f}")

        start_time = time.time()
        end_time = start_time + (duration_minutes * 60)

        trade_count = 0

        while time.time() < end_time:
            try:
                # Get live data
                live_data = self.get_live_data()
                current_price = float(live_data.iloc[-1]["close"].item())

                # Update environment
                self.update_environment_data(live_data)

                # Get observation
                obs = self.env.reset()
                if isinstance(obs, tuple):
                    obs = obs[0]

                # Get action
                discrete_action, continuous_action = self.get_action(obs)

                self.logger.info(
                    f"Price: {current_price:.0f}, Continuous Action: {continuous_action:.3f}, Discrete Action: {discrete_action}"
                )

                # Check risk limits
                if not self.check_risk_limits():
                    self.logger.error("Risk limits exceeded, stopping trading")
                    break

                # Execute trade
                if not self.dry_run:
                    self.execute_trade(discrete_action, current_price)
                    self.trade_count_hour += 1
                else:
                    self.logger.info(
                        f"DRY RUN - Would execute action {discrete_action}"
                    )

                # Log status
                self.logger.info(
                    f"Balance: {self.balance:.2f}, Position: {self.current_position:.4f}"
                )

                # Wait before next iteration
                time.sleep(60)  # 1 minute intervals

            except Exception as e:
                self.logger.error(f"Error in trading loop: {e}")
                time.sleep(10)  # Wait before retry

        self.logger.info(
            f"Trading completed. Final balance: {self.balance:.2f}, Trades executed: {trade_count}"
        )


def main():
    parser = argparse.ArgumentParser(description="SAC v396 Live Trading Bot")
    parser.add_argument(
        "--model-path",
        default="checkpoints/sac_session/sac_v396_50k_final.zip",
        help="Path to trained SAC model",
    )
    parser.add_argument(
        "--config-path",
        default="configs/sac_v396_optimized.json",
        help="Path to model configuration JSON",
    )
    parser.add_argument(
        "--duration-minutes",
        type=int,
        default=60,
        help="Trading duration in minutes",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        default=True,
        help="Enable dry run mode (default: True)",
    )
    parser.add_argument(
        "--max-position-size",
        type=float,
        default=0.01,
        help="Maximum position size as fraction",
    )
    parser.add_argument(
        "--transaction-cost",
        type=float,
        default=0.0005,
        help="Transaction cost as fraction",
    )
    parser.add_argument(
        "--max-daily-loss",
        type=float,
        default=0.05,
        help="Maximum daily loss as fraction (default: 0.05)",
    )
    parser.add_argument(
        "--max-trades-per-hour",
        type=int,
        default=10,
        help="Maximum trades per hour (default: 10)",
    )
    parser.add_argument(
        "--emergency-stop-balance",
        type=float,
        default=9000.0,
        help="Emergency stop balance threshold (default: 9000.0)",
    )

    args = parser.parse_args()

    trader = SACLiveTrader(
        model_path=args.model_path,
        config_path=args.config_path,
        dry_run=args.dry_run,
        max_position_size=args.max_position_size,
        transaction_cost=args.transaction_cost,
        max_daily_loss=args.max_daily_loss,
        max_trades_per_hour=args.max_trades_per_hour,
        emergency_stop_balance=args.emergency_stop_balance,
    )

    trader.run(duration_minutes=args.duration_minutes)


if __name__ == "__main__":
    main()
