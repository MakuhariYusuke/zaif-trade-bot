#!/usr/bin/env python3
"""
Heavy trading environment for reinforcement learning.
"""

import dataclasses
import logging
from typing import Any

import gymnasium as gym
import numpy as np
import pandas as pd
from gymnasium import spaces

from ztb.trading.environment.components.calculators.reward_calculator import RewardCalculator
from ztb.trading.environment.components.threshold_manager import ThresholdManager
from ztb.trading.environment.utils.config import EnvironmentConfig, RewardSettings
from ztb.types.protocols import TradingEnvironment

logger = logging.getLogger(__name__)


def _initial_portfolio_value_from_config(config: EnvironmentConfig) -> float:
    """Support both legacy training config and trading config field names."""
    initial_value = getattr(config, "initial_portfolio_value", None)
    if initial_value is None:
        initial_value = getattr(config, "initial_balance")
    return float(initial_value)

class HeavyTradingEnv(gym.Env, TradingEnvironment):
    """
    Heavy trading environment for reinforcement learning.
    """

    def __init__(
        self,
        data: pd.DataFrame,
        config: EnvironmentConfig,
        feature_columns: list[str] | None = None,
        reward_settings: dict[str, Any] | None = None,
    ):
        """
        Initialize trading environment.

        Args:
            data: Market data DataFrame
            config: Environment configuration
            feature_columns: list of feature column names
            reward_settings: Reward settings dictionary
        """
        super().__init__()

        self.data = data.copy()
        # Clean data: fill NaN values with forward fill, then 0
        self.data = self.data.ffill().fillna(0)
        self.config = config
        self.feature_columns = feature_columns or []
        self.reward_settings = reward_settings or {}

        logger.info(
            f"HeavyTradingEnv initialized with {len(self.feature_columns)} feature columns: {self.feature_columns}"
        )

        # Initialize ThresholdManager
        self.threshold_manager = ThresholdManager(self.config)

        # Pre-compute trading thresholds (legacy support / initial values)
        self.action_threshold = self.threshold_manager.base_threshold

        negative_threshold = getattr(
            self.config, "continuous_to_discrete_threshold_neg", None
        )
        self.negative_action_threshold = (
            float(negative_threshold)
            if negative_threshold is not None
            else -abs(float(self.action_threshold))
        )
        self.min_position_change = getattr(
            self.config,
            "min_position_change",
            getattr(self.config, "min_trade_size", 1e-4),
        )
        self._threshold_suppressed_actions = 0
        self._min_trade_suppressed_actions = 0

        # Action space: continuous action in [-1, 1] for SAC compatibility
        # negative = SELL, 0 = HOLD, positive = BUY
        self.action_space = spaces.Box(
            low=np.array([-1.0]), high=np.array([1.0]), dtype=np.float32
        )

        # Observation space (features only, matching training environment)
        obs_dim = len(self.feature_columns)  # features only (no account info)
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32
        )

        # Initialize reward calculator with action signal guidance
        reward_defaults: dict[str, Any] = {
            "base_profit_bonus_atr_coeff": 5.0,
            "base_profit_bonus_portfolio_coeff": 10.0,
            "base_action_penalty": 0.02,
            "loss_penalty_coeff": -1.0,
            "action_frequency_penalty": 0.005,
            "long_short_asymmetry": True,
            "risk_adjusted_bonus": True,
            "market_regime_penalty": True,
            "scalping_mode": True,
            "signal_guidance_integration": True,
            # HOLD penalty adjustments to reduce SELL bias - moved to custom_reward_params
            "custom_reward_params": {
                "hold_penalty_base": 0.00001,  # Further reduced from 0.001
                "hold_penalty_position_factor": 0.001,  # Further reduced from 0.01
                "hold_penalty_multiplier": 0.01,  # Further reduced from 0.5
            },
            # HOLD profit bonus adjustment
            "profit_bonus_multipliers": [
                1.0,
                1.0,
                1.0,
            ],  # HOLD gets same bonus as BUY/SELL
        }
        reward_defaults.update(self.reward_settings)
        # Filter reward_defaults to only include RewardSettings fields
        reward_settings_fields = {
            field.name for field in dataclasses.fields(RewardSettings)
        }
        filtered_reward_defaults = {
            k: v for k, v in reward_defaults.items() if k in reward_settings_fields
        }
        self.reward_settings = RewardSettings(**filtered_reward_defaults)
        initial_portfolio_value = _initial_portfolio_value_from_config(self.config)

        # Create trading EnvironmentConfig from training EnvironmentConfig
        from ztb.trading.environment.utils.config import (
            EnvironmentConfig as TradingEnvironmentConfig,
        )

        trading_config = TradingEnvironmentConfig(
            initial_portfolio_value=initial_portfolio_value,
            transaction_cost=self.config.commission,
            max_position_size=self.config.max_position_size,
            reward_scaling=self.config.reward_scaling,
            feature_set=self.config.feature_set,
            curriculum_stage=self.config.curriculum_stage,
            base_action_penalty=getattr(self.config, "base_action_penalty", 0.015),
            action_bonuses=getattr(self.config, "action_bonuses", None),
        )
        if hasattr(self.config, "behavior"):
            setattr(trading_config, "behavior", getattr(self.config, "behavior"))

        self.reward_calculator = RewardCalculator(
            config=trading_config,
            reward_settings=self.reward_settings,
            initial_portfolio_value=initial_portfolio_value,
        )
        # Expose optional MTFScheduler to the environment for diagnostics/tests
        self.mtf_scheduler = getattr(self.reward_calculator, "mtf_scheduler", None)

        # Initialize state
        self.reset()

    @property
    def position(self) -> float:
        """Get current position size."""
        return self._position

    @position.setter
    def position(self, value: float) -> None:
        """set current position size."""
        self._position = value

    @property
    def unrealized_pnl(self) -> float:
        """Get current unrealized P&L."""
        return self._unrealized_pnl

    @unrealized_pnl.setter
    def unrealized_pnl(self, value: float) -> None:
        """set current unrealized P&L."""
        self._unrealized_pnl = value

    def reset(self, *, seed=None, options=None) -> np.ndarray:
        """Reset environment to initial state."""
        if seed is not None:
            super().reset(seed=seed)

        self.current_step = 0
        self.balance = _initial_portfolio_value_from_config(self.config)
        self._position = 0.0  # Current position size
        self.entry_price = 0.0
        self._unrealized_pnl = 0.0
        self.total_pnl = 0.0
        self.trades_count = 0
        self.hold_duration = 0
        self._old_position = 0.0
        self.reward_history = []
        self.portfolio_value_history = []
        self._threshold_suppressed_actions = 0
        self._min_trade_suppressed_actions = 0

        return self._get_observation(), {}

    def step(
        self, action: np.ndarray
    ) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        """
        Execute one step in the environment.

        Args:
            action: Action array [position_size, hold_duration]

        Returns:
            observation, reward, done, info
        """
        # Convert continuous action: [-1, 1] -> position size
        # negative = SELL/short, 0 = HOLD/no position, positive = BUY/long
        action_value = float(action[0])

        # Check for NaN or invalid action
        if np.isnan(action_value) or np.isinf(action_value):
            action_value = 0.0  # Default to HOLD

        # Get current market data for adaptive threshold
        current_data = self.data.iloc[self.current_step]
        current_price = float(current_data["close"])

        # Get volatility (ATR)
        volatility = 0.0
        if "atr_14" in current_data:
            volatility = float(current_data["atr_14"])
        elif "volatility_10" in current_data:
            volatility = float(current_data["volatility_10"])

        # Convert to position size: absolute value determines position size, sign determines direction
        threshold = self.threshold_manager.get_threshold(volatility, current_price)

        # Update legacy attribute for compatibility
        self.action_threshold = threshold

        threshold_suppressed = False
        if abs(action_value) < threshold:
            new_position = 0.0
            self._threshold_suppressed_actions += 1
            threshold_suppressed = True
            if self._threshold_suppressed_actions % 50 == 0:
                logger.debug(
                    "Suppressed action %.4f due to threshold %.4f (suppressed=%s)",
                    action_value,
                    threshold,
                    self._threshold_suppressed_actions,
                )
        else:
            # Scale position size based on action intensity
            max_pos_size = getattr(self.config, "max_position_size", 1.0)
            new_position = float(np.clip(action_value, -max_pos_size, max_pos_size))
            if threshold_suppressed:
                logger.debug(
                    "Continuous action %.4f suppressed by threshold %.4f at step %s",
                    action_value,
                    threshold,
                    self.current_step,
                )

        # Execute trade if position changed significantly
        position_change_threshold = max(
            getattr(self, "min_position_change", 1e-4),
            getattr(self.config, "min_trade_size", 1e-4),
        )
        if abs(new_position - self.position) >= position_change_threshold:
            self._execute_trade(new_position)

        # Update unrealized P&L
        self._update_unrealized_pnl()

        # Update hold duration
        self.hold_duration = 5  # Fixed hold duration for simplicity

        # Calculate reward
        reward = self._calculate_reward(action_value)

        # Move to next step
        self.current_step += 1

        # Check if episode is done
        terminated = self.current_step >= len(self.data) - 1 or (
            self.config.max_steps is not None
            and self.current_step >= self.config.max_steps
        )
        truncated = False  # Not using truncation in this simple environment

        info = {
            "balance": self.balance,
            "position": self.position,
            "unrealized_pnl": self.unrealized_pnl,
            "total_pnl": self.total_pnl,
            "trades_count": self.trades_count,
            "current_step": self.current_step,
            "action_value": action_value,
            "position_change_threshold": position_change_threshold,
            "portfolio_value": self.balance
            + self.unrealized_pnl,  # ポートフォリオ価値を追加
            "btc_balance": getattr(self, "btc_balance", 0),  # BTC残高
            "current_price": self.data.iloc[self.current_step]["close"]
            if self.current_step < len(self.data)
            else 0,  # 現在価格
            "signal_strength": getattr(
                self.reward_calculator, "last_signal_strength", 0.0
            ),
            "signal_reward": getattr(self.reward_calculator, "last_signal_reward", 0.0),
            "threshold_suppressed_actions": self._threshold_suppressed_actions,
            "min_trade_suppressed_actions": self._min_trade_suppressed_actions,
            "suppressed_this_step": threshold_suppressed,
        }

        if self.current_step > 0 and self.current_step % 1000 == 0:
            logger.info(
                "Scalping diagnostics - step=%s trades=%s threshold_supp=%s min_trade_supp=%s last_signal=%.3f",
                self.current_step,
                self.trades_count,
                self._threshold_suppressed_actions,
                self._min_trade_suppressed_actions,
                info["signal_strength"],
            )

        return self._get_observation(), reward, terminated, truncated, info

    def _execute_trade(self, new_position: float) -> None:
        """Execute a trade."""
        if self.current_step >= len(self.data):
            return

        # Check for invalid position
        if np.isnan(new_position) or np.isinf(new_position):
            new_position = 0.0

        current_price = self.data.iloc[self.current_step]["close"].item()

        # Check for invalid price data
        if np.isnan(current_price) or np.isinf(current_price) or current_price <= 0:
            logger.warning(
                f"Invalid price data at step {self.current_step}: {current_price}"
            )
            return  # Skip trade execution

        # Close existing position if any
        if self.position != 0:
            # Calculate P&L from closing position
            pnl = self.position * (current_price - self.entry_price)
            commission = abs(self.position) * current_price * self.config.commission
            self.balance += pnl - commission
            self.total_pnl += pnl - commission

        # Open new position
        if new_position != 0:
            # Apply slippage (direction matters for short positions)
            min_trade_size = getattr(self.config, "min_trade_size", 1e-4)
            if abs(new_position) < min_trade_size:
                self._min_trade_suppressed_actions += 1
                logger.debug(
                    "Skipping trade execution - position %.6f below min_trade_size %.6f",
                    new_position,
                    min_trade_size,
                )
                if self._min_trade_suppressed_actions % 50 == 0:
                    logger.debug(
                        "Total min trade suppressions: %s",
                        self._min_trade_suppressed_actions,
                    )
                self.position = 0.0
                self.entry_price = 0.0
                return

            slippage = current_price * self.config.slippage * np.sign(new_position)
            execution_price = current_price + slippage

            # Check for invalid execution price
            if (
                np.isnan(execution_price)
                or np.isinf(execution_price)
                or execution_price <= 0
            ):
                logger.warning(f"Invalid execution price: {execution_price}")
                return

            # Apply commission
            commission = abs(new_position) * execution_price * self.config.commission
            self.balance -= commission

            self.position = new_position  # Can be negative for short positions
            self.entry_price = execution_price
            self.trades_count += 1

        else:
            self.position = 0.0
            self.entry_price = 0.0

    def _update_unrealized_pnl(self) -> None:
        """Update unrealized P&L based on current position and price."""
        if self.current_step >= len(self.data):
            return

        current_price = self.data.iloc[self.current_step]["close"].item()

        # Check for invalid price
        if np.isnan(current_price) or np.isinf(current_price) or current_price <= 0:
            return

        if self.position != 0 and self.entry_price > 0:
            if self.position > 0:  # Long position
                self.unrealized_pnl = self.position * (current_price - self.entry_price)
            else:  # Short position
                self.unrealized_pnl = abs(self.position) * (
                    self.entry_price - current_price
                )
        else:
            self.unrealized_pnl = 0.0

    def _calculate_reward(self, action_value: float = 0.0) -> float:
        """Calculate reward for current step using RewardCalculator."""
        if self.current_step == 0:
            return 0.0

        # Get current market data
        current_data = self.data.iloc[self.current_step]
        current_price = current_data["close"]

        # Check for invalid price data
        if np.isnan(current_price) or np.isinf(current_price) or current_price <= 0:
            logger.warning(
                f"Invalid price data for reward calculation at step {self.current_step}: {current_price}"
            )
            return 0.0

        # Calculate ATR (simplified)
        # logger.debug(
        #     f"ATR calculation: current_data keys: {list(current_data.keys()) if hasattr(current_data, 'keys') else 'no keys'}"
        # )
        # logger.debug(f"ATR calculation: current_data type: {type(current_data)}")
        if "atr_14" in current_data:
            atr = current_data["atr_14"].item()
            # logger.debug(f"Using atr_14: {atr}")
        elif "volatility_10" in current_data:
            atr = current_data["volatility_10"].item()
            # logger.debug(f"Using volatility_10: {atr}")
        else:
            atr = 0.01
            # logger.debug(f"Using default atr: {atr}")
        if np.isnan(atr) or np.isinf(atr) or atr <= 0:
            # logger.debug(f"ATR was invalid ({atr}), setting to 0.01")
            atr = 0.01

        # Calculate current P&L
        if self.position != 0:
            if self.position > 0:  # Long position
                current_pnl = self.position * (current_price - self.entry_price)
            else:  # Short position
                current_pnl = abs(self.position) * (self.entry_price - current_price)
        else:
            current_pnl = 0.0

        # Check for invalid P&L
        if np.isnan(current_pnl) or np.isinf(current_pnl):
            logger.warning(f"Invalid P&L calculation: {current_pnl}")
            current_pnl = 0.0

        # Get observation
        observation = self._get_observation()

        # Determine action based on intent (action_value) vs current position
        # This allows rewarding the INTENT to trade even if suppressed
        max_pos_size = getattr(self.config, "max_position_size", 1.0)
        target_position = float(np.clip(action_value, -max_pos_size, max_pos_size))

        # Determine if this is a BUY, SELL or HOLD intent
        intent_threshold = 1e-4

        if target_position > self.position + intent_threshold:
            action = 1  # BUY intent
        elif target_position < self.position - intent_threshold:
            action = -1  # SELL intent
        else:
            action = 0  # HOLD intent

        # Calculate reward using RewardCalculator
        try:
            portfolio_value = self.balance + self.unrealized_pnl
            # Check for invalid portfolio value
            if np.isnan(portfolio_value) or np.isinf(portfolio_value):
                logger.warning(
                    f"Invalid portfolio value: balance={self.balance}, unrealized_pnl={self.unrealized_pnl}"
                )
                portfolio_value = self.balance  # Fallback to balance only

            reward = self.reward_calculator.calculate_reward(
                action=action,
                current_price=float(current_price),
                position=self.position,
                portfolio_value=float(portfolio_value),
                atr=atr,
                transaction_cost=self.config.commission,
                reward_scaling=self.config.reward_scaling,
                pnl=float(current_pnl),
                old_position=getattr(self, "_old_position", 0.0),
                step=self.current_step,
                observation=observation,
                reward_history=getattr(self, "reward_history", []),
                portfolio_value_history=getattr(self, "portfolio_value_history", []),
            )
            # Store old position for next step
            self._old_position = self.position
            return reward
        except Exception as e:
            logger.warning(f"RewardCalculator failed, using simple reward: {e}")
            # Fallback to simple reward calculation
            return self._calculate_simple_reward()

    def _calculate_simple_reward(self) -> float:
        """Simple reward calculation as fallback."""
        if self.current_step == 0:
            return 0.0

        # Reward based on P&L change
        prev_pnl = self.total_pnl
        current_price = self.data.iloc[self.current_step]["close"].item()

        if self.position != 0:
            # Calculate current unrealized P&L based on position direction
            if self.position > 0:  # Long position
                current_pnl = self.position * (current_price - self.entry_price)
            else:  # Short position
                current_pnl = abs(self.position) * (self.entry_price - current_price)
            pnl_change = current_pnl - prev_pnl
        else:
            pnl_change = 0.0

        # Scale reward
        reward = pnl_change * self.config.reward_scaling

        # Update history
        self.reward_history.append(reward)
        self.portfolio_value_history.append(self.balance + self.unrealized_pnl)

        # Keep history limited
        if len(self.reward_history) > 100:
            self.reward_history = self.reward_history[-100:]
            self.portfolio_value_history = self.portfolio_value_history[-100:]

        return reward

    def _get_observation(self) -> np.ndarray:
        """Get current observation (features only, matching training environment)."""
        if self.current_step >= len(self.data):
            # Return zero observation if beyond data
            return np.zeros(self.observation_space.shape[0], dtype=np.float32)

        # Get feature values only (no account information, matching training)
        features = []
        for col in self.feature_columns:
            if col in self.data.columns:
                value = self.data.iloc[self.current_step][col]
                # Check for NaN/inf and replace with 0.0
                if np.isnan(value) or np.isinf(value):
                    features.append(0.0)
                else:
                    features.append(float(value))
            else:
                features.append(0.0)

        # Return only features (no account info: balance, position, unrealized_pnl)
        return np.array(features, dtype=np.float32)

    def render(self, mode: str = "human") -> None:
        """Render environment."""
        if mode == "human":
            logger.debug(
                f"Step: {self.current_step}, Balance: {self.balance:.2f}, "
                f"Position: {self.position:.4f}, P&L: {self.total_pnl:.2f}"
            )

    def close(self) -> None:
        """Close environment."""
        pass
