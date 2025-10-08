"""
Reward Calculator - Handles reward calculation logic for trading environment.

This module separates the complex reward calculation logic from the main environment class.
"""

# mypy: disable-error-code=literal-required

import logging
import math
from typing import Any, List, Optional

from ztb.trading.constants import (
    ACTION_HOLD,
    ACTION_BUY,
    ACTION_SELL,
    MULTIPLIER_INDEX_BUY,
    MULTIPLIER_INDEX_SELL,
    MULTIPLIER_INDEX_HOLD,
)
from ztb.trading.environment.utils.config import RewardSettings


class RewardCalculator:
    """
    Calculates rewards for trading actions with curriculum learning stages.
    
    This class encapsulates all reward-related logic including:
    - Curriculum learning stages (forced_balance, balanced_transition, etc.)
    - Profit bonuses and loss penalties
    - Position penalties
    - Drawdown, stagnation, growth, and win streak calculations
    """

    def __init__(
        self,
        config: Any,  # EnvironmentConfig
        reward_settings: RewardSettings,
        initial_portfolio_value: float,
    ):
        """
        Initialize RewardCalculator.
        
        Args:
            config: Environment configuration
            reward_settings: Dictionary of reward settings
            initial_portfolio_value: Initial portfolio value
        """
        self.config = config
        self.reward_settings = reward_settings
        self.initial_portfolio_value = initial_portfolio_value
        self.logger = logging.getLogger(__name__)
        
        # Internal state for tracking
        self._action_counts: List[int] = [0, 0, 0]  # [HOLD, BUY, SELL]
        
    def get_setting_int(self, key: str, default: int) -> int:
        """Get integer reward setting with fallback."""
        if self.reward_settings and key in self.reward_settings:
            value = self.reward_settings[key]
            if isinstance(value, (int, float)):
                return int(value)
        return default

    def get_setting_float(self, key: str, default: float) -> float:
        """Get float reward setting with fallback."""
        if self.reward_settings and key in self.reward_settings:
            value = self.reward_settings[key]
            if isinstance(value, (int, float)):
                return float(value)
        return default

    def get_setting_bool(self, key: str, default: bool) -> bool:
        """Get boolean reward setting with fallback."""
        if self.reward_settings and key in self.reward_settings:
            value = self.reward_settings[key]
            if isinstance(value, bool):
                return value
            if isinstance(value, (int, float)):
                return bool(value)
            if isinstance(value, str):
                return value.lower() in {"true", "1", "yes", "y", "on"}
        return default

    def calculate_reward(
        self,
        action: int,
        current_price: float,
        position: float,
        portfolio_value: float,
        atr: float,
        transaction_cost: float,
        reward_scaling: float,
        pnl: float,
        old_position: float,
        step: int,
        observation: Optional[Any],
        reward_history: List[float],
        portfolio_value_history: List[float],
    ) -> float:
        """
        Calculate reward with curriculum learning stages.
        
        Args:
            action: Action taken (0=HOLD, 1=BUY, 2=SELL)
            current_price: Current market price
            position: Current position
            portfolio_value: Current portfolio value
            atr: Average True Range
            transaction_cost: Transaction cost
            reward_scaling: Reward scaling factor
            pnl: Profit/Loss from action
            old_position: Position before action
            step: Current step number
            observation: Current observation (for feature access)
            reward_history: History of rewards
            portfolio_value_history: History of portfolio values
            
        Returns:
            Calculated reward value
        """
        curriculum_stage = self.config.curriculum_stage
        self.logger.debug(
            "Curriculum stage: %s, position: %.2f, action: %d",
            curriculum_stage, position, action
        )

        eps = 1e-8
        atr = atr if atr > eps else 1.0
        max_position_size = max(eps, self.config.max_position_size)

        atr_normalised = pnl / atr
        portfolio_return = pnl / max(abs(self.initial_portfolio_value), eps)

        # Curriculum learning stages
        if curriculum_stage == "forced_balance":
            return self._calculate_forced_balance_reward(action)
        elif curriculum_stage == "balanced_transition":
            return self._calculate_balanced_transition_reward(
                action, atr_normalised, portfolio_return, position,
                max_position_size, current_price, atr, pnl
            )
        elif curriculum_stage == "pnl_focused":
            return self._calculate_pnl_focused_reward(
                action, atr_normalised, portfolio_return, position,
                max_position_size, current_price, atr, pnl, observation
            )
        else:
            # Default stage
            return self._calculate_default_reward(
                action, atr_normalised, portfolio_return, position,
                max_position_size, current_price, atr, pnl
            )

    def _calculate_forced_balance_reward(self, action: int) -> float:
        """Stage 0: Force balanced action distribution (33% each action)."""
        self._action_counts[action] += 1
        total_actions = sum(self._action_counts)
        
        if total_actions >= 3:
            action_ratios = [count / total_actions for count in self._action_counts]
            target_ratio = 1.0 / 3.0

            balance_penalty = sum(abs(ratio - target_ratio) for ratio in action_ratios)

            if balance_penalty < 0.1:
                return 2.0
            elif balance_penalty < 0.2:
                return 1.0
            elif balance_penalty < 0.3:
                return 0.5
            else:
                return -1.0
        else:
            return 0.1

    def _calculate_balanced_transition_reward(
        self,
        action: int,
        atr_normalised: float,
        portfolio_return: float,
        position: float,
        max_position_size: float,
        current_price: float,
        atr: float,
        pnl: float,
    ) -> float:
        """Stage 1: Normal reward with balance penalty."""
        self._action_counts[action] += 1
        total_actions = sum(self._action_counts)
        balance_penalty = 0.0

        if total_actions >= 10:
            action_ratios = [count / total_actions for count in self._action_counts]
            target_ratio = 1.0 / 3.0
            
            for ratio in action_ratios:
                if abs(ratio - target_ratio) > 0.15:
                    balance_penalty += 0.5

        # Calculate base reward
        base_reward = self._calculate_base_reward(
            action, atr_normalised, portfolio_return, position,
            max_position_size, current_price, atr, pnl
        )
        
        return base_reward - balance_penalty

    def _calculate_pnl_focused_reward(
        self,
        action: int,
        atr_normalised: float,
        portfolio_return: float,
        position: float,
        max_position_size: float,
        current_price: float,
        atr: float,
        pnl: float,
        observation: Optional[Any],
    ) -> float:
        """Stage 2: PnL-focused reward with trend analysis."""
        base_profit_bonus = max(0.0, 1.5 * atr_normalised + 1.2 * portfolio_return) if pnl > 0 else 0.0

        # Trend analysis if observation available
        trend_multiplier = 1.0
        if observation is not None and hasattr(observation, '__getitem__'):
            try:
                # Assume RSI and MACD are in observation
                rsi_idx = -2 if len(observation) > 2 else None
                macd_idx = -1 if len(observation) > 1 else None
                
                if rsi_idx is not None and macd_idx is not None:
                    rsi = float(observation[rsi_idx])
                    macd_hist = float(observation[macd_idx])
                    
                    # Trend ratio
                    trend_ratio = (rsi / 50.0) * (1.0 + macd_hist / 100.0)
                    
                    if trend_ratio > 1.0 and action == ACTION_BUY:
                        trend_multiplier = 1.2
                    elif trend_ratio < 1.0 and action == ACTION_SELL:
                        trend_multiplier = 1.2
                    
                    # Oversold/Overbought signals
                    if rsi > 60.0 and trend_ratio > 1.0 and action == ACTION_SELL:
                        trend_multiplier *= 1.3
                    elif rsi < 40.0 and trend_ratio < 1.0 and action == ACTION_SELL:
                        trend_multiplier *= 1.3
            except (IndexError, TypeError, ValueError):
                pass

        # Balance BUY/SELL actions
        # profit_bonus_multipliers array order: [BUY, SELL, HOLD]
        multipliers_raw = self.reward_settings.get("profit_bonus_multipliers", [1.0, 1.0, 0.8])
        if len(multipliers_raw) >= 3:
            multipliers = [float(x) for x in multipliers_raw[:3]]
        else:
            multipliers = [1.0, 1.0, 0.8]
        
        if action == ACTION_BUY:
            profit_bonus = base_profit_bonus * multipliers[MULTIPLIER_INDEX_BUY] * trend_multiplier
        elif action == ACTION_SELL:
            profit_bonus = base_profit_bonus * multipliers[MULTIPLIER_INDEX_SELL] * trend_multiplier
        else:  # HOLD
            profit_bonus = base_profit_bonus * multipliers[MULTIPLIER_INDEX_HOLD] * trend_multiplier

        # Action penalties with configurable per-action penalties
        # Get action-specific penalties from settings (default: 0.0)
        hold_penalty = self.get_setting_float("hold_action_penalty", 0.0)
        buy_penalty = self.get_setting_float("buy_action_penalty", 0.0)
        sell_penalty = self.get_setting_float("sell_action_penalty", 0.0)
        
        # Base action penalties (legacy behavior)
        base_action_penalty = 0.015 if action in [ACTION_BUY, ACTION_SELL] else 0.0
        
        if action == ACTION_HOLD:
            position_size_factor = abs(position) / max_position_size
            volatility_factor = min(atr / (current_price * 0.01), 1.0)
            base_action_penalty = 0.01 + (0.04 * position_size_factor * volatility_factor)
            # Add configured HOLD penalty
            action_penalty = base_action_penalty + hold_penalty
        elif action == ACTION_BUY:
            # Add configured BUY penalty (negative value = reward)
            action_penalty = base_action_penalty + buy_penalty
        else:  # ACTION_SELL
            # Add configured SELL penalty (negative value = reward)
            action_penalty = base_action_penalty + sell_penalty

        # Loss penalty
        loss_penalty = -0.2 * abs(atr_normalised) if pnl < 0 else 0.0

        # Position penalty
        position_penalty = self._calculate_position_penalty(position, max_position_size)

        return profit_bonus - action_penalty + loss_penalty - position_penalty

    def _calculate_default_reward(
        self,
        action: int,
        atr_normalised: float,
        portfolio_return: float,
        position: float,
        max_position_size: float,
        current_price: float,
        atr: float,
        pnl: float,
    ) -> float:
        """Default reward calculation."""
        base_profit_bonus = max(0.0, 1.5 * atr_normalised + 1.2 * portfolio_return) if pnl > 0 else 0.0

        # profit_bonus_multipliers array order: [BUY, SELL, HOLD]
        multipliers = self.reward_settings.get("profit_bonus_multipliers", [1.0, 1.0, 0.8])
        
        # Get action-specific penalties from settings (default: 0.0)
        hold_penalty = self.get_setting_float("hold_action_penalty", 0.0)
        buy_penalty = self.get_setting_float("buy_action_penalty", 0.0)
        sell_penalty = self.get_setting_float("sell_action_penalty", 0.0)
        
        if action == ACTION_BUY:
            profit_bonus = base_profit_bonus * multipliers[MULTIPLIER_INDEX_BUY]
            action_penalty = 0.015 + buy_penalty
        elif action == ACTION_SELL:
            profit_bonus = base_profit_bonus * multipliers[MULTIPLIER_INDEX_SELL]
            action_penalty = 0.015 + sell_penalty
        else:  # HOLD
            profit_bonus = base_profit_bonus * multipliers[MULTIPLIER_INDEX_HOLD]
            position_size_factor = abs(position) / max_position_size
            volatility_factor = min(atr / (current_price * 0.01), 1.0)
            action_penalty = 0.01 + (0.04 * position_size_factor * volatility_factor) + hold_penalty

        loss_penalty = -0.2 * abs(atr_normalised) if pnl < 0 else 0.0
        
        # Forced diversity bonus
        diversity_bonus = 0.0
        if self.get_setting_bool("enable_forced_diversity", False):
            diversity_bonus = self._calculate_diversity_bonus(action)

        position_penalty = self._calculate_position_penalty(position, max_position_size)

        return profit_bonus - action_penalty + loss_penalty + diversity_bonus - position_penalty

    def _calculate_base_reward(
        self,
        action: int,
        atr_normalised: float,
        portfolio_return: float,
        position: float,
        max_position_size: float,
        current_price: float,
        atr: float,
        pnl: float,
    ) -> float:
        """Calculate base reward components."""
        return self._calculate_default_reward(
            action, atr_normalised, portfolio_return, position,
            max_position_size, current_price, atr, pnl
        )

    def _calculate_position_penalty(self, position: float, max_position_size: float) -> float:
        """Calculate penalty for excessive position usage."""
        position_utilisation = abs(position) / max_position_size
        soft_cap = self.get_setting_float("position_soft_cap", 0.8)
        
        if position_utilisation > soft_cap:
            overuse = position_utilisation - soft_cap
            scale = self.get_setting_float("position_penalty_scale", 0.5)
            exponent = self.get_setting_float("position_penalty_exp", 2.0)
            return scale * (math.exp(exponent * overuse) - 1.0)
        
        return 0.0

    def _calculate_diversity_bonus(self, action: int) -> float:
        """Calculate bonus for action diversity."""
        self._action_counts[action] += 1
        total_actions = sum(self._action_counts)
        
        if total_actions < 5:
            return 0.0

        action_ratios = [count / total_actions for count in self._action_counts]
        min_required_ratio = 0.1

        unused_penalty = sum(1.0 for count in self._action_counts if count == 0)
        
        underused_penalty = sum(
            max(0.0, min_required_ratio - ratio) * 2.0
            for ratio in action_ratios
        )

        return -(unused_penalty + underused_penalty)

    def calculate_drawdown_penalty(self, reward_history: List[float]) -> float:
        """Calculate drawdown penalty (when drawdown exceeds 50%)."""
        if len(reward_history) < 20:
            return 0.0

        recent_rewards = reward_history[-20:]
        cumulative_reward = sum(recent_rewards)

        if len(reward_history) >= 30:
            initial_rewards = reward_history[-30:-20]
            initial_cumulative = sum(initial_rewards)

            if initial_cumulative > 0:
                drawdown_ratio = (initial_cumulative - cumulative_reward) / initial_cumulative
                if drawdown_ratio > 0.5:
                    return drawdown_ratio * 0.05

        return 0.0

    def calculate_stagnation_penalty(self, portfolio_value_history: List[float]) -> float:
        """Calculate stagnation penalty (when portfolio isn't growing)."""
        if len(portfolio_value_history) < 30:
            return 0.0

        recent_values = portfolio_value_history[-30:]
        initial_value = recent_values[0]
        final_value = recent_values[-1]

        if initial_value > 0:
            growth_rate = (final_value - initial_value) / initial_value
            stagnation_threshold = -0.005

            if growth_rate < stagnation_threshold:
                return min(0.02, abs(growth_rate - stagnation_threshold) * 0.5)

        return 0.0

    def calculate_growth_bonus(self, portfolio_value_history: List[float]) -> float:
        """Calculate growth bonus (when portfolio is increasing)."""
        if len(portfolio_value_history) < 30:
            return 0.0

        recent_values = portfolio_value_history[-30:]
        initial_value = recent_values[0]
        final_value = recent_values[-1]

        if initial_value > 0:
            growth_rate = (final_value - initial_value) / initial_value
            growth_threshold = 0.005

            if growth_rate > growth_threshold:
                return min(0.05, growth_rate * 0.5)

        return 0.0

    def calculate_win_streak_bonus(self, reward_history: List[float]) -> float:
        """Calculate win streak bonus."""
        if len(reward_history) < 5:
            return 0.0

        recent_rewards = reward_history[-5:]
        win_count = sum(1 for r in recent_rewards if r > 0)

        if win_count >= 3:
            return win_count * 0.01

        return 0.0

    def reset(self) -> None:
        """Reset internal state."""
        self._action_counts = [0, 0, 0]


__all__ = ["RewardCalculator"]
