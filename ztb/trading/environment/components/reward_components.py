"""
Reward Components - Individual reward calculation components.

This module contains individual reward calculation components that can be combined
to create complex reward functions.
"""

from typing import Any, Dict, List, Optional, cast

import numpy as np

from ztb.trading.constants import (
    ACTION_BUY,
    ACTION_HOLD,
    ACTION_SELL,
    MULTIPLIER_INDEX_BUY,
    MULTIPLIER_INDEX_HOLD,
    MULTIPLIER_INDEX_SELL,
)
from ztb.trading.environment.components.reward_utils import RewardUtils


class RewardComponents:
    """Individual reward calculation components."""

    def __init__(self, reward_settings: Optional[Dict[str, Any]] = None):
        """Initialize reward components with settings."""
        self.reward_settings = cast(Dict[str, Any], reward_settings or {})

    def calculate_drawdown_penalty(
        self,
        current_portfolio_value: float,
        portfolio_value_history: List[float],
        reward_history: List[float],
    ) -> float:
        """Calculate drawdown penalty (prevent large losses)."""
        drawdown_window = RewardUtils.get_setting_int(
            self.reward_settings, "drawdown_window", 10
        )
        if (
            len(portfolio_value_history) < drawdown_window
            or len(reward_history) < drawdown_window
        ):
            return 0.0

        recent_values = portfolio_value_history[-drawdown_window:]
        recent_rewards = reward_history[-drawdown_window:]
        cumulative_reward = sum(recent_rewards)

        stagnation_window = RewardUtils.get_setting_int(
            self.reward_settings, "stagnation_window", 30
        )
        if len(reward_history) >= stagnation_window:
            initial_window = stagnation_window - drawdown_window
            initial_rewards = reward_history[-stagnation_window:-initial_window]
            initial_cumulative = sum(initial_rewards)

            if initial_cumulative > 0:
                drawdown_ratio = (
                    initial_cumulative - cumulative_reward
                ) / initial_cumulative
                drawdown_threshold = 0.5
                if drawdown_ratio > drawdown_threshold:
                    drawdown_penalty_coeff = RewardUtils.get_setting_float(
                        self.reward_settings, "drawdown_penalty_coeff", 0.05
                    )
                    return drawdown_ratio * drawdown_penalty_coeff

        return 0.0

    def calculate_stagnation_penalty(
        self, portfolio_value_history: List[float]
    ) -> float:
        """Calculate stagnation penalty (when portfolio isn't growing)."""
        stagnation_window = RewardUtils.get_setting_int(
            self.reward_settings, "stagnation_window", 30
        )
        if len(portfolio_value_history) < stagnation_window:
            return 0.0

        recent_values = portfolio_value_history[-stagnation_window:]
        initial_value = recent_values[0]
        final_value = recent_values[-1]

        if initial_value > 0:
            growth_rate = (final_value - initial_value) / initial_value
            stagnation_threshold = RewardUtils.get_setting_float(
                self.reward_settings, "stagnation_threshold", -0.005
            )

            if growth_rate < stagnation_threshold:
                stagnation_penalty_max = RewardUtils.get_setting_float(
                    self.reward_settings, "stagnation_penalty_max", 0.02
                )
                return min(
                    stagnation_penalty_max,
                    abs(growth_rate - stagnation_threshold) * 0.5,
                )

        return 0.0

    def calculate_growth_bonus(self, portfolio_value_history: List[float]) -> float:
        """Calculate growth bonus (when portfolio is increasing)."""
        stagnation_window = RewardUtils.get_setting_int(
            self.reward_settings, "stagnation_window", 30
        )
        if len(portfolio_value_history) < stagnation_window:
            return 0.0

        recent_values = portfolio_value_history[-stagnation_window:]
        initial_value = recent_values[0]
        final_value = recent_values[-1]

        if initial_value > 0:
            growth_rate = (final_value - initial_value) / initial_value
            growth_threshold = RewardUtils.get_setting_float(
                self.reward_settings, "growth_threshold", 0.005
            )

            if growth_rate > growth_threshold:
                growth_bonus_max = RewardUtils.get_setting_float(
                    self.reward_settings, "growth_bonus_max", 0.05
                )
                return min(growth_bonus_max, growth_rate * 0.5)

        return 0.0

    def calculate_win_streak_bonus(self, reward_history: List[float]) -> float:
        """Calculate win streak bonus."""
        win_streak_window = RewardUtils.get_setting_int(
            self.reward_settings, "win_streak_window", 5
        )
        if len(reward_history) < win_streak_window:
            return 0.0

        recent_rewards = reward_history[-win_streak_window:]
        win_count = sum(1 for r in recent_rewards if r > 0)

        win_streak_min = 3
        if win_count >= win_streak_min:
            win_streak_bonus_per_win = RewardUtils.get_setting_float(
                self.reward_settings, "win_streak_bonus_per_win", 0.01
            )
            return win_count * win_streak_bonus_per_win

        return 0.0

    def _calculate_win_rate_bonus(self, discrete_action: int, pnl: float) -> float:
        """Calculate win rate bonus based on recent trading performance."""
        # This is a placeholder - actual implementation would track win/loss history
        return 0.0

    def _calculate_diversity_bonus(self, action: int) -> float:
        """Calculate diversity bonus for action variety."""
        # This is a placeholder - actual implementation would track action diversity
        return 0.0

    def _calculate_position_penalty(
        self, position: float, effective_max_position: float
    ) -> float:
        """Calculate position size penalty."""
        if effective_max_position <= 0:
            return 0.0

        position_ratio = abs(position) / effective_max_position

        # Penalty increases with position size to discourage over-leveraging
        if position_ratio > 0.8:
            penalty = RewardUtils.get_setting_float(
                self.reward_settings, "high_position_penalty", 0.05
            )
        elif position_ratio > 0.5:
            penalty = RewardUtils.get_setting_float(
                self.reward_settings, "medium_position_penalty", 0.02
            )
        else:
            penalty = 0.0

        return penalty * position_ratio

    def _calculate_base_reward(
        self,
        action: int,
        atr_normalised: float,
        portfolio_return: float,
        position: float,
        effective_max_position: float,
        current_price: float,
        atr: float,
        pnl: float,
        observation: Optional[np.ndarray] = None,
    ) -> float:
        """Calculate base reward components."""
        # Profit bonus
        base_profit_bonus = (
            max(
                0.0,
                RewardUtils.get_setting_float(
                    self.reward_settings, "base_profit_bonus_atr_coeff", 1.5
                )
                * atr_normalised
                + RewardUtils.get_setting_float(
                    self.reward_settings, "base_profit_bonus_portfolio_coeff", 1.2
                )
                * portfolio_return,
            )
            if pnl > 0
            else 0.0
        )

        # Profit bonus multipliers
        multipliers = self.reward_settings.get(
            "profit_bonus_multipliers", [1.0, 1.0, 0.8]
        )

        if action == ACTION_BUY:
            profit_bonus = base_profit_bonus * multipliers[MULTIPLIER_INDEX_BUY]
        elif action == ACTION_SELL:
            profit_bonus = base_profit_bonus * multipliers[MULTIPLIER_INDEX_SELL]
        else:  # HOLD
            profit_bonus = base_profit_bonus * multipliers[MULTIPLIER_INDEX_HOLD]

        # Action penalties
        hold_penalty = RewardUtils.get_setting_float(
            self.reward_settings, "hold_action_penalty", 0.0
        )
        buy_penalty = RewardUtils.get_setting_float(
            self.reward_settings, "buy_action_penalty", 0.0
        )
        sell_penalty = RewardUtils.get_setting_float(
            self.reward_settings, "sell_action_penalty", 0.0
        )

        base_action_penalty = (
            RewardUtils.get_setting_float(
                self.reward_settings, "base_action_penalty", 0.015
            )
            if action in [ACTION_BUY, ACTION_SELL]
            else 0.0
        )

        if action == ACTION_HOLD:
            position_size_factor = abs(position) / max(effective_max_position, 0.01)
            volatility_factor = min(atr / (current_price * 0.01), 1.0)
            base_action_penalty = RewardUtils.get_setting_float(
                self.reward_settings, "hold_penalty_base", 0.01
            ) + (
                RewardUtils.get_setting_float(
                    self.reward_settings, "hold_penalty_position_factor", 0.04
                )
                * position_size_factor
                * volatility_factor
            )
            action_penalty = base_action_penalty + hold_penalty
        elif action == ACTION_BUY:
            action_penalty = base_action_penalty + buy_penalty
        else:  # ACTION_SELL
            action_penalty = base_action_penalty + sell_penalty

        # Loss penalty
        loss_penalty = (
            RewardUtils.get_setting_float(
                self.reward_settings, "loss_penalty_coeff", -0.2
            )
            * abs(atr_normalised)
            if pnl < 0
            else 0.0
        )

        # Position penalty
        position_penalty = self._calculate_position_penalty(
            position, effective_max_position
        )

        reward = profit_bonus - action_penalty + loss_penalty - position_penalty
        return reward

    def _calculate_pnl_focused_reward(
        self,
        action: int,
        atr_normalised: float,
        portfolio_return: float,
        position: float,
        effective_max_position: float,
        current_price: float,
        atr: float,
        pnl: float,
        observation: Optional[np.ndarray] = None,
    ) -> float:
        """Calculate PnL-focused reward."""
        base_profit_bonus = (
            max(
                0.0,
                RewardUtils.get_setting_float(
                    self.reward_settings, "base_profit_bonus_atr_coeff", 1.5
                )
                * atr_normalised
                + RewardUtils.get_setting_float(
                    self.reward_settings, "base_profit_bonus_portfolio_coeff", 1.2
                )
                * portfolio_return,
            )
            if pnl > 0
            else 0.0
        )

        multipliers = self.reward_settings.get(
            "profit_bonus_multipliers", [1.0, 1.0, 0.8]
        )

        # Trend multiplier based on observation (if available)
        trend_multiplier = 1.0
        if observation is not None and len(observation) >= 2:
            try:
                # Assume last elements are trend indicators
                trend_indicator = float(observation[-1])
                if abs(trend_indicator) > 0.1:  # Significant trend
                    trend_multiplier = 1.2
            except (IndexError, TypeError, ValueError):
                pass

        if action == ACTION_BUY:
            profit_bonus = (
                base_profit_bonus * multipliers[MULTIPLIER_INDEX_BUY] * trend_multiplier
            )
        elif action == ACTION_SELL:
            profit_bonus = (
                base_profit_bonus
                * multipliers[MULTIPLIER_INDEX_SELL]
                * trend_multiplier
            )
        else:  # HOLD
            profit_bonus = (
                base_profit_bonus
                * multipliers[MULTIPLIER_INDEX_HOLD]
                * trend_multiplier
            )

        # Action penalties
        hold_penalty = RewardUtils.get_setting_float(
            self.reward_settings, "hold_action_penalty", 0.0
        )
        buy_penalty = RewardUtils.get_setting_float(
            self.reward_settings, "buy_action_penalty", 0.0
        )
        sell_penalty = RewardUtils.get_setting_float(
            self.reward_settings, "sell_action_penalty", 0.0
        )

        base_action_penalty = (
            RewardUtils.get_setting_float(
                self.reward_settings, "base_action_penalty", 0.015
            )
            if action in [ACTION_BUY, ACTION_SELL]
            else 0.0
        )

        if action == ACTION_HOLD:
            position_size_factor = abs(position) / effective_max_position
            volatility_factor = min(atr / (current_price * 0.01), 1.0)
            base_action_penalty = RewardUtils.get_setting_float(
                self.reward_settings, "hold_penalty_base", 0.01
            ) + (
                RewardUtils.get_setting_float(
                    self.reward_settings, "hold_penalty_position_factor", 0.04
                )
                * position_size_factor
                * volatility_factor
            )
            action_penalty = base_action_penalty + hold_penalty
        elif action == ACTION_BUY:
            action_penalty = base_action_penalty + buy_penalty
        else:  # ACTION_SELL
            action_penalty = base_action_penalty + sell_penalty

        # Loss penalty
        loss_penalty = (
            RewardUtils.get_setting_float(
                self.reward_settings, "loss_penalty_coeff", -0.2
            )
            * abs(atr_normalised)
            if pnl < 0
            else 0.0
        )

        # Position penalty
        position_penalty = self._calculate_position_penalty(
            position, effective_max_position
        )

        reward = profit_bonus - action_penalty + loss_penalty - position_penalty
        return reward

    def _calculate_default_reward(
        self,
        action: int,
        atr_normalised: float,
        portfolio_return: float,
        position: float,
        effective_max_position: float,
        current_price: float,
        atr: float,
        pnl: float,
        observation: Optional[np.ndarray] = None,
    ) -> float:
        """Default reward calculation."""
        base_profit_bonus = (
            max(
                0.0,
                RewardUtils.get_setting_float(
                    self.reward_settings, "base_profit_bonus_atr_coeff", 1.5
                )
                * atr_normalised
                + RewardUtils.get_setting_float(
                    self.reward_settings, "base_profit_bonus_portfolio_coeff", 1.2
                )
                * portfolio_return,
            )
            if pnl > 0
            else 0.0
        )

        multipliers = self.reward_settings.get(
            "profit_bonus_multipliers", [1.0, 1.0, 0.8]
        )

        hold_penalty = RewardUtils.get_setting_float(
            self.reward_settings, "hold_action_penalty", 0.0
        )
        buy_penalty = RewardUtils.get_setting_float(
            self.reward_settings, "buy_action_penalty", 0.0
        )
        sell_penalty = RewardUtils.get_setting_float(
            self.reward_settings, "sell_action_penalty", 0.0
        )

        if action == ACTION_BUY:
            profit_bonus = base_profit_bonus * multipliers[MULTIPLIER_INDEX_BUY]
            action_penalty = 0.015 + buy_penalty
        elif action == ACTION_SELL:
            profit_bonus = base_profit_bonus * multipliers[MULTIPLIER_INDEX_SELL]
            action_penalty = 0.015 + sell_penalty
        else:  # HOLD
            profit_bonus = base_profit_bonus * multipliers[MULTIPLIER_INDEX_HOLD]
            position_size_factor = abs(position) / effective_max_position
            volatility_factor = min(atr / (current_price * 0.01), 1.0)
            action_penalty = (
                0.01 + (0.04 * position_size_factor * volatility_factor) + hold_penalty
            )

        loss_penalty = -0.2 * abs(atr_normalised) if pnl < 0 else 0.0

        # Forced diversity bonus
        diversity_bonus = 0.0
        if RewardUtils.get_setting_bool(
            self.reward_settings, "enable_forced_diversity", False
        ):
            diversity_bonus = self._calculate_diversity_bonus(action)

        # Win rate bonus
        win_rate_bonus = self._calculate_win_rate_bonus(action, pnl)

        # Momentum bonus (based on RSI and MACD from observation)
        momentum_bonus = 0.0
        if observation is not None and len(observation) >= 2:
            try:
                rsi_idx = -2 if len(observation) > 2 else None
                macd_idx = -1 if len(observation) > 1 else None
                if rsi_idx is not None and macd_idx is not None:
                    rsi = float(observation[rsi_idx])
                    macd_hist = float(observation[macd_idx])
                    momentum_multiplier = RewardUtils.get_setting_float(
                        self.reward_settings, "momentum_bonus", 0.0
                    )
                    if action == ACTION_BUY and rsi > 50 and macd_hist > 0:
                        momentum_bonus = momentum_multiplier
                    elif action == ACTION_SELL and rsi < 50 and macd_hist < 0:
                        momentum_bonus = momentum_multiplier
            except (IndexError, TypeError, ValueError):
                pass

        total_reward = (
            profit_bonus
            - action_penalty
            + loss_penalty
            + diversity_bonus
            + win_rate_bonus
            + momentum_bonus
        )
        return total_reward
