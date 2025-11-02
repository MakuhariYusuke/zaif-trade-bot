"""
PnL Focused Reward Calculator Component.

This component handles PnL-focused reward calculation with trend analysis.
"""

from typing import Optional

import numpy as np

from ztb.trading.constants import (
    ACTION_BUY,
    ACTION_HOLD,
    ACTION_SELL,
    MULTIPLIER_INDEX_BUY,
    MULTIPLIER_INDEX_HOLD,
    MULTIPLIER_INDEX_SELL,
)
from ztb.trading.environment.constants import EPSILON

from .base_reward_calculator import BaseRewardCalculator
from .position_penalty import PositionPenaltyCalculator


class PnLFocusedRewardCalculator(BaseRewardCalculator):
    """
    Calculates PnL-focused rewards with trend analysis.

    This component specializes in calculating rewards based on profit/loss
    with trend-aware multipliers and balanced action penalties.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.position_penalty_calculator = PositionPenaltyCalculator()

    def calculate_pnl_focused_reward(
        self,
        action: int,
        atr_normalised: float,
        portfolio_return: float,
        position: float,
        effective_max_position: float,
        current_price: float,
        atr: float,
        pnl: float,
        reward_scaling: float,
        observation: Optional[np.ndarray],
        step: int,
    ) -> float:
        """Calculate PnL-focused reward with trend analysis and fair action penalties."""
        # Base profit bonus
        base_profit_bonus = (
            max(
                0.0,
                self.get_setting_float("base_profit_bonus_atr_coeff", 1.5)
                * atr_normalised
                + self.get_setting_float("base_profit_bonus_portfolio_coeff", 1.2)
                * portfolio_return,
            )
            if pnl > 0
            else 0.0
        )

        # Trend analysis if observation available
        trend_multiplier = self._calculate_trend_multiplier(action, observation)

        # Fair profit bonus multipliers (same for BUY and SELL)
        multipliers_raw = self.reward_settings.custom_reward_params.get(
            "profit_bonus_multipliers", [1.0, 1.0, 0.8]
        )
        if isinstance(multipliers_raw, list) and len(multipliers_raw) >= 3:
            multipliers = [float(x) for x in multipliers_raw[:3]]
        else:
            multipliers = [1.0, 1.0, 0.8]

        self.logger.debug(f"Profit bonus multipliers: {multipliers}")

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

        # Fair action penalties (same for BUY and SELL)
        action_penalty = self._calculate_fair_action_penalty(
            action, position, effective_max_position, current_price, atr
        )

        # Loss penalty
        loss_penalty = (
            self.get_setting_float("loss_penalty_coeff", -0.2) * abs(atr_normalised)
            if pnl < 0
            else 0.0
        )

        # Position penalty
        position_penalty = self.position_penalty_calculator.calculate(
            position, effective_max_position
        )

        reward = profit_bonus - action_penalty + loss_penalty - position_penalty

        self.logger.debug(
            f"PnL focused reward components: profit_bonus={profit_bonus:.4f}, "
            f"action_penalty={action_penalty:.4f}, loss_penalty={loss_penalty:.4f}, "
            f"position_penalty={position_penalty:.4f}, final={reward:.4f}"
        )

        return reward

    def _calculate_trend_multiplier(
        self, action: int, observation: Optional[np.ndarray]
    ) -> float:
        """Calculate trend-based multiplier for profit bonus."""
        if observation is None or not hasattr(observation, "__getitem__"):
            return 1.0

        try:
            # Assume RSI and MACD are in observation
            rsi_idx = -2 if len(observation) > 2 else None
            macd_idx = -1 if len(observation) > 1 else None

            if rsi_idx is not None and macd_idx is not None:
                rsi = float(observation[rsi_idx])
                macd_hist = float(observation[macd_idx])

                # Check for NaN/inf
                if (
                    np.isnan(rsi)
                    or np.isinf(rsi)
                    or np.isnan(macd_hist)
                    or np.isinf(macd_hist)
                ):
                    return 1.0

                # Trend ratio
                trend_ratio = (rsi / 50.0) * (1.0 + macd_hist / 100.0)

                if trend_ratio > 1.0 and action == ACTION_BUY:
                    return 1.2
                elif trend_ratio < 1.0 and action == ACTION_SELL:
                    return 1.2

                # Oversold/Overbought signals
                if rsi > 60.0 and trend_ratio > 1.0 and action == ACTION_SELL:
                    return 1.3
                elif rsi < 40.0 and trend_ratio < 1.0 and action == ACTION_SELL:
                    return 1.3
        except (IndexError, TypeError, ValueError):
            pass

        return 1.0

    def _calculate_fair_action_penalty(
        self,
        action: int,
        position: float,
        effective_max_position: float,
        current_price: float,
        atr: float,
    ) -> float:
        """Calculate fair action penalties (same for BUY and SELL)."""
        # Get action bonuses settings
        buy_action_bonus = self.get_setting_float(
            "action_bonuses.buy_action_bonus", 0.0
        )
        sell_action_bonus = self.get_setting_float(
            "action_bonuses.sell_action_bonus", 0.0
        )
        hold_action_bonus = self.get_setting_float(
            "action_bonuses.hold_action_bonus", 0.0
        )

        # Base action penalties (legacy behavior)
        base_action_penalty = (
            self.get_setting_float("base_action_penalty", 0.015)
            if action in [ACTION_BUY, ACTION_SELL]
            else 0.0
        )

        if action == ACTION_HOLD:
            # Get HOLD penalty settings from custom_reward_params
            hold_penalty_base = self.reward_settings.custom_reward_params.get(
                "hold_penalty_base", 0.01
            )
            hold_penalty_position_factor = self.reward_settings.custom_reward_params.get(
                "hold_penalty_position_factor", 0.04
            )
            hold_penalty_multiplier = self.reward_settings.custom_reward_params.get(
                "hold_penalty_multiplier", 1.0
            )

            self.logger.debug(
                f"HOLD penalty settings - base: {hold_penalty_base}, position_factor: {hold_penalty_position_factor}, multiplier: {hold_penalty_multiplier}"
            )

            position_size_factor = abs(position) / max(effective_max_position, EPSILON)
            volatility_factor = min(atr / (current_price * 0.01), 1.0)
            base_action_penalty = (
                hold_penalty_base
                + hold_penalty_position_factor
                * position_size_factor
                * volatility_factor
            )
            base_action_penalty *= hold_penalty_multiplier
            # Add action bonus for HOLD
            return base_action_penalty + hold_action_bonus
        elif action == ACTION_BUY:
            # Apply action bonus for BUY
            return base_action_penalty + buy_action_bonus
        elif action == ACTION_SELL:
            # Apply action bonus for SELL
            return base_action_penalty + sell_action_bonus

        return 0.0
