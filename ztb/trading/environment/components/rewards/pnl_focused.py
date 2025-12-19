import logging
from typing import Any

import numpy as np

from .base import RewardComponent, RewardContext


class PnlFocusedReward(RewardComponent):
    """
    Stage 2: PnL-focused reward with trend analysis.
    Ported from RewardCalculator._calculate_pnl_focused_reward.
    """

    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.ACTION_BUY = 1
        self.ACTION_SELL = -1
        self.ACTION_HOLD = 0
        self.MULTIPLIER_INDEX_BUY = 0
        self.MULTIPLIER_INDEX_SELL = 1
        self.MULTIPLIER_INDEX_HOLD = 2

    def get_name(self) -> str:
        return "pnl_focused"


    def calculate(self, context: RewardContext) -> float:
        # Base profit bonus calculation
        base_profit_bonus_atr_coeff = self._get_setting_float(
            context, "base_profit_bonus_atr_coeff", 1.5
        )
        base_profit_bonus_portfolio_coeff = self._get_setting_float(
            context, "base_profit_bonus_portfolio_coeff", 1.2
        )

        # Use pre-calculated values from context if available/appropriate
        atr_normalised = context.atr_normalised
        portfolio_return = context.portfolio_return

        base_profit_bonus = 0.0
        if context.pnl > 0:
            base_profit_bonus = max(
                0.0,
                base_profit_bonus_atr_coeff * atr_normalised
                + base_profit_bonus_portfolio_coeff * portfolio_return,
            )

        # Trend analysis if observation available
        trend_multiplier = 1.0
        if context.observation is not None and hasattr(
            context.observation, "__getitem__"
        ):
            try:
                # Assume RSI and MACD are in observation
                rsi_idx = -2 if len(context.observation) > 2 else None
                macd_idx = -1 if len(context.observation) > 1 else None

                if rsi_idx is not None and macd_idx is not None:
                    rsi = float(context.observation[rsi_idx])
                    macd_hist = float(context.observation[macd_idx])

                    # Check for NaN/inf
                    if not (
                        np.isnan(rsi)
                        or np.isinf(rsi)
                        or np.isnan(macd_hist)
                        or np.isinf(macd_hist)
                    ):
                        # Trend ratio
                        trend_ratio = (rsi / 50.0) * (1.0 + macd_hist / 100.0)

                        if trend_ratio > 1.0 and context.action == self.ACTION_BUY:
                            trend_multiplier = 1.2
                        elif trend_ratio < 1.0 and context.action == self.ACTION_SELL:
                            trend_multiplier = 1.2

                        # Oversold/Overbought signals (Symmetric Implementation)
                        if (
                            rsi > 60.0
                            and trend_ratio > 1.0
                            and context.action == self.ACTION_SELL
                        ):
                            trend_multiplier *= 1.3
                        elif (
                            rsi < 40.0
                            and trend_ratio < 1.0
                            and context.action == self.ACTION_BUY
                        ):
                            trend_multiplier *= 1.3
            except (IndexError, TypeError, ValueError):
                pass

        # Balance BUY/SELL actions
        multipliers_raw = self._get_setting(context, "profit_bonus_multipliers", None)

        if isinstance(multipliers_raw, list) and len(multipliers_raw) >= 3:
            multipliers = [float(x) for x in multipliers_raw[:3]]
        else:
            multipliers = [1.0, 1.0, 0.8]

        if context.action == self.ACTION_BUY:
            profit_bonus = (
                base_profit_bonus
                * multipliers[self.MULTIPLIER_INDEX_BUY]
                * trend_multiplier
            )
        elif context.action == self.ACTION_SELL:
            profit_bonus = (
                base_profit_bonus
                * multipliers[self.MULTIPLIER_INDEX_SELL]
                * trend_multiplier
            )
        else:  # HOLD
            profit_bonus = (
                base_profit_bonus
                * multipliers[self.MULTIPLIER_INDEX_HOLD]
                * trend_multiplier
            )

        # Action penalties
        hold_penalty = self._get_setting_float(context, "hold_action_penalty", 0.0)
        buy_penalty = self._get_setting_float(context, "buy_action_penalty", 0.0)
        sell_penalty = self._get_setting_float(context, "sell_action_penalty", 0.0)

        base_action_penalty = 0.0
        if context.action in [self.ACTION_BUY, self.ACTION_SELL]:
            base_action_penalty = self._get_setting_float(
                context, "base_action_penalty", 0.015
            )

        action_penalty = 0.0
        if context.action == self.ACTION_HOLD:
            position_size_factor = abs(context.position) / max(
                context.effective_max_position, 1e-8
            )
            volatility_factor = (
                min(context.atr / (context.current_price * 0.01), 1.0)
                if context.current_price > 0
                else 1.0
            )

            base_hold_penalty = self._get_setting_float(
                context, "hold_penalty_base", 0.01
            ) + (
                self._get_setting_float(context, "hold_penalty_position_factor", 0.04)
                * position_size_factor
                * volatility_factor
            )
            base_hold_penalty *= self._get_setting_float(
                context, "hold_penalty_multiplier", 1.0
            )

            action_penalty = base_hold_penalty + hold_penalty

        elif context.action == self.ACTION_BUY:
            action_penalty = base_action_penalty + buy_penalty
        elif context.action == self.ACTION_SELL:
            action_penalty = base_action_penalty + sell_penalty

        # Loss penalty
        loss_penalty = 0.0
        if context.pnl < 0:
            loss_penalty = self._get_setting_float(
                context, "loss_penalty_coeff", -0.2
            ) * abs(atr_normalised)

        # Position penalty (simplified version of _calculate_position_penalty)
        position_penalty = 0.0

        reward = profit_bonus - action_penalty + loss_penalty - position_penalty

        return reward
