import logging

from .base import RewardComponent, RewardContext
from .utils import RewardUtils

class ProfitOptimizedReward(RewardComponent):
    """
    Stage: Profit-optimized reward that maximizes profitable trading while minimizing losses.
    Ported from RewardCalculator._calculate_profit_optimized_reward.
    """

    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.ACTION_BUY = 1
        self.ACTION_SELL = -1
        self.ACTION_HOLD = 0

    def get_name(self) -> str:
        return "profit_optimized"

    def calculate(self, context: RewardContext) -> float:
        # 1. Calculate base reward from PnL
        # Original: _calculate_base_reward -> _calculate_pnl_reward(pnl, 1.0) -> pnl
        base_reward = context.pnl

        # 2. Apply profit/loss modifiers
        profit_multiplier = self._get_setting_float(context, "profit_multiplier", 2.0)
        loss_penalty_multiplier = self._get_setting_float(
            context, "loss_penalty_multiplier", 1.5
        )
        profit_sell_penalty_rate = self._get_setting_float(
            context, "profit_sell_penalty_rate", 0.0
        )
        profit_hold_bonus_rate = self._get_setting_float(
            context, "profit_hold_bonus_rate", 0.0
        )

        pnl_normalizer = (
            context.atr * context.effective_max_position * context.current_price
        )
        normalized_pnl = context.pnl / max(pnl_normalizer, 1e-8)

        if context.pnl > 0:
            profit_bonus = normalized_pnl * profit_multiplier
            base_reward += profit_bonus

            if context.action == self.ACTION_SELL and profit_sell_penalty_rate > 0:
                profit_sell_penalty = normalized_pnl * profit_sell_penalty_rate
                base_reward -= profit_sell_penalty

            if context.action == self.ACTION_HOLD and profit_hold_bonus_rate > 0:
                profit_hold_bonus = normalized_pnl * profit_hold_bonus_rate
                base_reward += profit_hold_bonus

        elif context.pnl < 0:
            loss_penalty = abs(normalized_pnl) * loss_penalty_multiplier
            base_reward -= loss_penalty

        # 3. Apply common rewards/penalties for trading actions
        if context.action in [self.ACTION_BUY, self.ACTION_SELL]:
            # _calculate_base_trading_reward logic

            # Trading bonus
            trading_bonus_multiplier = self._get_setting_float(
                context, "trading_bonus_multiplier", 3.0
            )
            trading_bonus_base = self._get_setting_float(context, "trading_bonus", 0.01)
            trading_bonus = trading_bonus_base * trading_bonus_multiplier
            base_reward += trading_bonus

            # Position size bonus
            position_size_bonus_rate = self._get_setting_float(
                context, "position_size_bonus_rate", 0.05
            )
            base_reward += RewardUtils.calculate_position_size_bonus(
                context.position,
                context.effective_max_position,
                bonus_rate=position_size_bonus_rate,
            )

            # Activity incentive bonus
            activity_bonus_rate = self._get_setting_float(
                context, "activity_bonus_rate", 0.02
            )
            base_reward += RewardUtils.calculate_activity_bonus(
                context.recent_actions,
                bonus_rate=activity_bonus_rate,
                action_hold=self.ACTION_HOLD,
            )

        elif context.action == self.ACTION_HOLD:
            hold_penalty_rate = self._get_setting_float(
                context, "hold_penalty_rate", 0.1
            )
            hold_penalty = (
                hold_penalty_rate
                * abs(context.position)
                / max(context.effective_max_position, 0.01)
            )
            base_reward -= hold_penalty

        # 4. Apply balance penalty
        target_ratios = [0.15, 0.425, 0.425]  # [HOLD, BUY, SELL]
        tolerance = self._get_setting_float(context, "balance_penalty_tolerance", 0.05)
        penalty_coeff = self._get_setting_float(context, "balance_penalty", 6.0)

        balance_penalty = RewardUtils.calculate_balance_penalty(
            context.action_counts, target_ratios, tolerance, penalty_coeff
        )

        final_reward = base_reward - balance_penalty
        return final_reward * context.reward_scaling
