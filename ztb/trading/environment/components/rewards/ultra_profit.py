import logging

import numpy as np

from .base import RewardComponent, RewardContext
from .utils import RewardUtils


class UltraProfitReward(RewardComponent):
    """
    Simplified ultra-profit reward that focuses on basic trading principles.
    Ported from RewardCalculator._calculate_ultra_profit_reward.
    """

    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.ACTION_BUY = 1
        self.ACTION_SELL = 2
        self.ACTION_HOLD = 0

    def get_name(self) -> str:
        return "ultra_profit"

    def _get_setting_float(
        self, context: RewardContext, key: str, default: float
    ) -> float:
        if context.reward_settings:
            if hasattr(context.reward_settings, "get"):
                val = context.reward_settings.get(key)
                if val is not None:
                    try:
                        return float(val)
                    except (ValueError, TypeError):
                        pass
            elif hasattr(context.reward_settings, key):
                val = getattr(context.reward_settings, key)
                if val is not None:
                    try:
                        return float(val)
                    except (ValueError, TypeError):
                        pass

        if hasattr(context.config, "get"):
            val = context.config.get(key, default)
            try:
                return float(val)
            except (ValueError, TypeError):
                return default
        return default

    def calculate(self, context: RewardContext) -> float:
        reward = 0.0

        # Profit/Loss component - normalized by ATR
        # Note: context.atr_normalised is passed in context
        if context.pnl > 0:
            reward += context.atr_normalised * 2.0
        elif context.pnl < 0:
            reward -= abs(context.atr_normalised) * 1.0

        # Position penalty
        reward -= RewardUtils.calculate_position_penalty(
            context.position,
            context.effective_max_position,
            threshold=0.5,
            penalty_coeff=0.2,
        )

        # Action diversity encouragement (Balance Penalty)
        # Target ratios: [0.2, 0.4, 0.4] (HOLD, BUY, SELL)
        # Tolerance: 0.15
        # Penalty Coeff: 0.005
        reward -= RewardUtils.calculate_balance_penalty(
            context.action_counts,
            target_ratios=[0.2, 0.4, 0.4],
            tolerance=0.15,
            penalty_coeff=0.005,
        )

        # Strong trading bonus
        reward += RewardUtils.calculate_trading_bonus(
            context.action,
            bonus_amount=0.1,
            action_buy=self.ACTION_BUY,
            action_sell=self.ACTION_SELL,
        )

        # Apply scaling and clipping
        reward *= context.reward_scaling

        reward_clip_min = self._get_setting_float(context, "reward_clip_min", -1.0)
        reward_clip_max = self._get_setting_float(context, "reward_clip_max", 1.0)

        reward = np.clip(reward, reward_clip_min, reward_clip_max)

        return float(reward)
