"""
Reward Kernel - Stateless core logic for reward calculation.

This module provides the fundamental reward calculation formulas used across
both HeavyTradingEnv and LiteTradingEnv to ensure consistency.
"""

from dataclasses import dataclass
from typing import Optional

from ztb.trading.constants import (
    ACTION_BUY,
    ACTION_HOLD,
    ACTION_SELL,
)

@dataclass(frozen=True)
class RewardParams:
    """Parameters for reward calculation."""
    reward_scaling: float = 1.0
    hold_penalty_multiplier: float = 1.0
    trade_frequency_bonus: float = 0.0
    bankruptcy_penalty: float = -100.0
    position_change_penalty: float = 0.0
    position_change_threshold: float = 0.1
    reward_clip_value: Optional[float] = None

class RewardKernel:
    """Stateless core logic for reward calculation."""

    @staticmethod
    def calculate_basic_reward(
        pnl: float,
        action: int,
        params: RewardParams,
        old_position: float = 0.0,
        current_position: float = 0.0,
        portfolio_value: float = 1.0,
    ) -> float:
        """
        Calculate a basic reward based on PnL and behavioral factors.

        This implements the core logic shared between Heavy and Lite environments.
        """
        # 1. Base PnL scaling
        reward = float(pnl * params.reward_scaling)

        # 2. Behavioral Adjustments
        if action == ACTION_HOLD:
            reward *= params.hold_penalty_multiplier
        elif action in (ACTION_BUY, ACTION_SELL):
            reward += params.trade_frequency_bonus

        # 3. Position change penalty (v431/v440 logic)
        position_change = abs(current_position - old_position)
        if params.position_change_penalty > 0 and position_change > params.position_change_threshold:
            reward -= params.position_change_penalty

        # 4. Bankruptcy penalty
        if portfolio_value <= 0:
            reward += params.bankruptcy_penalty

        # 5. Clipping
        if params.reward_clip_value is not None:
            clip = params.reward_clip_value
            reward = max(-clip, min(clip, reward))

        return reward
