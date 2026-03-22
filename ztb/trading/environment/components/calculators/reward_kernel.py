"""
Reward Kernel - Stateless core logic for reward calculation.

This module provides the fundamental reward calculation formulas used across
both HeavyTradingEnv and LiteTradingEnv to ensure consistency.
It delegates complex behavioral calculations to RewardUtils.
"""

from dataclasses import dataclass, field
from typing import Optional, List

from ztb.trading.constants import (
    ACTION_BUY,
    ACTION_HOLD,
    ACTION_SELL,
)
from ztb.trading.environment.components.rewards.utils import RewardUtils

@dataclass(frozen=True)
class RewardParams:
    """Parameters for reward calculation."""
    reward_scaling: float = 1.0
    hold_penalty_multiplier: float = 1.0
    trade_frequency_bonus: float = 0.0
    bankruptcy_penalty: float = -100.0
    position_change_penalty: float = 0.0
    position_change_threshold: float = 0.1
    reward_clip_value: Optional[float] = 10.0
    
    # Advanced behavioral params (from RewardUtils)
    position_size_bonus_rate: float = 0.0
    activity_bonus_rate: float = 0.0
    balance_penalty_coeff: float = 0.0
    balance_penalty_tolerance: float = 0.05
    balance_penalty_targets: List[float] = field(default_factory=lambda: [0.4, 0.3, 0.3]) # [HOLD, BUY, SELL]

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
        effective_max_position: float = 0.0,
        recent_actions: Optional[List[int]] = None,
        action_counts: Optional[List[int]] = None,
    ) -> float:
        """
        Calculate a basic reward based on PnL and behavioral factors.

        Delegates behavioral calculations to RewardUtils to ensure consistency.
        """
        # 1. Base PnL scaling (via RewardUtils helper)
        reward = RewardUtils.calculate_pnl_reward(pnl, params.reward_scaling)

        # 2. Basic Behavioral Adjustments
        if action == ACTION_HOLD:
            reward *= params.hold_penalty_multiplier
        elif action in (ACTION_BUY, ACTION_SELL):
            # Trading bonus
            reward += RewardUtils.calculate_trading_bonus(action, params.trade_frequency_bonus)

        # 3. Position change penalty (v431/v440 logic)
        position_change = abs(current_position - old_position)
        if params.position_change_penalty > 0 and position_change > params.position_change_threshold:
            reward -= params.position_change_penalty

        # 4. Advanced Behavioral Components (from RewardUtils)
        
        # Position size bonus
        if params.position_size_bonus_rate > 0 and effective_max_position > 0:
            reward += RewardUtils.calculate_position_size_bonus(
                current_position, 
                effective_max_position, 
                bonus_rate=params.position_size_bonus_rate
            )
            
        # Activity bonus
        if params.activity_bonus_rate > 0 and recent_actions:
            reward += RewardUtils.calculate_activity_bonus(
                recent_actions,
                bonus_rate=params.activity_bonus_rate,
                action_hold=ACTION_HOLD
            )
            
        # Balance penalty
        if params.balance_penalty_coeff > 0 and action_counts:
            reward -= RewardUtils.calculate_balance_penalty(
                action_counts,
                params.balance_penalty_targets,
                params.balance_penalty_tolerance,
                params.balance_penalty_coeff
            )

        # 5. Bankruptcy penalty
        if portfolio_value <= 0:
            reward += params.bankruptcy_penalty

        # 6. Clipping
        if params.reward_clip_value is not None:
            clip = params.reward_clip_value
            reward = max(-clip, min(clip, reward))

        return reward
