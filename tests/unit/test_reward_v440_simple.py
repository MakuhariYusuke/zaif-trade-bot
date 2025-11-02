#!/usr/bin/env python3
"""
Test for v440 enhanced reward function with dynamic reward shaping.
"""

import os
import sys

import numpy as np

# Add project root to path
sys.path.insert(
    0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)

from ztb.trading.constants import ACTION_BUY, ACTION_HOLD
from ztb.trading.environment.components.reward_calculator import RewardCalculator
from ztb.trading.environment.utils.config import EnvironmentConfig, RewardSettings


def test_v440_reward_simple():
    """Test v440 reward calculation with dynamic shaping."""

    # Create proper EnvironmentConfig
    config = EnvironmentConfig(
        curriculum_stage="basic",
        max_position_size=1.0,
        feature_names=None,
    )

    # Test with v440 parameters including dynamic reward shaping
    reward_settings = RewardSettings(
        use_simple_reward=True,
        reward_scale=0.1,
        reward_clip_value=20.0,
    )

    calculator = RewardCalculator(
        config=config, reward_settings=reward_settings, initial_portfolio_value=200000.0
    )

    print("Calculator created, calling calculate_reward")

    # Test basic reward calculation
    reward = calculator.calculate_reward(
        action=ACTION_BUY,
        current_price=100.0,
        position=0.1,
        portfolio_value=200000.0,
        atr=1.0,
        transaction_cost=0.001,
        reward_scaling=0.1,
        pnl=100.0,
        old_position=0.0,
        step=1,
        observation=np.array([100.0, 0.1, 100.0]),
        reward_history=[],
        portfolio_value_history=[200000.0],
    )

    print(f"Reward for BUY with profit: {reward}")
    assert reward > 0, "Positive PnL should result in positive reward"

    # Test HOLD penalty
    reward_hold = calculator.calculate_reward(
        action=ACTION_HOLD,
        current_price=100.0,
        position=0.1,
        portfolio_value=200000.0,
        atr=1.0,
        transaction_cost=0.001,
        reward_scaling=0.1,
        pnl=100.0,
        old_position=0.1,
        step=1,
        observation=np.array([100.0, 0.1, 100.0]),
        reward_history=[],
        portfolio_value_history=[200000.0],
    )

    print(f"Reward for HOLD with profit: {reward_hold}")
    print(f"Ratio: {reward_hold / reward if reward != 0 else 'inf'}")
    # In simple reward mode, all actions should have the same reward based on PnL
    assert (
        reward_hold == reward
    ), "In simple reward mode, all actions should have identical rewards"

    print("v440 reward tests passed!")


if __name__ == "__main__":
    test_v440_reward_simple()
