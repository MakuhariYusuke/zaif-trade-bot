#!/usr/bin/env python3
"""
Test for v440 reward function enhancements with dynamic reward shaping.
"""

import os
import sys

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from ztb.trading.constants import ACTION_BUY, ACTION_HOLD, ACTION_SELL
from ztb.trading.environment.components.reward_calculator import RewardCalculator


def test_v440_dynamic_reward_shaping():
    """Test v440 reward calculation with dynamic shaping enabled."""
    
    # Mock config
    class MockConfig:
        def __init__(self):
            self.curriculum_stage = "basic"
            self.max_position_size = 1.0
            self.signal_guidance_enabled = False
            self.signal_guidance = {}
            self.feature_names = None

    config = MockConfig()

    # Test with v440 parameters including dynamic reward shaping
    reward_settings = {
        "use_simple_reward": True,
        "hold_penalty_multiplier": 0.5,  # Changed from 2.0 to 0.5 for penalty
        "trade_frequency_bonus": 0.001,
        "reward_scaling": 0.1,
        "reward_clip_value": 20.0,
        "dynamic_reward_shaping": {
            "enabled": True,
            "market_regime_awareness": True,
            "volatility_adjusted_rewards": True,
            "trend_strength_bonus": True,
            "regime_detection_window": 20,
            "adaptation_frequency": 10,
            "regime_coefficients": {
                "bull_market_bonus_coeff": 1.2,
                "bear_market_penalty_coeff": 0.8,
                "sideways_market_penalty_coeff": 0.9,
                "volatile_market_bonus_coeff": 1.1,
            },
            "volatility_coefficients": {
                "high_volatility_threshold": 0.02,
                "low_volatility_threshold": 0.005,
                "high_volatility_bonus": 1.3,
                "low_volatility_penalty": 0.7,
            },
            "trend_coefficients": {
                "trend_strength_threshold": 0.001,
                "strong_trend_bonus": 1.2,
                "weak_trend_penalty": 0.9,
            },
        },
        "long_position_reward_multiplier": 1.3,
        "short_position_reward_multiplier": 0.7,
        "long_position_penalty_multiplier": 0.9,
        "short_position_penalty_multiplier": 0.95,
    }

    calculator = RewardCalculator(
        config=config, reward_settings=reward_settings, initial_portfolio_value=200000.0
    )

    print("Calculator created, calling calculate_reward_simple")

    # Test basic reward calculation
    reward = calculator.calculate_reward_simple(
        pnl=100.0,
        portfolio_value=200000.0,
        position=0.1,
        old_position=0.0,
        action=ACTION_BUY,
        reward_history=[],
        portfolio_value_history=[200000.0],
        current_price=100.0,
        step=10
    )
    print(f"Reward for BUY with profit: {reward}")
    assert reward > 0, "Positive PnL should result in positive reward"

    # Test HOLD penalty
    reward_hold = calculator.calculate_reward_simple(
        pnl=100.0,
        portfolio_value=200000.0,
        position=0.1,
        old_position=0.1,
        action=ACTION_HOLD,
        reward_history=[],
        portfolio_value_history=[200000.0],
        current_price=100.0,
        step=10
    )
    print(f"Reward for HOLD with profit: {reward_hold}")
    print(f"Ratio: {reward_hold / reward if reward != 0 else 'inf'}")
    assert reward_hold < reward, "HOLD should have penalty applied"

    print("v440 dynamic reward shaping tests passed!")


if __name__ == "__main__":
    test_v440_dynamic_reward_shaping()
