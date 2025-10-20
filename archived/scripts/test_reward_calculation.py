#!/usr/bin/env python3
"""
Test script to debug reward calculation issues.
"""

import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ztb.trading.constants import ACTION_BUY, ACTION_HOLD, ACTION_SELL
from ztb.trading.environment.components.reward_calculator import RewardCalculator
from ztb.trading.environment.utils.config import RewardSettings


def test_reward_calculation():
    """Test reward calculation with different parameters."""

    # Mock config
    class MockConfig:
        def __init__(self):
            self.curriculum_stage = "ultra_profit"
            self.max_position_size = 1.0

    config = MockConfig()

    # Test with extreme aggressive parameters (current settings)
    reward_settings = RewardSettings(
        profit_weight=5.0,
        risk_weight=0.05,
        consistency_weight=0.2,
        ultra_profit_multiplier=2.0,
        ultra_risk_multiplier=0.5,
        reward_scale=1.0,  # Fixed scaling
        reward_clip_min=-10.0,
        reward_clip_max=10.0,
        use_simple_reward=False,
        # Add other required settings with defaults
        profit_multiplier=0.01,
        loss_penalty_multiplier=0.01,
        hold_penalty_rate=0.01,
        balance_penalty_tolerance=0.15,
        balance_penalty=1.0,
        trading_bonus_multiplier=2.0,
        trading_bonus=0.01,
    )

    calculator = RewardCalculator(
        config=config, reward_settings=reward_settings, initial_portfolio_value=200000.0
    )

    # Test scenarios
    test_cases = [
        {
            "name": "Profitable BUY",
            "action": ACTION_BUY,
            "pnl": 1000.0,
            "position": 0.5,
            "current_price": 5000000.0,
            "atr": 50000.0,
        },
        {
            "name": "Loss SELL",
            "action": ACTION_SELL,
            "pnl": -500.0,
            "position": -0.3,
            "current_price": 4800000.0,
            "atr": 45000.0,
        },
        {
            "name": "HOLD with position",
            "action": ACTION_HOLD,
            "pnl": 0.0,
            "position": 0.8,
            "current_price": 4900000.0,
            "atr": 40000.0,
        },
    ]

    print("Testing reward calculation with extreme aggressive parameters:")
    print("=" * 60)

    for test_case in test_cases:
        reward = calculator._calculate_ultra_profit_reward(
            action=test_case["action"],
            atr_normalised=test_case["pnl"] / test_case["atr"],
            portfolio_return=test_case["pnl"] / 200000.0,
            position=test_case["position"],
            effective_max_position=1.0,
            current_price=test_case["current_price"],
            atr=test_case["atr"],
            pnl=test_case["pnl"],
            reward_scaling=1.0,  # Fixed scaling
        )

        print(f"{test_case['name']}:")
        print(f"  Action: {test_case['action']} (0=HOLD, 1=BUY, 2=SELL)")
        print(f"  PnL: {test_case['pnl']:.2f}")
        print(f"  Position: {test_case['position']:.2f}")
        print(f"  Reward: {reward:.4f}")
        print()


if __name__ == "__main__":
    test_reward_calculation()
