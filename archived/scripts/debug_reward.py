#!/usr/bin/env python3
"""
Test script for reward calculator debugging.
"""

import logging
import sys
from pathlib import Path

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

# Enable debug logging
logging.basicConfig(level=logging.DEBUG)

from typing import cast

from ztb.trading.environment.components.reward_calculator import RewardCalculator
from ztb.trading.environment.utils.config import RewardSettings


def test_reward_scenarios():
    """Test reward calculation with various scenarios."""
    print("=== Reward Function Debug Test ===\n")

    # Current SAC v399 balanced settings
    reward_settings_current = {
        "use_simple_reward": True,
        "reward_scale": 1000.0,
        "reward_clip_min": -10.0,
        "reward_clip_max": 10.0,
        "enable_inactivity_penalty": True,
        "inactivity_penalty_rate": 0.001,
        "enable_opportunity_cost": True,
        "opportunity_cost_rate": 0.001,
        "enable_trade_execution_bonus": True,
        "trade_execution_bonus_rate": 0.05,
    }

    # Proposed aggressive settings for 80%+ win rate target
    reward_settings_improved = {
        "use_simple_reward": True,
        "reward_scale": 8000.0,  # Doubled from 4000.0 for extreme sensitivity
        "reward_clip_min": -80.0,  # Doubled from -40.0 for aggressive rewards
        "reward_clip_max": 80.0,  # Doubled from 40.0 for aggressive rewards
        "enable_inactivity_penalty": True,
        "inactivity_penalty_rate": 0.0005,
        "enable_opportunity_cost": True,
        "opportunity_cost_rate": 0.0005,
        "enable_trade_execution_bonus": True,
        "trade_execution_bonus_rate": 0.1,
        "buy_action_penalty": 0.0,  # Neutral penalty for BUY actions
        "sell_action_penalty": 0.0,  # Neutral penalty for SELL actions
        "action_threshold_buy": 0.05,  # Adjusted for SELL bias correction
        "action_threshold_sell": -0.3,  # Adjusted for SELL bias correction
    }

    # Create environment config mock
    class MockConfig:
        def __init__(self):
            self.curriculum_stage = "default"
            self.max_position_size = 0.01
            self.initial_balance = 200000.0

    config = MockConfig()

    print("Comparing Current vs Improved Reward Settings:")
    print("=" * 80)
    print(
        f"Current Settings:  scale={reward_settings_current['reward_scale']}, clip=[{reward_settings_current['reward_clip_min']}, {reward_settings_current['reward_clip_max']}]"
    )
    print(
        f"Improved Settings: scale={reward_settings_improved['reward_scale']}, clip=[{reward_settings_improved['reward_clip_min']}, {reward_settings_improved['reward_clip_max']}]"
    )
    print()

    # Test scenarios
    test_cases = [
        {
            "name": "Small Profit Trade",
            "pnl": 1000.0,  # 0.5% profit
            "position": 0.005,
            "old_position": 0.0,
            "action": 1,  # BUY
            "description": "Small profitable trade",
        },
        {
            "name": "Small Loss Trade",
            "pnl": -1000.0,  # 0.5% loss
            "position": -0.005,
            "old_position": 0.0,
            "action": 2,  # SELL
            "description": "Small losing trade",
        },
        {
            "name": "HOLD without Position",
            "pnl": 0.0,  # No change
            "position": 0.0,
            "old_position": 0.0,
            "action": 0,  # HOLD
            "description": "Holding no position (idle)",
        },
        {
            "name": "Large Profit Trade",
            "pnl": 5000.0,  # 2.5% profit
            "position": 0.01,
            "old_position": 0.0,
            "action": 1,  # BUY
            "description": "Large profitable trade",
        },
        {
            "name": "Continuous BUY Action (>0.2)",
            "pnl": 2000.0,  # 1% profit
            "position": 0.005,
            "old_position": 0.0,
            "action": 0.8,  # Continuous BUY signal
            "description": "Continuous action > 0.2 should map to BUY",
        },
        {
            "name": "Continuous SELL Action (<-0.2)",
            "pnl": -2000.0,  # 1% loss
            "position": -0.005,
            "old_position": 0.0,
            "action": -0.8,  # Continuous SELL signal
            "description": "Continuous action < -0.2 should map to SELL",
        },
        {
            "name": "Continuous HOLD Action (-0.2 to 0.2)",
            "pnl": 0.0,  # No change
            "position": 0.0,
            "old_position": 0.0,
            "action": 0.1,  # Continuous HOLD signal
            "description": "Continuous action between -0.2 and 0.2 should map to HOLD",
        },
    ]

    print("Test Results Comparison:")
    print("-" * 120)

    for test_case in test_cases:
        # Current settings
        calculator_current = RewardCalculator(
            config=config,
            reward_settings=cast(RewardSettings, reward_settings_current),
            initial_portfolio_value=200000.0,
        )

        reward_current = calculator_current.calculate_reward_simple(
            pnl=test_case["pnl"],
            portfolio_value=config.initial_balance,
            position=test_case["position"],
            old_position=test_case["old_position"],
            action=test_case["action"],
        )

        # Improved settings
        calculator_improved = RewardCalculator(
            config=config,
            reward_settings=cast(RewardSettings, reward_settings_improved),
            initial_portfolio_value=200000.0,
        )

        reward_improved = calculator_improved.calculate_reward_simple(
            pnl=test_case["pnl"],
            portfolio_value=config.initial_balance,
            position=test_case["position"],
            old_position=test_case["old_position"],
            action=test_case["action"],
        )

        pnl_ratio = test_case["pnl"] / config.initial_balance

        print(f"{test_case['name']}")
        print(f"   PnL: {test_case['pnl']:.1f} JPY ({pnl_ratio*100:.2f}%)")
        print(f"   Action Input: {test_case['action']}")
        print(f"   Current Reward: {reward_current:+.4f}")
        print(f"   Improved Reward: {reward_improved:+.4f}")
        print(f"   Improvement: {reward_improved - reward_current:+.4f}")
        print()


def main():
    test_reward_scenarios()


if __name__ == "__main__":
    main()
