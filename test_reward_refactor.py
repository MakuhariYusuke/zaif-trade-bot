#!/usr/bin/env python3
"""Test script for the refactored reward calculator."""

import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

from ztb.trading.environment.components.reward_calculator import RewardCalculator

def test_reward_calculator():
    """Test the refactored reward calculator."""
    print("Testing refactored reward calculator...")

    # Create calculator instance
    calculator = RewardCalculator(
        config=None,
        reward_settings={
            'reward_scale': 1000.0,
            'reward_clip_min': -40.0,
            'reward_clip_max': 40.0,
            'action_threshold_buy': 0.2,
            'action_threshold_sell': -0.2
        },
        initial_portfolio_value=200000.0
    )

    # Test cases
    test_cases = [
        {
            "pnl": 100.0,
            "position": 0.01,
            "old_position": 0.0,
            "action": 0,  # HOLD
            "description": "HOLD with small profit"
        },
        {
            "pnl": 200.0,
            "position": 0.02,
            "old_position": 0.01,
            "action": 1,  # BUY
            "description": "BUY with profit"
        },
        {
            "pnl": -100.0,
            "position": 0.0,
            "old_position": 0.02,
            "action": 2,  # SELL
            "description": "SELL with loss"
        }
    ]

    print("Running test cases...")
    for i, case in enumerate(test_cases):
        reward = calculator.calculate_reward_simple(
            pnl=case["pnl"],
            portfolio_value=200000.0,
            position=case["position"],
            old_position=case["old_position"],
            action=case["action"],
            portfolio_value_history=[200000.0] * 30
        )
        print(f"Test {i+1}: {case['description']} -> Reward: {reward:.4f}")

    print("Test completed successfully!")

if __name__ == "__main__":
    test_reward_calculator()