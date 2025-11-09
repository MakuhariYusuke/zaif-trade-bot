#!/usr/bin/env python3
"""
Test the new simplified reward functions in metrics.py
"""

import sys
from pathlib import Path

import pandas as pd

project_root = Path(__file__).resolve().parent
sys.path.insert(0, str(project_root))

from ztb.metrics.metrics import (
    calculate_downside_risk_reward,
    calculate_risk_adjusted_reward,
    calculate_trading_reward,
)


def test_calculate_trading_reward():
    """Test the basic trading reward calculation"""
    print("Testing calculate_trading_reward...")

    # Test case 1: Profitable trade with no penalties
    reward = calculate_trading_reward(
        pnl=1000.0,
        transaction_cost=10.0,
        position=1.0,
        old_position=0.0,
        reward_scaling=1.0,
    )
    expected = 990.0  # 1000 - 10
    assert abs(reward - expected) < 1e-6, f"Expected {expected}, got {reward}"
    print("✓ Basic profitable trade test passed")

    # Test case 2: Trade with opportunity cost penalty
    reward = calculate_trading_reward(
        pnl=100.0,
        transaction_cost=5.0,
        position=0.0,  # No position change
        old_position=0.0,
        reward_scaling=1.0,
        opportunity_cost_penalty=0.01,
        consecutive_idle_steps=10,
    )
    expected = 95.0 - 0.1  # 100 - 5 - 0.01 * 10
    assert abs(reward - expected) < 1e-6, f"Expected {expected}, got {reward}"
    print("✓ Opportunity cost penalty test passed")

    # Test case 3: Trade with stagnation penalty
    reward = calculate_trading_reward(
        pnl=200.0,
        transaction_cost=8.0,
        position=1.0,  # Has position
        old_position=1.0,
        reward_scaling=1.0,
        stagnation_penalty=0.005,
        consecutive_position_hold_steps=20,
    )
    expected = 192.0 - 0.1  # 200 - 8 - 0.005 * 20
    assert abs(reward - expected) < 1e-6, f"Expected {expected}, got {reward}"
    print("✓ Stagnation penalty test passed")

    # Test case 4: Reward scaling
    reward = calculate_trading_reward(
        pnl=100.0,
        transaction_cost=5.0,
        position=1.0,
        old_position=0.0,
        reward_scaling=2.0,
    )
    expected = 190.0  # (100 - 5) * 2
    assert abs(reward - expected) < 1e-6, f"Expected {expected}, got {reward}"
    print("✓ Reward scaling test passed")


def test_calculate_risk_adjusted_reward():
    """Test the risk-adjusted reward calculation"""
    print("\nTesting calculate_risk_adjusted_reward...")

    # Create sample returns data
    returns = pd.Series([0.01, 0.02, -0.005, 0.015, -0.01])

    # Test case 1: Positive Sharpe ratio
    reward = calculate_risk_adjusted_reward(
        returns=returns,
        current_pnl=100.0,
        transaction_cost=5.0,
        risk_free_rate=0.0,
        reward_scaling=1.0,
    )
    assert reward > 0, f"Expected positive reward, got {reward}"
    print("✓ Risk-adjusted reward with positive Sharpe test passed")

    # Test case 2: Empty returns (fallback to basic calculation)
    reward = calculate_risk_adjusted_reward(
        returns=pd.Series([]),
        current_pnl=100.0,
        transaction_cost=5.0,
        reward_scaling=1.0,
    )
    expected = 95.0
    assert abs(reward - expected) < 1e-6, f"Expected {expected}, got {reward}"
    print("✓ Risk-adjusted reward with empty returns test passed")


def test_calculate_downside_risk_reward():
    """Test the downside risk reward calculation"""
    print("\nTesting calculate_downside_risk_reward...")

    # Create sample returns with some negative values
    returns = pd.Series([0.01, 0.02, -0.005, 0.015, -0.01, -0.02])

    # Test case 1: Basic functionality
    reward = calculate_downside_risk_reward(
        returns=returns,
        current_pnl=100.0,
        transaction_cost=5.0,
        reward_scaling=1.0,
    )
    assert reward > 0, f"Expected positive reward, got {reward}"
    print("✓ Downside risk reward test passed")

    # Test case 2: Empty returns (fallback to basic calculation)
    reward = calculate_downside_risk_reward(
        returns=pd.Series([]),
        current_pnl=100.0,
        transaction_cost=5.0,
        reward_scaling=1.0,
    )
    expected = 95.0
    assert abs(reward - expected) < 1e-6, f"Expected {expected}, got {reward}"
    print("✓ Downside risk reward with empty returns test passed")


def test_reward_function_comparison():
    """Compare the new simplified functions with expected behavior"""
    print("\nTesting reward function comparison...")

    # Test parameters
    pnl = 500.0
    transaction_cost = 25.0
    position = 0.5
    old_position = 0.0

    # Calculate rewards from different functions
    basic_reward = calculate_trading_reward(
        pnl=pnl,
        transaction_cost=transaction_cost,
        position=position,
        old_position=old_position,
    )

    # Create sample returns for risk-adjusted functions
    returns = pd.Series([0.01, 0.005, -0.002, 0.008, -0.003])

    risk_adjusted_reward = calculate_risk_adjusted_reward(
        returns=returns,
        current_pnl=pnl,
        transaction_cost=transaction_cost,
    )

    downside_reward = calculate_downside_risk_reward(
        returns=returns,
        current_pnl=pnl,
        transaction_cost=transaction_cost,
    )

    # All should be positive and reasonable
    assert basic_reward > 0, f"Basic reward should be positive, got {basic_reward}"
    assert (
        risk_adjusted_reward > 0
    ), f"Risk-adjusted reward should be positive, got {risk_adjusted_reward}"
    assert (
        downside_reward > 0
    ), f"Downside risk reward should be positive, got {downside_reward}"

    print(f"✓ Basic reward: {basic_reward:.2f}")
    print(f"✓ Risk-adjusted reward: {risk_adjusted_reward:.2f}")
    print(f"✓ Downside risk reward: {downside_reward:.2f}")
    print("✓ All reward functions produce reasonable positive values")


def main():
    """Run all tests"""
    print("Running simplified reward function tests...\n")

    try:
        test_calculate_trading_reward()
        test_calculate_risk_adjusted_reward()
        test_calculate_downside_risk_reward()
        test_reward_function_comparison()

        print(
            "\n✅ All tests passed! The simplified reward functions are working correctly."
        )
        return True

    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback

        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
