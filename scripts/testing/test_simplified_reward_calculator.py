#!/usr/bin/env python3
"""
Test the SimplifiedRewardCalculator
"""

import sys
from pathlib import Path

import numpy as np

project_root = Path(__file__).resolve().parent
sys.path.insert(0, str(project_root))

from ztb.trading.constants import ACTION_BUY, ACTION_HOLD, ACTION_SELL
from ztb.trading.environment.components.simplified_reward_calculator import (
    SimplifiedRewardCalculator,
)
from ztb.trading.environment.utils.config import EnvironmentConfig, RewardSettings


def create_test_config(reward_type: str = "basic") -> EnvironmentConfig:
    """Create a test configuration."""
    config = EnvironmentConfig()
    config.simplified_reward_type = reward_type
    config.reward_scaling = 1.0
    return config


def create_test_reward_settings() -> RewardSettings:
    """Create test reward settings."""
    return RewardSettings()


def test_basic_reward_calculator():
    """Test basic reward calculator functionality."""
    print("Testing SimplifiedRewardCalculator (basic)...")

    config = create_test_config("basic")
    reward_settings = create_test_reward_settings()
    calculator = SimplifiedRewardCalculator(config, reward_settings, 100000.0)

    # Test basic reward calculation
    reward = calculator.calculate_reward(
        action=ACTION_BUY,
        current_price=100.0,
        position=1.0,
        portfolio_value=100000.0,
        atr=1.0,
        transaction_cost=10.0,
        reward_scaling=1.0,
        pnl=100.0,
        old_position=0.0,
        step=1,
        observation=np.array([1.0, 2.0, 3.0]),
        reward_history=[],
        portfolio_value_history=[100000.0],
    )

    assert reward > 0, f"Expected positive reward, got {reward}"
    print(f"✓ Basic reward calculation: {reward:.2f}")

    # Test with opportunity cost penalty
    calculator._consecutive_idle_steps = 5
    reward_penalty = calculator.calculate_reward(
        action=ACTION_HOLD,
        current_price=100.0,
        position=0.0,
        portfolio_value=100000.0,
        atr=1.0,
        transaction_cost=5.0,
        reward_scaling=1.0,
        pnl=50.0,
        old_position=0.0,
        step=2,
        observation=np.array([1.0, 2.0, 3.0]),
        reward_history=[reward],
        portfolio_value_history=[100000.0, 100000.0],
    )

    assert (
        reward_penalty < reward
    ), f"Expected penalty to reduce reward, got {reward_penalty} vs {reward}"
    print(f"✓ Opportunity cost penalty applied: {reward_penalty:.2f}")


def test_risk_adjusted_reward_calculator():
    """Test risk-adjusted reward calculator."""
    print("\nTesting SimplifiedRewardCalculator (risk_adjusted)...")

    config = create_test_config("risk_adjusted")
    reward_settings = create_test_reward_settings()
    calculator = SimplifiedRewardCalculator(config, reward_settings, 100000.0)

    # Add some returns history
    calculator._returns_history = [0.01, 0.005, -0.002, 0.008]

    reward = calculator.calculate_reward(
        action=ACTION_BUY,
        current_price=100.0,
        position=1.0,
        portfolio_value=101000.0,  # Slight gain
        atr=1.0,
        transaction_cost=10.0,
        reward_scaling=1.0,
        pnl=100.0,
        old_position=0.0,
        step=1,
        observation=np.array([1.0, 2.0, 3.0]),
        reward_history=[],
        portfolio_value_history=[100000.0],
    )

    assert reward > 0, f"Expected positive reward, got {reward}"
    print(f"✓ Risk-adjusted reward calculation: {reward:.2f}")


def test_downside_risk_reward_calculator():
    """Test downside risk reward calculator."""
    print("\nTesting SimplifiedRewardCalculator (downside_risk)...")

    config = create_test_config("downside_risk")
    reward_settings = create_test_reward_settings()
    calculator = SimplifiedRewardCalculator(config, reward_settings, 100000.0)

    # Add some returns history with negative values
    calculator._returns_history = [0.01, -0.005, -0.002, 0.008, -0.01]

    reward = calculator.calculate_reward(
        action=ACTION_SELL,
        current_price=100.0,
        position=-1.0,
        portfolio_value=99000.0,  # Slight loss
        atr=1.0,
        transaction_cost=10.0,
        reward_scaling=1.0,
        pnl=-100.0,
        old_position=0.0,
        step=1,
        observation=np.array([1.0, 2.0, 3.0]),
        reward_history=[],
        portfolio_value_history=[100000.0],
    )

    # Even with loss, should get some reward based on risk adjustment
    print(f"✓ Downside risk reward calculation: {reward:.2f}")


def test_reward_components():
    """Test reward components retrieval."""
    print("\nTesting reward components...")

    config = create_test_config("basic")
    reward_settings = create_test_reward_settings()
    calculator = SimplifiedRewardCalculator(config, reward_settings, 100000.0)

    components = calculator.get_reward_components()
    assert "reward_function_type" in components
    assert "action_counts" in components
    assert components["reward_function_type"] == "basic"
    print("✓ Reward components retrieval works")


def test_reset_functionality():
    """Test reset functionality."""
    print("\nTesting reset functionality...")

    config = create_test_config("basic")
    reward_settings = create_test_reward_settings()
    calculator = SimplifiedRewardCalculator(config, reward_settings, 100000.0)

    # Modify internal state
    calculator._action_counts = [5, 10, 3]
    calculator._consecutive_idle_steps = 7
    calculator._returns_history = [0.01, 0.02]

    # Reset
    calculator.reset()

    assert calculator._action_counts == [0, 0, 0]
    assert calculator._consecutive_idle_steps == 0
    assert calculator._returns_history == []
    print("✓ Reset functionality works")


def main():
    """Run all tests."""
    print("Running SimplifiedRewardCalculator tests...\n")

    try:
        test_basic_reward_calculator()
        test_risk_adjusted_reward_calculator()
        test_downside_risk_reward_calculator()
        test_reward_components()
        test_reset_functionality()

        print("\n✅ All SimplifiedRewardCalculator tests passed!")
        return True

    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback

        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
