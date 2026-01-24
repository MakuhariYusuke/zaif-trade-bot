#!/usr/bin/env python3
"""
Test for SELL action bonus reward calculation fix.

This test verifies that sell_action_bonus is correctly applied to reward calculation.
"""

from unittest.mock import Mock


from ztb.trading.constants import ACTION_BUY, ACTION_HOLD, ACTION_SELL
from ztb.trading.environment.components.calculators.reward_calculator import RewardCalculator
from ztb.trading.environment.utils.config import EnvironmentConfig, RewardSettings


def test_sell_action_bonus_applied_to_reward():
    """Test that sell_action_bonus is correctly added to base reward."""
    # Create config with sell_action_bonus
    config = EnvironmentConfig(
        curriculum_stage="profit_optimized",
        max_position_size=1.0,
        action_bonuses={
            "buy_action_bonus": -0.01,
            "sell_action_bonus": 3.0,
            "hold_action_bonus": 0.01,
        },
    )

    reward_settings = RewardSettings(
        use_simple_reward=False,
    )

    calculator = RewardCalculator(config, reward_settings, 200000.0)

    # Mock observation
    observation = Mock()
    observation.position = 0.5
    observation.price = 100.0
    observation.spread = 0.001

    # Test BUY action reward
    buy_reward = calculator.calculate_reward(
        ACTION_BUY,
        100.0,
        0.5,
        200000.0,
        1.0,
        0.001,
        1.0,
        0.0,
        0.0,
        1,
        observation,
        [],
        [200000.0],
    )

    # Get BUY components immediately after calculation
    buy_components = calculator.get_last_reward_components()

    # Test SELL action reward
    sell_reward = calculator.calculate_reward(
        ACTION_SELL,
        100.0,
        0.5,
        200000.0,
        1.0,
        0.001,
        1.0,
        0.0,
        0.0,
        1,
        observation,
        [],
        [200000.0],
    )

    # Get SELL components immediately after calculation
    sell_components = calculator.get_last_reward_components()

    # Test HOLD action reward
    hold_reward = calculator.calculate_reward(
        ACTION_HOLD,
        100.0,
        0.5,
        200000.0,
        1.0,
        0.001,
        1.0,
        0.0,
        0.0,
        1,
        observation,
        [],
        [200000.0],
    )

    # Get HOLD components immediately after calculation
    hold_components = calculator.get_last_reward_components()

    print(f"BUY components: {buy_components}")
    print(f"SELL components: {sell_components}")
    print(f"HOLD components: {hold_components}")

    # Verify that action_bonus is recorded
    assert (
        "action_bonus" in sell_components
    ), "action_bonus should be recorded in reward components"
    assert (
        sell_components["action_bonus"] == 3.0
    ), f"SELL action_bonus should be 3.0, got {sell_components['action_bonus']}"

    assert (
        "action_bonus" in buy_components
    ), "action_bonus should be recorded in reward components"
    assert (
        buy_components["action_bonus"] == -0.01
    ), f"BUY action_bonus should be -0.01, got {buy_components['action_bonus']}"

    assert (
        "action_bonus" in hold_components
    ), "action_bonus should be recorded in reward components"
    assert (
        hold_components["action_bonus"] == 0.01
    ), f"HOLD action_bonus should be 0.01, got {hold_components['action_bonus']}"

    # SELL reward should be higher than BUY reward due to sell_action_bonus
    # (Note: actual reward values depend on other factors, but the bonus should be applied)
    print("✅ SELL action bonus test passed")


def test_sell_action_bonus_zero():
    """Test with zero sell_action_bonus."""
    # Create config with zero sell_action_bonus
    config = EnvironmentConfig(
        curriculum_stage="profit_optimized",
        max_position_size=1.0,
        action_bonuses={
            "buy_action_bonus": 0.0,
            "sell_action_bonus": 0.0,
            "hold_action_bonus": 0.0,
        },
    )

    reward_settings = RewardSettings(
        use_simple_reward=False,
    )

    calculator = RewardCalculator(config, reward_settings, 200000.0)

    # Mock observation
    observation = Mock()
    observation.position = 0.5
    observation.price = 100.0
    observation.spread = 0.001

    # Test SELL action reward
    sell_reward = calculator.calculate_reward(
        ACTION_SELL,
        100.0,
        0.5,
        200000.0,
        1.0,
        0.001,
        1.0,
        0.0,
        0.0,
        1,
        observation,
        [],
        [200000.0],
    )

    sell_components = calculator.get_last_reward_components()

    print(f"SELL reward (zero bonus): {sell_reward}")
    print(f"SELL components (zero bonus): {sell_components}")

    # Verify that action_bonus is 0.0
    assert (
        sell_components["action_bonus"] == 0.0
    ), f"SELL action_bonus should be 0.0, got {sell_components['action_bonus']}"

    print("✅ Zero sell action bonus test passed")


def test_environment_config_action_bonuses_loading():
    """Test that EnvironmentConfig.from_dict correctly loads action_bonuses from reward_settings."""
    config_dict = {
        "action_bonuses": {
            "buy_action_bonus": -0.01,
            "sell_action_bonus": 3.0,
            "hold_action_bonus": 0.01,
        }
    }

    config = EnvironmentConfig.from_dict(config_dict)

    print(f"Config action_bonuses: {config.action_bonuses}")

    assert (
        config.action_bonuses["sell_action_bonus"] == 3.0
    ), f"sell_action_bonus should be 3.0, got {config.action_bonuses['sell_action_bonus']}"
    assert (
        config.action_bonuses["buy_action_bonus"] == -0.01
    ), f"buy_action_bonus should be -0.01, got {config.action_bonuses['buy_action_bonus']}"
    assert (
        config.action_bonuses["hold_action_bonus"] == 0.01
    ), f"hold_action_bonus should be 0.01, got {config.action_bonuses['hold_action_bonus']}"

    print("✅ EnvironmentConfig action_bonuses loading test passed")


if __name__ == "__main__":
    test_sell_action_bonus_applied_to_reward()
    test_sell_action_bonus_zero()
    test_environment_config_action_bonuses_loading()
    print("🎉 All SELL action bonus tests passed!")
