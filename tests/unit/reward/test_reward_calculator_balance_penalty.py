"""Test for RewardCalculator balance penalty mechanism."""

from unittest.mock import Mock

from ztb.trading.constants import ACTION_BUY, ACTION_HOLD, ACTION_SELL
from ztb.trading.environment.components.reward_calculator import RewardCalculator
from ztb.trading.environment.utils.config import EnvironmentConfig, RewardSettings
from ztb.trading.environment.components.behavioral_penalty_calculator import BehavioralPenaltyCalculator


def test_reward_calculator_balance_penalty():
    """Test that RewardCalculator applies balance penalty correctly."""
    # Create config with forced_balance curriculum to enable balance penalty
    config = EnvironmentConfig(
        curriculum_stage="forced_balance",
        max_position_size=1.0,
        behavior_optimization={
            "balance_penalty": 1000.0,  # High penalty for testing
            "action_balance_target": 0.333,
        }
    )

    reward_settings = RewardSettings(
        balance_penalty=1000.0,
        use_simple_reward=False,
    )

    calculator = RewardCalculator(config, reward_settings, 200000.0)

    # Extend the behavioral lookback so the forced balance considers the recent history
    config.reward_settings = {"skewness_lookback": 50, "action_entropy_lookback": 50}
    calculator.behavioral_penalty_calculator = BehavioralPenaltyCalculator(config)

    # Mock observation
    observation = Mock()
    observation.position = 0.5
    observation.price = 100.0
    observation.spread = 0.001

    # Populate the behavioral penalty calculator (sliding-window counts) with imbalanced SELL actions
    for _ in range(20):
        calculator.behavioral_penalty_calculator.record_action(ACTION_SELL)

    # Test BUY action - should get high penalty due to imbalance
    buy_reward = calculator.calculate_reward(
        ACTION_BUY, 100.0, 0.5, 200000.0, 1.0, 0.001, 1.0, 0.0, 0.0, 1,
        observation, [], [200000.0]
    )
    print(f"BUY reward with imbalanced actions: {buy_reward}")

    # Test SELL action - should get penalty but less than BUY due to asymmetry
    sell_reward = calculator.calculate_reward(
        ACTION_SELL, 100.0, 0.5, 200000.0, 1.0, 0.001, 1.0, 0.0, 0.0, 1,
        observation, [], [200000.0]
    )
    print(f"SELL reward with imbalanced actions: {sell_reward}")

    # Test HOLD action - should get penalty
    hold_reward = calculator.calculate_reward(
        ACTION_HOLD, 100.0, 0.5, 200000.0, 1.0, 0.001, 1.0, 0.0, 0.0, 1,
        observation, [], [200000.0]
    )
    print(f"HOLD reward with imbalanced actions: {hold_reward}")

    # The imbalance should cause at least one strongly negative reward (SELL gets penalized due to sell-biased history)
    assert sell_reward < 0, f"SELL reward should be negative, got {sell_reward}"

    # BUY should have higher reward than SELL and HOLD due to asymmetric targets
    # (BUY target 0.4 > SELL target 0.25, so BUY gets less penalty)
    assert buy_reward > sell_reward, f"BUY reward {buy_reward} should be higher than SELL reward {sell_reward}"
    assert buy_reward > hold_reward, f"BUY reward {buy_reward} should be higher than HOLD reward {hold_reward}"


def test_reward_calculator_zero_balance_penalty():
    """Test RewardCalculator with zero balance penalty."""
    # Create config with forced_balance but zero penalty
    config = EnvironmentConfig(
        curriculum_stage="forced_balance",
        max_position_size=1.0,
        behavior_optimization={
            "balance_penalty": 0.0,  # Zero penalty
            "action_balance_target": 0.333,
        }
    )

    reward_settings = RewardSettings(
        balance_penalty=0.0,
        use_simple_reward=False,
    )

    calculator = RewardCalculator(config, reward_settings, 200000.0)

    # Mock observation
    observation = Mock()
    observation.position = 0.5
    observation.price = 100.0
    observation.spread = 0.001

    # Populate recent actions with imbalanced SELL actions
    for _ in range(20):
        calculator._recent_actions.append(ACTION_SELL)

    # Test actions - should not get balance penalty
    buy_reward = calculator.calculate_reward(
        ACTION_BUY, 100.0, 0.5, 200000.0, 1.0, 0.001, 1.0, 0.0, 0.0, 1,
        observation, [], [200000.0]
    )
    sell_reward = calculator.calculate_reward(
        ACTION_SELL, 100.0, 0.5, 200000.0, 1.0, 0.001, 1.0, 0.0, 0.0, 1,
        observation, [], [200000.0]
    )
    hold_reward = calculator.calculate_reward(
        ACTION_HOLD, 100.0, 0.5, 200000.0, 1.0, 0.001, 1.0, 0.0, 0.0, 1,
        observation, [], [200000.0]
    )

    print(f"BUY reward with zero penalty: {buy_reward}")
    print(f"SELL reward with zero penalty: {sell_reward}")
    print(f"HOLD reward with zero penalty: {hold_reward}")

    # Rewards should be close to zero (no pnl, some bonuses/penalties but no balance penalty)
    # Exact values depend on other components, but should not be highly negative
    assert buy_reward > -1.0, f"BUY reward should not be highly negative, got {buy_reward}"
    assert sell_reward > -1.0, f"SELL reward should not be highly negative, got {sell_reward}"
    assert hold_reward > -1.0, f"HOLD reward should not be highly negative, got {hold_reward}"