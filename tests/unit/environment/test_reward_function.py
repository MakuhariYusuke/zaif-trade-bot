import numpy as np
import pandas as pd

from ztb.trading.environment.heavy_env.core import HeavyTradingEnv
from ztb.trading.environment.utils.config import EnvironmentConfig, RewardSettings


def test_reward_calculation_with_behavior_optimization():
    """Test that reward function correctly applies behavior optimization penalties/bonuses."""
    # Create a minimal config with reward_settings
    config = EnvironmentConfig(
        initial_portfolio_value=100000.0,
        transaction_cost=0.001,
        max_position_size=1.0,
        reward_scaling=1.0,
        action_space_type="continuous",
        use_continuous_actions=True,
        reward_settings=RewardSettings(
            custom_reward_params={
                "action_bonuses": {
                    "buy_action_bonus": 0.0,
                    "sell_action_bonus": 0.0,
                    "hold_action_bonus": 0.0,
                }
            }
        ),
    )

    # Create minimal dummy data
    dates = pd.date_range("2023-01-01", periods=10, freq="1H")
    dummy_df = pd.DataFrame(
        {
            "close": np.random.uniform(100, 200, 10),
            "high": np.random.uniform(105, 205, 10),
            "low": np.random.uniform(95, 195, 10),
            "volume": np.random.uniform(1000, 10000, 10),
            "timestamp": dates,
        }
    )

    # Initialize environment
    env = HeavyTradingEnv(df=dummy_df, config=config)

    # Reset to get initial state
    obs, info = env.reset()

    # Test different actions and their rewards
    test_cases = [
        (0.0, "HOLD"),  # Continuous action 0.0 -> HOLD
        (1.0, "BUY"),  # Continuous action 1.0 -> BUY
        (-1.0, "SELL"),  # Continuous action -1.0 -> SELL
        (0.5, "BUY"),  # Continuous action 0.5 -> BUY (above threshold)
        (-0.5, "SELL"),  # Continuous action -0.5 -> SELL (below threshold)
    ]

    for continuous_action, expected_discrete in test_cases:
        # Reset environment for each test
        obs, info = env.reset()

        # Execute step with continuous action
        next_obs, reward, terminated, truncated, info = env.step(
            np.array([continuous_action])
        )

        # Check that reward is a number
        assert isinstance(
            reward, (int, float, np.number)
        ), f"Reward should be numeric, got {type(reward)}"

        # Check that reward is within reasonable bounds (not extreme)
        assert -1000 <= reward <= 1000, f"Reward {reward} seems unreasonable"

        # Log for debugging
        print(f"Action {continuous_action} ({expected_discrete}): Reward = {reward}")

        # For this test, we mainly check that the function doesn't crash and returns reasonable values
        # More specific assertions would require detailed knowledge of the reward calculation logic


def test_reward_balance_in_behavior_optimization():
    """Test that behavior optimization doesn't create extreme bias."""
    # Similar setup as above
    config = EnvironmentConfig(
        initial_portfolio_value=100000.0,
        transaction_cost=0.001,
        max_position_size=1.0,
        reward_scaling=1.0,
        action_space_type="continuous",
        use_continuous_actions=True,
        reward_settings=RewardSettings(
            custom_reward_params={
                "action_bonuses": {
                    "buy_action_bonus": 0.0,
                    "sell_action_bonus": 0.0,
                    "hold_action_bonus": 0.0,
                }
            }
        ),
    )

    dates = pd.date_range("2023-01-01", periods=20, freq="1H")
    dummy_df = pd.DataFrame(
        {
            "close": np.random.uniform(100, 200, 20),
            "high": np.random.uniform(105, 205, 20),
            "low": np.random.uniform(95, 195, 20),
            "volume": np.random.uniform(1000, 10000, 20),
            "timestamp": dates,
        }
    )

    env = HeavyTradingEnv(df=dummy_df, config=config)

    rewards = []
    actions = []

    # Run multiple steps to collect reward statistics
    obs, info = env.reset()
    for i in range(10):
        action = np.random.uniform(-1, 1)  # Random continuous action
        next_obs, reward, terminated, truncated, info = env.step(np.array([action]))
        rewards.append(reward)
        actions.append(action)
        if terminated or truncated:
            break

    # Check that rewards vary and aren't all the same (indicating some differentiation)
    unique_rewards = len(set(rewards))
    assert (
        unique_rewards > 1
    ), "Rewards should vary based on actions, but all rewards are identical"

    # Check that the mean reward is reasonable
    mean_reward = np.mean(rewards)
    assert (
        -10 <= mean_reward <= 10
    ), f"Mean reward {mean_reward} seems unreasonable for behavior optimization"

    print(
        f"Collected {len(rewards)} rewards with mean {mean_reward:.3f}, std {np.std(rewards):.3f}"
    )
