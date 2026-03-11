import numpy as np

from tests.helpers import (
    make_exchange_random_walk_ohlcv_data,
    make_schema_feature_env_config,
)
from ztb.trading.environment.heavy_env.core import HeavyTradingEnv
from ztb.trading.environment.utils.config import RewardSettings


def _make_reward_env(rows: int, *, seed: int = 42) -> HeavyTradingEnv:
    dummy_df = make_exchange_random_walk_ohlcv_data(
        rows=rows,
        seed=seed,
        start="2023-01-01",
        freq="1H",
        base_price=150.0,
        return_scale=0.01,
        intrabar_scale=0.02,
        open_scale=0.01,
        volume_logmean=8.5,
        volume_logsigma=0.25,
        include_timestamp=True,
    )

    config = make_schema_feature_env_config(
        dummy_df,
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
    return HeavyTradingEnv(df=dummy_df, config=config)


def test_reward_calculation_with_behavior_optimization():
    """Test that reward function correctly applies behavior optimization penalties/bonuses."""
    env = _make_reward_env(8, seed=42)

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

        # For this test, we mainly check that the function doesn't crash and returns reasonable values
        # More specific assertions would require detailed knowledge of the reward calculation logic


def test_reward_balance_in_behavior_optimization():
    """Test that behavior optimization doesn't create extreme bias."""
    env = _make_reward_env(12, seed=43)

    rewards = []
    rng = np.random.default_rng(44)

    # Run multiple steps to collect reward statistics
    obs, info = env.reset()
    for i in range(6):
        action = rng.uniform(-1, 1)  # Random continuous action
        next_obs, reward, terminated, truncated, info = env.step(np.array([action]))
        rewards.append(reward)
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
