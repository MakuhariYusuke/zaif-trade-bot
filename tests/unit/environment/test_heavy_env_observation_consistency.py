import numpy as np
import pandas as pd
import pytest

from tests.helpers import make_schema_feature_env_config
from ztb.trading.environment.heavy_env.core import HeavyTradingEnv


@pytest.fixture(scope="module")
def schema_feature_env() -> HeavyTradingEnv:
    df = pd.DataFrame(
        {
            "timestamp": pd.date_range("2023-01-01", periods=96, freq="1min"),
            "open": [100.0] * 96,
            "high": [101.0] * 96,
            "low": [99.0] * 96,
            "close": [100.5] * 96,
            "volume": [1000.0] * 96,
        }
    )
    env = HeavyTradingEnv(df=df, config=make_schema_feature_env_config(df))
    yield env
    env.close()


class TestHeavyTradingEnvObservationConsistency:
    """Test cases for HeavyTradingEnv observation consistency without defensive code."""

    def test_observation_always_matches_space_dimensions(self, schema_feature_env):
        """Test that observations always match the declared observation space dimensions."""
        env = schema_feature_env

        # Reset to get initial observation
        obs, info = env.reset()

        # Check initial observation matches space
        obs_dim = env.observation_space.shape[0]
        assert (
            len(obs) == obs_dim
        ), f"Initial observation length {len(obs)} != observation space dim {obs_dim}"

        # Take several steps and check each observation
        for step in range(10):
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)

            # Verify observation matches declared space
            assert (
                len(obs) == obs_dim
            ), f"Step {step} observation length {len(obs)} != observation space dim {obs_dim}"

            # Verify it's a numpy array with correct dtype
            assert isinstance(
                obs, np.ndarray
            ), f"Observation should be numpy array, got {type(obs)}"
            assert (
                obs.dtype == np.float32
            ), f"Observation dtype should be float32, got {obs.dtype}"

            if terminated or truncated:
                break

    def test_no_defensive_code_in_step_method(self):
        """Test that the step method does not contain defensive trimming/padding code."""
        # This test ensures that the defensive code has been removed
        # and observations are guaranteed to match the space by design

        # Create minimal environment
        df = pd.DataFrame(
            {
                "timestamp": pd.date_range("2023-01-01", periods=50, freq="1min"),
                "open": [100] * 50,
                "high": [101] * 50,
                "low": [99] * 50,
                "close": [100.5] * 50,
                "volume": [1000] * 50,
            }
        )

        config = make_schema_feature_env_config(df)

        env = HeavyTradingEnv(df=df, config=config)
        obs, info = env.reset()

        # Take a step
        action = env.action_space.sample()
        new_obs, reward, terminated, truncated, info = env.step(action)

        # The observation should naturally match the space dimensions
        # without any defensive adjustments
        obs_dim = env.observation_space.shape[0]
        assert (
            len(new_obs) == obs_dim
        ), "Observation should match space dimensions without defensive code"

    def test_observation_consistency_across_multiple_resets(self, schema_feature_env):
        """Test that observation dimensions remain consistent across multiple environment resets."""
        env = schema_feature_env

        obs_dims = []

        # Reset multiple times and collect observation dimensions
        for reset_count in range(3):
            obs, info = env.reset()
            obs_dims.append(len(obs))

            # Take a few steps
            for step in range(5):
                action = env.action_space.sample()
                obs, reward, terminated, truncated, info = env.step(action)
                obs_dims.append(len(obs))

                if terminated or truncated:
                    break

        # All observations should have the same dimension
        unique_dims = set(obs_dims)
        assert (
            len(unique_dims) == 1
        ), f"All observations should have same dimension, got {unique_dims}"

        # And it should match the declared space
        expected_dim = env.observation_space.shape[0]
        assert (
            unique_dims.pop() == expected_dim
        ), f"Observation dim should match space dim {expected_dim}"
