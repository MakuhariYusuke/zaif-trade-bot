"""
Unit tests for ztb.trading.environment module.
"""

import numpy as np
import pandas as pd

from ztb.trading.environment import HeavyTradingEnv


class TestHeavyTradingEnv:
    """Test HeavyTradingEnv functionality."""

    def test_initialization(self):
        """Test environment initialization."""
        # Create sample data
        dates = pd.date_range("2023-01-01", periods=100, freq="D")
        data = {
            "open": [100 + i * 0.1 for i in range(100)],
            "high": [105 + i * 0.1 for i in range(100)],
            "low": [95 + i * 0.1 for i in range(100)],
            "close": [102 + i * 0.1 for i in range(100)],
            "volume": [1000 + i * 10 for i in range(100)],
        }
        df = pd.DataFrame(data, index=dates)

        # Test initialization
        env = HeavyTradingEnv(df=df)
        assert env is not None
        assert hasattr(env, "reset")
        assert hasattr(env, "step")
        assert hasattr(env, "observation_space")
        assert hasattr(env, "action_space")

    def test_reset(self):
        """Test environment reset."""
        # Create sample data
        dates = pd.date_range("2023-01-01", periods=50, freq="D")
        data = {
            "open": [100 + i * 0.1 for i in range(50)],
            "high": [105 + i * 0.1 for i in range(50)],
            "low": [95 + i * 0.1 for i in range(50)],
            "close": [102 + i * 0.1 for i in range(50)],
            "volume": [1000 + i * 10 for i in range(50)],
        }
        df = pd.DataFrame(data, index=dates)

        env = HeavyTradingEnv(df=df)
        obs, info = env.reset()

        assert isinstance(obs, np.ndarray)
        assert isinstance(info, dict)
        assert len(obs) > 0  # Should have observation features

    def test_step(self):
        """Test environment step."""
        # Create sample data
        dates = pd.date_range("2023-01-01", periods=30, freq="D")
        data = {
            "open": [100 + i * 0.1 for i in range(30)],
            "high": [105 + i * 0.1 for i in range(30)],
            "low": [95 + i * 0.1 for i in range(30)],
            "close": [102 + i * 0.1 for i in range(30)],
            "volume": [1000 + i * 10 for i in range(30)],
        }
        df = pd.DataFrame(data, index=dates)

        env = HeavyTradingEnv(df=df)
        env.reset()

        # Test step with random action
        action = env.action_space.sample()
        result = env.step(action)
        obs, reward, terminated, truncated, info = result
        done = terminated or truncated

        assert isinstance(obs, np.ndarray)
        assert isinstance(reward, (int, float))
        assert isinstance(done, bool)
        assert isinstance(info, dict)

    def test_action_space(self):
        """Test action space properties."""
        dates = pd.date_range("2023-01-01", periods=20, freq="D")
        df = pd.DataFrame(
            {
                "close": [100 + i for i in range(20)],
                "open": [99 + i for i in range(20)],
                "high": [105 + i for i in range(20)],
                "low": [95 + i for i in range(20)],
                "volume": [1000 + i * 10 for i in range(20)],
            },
            index=dates,
        )

        env = HeavyTradingEnv(df=df)
        assert env.action_space is not None
        # Sample some actions
        for _ in range(5):
            action = env.action_space.sample()
            assert action is not None

    def test_observation_space(self):
        """Test observation space properties."""
        dates = pd.date_range("2023-01-01", periods=20, freq="D")
        df = pd.DataFrame(
            {
                "close": [100 + i for i in range(20)],
                "open": [99 + i for i in range(20)],
                "high": [105 + i for i in range(20)],
                "low": [95 + i for i in range(20)],
                "volume": [1000 + i * 10 for i in range(20)],
            },
            index=dates,
        )

        env = HeavyTradingEnv(df=df)
        assert env.observation_space is not None

        obs, _ = env.reset()
        assert env.observation_space.contains(obs)

    def test_reward_calculation(self):
        """Test reward calculation."""
        dates = pd.date_range("2023-01-01", periods=25, freq="D")
        df = pd.DataFrame(
            {
                "close": [100 + i for i in range(25)],
                "open": [99 + i for i in range(25)],
                "high": [105 + i for i in range(25)],
                "low": [95 + i for i in range(25)],
                "volume": [1000 + i * 10 for i in range(25)],
            },
            index=dates,
        )

        env = HeavyTradingEnv(df=df)
        env.reset()

        # Take a few steps and check rewards are reasonable
        for _ in range(5):
            action = env.action_space.sample()
            result = env.step(action)
            _, reward, _, _, _ = result
            assert isinstance(reward, (int, float))
            # Reward should be finite
            assert np.isfinite(reward)

    def test_episode_completion(self):
        """Test episode completion."""
        dates = pd.date_range("2023-01-01", periods=15, freq="D")
        df = pd.DataFrame(
            {
                "close": [100 + i for i in range(15)],
                "open": [99 + i for i in range(15)],
                "high": [105 + i for i in range(15)],
                "low": [95 + i for i in range(15)],
                "volume": [1000 + i * 10 for i in range(15)],
            },
            index=dates,
        )

        env = HeavyTradingEnv(df=df)
        env.reset()

        done = False
        steps = 0
        while not done and steps < 20:  # Safety limit
            action = env.action_space.sample()
            result = env.step(action)
            _, _, terminated, truncated, _ = result
            done = terminated or truncated
            steps += 1

        # Should eventually complete or reach safety limit
        assert steps > 0
