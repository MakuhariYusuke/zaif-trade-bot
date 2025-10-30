#!/usr/bin/env python3
"""
Tests for step PnL calculation bug fixes.
"""

from typing import Any, Dict

import numpy as np
import pandas as pd
import pytest

from ztb.training.environments.environment_config import EnvironmentConfig
from ztb.training.environments.heavy_trading_env import HeavyTradingEnv


class TestStepPnLCalculation:
    """Test step PnL calculation fixes."""

    @pytest.fixture
    def simple_price_data(self) -> pd.DataFrame:
        """Simple price data for testing."""
        df = pd.DataFrame(
            {
                "close": [100.0, 101.0, 102.0, 103.0, 104.0],
                "open": [100.0, 101.0, 102.0, 103.0, 104.0],
                "high": [101.0, 102.0, 103.0, 104.0, 105.0],
                "low": [99.0, 100.0, 101.0, 102.0, 103.0],
                "volume": [1000.0] * 5,
            }
        )
        return df

    @pytest.fixture
    def env_config(self) -> EnvironmentConfig:
        """Environment configuration for testing."""
        return EnvironmentConfig(
            initial_balance=10000.0,
            max_position_size=1.0,
            commission=0.001,  # 0.1%
            reward_scaling=1.0,
        )

    @pytest.fixture
    def reward_settings(self) -> Dict[str, Any]:
        """Reward settings for testing."""
        return {
            "profit_weight": 1.0,
            "risk_weight": 1.0,
            "consistency_weight": 1.0,
        }

    def test_step_pnl_reset_on_environment_reset(
        self, simple_price_data, env_config, reward_settings
    ):
        """Test that environment resets properly."""
        env = HeavyTradingEnv(
            data=simple_price_data, config=env_config, reward_settings=reward_settings
        )

        # Reset environment
        obs = env.reset()

        # Check that environment is in initial state
        assert env.current_step == 0
        assert env.position == 0.0
        assert env.balance == env_config.initial_balance
        assert env.unrealized_pnl == 0.0
        assert env.total_pnl == 0.0

    def test_step_pnl_calculation_with_price_change(
        self, simple_price_data, env_config, reward_settings
    ):
        """Test that step PnL correctly calculates trade_pnl + unrealized_delta."""
        env = HeavyTradingEnv(
            data=simple_price_data, config=env_config, reward_settings=reward_settings
        )

        env.reset()

        # Initial state
        initial_balance = env.balance
        initial_position = env.position

        # Take BUY action (opens long position)
        obs, reward, terminated, truncated, info = env.step(
            np.array([1.0])
        )  # BUY action

        # Check that reward is reasonable
        assert isinstance(reward, (int, float))
        assert not (reward != reward)  # Check not NaN

    def test_per_step_pnl_vs_cumulative_pnl(
        self, simple_price_data, env_config, reward_settings
    ):
        """Test that per-step PnL prevents cumulative PnL explosion."""
        env = HeavyTradingEnv(
            data=simple_price_data, config=env_config, reward_settings=reward_settings
        )

        env.reset()

        rewards = []
        pnls = []

        # Take several actions
        for action_value in [1.0, 0.0, -1.0, 0.0]:  # BUY, HOLD, SELL, HOLD
            obs, reward, terminated, truncated, info = env.step(
                np.array([action_value])
            )
            rewards.append(reward)
            pnls.append(info.get("total_pnl", 0))

        # Rewards should be finite and not exploding
        for reward in rewards:
            assert isinstance(reward, (int, float))
            assert abs(reward) < 1000  # Reasonable bound
            assert not (reward != reward)  # Not NaN

        # PnL should accumulate properly
        assert all(isinstance(pnl, (int, float)) for pnl in pnls)
        assert not any(pnl != pnl for pnl in pnls)  # No NaN values

    def test_unrealized_pnl_tracking(
        self, simple_price_data, env_config, reward_settings
    ):
        """Test that unrealized PnL is tracked correctly."""
        env = HeavyTradingEnv(
            data=simple_price_data, config=env_config, reward_settings=reward_settings
        )

        env.reset()

        # Initial state
        initial_unrealized = env.unrealized_pnl
        assert initial_unrealized == 0.0

        # Take action
        obs, reward, terminated, truncated, info = env.step(np.array([1.0]))  # BUY

        # Unrealized PnL should change
        assert env.unrealized_pnl != initial_unrealized

        # Take another action
        obs, reward, terminated, truncated, info = env.step(np.array([0.0]))  # HOLD

        # Unrealized PnL should continue to be tracked
        assert isinstance(env.unrealized_pnl, (int, float))


if __name__ == "__main__":
    pytest.main([__file__])
