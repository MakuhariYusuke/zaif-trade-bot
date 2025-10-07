"""
Unit tests for HeavyTradingEnv class.
"""

import numpy as np
import pandas as pd
import pytest

from ztb.trading.environment.environment import HeavyTradingEnv
from ztb.tests.test_utils import (
    get_sample_trading_data,
    get_default_env_config,
    create_test_env,
    assert_env_initialized_correctly,
)


class TestHeavyTradingEnv:
    """Test suite for HeavyTradingEnv class."""

    @pytest.fixture
    def sample_data(self) -> pd.DataFrame:
        """Create sample trading data for testing."""
        return get_sample_trading_data()

    @pytest.fixture
    def env_config(self) -> dict:
        """Create default environment configuration for testing."""
        return get_default_env_config()

    def test_initialization_with_dataframe(self, sample_data, env_config):
        """Test that environment initializes correctly with a DataFrame."""
        env = create_test_env(sample_data, env_config)
        assert_env_initialized_correctly(env, len(sample_data))

    def test_action_space(self, sample_data, env_config):
        """Test that action space is correctly defined."""
        env = create_test_env(sample_data, env_config)
        assert env.action_space.n == 3  # hold, buy, sell
