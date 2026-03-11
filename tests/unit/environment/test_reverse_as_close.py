#!/usr/bin/env python3
"""
Unit tests for Reverse-as-Close functionality.

Tests verify that:
1. allow_reverse=True behaves as before (default, backward compatible)
2. allow_reverse=False prevents immediate reversal
3. Position transitions are correct
4. PnL and transaction costs are as expected
"""

import pandas as pd
import pytest

from tests.helpers import make_schema_feature_env_config, make_trending_ohlcv_data
from ztb.trading.environment.environment import EnvironmentConfig, HeavyTradingEnv


@pytest.fixture(scope="module")
def sample_df():
    """Create sample DataFrame for testing."""
    return make_trending_ohlcv_data(
        rows=6,
        seed=7,
        start_price=100.0,
        end_price=105.0,
        noise_scale=0.0,
        volume_low=1000.0,
        volume_high=1000.0,
        include_timestamp=False,
    )

def _make_env(sample_df, **overrides) -> HeavyTradingEnv:
    config = make_schema_feature_env_config(
        sample_df,
        curriculum_stage="full",
        max_position_size=1.0,
        **overrides,
    )
    return HeavyTradingEnv(df=sample_df.copy(), config=config)


@pytest.fixture(scope="module")
def env_allow_reverse_true(sample_df: pd.DataFrame) -> HeavyTradingEnv:
    env = _make_env(sample_df, allow_reverse=True, transaction_cost=0.001)
    yield env
    env.close()


@pytest.fixture(scope="module")
def env_allow_reverse_false(sample_df: pd.DataFrame) -> HeavyTradingEnv:
    env = _make_env(sample_df, allow_reverse=False, transaction_cost=0.001)
    yield env
    env.close()


@pytest.fixture(scope="module")
def env_allow_reverse_true_high_fee(sample_df: pd.DataFrame) -> HeavyTradingEnv:
    env = _make_env(sample_df, allow_reverse=True, transaction_cost=0.01)
    yield env
    env.close()


@pytest.fixture(scope="module")
def env_allow_reverse_false_high_fee(sample_df: pd.DataFrame) -> HeavyTradingEnv:
    env = _make_env(sample_df, allow_reverse=False, transaction_cost=0.01)
    yield env
    env.close()


@pytest.fixture(scope="module")
def env_allow_reverse_default(sample_df: pd.DataFrame) -> HeavyTradingEnv:
    env = HeavyTradingEnv(
        df=sample_df.copy(),
        config=make_schema_feature_env_config(
            sample_df,
            transaction_cost=0.001,
            curriculum_stage="full",
        ),
    )
    yield env
    env.close()


class TestReverseAsClose:
    """Test suite for allow_reverse flag."""

    def test_allow_reverse_true_default(self, env_allow_reverse_true):
        """Test default behavior: allow_reverse=True."""
        env = env_allow_reverse_true
        obs, info = env.reset()

        # Initial: position=0 (Flat)
        assert env.position == 0.0

        # Step 1: BUY → Long
        obs, reward, done, truncated, info = env.step(1)  # BUY
        long_position = env.position
        assert long_position > 0.0, "BUY from Flat should open Long"

        # Step 2: SELL → Close Long + Open Short (immediate reversal)
        obs, reward, done, truncated, info = env.step(2)  # SELL
        assert env.position < 0.0, "SELL from Long should reverse to Short (allow_reverse=True)"
        assert abs(env.position) > 0.0

    def test_allow_reverse_false_no_reversal(self, env_allow_reverse_false):
        """Test reverse禁止モード: allow_reverse=False."""
        env = env_allow_reverse_false
        obs, info = env.reset()

        # Initial: position=0 (Flat)
        assert env.position == 0.0

        # Step 1: BUY → Long
        obs, reward, done, truncated, info = env.step(1)  # BUY
        assert env.position > 0.0, "BUY from Flat should open Long"

        # Step 2: SELL → Close Long ONLY (no reversal to Short)
        obs, reward, done, truncated, info = env.step(2)  # SELL
        assert (
            env.position == 0.0
        ), "SELL from Long should close to Flat (allow_reverse=False)"

    def test_allow_reverse_false_short_to_flat(self, env_allow_reverse_false):
        """Test Short→BUY→Flat (no reversal)."""
        env = env_allow_reverse_false
        obs, info = env.reset()

        # Step 1: SELL → Short
        obs, reward, done, truncated, info = env.step(2)  # SELL
        assert env.position < 0.0, "SELL from Flat should open Short"

        # Step 2: BUY → Close Short ONLY (no reversal to Long)
        obs, reward, done, truncated, info = env.step(1)  # BUY
        assert (
            env.position == 0.0
        ), "BUY from Short should close to Flat (allow_reverse=False)"

    def test_flat_to_long_short_unaffected(
        self,
        env_allow_reverse_true,
        env_allow_reverse_false,
    ):
        """Test that Flat→Long/Short is unaffected by allow_reverse."""
        # Test with allow_reverse=True
        env_true = env_allow_reverse_true
        env_true.reset()
        env_true.step(1)  # BUY
        assert env_true.position > 0.0

        # Test with allow_reverse=False
        env_false = env_allow_reverse_false
        env_false.reset()
        env_false.step(1)  # BUY
        assert env_false.position > 0.0

        # Both should be identical
        assert env_true.position == env_false.position

    def test_transaction_cost_count(
        self,
        env_allow_reverse_true_high_fee,
        env_allow_reverse_false_high_fee,
    ):
        """Test that allow_reverse=False reduces transaction costs."""
        # Scenario: Flat→BUY→SELL
        # allow_reverse=True: 3 trades (BUY open, SELL close, SELL open)
        # allow_reverse=False: 2 trades (BUY open, SELL close)

        env_true = env_allow_reverse_true_high_fee
        env_true.reset()
        env_true.step(1)  # BUY
        env_true.step(2)  # SELL
        trades_true = env_true.trades_count

        env_false = env_allow_reverse_false_high_fee
        env_false.reset()
        env_false.step(1)  # BUY
        env_false.step(2)  # SELL
        trades_false = env_false.trades_count

        assert (
            trades_true > trades_false
        ), f"allow_reverse=True should have more trades ({trades_true} vs {trades_false})"

    def test_position_transitions_detailed(self, env_allow_reverse_false):
        """Test detailed position transitions with both allow_reverse modes."""
        # Test allow_reverse=False: Long→SELL→Flat→SELL→Short
        env = env_allow_reverse_false
        env.reset()

        # Flat → BUY → Long
        env.step(1)
        assert env.position > 0.0, "Should be Long"

        # Long → SELL → Flat (no reversal)
        env.step(2)
        assert env.position == 0.0, "Should be Flat (allow_reverse=False)"

        # Flat → SELL → Short (normal open)
        env.step(2)
        assert env.position < 0.0, "Should be Short from Flat"

        # Short → BUY → Flat (no reversal)
        env.step(1)
        assert env.position == 0.0, "Should be Flat (allow_reverse=False)"

    def test_config_from_dict_allow_reverse(self):
        """Test that allow_reverse is correctly parsed from dict."""
        # Test default (True)
        config_default = EnvironmentConfig.from_dict({})
        assert config_default.allow_reverse is True

        # Test explicit True
        config_true = EnvironmentConfig.from_dict({"allow_reverse": True})
        assert config_true.allow_reverse is True

        # Test explicit False
        config_false = EnvironmentConfig.from_dict({"allow_reverse": False})
        assert config_false.allow_reverse is False

        # Test string conversion
        config_str_true = EnvironmentConfig.from_dict({"allow_reverse": "true"})
        assert config_str_true.allow_reverse is True

        config_str_false = EnvironmentConfig.from_dict({"allow_reverse": "false"})
        assert config_str_false.allow_reverse is False

    def test_backward_compatibility(self, env_allow_reverse_default):
        """Test that existing code without allow_reverse still works."""
        # Old code that doesn't specify allow_reverse
        env = env_allow_reverse_default
        obs, info = env.reset()

        # Should default to allow_reverse=True (backward compatible)
        assert env.config.allow_reverse is True

        # Test reversal behavior
        env.step(1)  # BUY
        env.step(2)  # SELL
        assert env.position < 0.0, "Should allow reversal by default"
