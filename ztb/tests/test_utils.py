"""
Common test utilities for ztb testing.
"""

import sys
import types
from typing import Any, Dict
from unittest.mock import Mock

import numpy as np
import pandas as pd
import pytest

from ztb.trading.env_config import get_trading_env_config
from ztb.trading.environment.environment import HeavyTradingEnv


def create_mock_feature_engine():
    """Create a mock feature engine for testing."""
    fake_features = types.ModuleType("ztb.features")
    sys.modules["ztb.features"] = fake_features

    fake_feature_engine = types.ModuleType("ztb.features.feature_engine")

    def _compute_features_batch(df, feature_names=None, **_kwargs):
        """Mock feature computation that returns empty DataFrame."""
        return pd.DataFrame(index=df.index)

    fake_feature_engine.compute_features_batch = _compute_features_batch
    sys.modules["ztb.features.feature_engine"] = fake_feature_engine


def create_mock_observability():
    """Create a mock observability module for testing."""
    fake_observability = types.ModuleType("ztb.utils.observability")
    sys.modules["ztb.utils.observability"] = fake_observability

    def generate_correlation_id() -> str:
        return "test-correlation-id"

    fake_observability.generate_correlation_id = generate_correlation_id


def setup_test_modules():
    """Setup all mock modules needed for testing."""
    create_mock_feature_engine()
    create_mock_observability()


def get_sample_trading_data() -> pd.DataFrame:
    """Create sample OHLCV trading data for testing."""
    np.random.seed(42)
    n_steps = 100

    # Generate realistic price data
    base_price = 5000000.0  # JPY-based price
    price_changes = np.random.normal(0, 0.01, n_steps)  # 1% volatility
    prices = base_price * np.cumprod(1 + price_changes)

    # Generate OHLCV data
    high_mult = 1 + np.abs(np.random.normal(0, 0.005, n_steps))
    low_mult = 1 - np.abs(np.random.normal(0, 0.005, n_steps))
    volume_mult = np.random.uniform(0.5, 2.0, n_steps)

    data = {
        "open": prices * (1 + np.random.normal(0, 0.002, n_steps)),
        "high": prices * high_mult,
        "low": prices * low_mult,
        "close": prices,
        "volume": 1000 * volume_mult,
    }

    df = pd.DataFrame(data)
    # Ensure high >= max(open, close) and low <= min(open, close)
    df["high"] = np.maximum(df[["open", "close"]].max(axis=1), df["high"])
    df["low"] = np.minimum(df[["open", "close"]].min(axis=1), df["low"])

    return df


def get_default_env_config() -> Dict[str, Any]:
    """Create default environment configuration for testing."""
    return get_trading_env_config()


@pytest.fixture
def mock_feature_registry():
    """Create a mock feature registry for testing."""
    mock_registry = Mock()
    mock_registry.compute_features.return_value = pd.DataFrame(
        {
            "close": [100.0, 101.0, 102.0],
            "volume": [1000, 1100, 1200],
            "sma_20": [99.0, 100.0, 101.0],
            "rsi_14": [50.0, 55.0, 60.0],
        }
    )
    mock_registry.is_cache_enabled.return_value = False
    return mock_registry


@pytest.fixture
def mock_fee_model():
    """Create a mock fee model for testing."""
    mock_fee = Mock()
    mock_fee.calculate_fee.return_value = 0.1
    return mock_fee


def create_test_env(
    sample_data: pd.DataFrame, config: Dict[str, Any]
) -> HeavyTradingEnv:
    """Helper function to create HeavyTradingEnv with common test setup."""
    return HeavyTradingEnv(df=sample_data, config=config)


def assert_env_initialized_correctly(env: HeavyTradingEnv, expected_steps: int):
    """Common assertions for environment initialization."""
    assert env is not None
    assert hasattr(env, "df")
    assert len(env.df) == expected_steps
    assert hasattr(env, "current_step")
    assert env.current_step == 0
    assert hasattr(env, "position")
    assert env.position == 0.0
