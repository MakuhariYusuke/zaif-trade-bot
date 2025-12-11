from unittest.mock import MagicMock

import pytest

from ztb.trading.environment.components.rewards.base import RewardContext
from ztb.trading.environment.components.rewards.smart_incentive import (
    SmartIncentiveReward,
)


@pytest.fixture
def reward_component():
    return SmartIncentiveReward()


@pytest.fixture
def mock_config():
    config = MagicMock()
    config.get.side_effect = lambda k, d: d  # Default behavior
    return config


def test_calculate_basic_pnl(reward_component, mock_config):
    context = RewardContext(
        action=1,  # BUY
        current_price=100.0,
        position=1.0,
        portfolio_value=1000.0,
        atr=1.0,
        transaction_cost=0.0,
        reward_scaling=1.0,
        pnl=10.0,  # Positive PnL
        old_position=0.0,
        step=100,
        observation=None,
        reward_history=[],
        portfolio_value_history=[],
        config=mock_config,
    )

    # Normal PnL / ATR = 10 / 1 = 10
    # Multiplier default 1.0
    # Volatility ratio = 1/100 = 0.01 > 0.005 (threshold) -> Multiplier 1.1

    reward = reward_component.calculate(context)
    assert reward == pytest.approx(11.0)


def test_calculate_low_volatility(reward_component, mock_config):
    context = RewardContext(
        action=1,
        current_price=1000.0,
        position=1.0,
        portfolio_value=1000.0,
        atr=1.0,  # Low volatility relative to price (0.001)
        transaction_cost=0.0,
        reward_scaling=1.0,
        pnl=10.0,
        old_position=0.0,
        step=100,
        observation=None,
        reward_history=[],
        portfolio_value_history=[],
        config=mock_config,
    )

    # Volatility ratio = 1/1000 = 0.001 < 0.005 -> Multiplier 1.0
    reward = reward_component.calculate(context)
    assert reward == pytest.approx(10.0)
