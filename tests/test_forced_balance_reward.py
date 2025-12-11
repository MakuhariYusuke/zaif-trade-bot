from unittest.mock import MagicMock

import pytest

from ztb.trading.environment.components.rewards.base import RewardContext
from ztb.trading.environment.components.rewards.forced_balance import (
    ForcedBalanceReward,
)


@pytest.fixture
def reward_component():
    return ForcedBalanceReward()


@pytest.fixture
def mock_config():
    config = MagicMock()
    config.get.side_effect = lambda k, d: d  # Default behavior
    return config


def test_calculate_balanced(reward_component, mock_config):
    context = RewardContext(
        action=0,  # HOLD
        current_price=100.0,
        position=0.0,
        portfolio_value=1000.0,
        atr=1.0,
        transaction_cost=0.0,
        reward_scaling=1.0,
        pnl=0.0,
        old_position=0.0,
        step=100,
        observation=None,
        reward_history=[],
        portfolio_value_history=[],
        config=mock_config,
        action_counts=[33, 33, 33],  # Balanced
        target_ratios={"HOLD": 0.33, "BUY": 0.33, "SELL": 0.33},
    )

    reward = reward_component.calculate(context)
    # Should return balanced reward (default 2.0)
    assert reward == 2.0


def test_calculate_imbalanced_penalty(reward_component, mock_config):
    # Force imbalance: Too many HOLDs
    context = RewardContext(
        action=0,  # HOLD (adding to imbalance)
        current_price=100.0,
        position=0.0,
        portfolio_value=1000.0,
        atr=1.0,
        transaction_cost=0.0,
        reward_scaling=1.0,
        pnl=0.0,
        old_position=0.0,
        step=100,
        observation=None,
        reward_history=[],
        portfolio_value_history=[],
        config=mock_config,
        action_counts=[80, 10, 10],  # 80% HOLD
        target_ratios={"HOLD": 0.33, "BUY": 0.33, "SELL": 0.33},
    )

    # Mock settings to ensure we trigger penalty
    def get_setting(key, default):
        if key == "forced_balance_threshold":
            return 0.1
        return default

    mock_config.get.side_effect = get_setting

    reward = reward_component.calculate(context)
    assert reward < 0  # Should be penalized


def test_calculate_corrective_bonus(reward_component, mock_config):
    # Force imbalance: Too many HOLDs, but we take BUY action (under-represented)
    context = RewardContext(
        action=1,  # BUY (corrective)
        current_price=100.0,
        position=0.0,
        portfolio_value=1000.0,
        atr=1.0,
        transaction_cost=0.0,
        reward_scaling=1.0,
        pnl=0.0,
        old_position=0.0,
        step=100,
        observation=None,
        reward_history=[],
        portfolio_value_history=[],
        config=mock_config,
        action_counts=[80, 10, 10],  # 80% HOLD
        target_ratios={"HOLD": 0.33, "BUY": 0.33, "SELL": 0.33},
    )

    def get_setting(key, default):
        if key == "forced_balance_threshold":
            return 0.1
        return default

    mock_config.get.side_effect = get_setting

    reward = reward_component.calculate(context)
    assert reward > 0  # Should be rewarded
