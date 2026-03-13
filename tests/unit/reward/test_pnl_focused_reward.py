from unittest.mock import MagicMock

import numpy as np
import pytest

from ztb.trading.environment.components.rewards.base import RewardContext
from ztb.trading.environment.components.rewards.pnl_focused import PnlFocusedReward


@pytest.fixture
def reward_component():
    return PnlFocusedReward()


@pytest.fixture
def mock_settings():
    settings = MagicMock()
    # Default settings
    settings_dict = {
        "base_profit_bonus_atr_coeff": 1.5,
        "base_profit_bonus_portfolio_coeff": 1.2,
        "profit_bonus_multipliers": [1.0, 1.0, 0.8],
        "hold_action_penalty": 0.0,
        "buy_action_penalty": 0.0,
        "sell_action_penalty": 0.0,
        "base_action_penalty": 0.015,
        "hold_penalty_base": 0.01,
        "hold_penalty_position_factor": 0.04,
        "hold_penalty_multiplier": 1.0,
        "loss_penalty_coeff": -0.2,
        "position_penalty_coeff": 0.05,
        "max_position_penalty": 0.1,
    }

    def get_setting(key, default=None):
        return settings_dict.get(key, default)

    settings.get.side_effect = get_setting
    return settings


@pytest.fixture
def mock_config():
    return MagicMock()


def test_calculate_basic_profit(reward_component, mock_settings, mock_config):
    context = RewardContext(
        action=1,  # BUY
        current_price=100.0,
        position=0.0,
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
        reward_settings=mock_settings,
        atr_normalised=2.0,
        portfolio_return=0.01,
        effective_max_position=1.0,
        action_counts=[10, 10, 10],
    )

    # base_profit_bonus = 1.5 * 2.0 + 1.2 * 0.01 = 3.0 + 0.012 = 3.012
    # multipliers[BUY] = 1.0
    # trend_multiplier = 1.0 (no observation)
    # profit_bonus = 3.012 * 1.0 * 1.0 = 3.012

    # action_penalty (BUY) = base_action_penalty (0.015) + buy_penalty (0.0) = 0.015
    # loss_penalty = 0.0 (pnl > 0)
    # position_penalty = 0.0 (position 0)

    # reward = 3.012 - 0.015 + 0 - 0 = 2.997

    reward = reward_component.calculate(context)
    assert reward == pytest.approx(2.997)


def test_calculate_loss_penalty(reward_component, mock_settings, mock_config):
    context = RewardContext(
        action=1,  # BUY
        current_price=100.0,
        position=0.0,
        portfolio_value=1000.0,
        atr=1.0,
        transaction_cost=0.0,
        reward_scaling=1.0,
        pnl=-10.0,  # Negative PnL
        old_position=0.0,
        step=100,
        observation=None,
        reward_history=[],
        portfolio_value_history=[],
        config=mock_config,
        reward_settings=mock_settings,
        atr_normalised=-2.0,
        portfolio_return=-0.01,
        effective_max_position=1.0,
        action_counts=[10, 10, 10],
    )

    # base_profit_bonus = 0.0 (pnl < 0)
    # profit_bonus = 0.0

    # action_penalty (BUY) = 0.015

    # loss_penalty = -0.2 * abs(-2.0) = -0.4

    # reward = 0.0 - 0.015 - 0.4 - 0.0 = -0.415

    reward = reward_component.calculate(context)
    assert reward == pytest.approx(-0.415)


def test_trend_multiplier_buy(reward_component, mock_settings, mock_config):
    # Mock observation with RSI and MACD
    # RSI > 50, MACD > 0 -> Trend Ratio > 1.0 -> Buy Multiplier 1.2

    # Assuming observation size is large enough, RSI at -2, MACD at -1
    obs = np.zeros(10)
    obs[-2] = 60.0  # RSI
    obs[-1] = 10.0  # MACD Hist

    context = RewardContext(
        action=1,  # BUY
        current_price=100.0,
        position=0.0,
        portfolio_value=1000.0,
        atr=1.0,
        transaction_cost=0.0,
        reward_scaling=1.0,
        pnl=10.0,
        old_position=0.0,
        step=100,
        observation=obs,
        reward_history=[],
        portfolio_value_history=[],
        config=mock_config,
        reward_settings=mock_settings,
        atr_normalised=2.0,
        portfolio_return=0.01,
        effective_max_position=1.0,
        action_counts=[10, 10, 10],
    )

    # trend_ratio = (60/50) * (1 + 10/100) = 1.2 * 1.1 = 1.32 > 1.0
    # trend_multiplier = 1.2

    # base_profit_bonus = 3.012
    # profit_bonus = 3.012 * 1.2 = 3.6144

    # reward = 3.6144 - 0.015 = 3.5994

    reward = reward_component.calculate(context)
    assert reward == pytest.approx(3.5994)


def test_hold_penalty(reward_component, mock_settings, mock_config):
    context = RewardContext(
        action=0,  # HOLD
        current_price=100.0,
        position=0.5,
        portfolio_value=1000.0,
        atr=1.0,
        transaction_cost=0.0,
        reward_scaling=1.0,
        pnl=0.0,
        old_position=0.5,
        step=100,
        observation=None,
        reward_history=[],
        portfolio_value_history=[],
        config=mock_config,
        reward_settings=mock_settings,
        atr_normalised=0.0,
        portfolio_return=0.0,
        effective_max_position=1.0,
        action_counts=[10, 10, 10],
    )

    # base_profit_bonus = 0.0
    # profit_bonus = 0.0

    # position_size_factor = 0.5 / 1.0 = 0.5
    # volatility_factor = min(1.0 / (100 * 0.01), 1.0) = 1.0
    # base_action_penalty = 0.01 + (0.04 * 0.5 * 1.0) = 0.01 + 0.02 = 0.03
    # base_action_penalty *= 1.0
    # action_penalty = 0.03 + 0.0 = 0.03

    # reward = 0.0 - 0.03 - 0.0 - 0.0 = -0.03

    reward = reward_component.calculate(context)
    assert reward == pytest.approx(-0.03)
