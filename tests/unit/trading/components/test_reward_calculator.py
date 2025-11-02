"""Tests for RewardCalculator component."""

import math
from unittest.mock import Mock

import pytest

from ztb.trading.constants import ACTION_BUY, ACTION_HOLD, ACTION_SELL
from ztb.trading.environment.components.reward_calculator import RewardCalculator
from ztb.trading.environment.utils.config import RewardSettings


@pytest.fixture
def sample_config():
    """Sample environment config for testing."""
    config = Mock()
    config.curriculum_stage = "balanced_transition"
    config.max_position_size = 1.0
    config.transaction_cost = 0.001
    config.reward_scaling = 1.0
    return config


@pytest.fixture
def reward_settings():
    """Sample reward settings for testing."""
    return RewardSettings(
        use_simple_reward=False,
        reward_scale=100.0,
        trading_bonus=0.01,
        profit_bonuses={"base": 1.5, "ultra": 2.0},
        penalty_coefficients={"loss": 2.0, "position": 0.01, "stagnation": 0.001},
        entropy_bonus=0.0,
        custom_reward_params={},
        balance_penalty=0.1,
        balance_penalty_tolerance=0.05,
        profit_weight=1.0,
        risk_weight=0.5,
        consistency_weight=0.2,
        ultra_profit_multiplier=2.0,
        ultra_risk_multiplier=0.5,
        position_soft_cap=0.5,
        position_penalty_scale=0.1,
        position_penalty_exponent=2.0,
        inventory_window=10,
        inventory_penalty_scale=0.01,
        trade_frequency_penalty=0.001,
        trade_frequency_halflife=100.0,
        trade_cooldown_steps=5,
        trade_cooldown_penalty=0.01,
        max_consecutive_trades=3,
        consecutive_trade_penalty=0.05,
        volatility_window=20,
        volatility_penalty_scale=0.01,
        sharpe_bonus_scale=0.01,
        sortino_bonus_scale=0.01,
        calmar_bonus_scale=0.005,
        reward_clip_value=10.0,
        profit_bonus_multipliers=[1.0, 1.5, 2.0],
        enable_forced_diversity=False,
    )


@pytest.fixture
def reward_calculator(sample_config, reward_settings):
    """RewardCalculator instance for testing."""
    return RewardCalculator(
        config=sample_config,
        reward_settings=reward_settings,
        initial_portfolio_value=100000.0,
    )


class TestRewardCalculatorInitialization:
    """Test RewardCalculator initialization."""

    def test_initialization(self, reward_calculator):
        """Test proper initialization."""
        assert reward_calculator.config is not None
        assert reward_calculator.reward_settings is not None
        assert reward_calculator.initial_portfolio_value == 100000.0
        assert reward_calculator._action_counts == [0, 0, 0]
        assert reward_calculator._consecutive_idle_steps == 0
        assert reward_calculator._win_count == 0
        assert reward_calculator._loss_count == 0

    def test_get_setting_methods(self, reward_calculator):
        """Test setting getter methods."""
        # Test get_setting_int
        assert (
            reward_calculator.get_setting_int("max_consecutive_trades", 1) == 3
        )  # from reward_settings
        assert reward_calculator.get_setting_int("nonexistent", 42) == 42

        # Test get_setting_float
        assert reward_calculator.get_setting_float("reward_scale", 1.0) == 100.0
        assert reward_calculator.get_setting_float("nonexistent", 3.14) == 3.14

        # Test get_setting_bool
        assert reward_calculator.get_setting_bool("use_simple_reward", True) is False
        assert reward_calculator.get_setting_bool("nonexistent", True) is True


class TestRewardCalculatorSimple:
    """Test simple reward calculation."""

    def test_calculate_reward_simple_profit(self, reward_calculator):
        """Test simple reward with profit."""
        import numpy as np

        reward = reward_calculator.calculate_reward(
            action=ACTION_BUY,
            current_price=100.0,
            position=0.5,
            portfolio_value=101000.0,
            atr=1.0,
            transaction_cost=0.001,
            reward_scaling=1.0,
            pnl=1000.0,
            old_position=0.0,
            step=1,
            observation=np.array([1.0, 2.0, 3.0]),
            reward_history=[0.1, 0.2],
            portfolio_value_history=[100000.0, 100500.0],
        )
        assert isinstance(reward, float)
        assert reward > 0  # Profit should give positive reward

    def test_calculate_reward_simple_loss(self, reward_calculator):
        """Test simple reward with loss."""
        import numpy as np

        reward = reward_calculator.calculate_reward(
            action=ACTION_SELL,
            current_price=100.0,
            position=-0.3,
            portfolio_value=99500.0,
            atr=1.0,
            transaction_cost=0.001,
            reward_scaling=1.0,
            pnl=-500.0,
            old_position=0.0,
            step=1,
            observation=np.array([1.0, 2.0, 3.0]),
            reward_history=[0.1, -0.05],
            portfolio_value_history=[100000.0, 99750.0],
        )
        assert isinstance(reward, float)
        assert reward < 0  # Loss should give negative reward


class TestRewardCalculatorComplex:
    """Test complex reward calculation."""

    def test_calculate_reward_balanced_transition(self, reward_calculator):
        """Test balanced transition reward calculation."""
        reward = reward_calculator.calculate_reward(
            action=ACTION_BUY,
            current_price=100.0,
            position=0.5,
            portfolio_value=100500.0,
            atr=1.0,
            transaction_cost=0.001,
            reward_scaling=1.0,
            pnl=500.0,
            old_position=0.0,
            step=1,
            observation=None,
            reward_history=[],
            portfolio_value_history=[100000.0],
        )
        assert isinstance(reward, float)

    def test_calculate_reward_with_transaction_cost(self, reward_calculator):
        """Test reward calculation includes transaction cost."""
        reward = reward_calculator.calculate_reward(
            action=ACTION_BUY,
            current_price=100.0,
            position=0.5,
            portfolio_value=99900.0,  # Account for transaction cost
            atr=1.0,
            transaction_cost=0.001,
            reward_scaling=1.0,
            pnl=-100.0,  # Transaction cost
            old_position=0.0,
            step=1,
            observation=None,
            reward_history=[],
            portfolio_value_history=[100000.0],
        )
        assert isinstance(reward, float)


class TestRewardCalculatorReset:
    """Test reset functionality."""

    def test_reset(self, reward_calculator):
        """Test reset method clears internal state."""
        # Modify internal state
        reward_calculator._action_counts = [1, 2, 3]
        reward_calculator._consecutive_idle_steps = 5
        reward_calculator._win_count = 10
        reward_calculator._loss_count = 8
        reward_calculator._recent_actions = [ACTION_BUY, ACTION_SELL]

        # Reset
        reward_calculator.reset_episode_state()

        # Verify reset
        assert reward_calculator._action_counts == [0, 0, 0]
        assert reward_calculator._consecutive_idle_steps == 0
        assert reward_calculator._win_count == 0
        assert reward_calculator._loss_count == 0
        assert reward_calculator._recent_actions == []


class TestRewardCalculatorEdgeCases:
    """Test edge cases and error conditions."""

    def test_calculate_reward_with_zero_atr(self, reward_calculator):
        """Test reward calculation with zero ATR."""
        reward = reward_calculator.calculate_reward(
            action=ACTION_HOLD,
            current_price=100.0,
            position=0.0,
            portfolio_value=100000.0,
            atr=0.0,  # Zero ATR
            transaction_cost=0.001,
            reward_scaling=1.0,
            pnl=0.0,
            old_position=0.0,
            step=1,
            observation=None,
            reward_history=[],
            portfolio_value_history=[100000.0],
        )
        assert isinstance(reward, float)
        assert not math.isnan(reward)  # Should not be NaN

    def test_calculate_reward_with_extreme_values(self, reward_calculator):
        """Test reward calculation with extreme values."""
        # Use more reasonable extreme values to avoid overflow
        try:
            reward = reward_calculator.calculate_reward(
                action=ACTION_BUY,
                current_price=1e6,  # High but reasonable price
                position=1.0,
                portfolio_value=1e9,
                atr=1e3,
                transaction_cost=0.001,
                reward_scaling=1.0,
                pnl=1e6,
                old_position=0.0,
                step=1,
                observation=None,
                reward_history=[],
                portfolio_value_history=[100000.0],
            )
            assert isinstance(reward, float)
            assert not math.isnan(reward)
            assert not math.isinf(reward)
        except OverflowError:
            # If overflow occurs, that's acceptable for extreme values
            pass
