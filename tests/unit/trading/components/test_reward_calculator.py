"""Tests for RewardCalculator component."""

import math

import pytest

from ztb.trading.constants import ACTION_BUY, ACTION_HOLD, ACTION_SELL
from ztb.trading.environment.components.reward_calculator import RewardCalculator
from ztb.trading.environment.utils.config import RewardSettings


@pytest.fixture
def sample_config():
    """Sample environment config for testing."""
    from ztb.trading.environment.utils.config import EnvironmentConfig

    config_dict = {
        "max_position_size": 1.0,
        "transaction_cost": 0.001,
        "exchange": "coincheck",
        "reward_scaling": 1.0,
        "action_space_type": "continuous",
        "use_continuous_actions": True,
        "feature_set": "minimal",
        "enable_action_masking": True,
        "use_standardized_observations": True,
        "random_start": True,
        "continuous_to_discrete_threshold": 0.08,
        "behavior_optimization": {
            "action_balance_target": 0.333,
            "entropy_regularization": 0.01,
            "action_smoothing": 0.1,
            "consistency_penalty": 0.05,
            "balance_penalty": 0.1,
            "redundant_trade_penalty": 5.0,
            'balance_penalty_min_actions': 1,
        },
        "action_bonuses": {
            "buy_action_bonus": 0.0,
            "sell_action_bonus": 0.0,
            "hold_action_bonus": 0.0,
        },
        "base_action_penalty": 0.015,
    }
    return EnvironmentConfig.from_dict(config_dict)


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


class TestRewardCalculatorBalancePenalty:
    """Test balance penalty functionality."""

    def test_balance_penalty_applied_to_actions(self, reward_calculator):
        """Test that balance_penalty is correctly applied to BUY/SELL actions."""
        # Test BUY action with balance_penalty
        reward_buy = reward_calculator.calculate_reward(
            action=ACTION_BUY,
            current_price=100.0,
            position=0.0,
            portfolio_value=100000.0,
            atr=1.0,
            transaction_cost=0.001,
            reward_scaling=1.0,
            pnl=0.0,
            old_position=0.0,
            step=1,
            observation=None,
            reward_history=[],
            portfolio_value_history=[100000.0],
        )

        # Test SELL action with balance_penalty
        reward_sell = reward_calculator.calculate_reward(
            action=ACTION_SELL,
            current_price=100.0,
            position=0.0,
            portfolio_value=100000.0,
            atr=1.0,
            transaction_cost=0.001,
            reward_scaling=1.0,
            pnl=0.0,
            old_position=0.0,
            step=1,
            observation=None,
            reward_history=[],
            portfolio_value_history=[100000.0],
        )

        # Test HOLD action (should not have balance_penalty)
        reward_hold = reward_calculator.calculate_reward(
            action=ACTION_HOLD,
            current_price=100.0,
            position=0.0,
            portfolio_value=100000.0,
            atr=1.0,
            transaction_cost=0.001,
            reward_scaling=1.0,
            pnl=0.0,
            old_position=0.0,
            step=1,
            observation=None,
            reward_history=[],
            portfolio_value_history=[100000.0],
        )

        # BUY and SELL should have balance_penalty applied (negative impact)
        # HOLD should not have balance_penalty
        assert (
            reward_buy < reward_hold
        ), f"BUY reward {reward_buy} should be less than HOLD {reward_hold} due to balance_penalty"
        assert (
            reward_sell < reward_hold
        ), f"SELL reward {reward_sell} should be less than HOLD {reward_hold} due to balance_penalty"

        # Verify penalty values are reasonable
        balance_penalty = reward_calculator.reward_settings.balance_penalty  # 0.1
        assert balance_penalty > 0, "balance_penalty should be positive"

        # The penalty should be applied as negative reward component
        buy_penalty_impact = reward_hold - reward_buy
        sell_penalty_impact = reward_hold - reward_sell

        assert (
            buy_penalty_impact > 0
        ), f"BUY should have penalty impact: {buy_penalty_impact}"
        assert (
            sell_penalty_impact > 0
        ), f"SELL should have penalty impact: {sell_penalty_impact}"


def test_mtf_weights_present_in_last_components(reward_calculator):
    """Ensure mtf_weights telemetry is present after reward calc."""
    reward = reward_calculator.calculate_reward(
        action=ACTION_HOLD,
        current_price=100.0,
        position=0.0,
        portfolio_value=100000.0,
        atr=1.0,
        transaction_cost=0.001,
        reward_scaling=1.0,
        pnl=0.0,
        old_position=0.0,
        step=1,
        observation=None,
        reward_history=[],
        portfolio_value_history=[100000.0],
    )
    components = reward_calculator.get_last_reward_components()
    assert "mtf_weights" in components

    def test_balance_penalty_with_action_bonuses(self, reward_calculator):
        """Test balance_penalty interaction with action bonuses."""
        # Create reward settings with action bonuses
        bonus_settings = RewardSettings(
            use_simple_reward=False,
            reward_scale=100.0,
            trading_bonus=0.01,
            profit_bonuses={"base": 1.5, "ultra": 2.0},
            penalty_coefficients={"loss": 2.0, "position": 0.01, "stagnation": 0.001},
            entropy_bonus=0.0,
            custom_reward_params={
                "buy_action_bonus": 10.0,
                "sell_action_bonus": 5.0,
                "hold_action_bonus": 2.0,
            },
            balance_penalty=200.0,  # High penalty
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

        # Create calculator with bonus settings
        bonus_calculator = RewardCalculator(
            config=reward_calculator.config,
            reward_settings=bonus_settings,
            initial_portfolio_value=100000.0,
        )

        # Test rewards with high balance_penalty and bonuses
        reward_buy_bonus = bonus_calculator.calculate_reward(
            action=ACTION_BUY,
            current_price=100.0,
            position=0.0,
            portfolio_value=100000.0,
            atr=1.0,
            transaction_cost=0.001,
            reward_scaling=1.0,
            pnl=0.0,
            old_position=0.0,
            step=1,
            observation=None,
            reward_history=[],
            portfolio_value_history=[100000.0],
        )

        reward_sell_bonus = bonus_calculator.calculate_reward(
            action=ACTION_SELL,
            current_price=100.0,
            position=0.0,
            portfolio_value=100000.0,
            atr=1.0,
            transaction_cost=0.001,
            reward_scaling=1.0,
            pnl=0.0,
            old_position=0.0,
            step=1,
            observation=None,
            reward_history=[],
            portfolio_value_history=[100000.0],
        )

        reward_hold_bonus = bonus_calculator.calculate_reward(
            action=ACTION_HOLD,
            current_price=100.0,
            position=0.0,
            portfolio_value=100000.0,
            atr=1.0,
            transaction_cost=0.001,
            reward_scaling=1.0,
            pnl=0.0,
            old_position=0.0,
            step=1,
            observation=None,
            reward_history=[],
            portfolio_value_history=[100000.0],
        )

        # With bonuses: BUY should be better than SELL due to higher bonus (10.0 vs 5.0)
        # But both should be worse than HOLD due to high balance_penalty (200.0)
        assert (
            reward_buy_bonus < reward_hold_bonus
        ), f"BUY with bonus {reward_buy_bonus} should be < HOLD {reward_hold_bonus}"
        assert (
            reward_sell_bonus < reward_hold_bonus
        ), f"SELL with bonus {reward_sell_bonus} should be < HOLD {reward_hold_bonus}"

        # BUY should be better than SELL due to higher bonus
        assert (
            reward_buy_bonus > reward_sell_bonus
        ), f"BUY bonus {reward_buy_bonus} should be > SELL bonus {reward_sell_bonus}"

        print(f"BUY reward with bonus: {reward_buy_bonus}")
        print(f"SELL reward with bonus: {reward_sell_bonus}")
        print(f"HOLD reward with bonus: {reward_hold_bonus}")
        print(f"BUY bonus advantage: {reward_buy_bonus - reward_sell_bonus}")


class TestRewardCalculatorComplex:
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

    def test_trend_integration_in_forced_balance(self, reward_calculator):
        """When trend favors BUY, BUY actions should be penalized less in forced_balance stage."""

        # Ensure forced_balance stage triggers by lowering min actions
        reward_calculator.reward_settings.custom_reward_params = {
            "forced_balance_min_actions": 1,
            "forced_balance_exploration_reward": 0.0,
        }

        # Create a stub TrendDetector
        class StubTrendDetector:
            def __init__(self, signal):
                self._signal = signal

            def get_trend_signal(self):
                return self._signal

            def update(self, price):
                pass

            def get_statistics(self):
                return {"samples": 1, "last_signal": self._signal}

        # Set up balanced counts that would produce a penalty for BUY
        reward_calculator._action_counts = [0, 8, 0]

        # Prepare two calculators: one with positive trend favoring BUY, one with negative
        reward_calculator.behavioral_penalty_calculator.trend_detector = (
            StubTrendDetector(0.6)
        )
        pos_trend_reward = reward_calculator._calculate_forced_balance_reward(
            action=ACTION_BUY, step=10
        )

        reward_calculator.behavioral_penalty_calculator.trend_detector = (
            StubTrendDetector(-0.6)
        )
        neg_trend_reward = reward_calculator._calculate_forced_balance_reward(
            action=ACTION_BUY, step=10
        )

        # With positive trend favoring BUY, the penalty should be smaller (higher reward)
        assert (
            pos_trend_reward > neg_trend_reward
        ), "Positive trend should reduce BUY penalty compared to negative trend"

    def test_reset_resets_trend_detector(self, reward_calculator):
        """Resetting RewardCalculator should reset TrendDetector state."""
        # Ensure trend detector exists
        td = reward_calculator.behavioral_penalty_calculator.trend_detector
        assert td is not None
        # Simulate updates
        td.update(100.0)
        td.update(101.0)
        assert td.update_count >= 1
        reward_calculator.reset()
        assert td.update_count == 0


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


def test_record_action_builds_recent_actions_from_action_counts(reward_calculator):
    """If RewardCalculator._action_counts is preset and recent_actions is empty, _record_action should build the deque from counts and then record the incoming action."""
    from collections import deque

    # Ensure recent_actions is empty first
    reward_calculator.behavioral_penalty_calculator.recent_actions = deque()

    # Preset counts: HOLD=0, BUY=2, SELL=1
    reward_calculator._action_counts = [0, 2, 1]

    # Record one BUY action
    reward_calculator._record_action(ACTION_BUY)

    # After recording, action_counts should reflect the added action
    assert reward_calculator._action_counts[1] == 3
    assert reward_calculator._action_counts[2] == 1
    # Recent actions deque should contain the expected counts (at least the built + recorded action)
    recent = list(reward_calculator.behavioral_penalty_calculator.recent_actions)
    assert recent.count(ACTION_BUY) >= 3
    assert recent.count(ACTION_SELL) >= 1


def test_record_action_clears_recent_actions_when_counts_zero(reward_calculator):
    """If _action_counts is zero but recent_actions contains items, _record_action should clear it before recording current action."""
    from collections import deque

    # Populate recent_actions with some history
    reward_calculator.behavioral_penalty_calculator.recent_actions = deque([ACTION_BUY, ACTION_SELL, ACTION_BUY])

    # Ensure we start with non-zero recent history
    assert len(reward_calculator.behavioral_penalty_calculator.recent_actions) > 0

    # Set action counts to zero
    reward_calculator._action_counts = [0, 0, 0]

    # Record a HOLD action
    reward_calculator._record_action(ACTION_HOLD)

    # After recording, recent should have been cleared and then include only the HOLD (or at least one HOLD)
    recent = list(reward_calculator.behavioral_penalty_calculator.recent_actions)
    assert recent.count(ACTION_HOLD) >= 1
    # And _action_counts should reflect the HOLD recorded
    assert reward_calculator._action_counts[0] >= 1

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
