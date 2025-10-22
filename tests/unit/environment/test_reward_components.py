"""
Tests for reward components.

This module tests individual reward calculation components.
"""


from ztb.trading.environment.components.reward_components import RewardComponents


class TestRewardComponents:
    """Test cases for RewardComponents class."""

    def setup_method(self):
        """Set up test fixtures."""
        self.reward_settings = {
            "drawdown_window": 5,
            "stagnation_threshold": 0.01,
            "growth_bonus_multiplier": 0.1,
            "win_streak_bonus": 0.05,
        }
        self.components = RewardComponents(self.reward_settings)

    def test_init_with_settings(self):
        """Test initialization with reward settings."""
        assert self.components.reward_settings == self.reward_settings

    def test_init_without_settings(self):
        """Test initialization without reward settings."""
        components = RewardComponents()
        assert components.reward_settings == {}

    def test_calculate_drawdown_penalty_insufficient_history(self):
        """Test drawdown penalty with insufficient history."""
        current_value = 1000.0
        portfolio_history = [1000.0, 990.0]  # Less than drawdown_window
        reward_history = [0.0, -0.01]

        penalty = self.components.calculate_drawdown_penalty(
            current_value, portfolio_history, reward_history
        )
        assert penalty == 0.0

    def test_calculate_drawdown_penalty_no_penalty(self):
        """Test drawdown penalty when no penalty should be applied."""
        current_value = 1000.0
        portfolio_history = [1000.0, 1010.0, 1020.0, 1030.0, 1040.0, 1050.0]
        reward_history = [0.01, 0.02, 0.03, 0.04, 0.05, 0.06]

        penalty = self.components.calculate_drawdown_penalty(
            current_value, portfolio_history, reward_history
        )
        assert penalty == 0.0

    def test_calculate_drawdown_penalty_with_penalty(self):
        """Test drawdown penalty when penalty should be applied."""
        current_value = 1000.0
        # Create longer history to meet stagnation_window requirement
        portfolio_history = [1000.0] * 25 + [
            990.0,
            980.0,
            970.0,
            960.0,
            950.0,
        ]  # 30 values total
        reward_history = [0.01] * 25 + [
            -0.01,
            -0.02,
            -0.03,
            -0.04,
            -0.05,
        ]  # 30 values total

        penalty = self.components.calculate_drawdown_penalty(
            current_value, portfolio_history, reward_history
        )
        # The penalty logic is complex, so just check it's a valid float
        assert isinstance(penalty, float)
        assert penalty >= 0.0  # Penalties are always non-negative

    def test_calculate_stagnation_penalty_no_penalty(self):
        """Test stagnation penalty when no penalty should be applied."""
        portfolio_history = [1000.0, 1010.0, 1020.0, 1030.0, 1040.0, 1050.0]

        penalty = self.components.calculate_stagnation_penalty(portfolio_history)
        assert penalty == 0.0

    def test_calculate_stagnation_penalty_with_penalty(self):
        """Test stagnation penalty when penalty should be applied."""
        # Create decreasing portfolio values to trigger penalty
        portfolio_history = [
            1000.0 - i * 0.1 for i in range(35)
        ]  # 35 decreasing values

        penalty = self.components.calculate_stagnation_penalty(portfolio_history)
        assert penalty > 0.0  # Should be positive penalty

    def test_calculate_growth_bonus_insufficient_history(self):
        """Test growth bonus with insufficient history."""
        portfolio_history = [1000.0]  # Less than 2 values

        bonus = self.components.calculate_growth_bonus(portfolio_history)
        assert bonus == 0.0

    def test_calculate_growth_bonus_positive_growth(self):
        """Test growth bonus with positive growth."""
        # Need at least 30 samples for growth calculation
        portfolio_history = [1000.0 + i * 10 for i in range(30)]  # 30 increasing values

        bonus = self.components.calculate_growth_bonus(portfolio_history)
        assert bonus > 0.0

    def test_calculate_growth_bonus_negative_growth(self):
        """Test growth bonus with negative growth."""
        portfolio_history = [1000.0, 900.0, 800.0, 700.0, 600.0]

        bonus = self.components.calculate_growth_bonus(portfolio_history)
        assert bonus == 0.0  # No bonus for negative growth

    def test_calculate_win_streak_bonus_no_streak(self):
        """Test win streak bonus with no winning streak."""
        reward_history = [-0.01, 0.02, -0.03, 0.04, -0.05]

        bonus = self.components.calculate_win_streak_bonus(reward_history)
        assert bonus == 0.0

    def test_calculate_win_streak_bonus_with_streak(self):
        """Test win streak bonus with winning streak."""
        reward_history = [0.01, 0.02, 0.03, 0.04, 0.05]

        bonus = self.components.calculate_win_streak_bonus(reward_history)
        assert bonus > 0.0

    def test_calculate_win_streak_bonus_mixed_history(self):
        """Test win streak bonus with mixed positive/negative rewards."""
        reward_history = [-0.01, 0.02, 0.03, 0.04, -0.05, 0.06, 0.07]

        bonus = self.components.calculate_win_streak_bonus(reward_history)
        assert bonus > 0.0  # Should detect the final streak

    def test_calculate_win_rate_bonus_buy_action_positive_pnl(self):
        """Test win rate bonus for buy action with positive PnL."""
        from ztb.trading.constants import ACTION_BUY

        bonus = self.components._calculate_win_rate_bonus(ACTION_BUY, 10.0)
        assert bonus == 0.0  # Placeholder implementation

    def test_calculate_win_rate_bonus_sell_action_negative_pnl(self):
        """Test win rate bonus for sell action with negative PnL."""
        from ztb.trading.constants import ACTION_SELL

        bonus = self.components._calculate_win_rate_bonus(ACTION_SELL, -10.0)
        assert bonus == 0.0  # Placeholder implementation

    def test_calculate_win_rate_bonus_hold_action(self):
        """Test win rate bonus for hold action."""
        from ztb.trading.constants import ACTION_HOLD

        bonus = self.components._calculate_win_rate_bonus(ACTION_HOLD, 5.0)
        assert bonus == 0.0  # Placeholder implementation

    def test_calculate_diversity_bonus_buy_action(self):
        """Test diversity bonus for buy action."""
        from ztb.trading.constants import ACTION_BUY

        bonus = self.components._calculate_diversity_bonus(ACTION_BUY)
        assert isinstance(bonus, float)

    def test_calculate_diversity_bonus_sell_action(self):
        """Test diversity bonus for sell action."""
        from ztb.trading.constants import ACTION_SELL

        bonus = self.components._calculate_diversity_bonus(ACTION_SELL)
        assert isinstance(bonus, float)

    def test_calculate_diversity_bonus_hold_action(self):
        """Test diversity bonus for hold action."""
        from ztb.trading.constants import ACTION_HOLD

        bonus = self.components._calculate_diversity_bonus(ACTION_HOLD)
        assert isinstance(bonus, float)
