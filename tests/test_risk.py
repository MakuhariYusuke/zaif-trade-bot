"""Unit tests for risk management module."""

import pytest
from unittest.mock import Mock
from ztb.risk.profiles import get_risk_profile, create_custom_risk_profile
from ztb.risk.rules import RiskRuleEngine
from ztb.risk.checks import RiskChecker
from ztb.risk import RiskManager


class TestRiskProfiles:
    """Test risk profile functionality."""

    def test_get_conservative_profile(self):
        """Test conservative risk profile."""
        profile = get_risk_profile("conservative")

        assert profile.max_position_notional == 50000
        assert profile.daily_loss_limit_pct == 0.02
        assert profile.max_drawdown_pct == 0.05
        assert profile.max_trades_per_hour == 2

    def test_get_balanced_profile(self):
        """Test balanced risk profile."""
        profile = get_risk_profile("balanced")

        assert profile.max_position_notional == 100000
        assert profile.daily_loss_limit_pct == 0.05
        assert profile.max_drawdown_pct == 0.10
        assert profile.max_trades_per_hour == 5

    def test_get_aggressive_profile(self):
        """Test aggressive risk profile."""
        profile = get_risk_profile("aggressive")

        assert profile.max_position_notional == 200000
        assert profile.daily_loss_limit_pct == 0.10
        assert profile.max_drawdown_pct == 0.20
        assert profile.max_trades_per_hour == 10

    def test_invalid_profile_name(self):
        """Test invalid profile name raises error."""
        with pytest.raises(ValueError, match="Unknown risk profile"):
            get_risk_profile("invalid")

    def test_create_custom_profile(self):
        """Test custom risk profile creation."""
        custom = create_custom_risk_profile(
            max_position_notional=75000,
            daily_loss_limit_pct=0.03,
            max_trades_per_hour=3
        )

        assert custom.max_position_notional == 75000
        assert custom.daily_loss_limit_pct == 0.03
        assert custom.max_trades_per_hour == 3


class TestRiskRuleEngine:
    """Test risk rule engine functionality."""

    def setup_method(self):
        """Set up test fixtures."""
        self.profile = get_risk_profile("balanced")
        self.engine = RiskRuleEngine(self.profile)

    def test_initialization(self):
        """Test engine initializes correctly."""
        assert self.engine.limits == self.profile
        assert hasattr(self.engine, 'check_daily_loss_limit')
        assert hasattr(self.engine, 'check_position_size')

    def test_check_position_limits_pass(self):
        """Test position limits check passes."""
        result, message = self.engine.check_position_size(
            position_notional=50000
        )

        assert result is True
        assert message == ""

    def test_check_position_limits_fail(self):
        """Test position limits check fails."""
        result, message = self.engine.check_position_size(
            position_notional=120000  # Above 100k limit
        )

        assert result is False
        assert "Position size exceeds limit" in message

    def test_check_daily_loss_limit_pass(self):
        """Test daily loss limit check passes."""
        # Set current loss below limit
        self.engine.daily_loss = 0.03  # 3% loss
        self.engine.daily_start_capital = 1.0  # 100% starting capital

        result, message = self.engine.check_daily_loss_limit()

        assert result is True
        # Message is empty when within limits
        assert message == ""

    def test_check_daily_loss_limit_fail(self):
        """Test daily loss limit check fails."""
        # Set current loss above limit
        self.engine.daily_loss = 0.06  # 6% loss
        self.engine.daily_start_capital = 1.0  # 100% starting capital

        result, message = self.engine.check_daily_loss_limit()

        assert result is False
        assert "Daily loss limit exceeded" in message

    def test_check_trade_frequency_pass(self):
        """Test trade frequency check passes."""
        # Reset trade count
        self.engine.trades_this_hour = 3  # Below 5 limit

        result, message = self.engine.check_trade_frequency()

        assert result is True
        # Message is empty when within limits
        assert message == ""

    def test_check_trade_frequency_fail(self):
        """Test trade frequency check fails."""
        # Set trade count above limit
        self.engine.trades_this_hour = 6  # Above 5 limit

        result, message = self.engine.check_trade_frequency()

        assert result is False
        assert "Trade frequency limit exceeded" in message

    def test_check_drawdown_limit_pass(self):
        """Test drawdown limit check passes."""
        self.engine.portfolio_value = 100000  # Same as peak, no drawdown
        result, message = self.engine.check_max_drawdown(
            peak_value=100000
        )

        assert result is True
        assert message == ""

    def test_check_drawdown_limit_fail(self):
        """Test drawdown limit check fails."""
        # Set current portfolio value to 85k with peak of 100k = 15% drawdown > 10% limit
        self.engine.portfolio_value = 85000
        result, message = self.engine.check_max_drawdown(
            peak_value=100000
        )

        assert result is False
        assert "Max drawdown exceeded" in message


class TestRiskChecker:
    """Test risk checker functionality."""

    def setup_method(self):
        """Set up test fixtures."""
        self.profile = get_risk_profile("conservative")
        self.checker = RiskChecker(self.profile)

    def test_initialization(self):
        """Test checker initializes correctly."""
        assert self.checker.engine.limits == self.profile
        assert hasattr(self.checker, 'pre_trade_check')

    def test_validate_trade_pass(self):
        """Test trade validation passes."""
        self.checker.engine.portfolio_value = 52000  # Same as peak, no drawdown
        result, message = self.checker.pre_trade_check(
            trade_notional=1000,  # 1000/52000 = 0.0192 < 0.02
            position_notional=20000,
            peak_value=52000
        )

        assert result is True
        assert message == ""

    def test_validate_trade_fail_position(self):
        """Test trade validation fails on position limit."""
        self.checker.engine.portfolio_value = 52000
        result, message = self.checker.pre_trade_check(
            trade_notional=1000,
            position_notional=60000,  # 60k > 50k limit
            peak_value=52000
        )

        assert result is False
        assert "Position size exceeds limit" in message

    def test_validate_trade_fail_drawdown(self):
        """Test trade validation fails on drawdown."""
        # Set portfolio value for drawdown calculation
        self.checker.engine.portfolio_value = 30000  # Current value 30k
        result, message = self.checker.pre_trade_check(
            trade_notional=10000,
            position_notional=20000,
            peak_value=55000  # Drawdown = (55k-30k)/55k = 45% > 5%
        )

        assert result is False
        assert "Max drawdown exceeded" in message


class TestRiskManager:
    """Test risk manager functionality."""

    def setup_method(self):
        """Set up test fixtures."""
        self.manager = RiskManager(profile_name="balanced")

    def test_initialization(self):
        """Test manager initializes correctly."""
        assert self.manager.limits is not None
        assert hasattr(self.manager, 'validate_and_execute_trade')
        assert hasattr(self.manager, 'get_status_report')

    def test_validate_and_execute_trade_approved(self):
        """Test trade validation and execution when approved."""
        mock_trade_func = Mock(return_value="TRADE_EXECUTED")

        # Set portfolio state for validation
        self.manager.checker.engine.portfolio_value = 100000

        allowed, result, message = self.manager.validate_and_execute_trade(
            trade_func=mock_trade_func,
            trade_notional=1000,  # 1000/100000 = 0.01 < 0.02
            position_notional=30000,
            peak_value=105000,
            symbol='BTC_JPY',
            side='buy',
            quantity=0.001
        )

        assert allowed is True
        assert result == "TRADE_EXECUTED"
        assert "Trade executed successfully" in message
        mock_trade_func.assert_called_once()

    def test_validate_and_execute_trade_rejected(self):
        """Test trade validation and execution when rejected."""
        mock_trade_func = Mock()

        # Set portfolio state to trigger rejection (high drawdown)
        self.manager.checker.engine.portfolio_value = 1000  # Very low current value
        self.manager.checker.engine.daily_start_capital = 1000

        allowed, result, message = self.manager.validate_and_execute_trade(
            trade_func=mock_trade_func,
            trade_notional=60000,
            position_notional=60000,  # Exceeds position limit
            peak_value=105000,
            symbol='BTC_JPY',
            side='buy',
            quantity=0.001
        )

        assert allowed is False
        assert result is None
        assert "Max drawdown exceeded" in message  # Drawdown check fails first
        mock_trade_func.assert_not_called()

    def test_get_status_report(self):
        """Test status report generation."""
        # Set portfolio state
        self.manager.checker.engine.portfolio_value = 100000

        # Execute a trade to change state
        mock_trade_func = Mock(return_value="TRADE_EXECUTED")
        self.manager.validate_and_execute_trade(
            trade_func=mock_trade_func,
            trade_notional=20000,
            position_notional=0,
            peak_value=100000,
            symbol='BTC_JPY',
            side='buy',
            quantity=0.001
        )

        status = self.manager.get_status_report()

        assert 'profile' in status
        assert 'current_status' in status
        assert 'limits' in status
        assert hasattr(status['profile'], 'max_position_notional')  # profile is RiskLimits object
        assert 'trades_this_hour' in status['current_status']


if __name__ == '__main__':
    pytest.main([__file__])