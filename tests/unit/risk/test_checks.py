"""Tests for RiskChecker and RiskManager components."""

from unittest.mock import Mock, patch

import pytest

from ztb.risk.checks import RiskChecker, RiskManager
from ztb.risk.profiles import RiskLimits


@pytest.fixture
def sample_risk_limits():
    """Sample risk limits for testing."""
    return RiskLimits(
        max_position_notional=100000.0,
        max_single_trade_pct=0.05,
        daily_loss_limit_pct=0.05,
        max_drawdown_pct=0.10,
        max_trades_per_hour=5,
        min_trade_interval_sec=600,
        max_volatility_pct=0.15,
        required_sharpe_ratio=1.0,
        stop_loss_pct=0.03,
        take_profit_pct=0.08,
    )


@pytest.fixture
def risk_checker(sample_risk_limits):
    """RiskChecker instance for testing."""
    return RiskChecker(sample_risk_limits)


@pytest.fixture
def risk_manager():
    """RiskManager instance for testing."""
    with patch("ztb.risk.profiles.get_risk_profile") as mock_get_profile:
        mock_limits = Mock()
        mock_limits.max_position_notional = 100000.0
        mock_limits.max_single_trade_pct = 0.05
        mock_limits.daily_loss_limit_pct = 0.05
        mock_limits.max_drawdown_pct = 0.10
        mock_limits.max_trades_per_hour = 5
        mock_limits.min_trade_interval_sec = 600
        mock_limits.max_volatility_pct = 0.15
        mock_limits.required_sharpe_ratio = 1.0
        mock_limits.stop_loss_pct = 0.03
        mock_limits.take_profit_pct = 0.08
        mock_get_profile.return_value = mock_limits
        manager = RiskManager("balanced")
        return manager


class TestRiskCheckerInitialization:
    """Test RiskChecker initialization."""

    def test_initialization(self, risk_checker, sample_risk_limits):
        """Test proper initialization."""
        assert hasattr(risk_checker, "engine")
        assert risk_checker.engine.limits == sample_risk_limits


class TestRiskCheckerPreTradeCheck:
    """Test pre-trade risk validation."""

    def test_pre_trade_check_allowed(self, risk_checker):
        """
        Test pre-trade risk validation when all conditions are met.

        Scenario: Normal market conditions with acceptable position size and performance metrics
        Expected: Trade should be approved with no blocking reason
        """
        # Set up engine state
        risk_checker.engine.portfolio_value = 100000.0
        risk_checker.engine.update_portfolio_state(100000.0, 0.1)

        allowed, reason = risk_checker.pre_trade_check(
            trade_notional=3000.0,
            position_notional=50000.0,
            peak_value=105000.0,
            sharpe_ratio=1.2,
        )

        assert allowed is True
        assert reason == ""

    def test_pre_trade_check_blocked(self, risk_checker):
        """
        Test pre-trade risk validation when position size exceeds limit.

        Scenario: Attempted trade would result in position exceeding max_position_notional
        Expected: Trade should be blocked with clear rejection reason
        """
        # Set up engine state to exceed position limit
        risk_checker.engine.portfolio_value = 100000.0

        allowed, reason = risk_checker.pre_trade_check(
            trade_notional=1000.0,
            position_notional=120000.0,  # Exceeds limit
            peak_value=100000.0,
        )

        assert allowed is False
        assert "Position size exceeds limit" in reason

    def test_pre_trade_check_small_trade(self, risk_checker):
        """
        Test pre-trade check with very small trade amount.

        Scenario: Micro trade that should be allowed under normal conditions
        Expected: Small trades should pass all risk checks
        """
        # Set up engine state
        risk_checker.engine.portfolio_value = 100000.0
        risk_checker.engine.update_portfolio_state(100000.0, 0.1)

        allowed, reason = risk_checker.pre_trade_check(
            trade_notional=25.0,  # Very small trade
            position_notional=25000.0,
            peak_value=105000.0,
            sharpe_ratio=1.2,
        )

        assert allowed is True
        assert reason == ""

    def test_pre_trade_check_extreme_volatility(self, risk_checker):
        """
        Test pre-trade check under extreme market volatility.

        Scenario: High volatility conditions that may trigger risk limits
        Expected: Trade should be blocked due to excessive volatility
        """
        # Set up engine state with high volatility
        risk_checker.engine.portfolio_value = 100000.0
        risk_checker.engine.update_portfolio_state(100000.0, 0.20)  # High volatility

        allowed, reason = risk_checker.pre_trade_check(
            trade_notional=3000.0,
            position_notional=50000.0,
            peak_value=105000.0,
            sharpe_ratio=0.5,  # Poor Sharpe ratio
        )

        # Should be blocked due to volatility or Sharpe ratio
        assert allowed is False


class TestRiskCheckerPostTradeUpdate:
    """Test post-trade state updates."""

    def test_post_trade_update_basic(self, risk_checker):
        """Test basic post-trade update."""
        risk_checker.post_trade_update(
            current_value=95000.0,
            volatility=0.12,
        )

        assert risk_checker.engine.portfolio_value == 95000.0
        assert risk_checker.engine.portfolio_volatility == 0.12

    def test_post_trade_update_with_trade_data(self, risk_checker):
        """Test post-trade update with trade data."""
        trade_data = {"pnl": 1000.0, "size": 50000.0}

        risk_checker.post_trade_update(
            current_value=95000.0,
            volatility=0.12,
            trade_data=trade_data,
        )

        assert risk_checker.engine.portfolio_value == 95000.0
        assert risk_checker.engine.portfolio_volatility == 0.12
        assert len(risk_checker.engine.trade_history) == 1
        assert risk_checker.engine.trade_history[0]["pnl"] == 1000.0


class TestRiskCheckerTrailingStop:
    """Test trailing stop functionality."""

    def test_update_trailing_stop(self, risk_checker):
        """Test trailing stop updates."""
        risk_checker.update_trailing_stop(100.0, "long")

        assert risk_checker.engine.trailing_stop_level is not None

    def test_check_trailing_stop_not_hit(self, risk_checker):
        """Test trailing stop check when not hit."""
        risk_checker.update_trailing_stop(100.0, "long")

        allowed, reason = risk_checker.check_trailing_stop(98.0, "long")

        assert allowed is True
        assert reason == ""

    def test_check_trailing_stop_hit(self, risk_checker):
        """Test trailing stop check when hit."""
        risk_checker.update_trailing_stop(100.0, "long")

        allowed, reason = risk_checker.check_trailing_stop(96.0, "long")

        assert allowed is False
        assert "Trailing stop hit" in reason


class TestRiskCheckerTakeProfit:
    """Test take profit functionality."""

    def test_check_take_profit_not_reached(self, risk_checker):
        """Test take profit check when not reached."""
        allowed, reason = risk_checker.check_take_profit(100.0, 105.0, "long")

        assert allowed is True
        assert reason == ""

    def test_check_take_profit_reached(self, risk_checker):
        """Test take profit check when reached."""
        allowed, reason = risk_checker.check_take_profit(100.0, 109.0, "long")

        assert allowed is False
        assert "Take profit target reached" in reason


class TestRiskCheckerRiskStatus:
    """Test risk status reporting."""

    def test_get_risk_status(self, risk_checker):
        """Test risk status retrieval."""
        # Set up some state
        risk_checker.engine.portfolio_value = 95000.0
        risk_checker.engine.portfolio_volatility = 0.12
        risk_checker.engine.daily_loss = 5000.0
        risk_checker.engine.trades_this_hour = 3

        status = risk_checker.get_risk_status()

        assert status["daily_loss"] == 5000.0
        assert status["portfolio_value"] == 95000.0
        assert status["portfolio_volatility"] == 0.12
        assert status["trades_this_hour"] == 3
        assert status["trailing_stop_level"] is None  # Not set


class TestRiskManagerInitialization:
    """Test RiskManager initialization."""

    def test_initialization_with_profile(self, risk_manager):
        """Test initialization with risk profile."""
        assert hasattr(risk_manager, "limits")
        assert hasattr(risk_manager, "checker")
        assert risk_manager.on_risk_violation is None
        assert risk_manager.on_trade_blocked is None

    def test_initialization_callbacks(self, risk_manager):
        """Test callback setup."""

        def risk_callback(msg):
            pass

        def trade_callback(msg):
            pass

        risk_manager.on_risk_violation = risk_callback
        risk_manager.on_trade_blocked = trade_callback

        assert risk_manager.on_risk_violation == risk_callback
        assert risk_manager.on_trade_blocked == trade_callback


class TestRiskManagerValidateAndExecuteTrade:
    """Test trade validation and execution."""

    def test_validate_and_execute_trade_allowed(self, risk_manager):
        """Test successful trade validation and execution."""
        # Mock the checker to allow trade
        risk_manager.checker.pre_trade_check = Mock(return_value=(True, ""))

        def mock_trade_func(amount, price):
            return {"executed": True, "amount": amount, "price": price}

        success, result, message = risk_manager.validate_and_execute_trade(
            trade_func=mock_trade_func,
            trade_notional=3000.0,
            position_notional=50000.0,
            peak_value=105000.0,
            amount=100,
            price=100.0,
        )

        assert success is True
        assert result == {"executed": True, "amount": 100, "price": 100.0}
        assert message == "Trade executed successfully"

    def test_validate_and_execute_trade_blocked(self, risk_manager):
        """Test trade blocking due to risk violation."""
        # Mock the checker to block trade
        risk_manager.checker.pre_trade_check = Mock(
            return_value=(False, "Risk violation")
        )

        def mock_trade_func():
            return {"executed": False}

        # Mock callback
        callback_called = []
        risk_manager.on_trade_blocked = lambda msg: callback_called.append(msg)

        success, result, message = risk_manager.validate_and_execute_trade(
            trade_func=mock_trade_func,
            trade_notional=3000.0,
            position_notional=120000.0,  # Would exceed limit
            peak_value=105000.0,
        )

        assert success is False
        assert result is None
        assert message == "Risk violation"
        assert callback_called == ["Risk violation"]

    def test_validate_and_execute_trade_execution_failure(self, risk_manager):
        """Test trade execution failure."""
        # Mock the checker to allow trade
        risk_manager.checker.pre_trade_check = Mock(return_value=(True, ""))

        def failing_trade_func():
            raise ValueError("Execution failed")

        # Mock callback
        callback_called = []
        risk_manager.on_risk_violation = lambda msg: callback_called.append(msg)

        success, result, message = risk_manager.validate_and_execute_trade(
            trade_func=failing_trade_func,
            trade_notional=3000.0,
            position_notional=50000.0,
            peak_value=105000.0,
        )

        assert success is False
        assert result is None
        assert "Trade execution failed" in message
        assert len(callback_called) == 1
        assert "Execution failed" in callback_called[0]


class TestRiskManagerMonitorPosition:
    """Test position monitoring."""

    def test_monitor_position_no_triggers(self, risk_manager):
        """Test position monitoring with no triggers."""
        # Mock checker methods
        risk_manager.checker.check_trailing_stop = Mock(return_value=(True, ""))
        risk_manager.checker.check_take_profit = Mock(return_value=(True, ""))

        triggers = risk_manager.monitor_position(
            current_price=105.0,
            entry_price=100.0,
            position_side="long",
        )

        assert triggers["trailing_stop"]["triggered"] is False
        assert triggers["take_profit"]["triggered"] is False

    def test_monitor_position_with_triggers(self, risk_manager):
        """Test position monitoring with triggers."""
        # Mock checker methods to trigger stops
        risk_manager.checker.check_trailing_stop = Mock(
            return_value=(False, "Stop hit")
        )
        risk_manager.checker.check_take_profit = Mock(
            return_value=(False, "Profit target reached")
        )

        triggers = risk_manager.monitor_position(
            current_price=109.0,
            entry_price=100.0,
            position_side="long",
        )

        assert triggers["trailing_stop"]["triggered"] is True
        assert triggers["trailing_stop"]["reason"] == "Stop hit"
        assert triggers["take_profit"]["triggered"] is True
        assert triggers["take_profit"]["reason"] == "Profit target reached"


class TestRiskManagerStatusReport:
    """Test status reporting."""

    def test_get_status_report(self, risk_manager):
        """Test comprehensive status report."""
        # Mock the checker status
        risk_manager.checker.get_risk_status = Mock(
            return_value={
                "daily_loss": 5000.0,
                "portfolio_value": 95000.0,
                "trades_this_hour": 3,
            }
        )

        report = risk_manager.get_status_report()

        assert "profile" in report
        assert "current_status" in report
        assert "limits" in report
        assert report["current_status"]["daily_loss"] == 5000.0
        assert report["current_status"]["portfolio_value"] == 95000.0
        assert report["current_status"]["trades_this_hour"] == 3

        # Check limits are included
        limits = report["limits"]
        assert "max_position_notional" in limits
        assert "max_single_trade_pct" in limits
        assert "daily_loss_limit_pct" in limits


class TestRiskManagerProfileVariations:
    """Test RiskManager with different risk profiles."""

    @pytest.mark.parametrize(
        "profile_name,expected_max_drawdown",
        [
            ("conservative", 0.05),  # Conservative: stricter limits
            ("balanced", 0.10),  # Balanced: moderate limits
            ("aggressive", 0.15),  # Aggressive: looser limits
        ],
    )
    def test_risk_manager_different_profiles(self, profile_name, expected_max_drawdown):
        """
        Test RiskManager initialization with different risk profiles.

        Verifies that different profiles load appropriate risk limits.
        """
        with patch("ztb.risk.profiles.get_risk_profile") as mock_get_profile:
            # Mock different profiles with varying limits
            mock_limits = Mock()
            if profile_name == "conservative":
                mock_limits.max_position_notional = 50000.0
                mock_limits.max_drawdown_pct = 0.05
                mock_limits.max_volatility_pct = 0.10
                mock_limits.required_sharpe_ratio = 1.5
            elif profile_name == "balanced":
                mock_limits.max_position_notional = 100000.0
                mock_limits.max_drawdown_pct = 0.10
                mock_limits.max_volatility_pct = 0.15
                mock_limits.required_sharpe_ratio = 1.0
            elif profile_name == "aggressive":
                mock_limits.max_position_notional = 200000.0
                mock_limits.max_drawdown_pct = 0.15
                mock_limits.max_volatility_pct = 0.20
                mock_limits.required_sharpe_ratio = 0.5

            mock_get_profile.return_value = mock_limits

            manager = RiskManager(profile_name)

            # Verify profile was loaded
            mock_get_profile.assert_called_once_with(profile_name)

            # Verify limits are set correctly
            assert hasattr(manager, "limits")
            assert manager.limits.max_drawdown_pct == expected_max_drawdown

    @pytest.mark.parametrize(
        "profile_name,max_position_size",
        [
            ("conservative", 50000.0),
            ("balanced", 100000.0),
            ("aggressive", 200000.0),
        ],
    )
    def test_position_limits_by_profile(self, profile_name, max_position_size):
        """
        Test that position size limits vary appropriately by risk profile.

        Conservative profiles should have smaller position limits.
        """
        with patch("ztb.risk.profiles.get_risk_profile") as mock_get_profile:
            mock_limits = Mock()
            mock_limits.max_position_notional = max_position_size
            mock_limits.max_single_trade_pct = 0.05
            mock_limits.daily_loss_limit_pct = 0.05
            mock_limits.max_drawdown_pct = 0.10
            mock_limits.max_trades_per_hour = 5
            mock_limits.min_trade_interval_sec = 600
            mock_limits.max_volatility_pct = 0.15
            mock_limits.required_sharpe_ratio = 1.0
            mock_limits.stop_loss_pct = 0.03
            mock_limits.take_profit_pct = 0.08

            mock_get_profile.return_value = mock_limits

            manager = RiskManager(profile_name)

            # Set up engine state to avoid other checks interfering
            manager.checker.engine.portfolio_value = max_position_size
            manager.checker.engine.update_portfolio_state(max_position_size, 0.05)

            # Test that the position limit is correctly applied
            test_position = max_position_size + 1000

            # This should be blocked for all profiles
            allowed, reason = manager.checker.pre_trade_check(
                trade_notional=1000.0,
                position_notional=test_position,
                peak_value=max_position_size,
            )

            assert allowed is False
            assert "Position size exceeds limit" in reason

    def test_invalid_profile_handling(self):
        """
        Test behavior when invalid risk profile is requested.

        Should handle gracefully or raise appropriate error.
        """
        with patch("ztb.risk.profiles.get_risk_profile") as mock_get_profile:
            # Simulate profile not found
            mock_get_profile.side_effect = ValueError("Invalid profile")

            with pytest.raises(ValueError, match="Invalid profile"):
                RiskManager("invalid_profile")
