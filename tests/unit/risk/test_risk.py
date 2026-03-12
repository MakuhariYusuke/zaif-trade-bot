"""
Unit tests for ztb.risk module.
"""

import pytest

from ztb.risk import (
    RiskChecker,
    RiskManager,
    RiskRuleEngine,
    create_custom_risk_profile,
    get_risk_profile,
)
from ztb.risk.profiles import RiskLimits


class TestRiskProfiles:
    """Test cases for risk profile functions."""

    def test_get_risk_profile_conservative(self):
        """Test getting conservative risk profile."""
        profile = get_risk_profile("conservative")

        assert isinstance(profile, RiskLimits)
        assert profile.max_position_notional == 50000.0
        assert profile.daily_loss_limit_pct == 0.02

    def test_get_risk_profile_balanced(self):
        """Test getting balanced risk profile."""
        profile = get_risk_profile("balanced")

        assert isinstance(profile, RiskLimits)
        assert profile.max_position_notional == 100000.0
        assert profile.daily_loss_limit_pct == 0.05

    def test_get_risk_profile_aggressive(self):
        """Test getting aggressive risk profile."""
        profile = get_risk_profile("aggressive")

        assert isinstance(profile, RiskLimits)
        assert profile.max_position_notional == 200000.0
        assert profile.daily_loss_limit_pct == 0.10

    def test_get_risk_profile_invalid(self):
        """Test getting invalid risk profile."""
        with pytest.raises(ValueError):
            get_risk_profile("invalid")

    def test_create_custom_risk_profile(self):
        """Test creating custom risk profile."""
        profile = create_custom_risk_profile(
            max_position_notional=75000.0,
            max_single_trade_pct=0.05,
            daily_loss_limit_pct=0.03,
            max_drawdown_pct=0.15,
            max_trades_per_hour=10,
            min_trade_interval_sec=30,
            max_volatility_pct=0.20,
            required_sharpe_ratio=1.0,
            stop_loss_pct=0.02,
            take_profit_pct=0.05,
        )

        assert isinstance(profile, RiskLimits)
        assert profile.max_position_notional == 75000.0
        assert profile.daily_loss_limit_pct == 0.03


class TestRiskRuleEngine:
    """Test cases for RiskRuleEngine."""

    def test_risk_rule_engine_init(self):
        """Test RiskRuleEngine initialization."""
        limits = get_risk_profile("conservative")
        engine = RiskRuleEngine(limits)

        assert engine.limits == limits

    def test_validate_trade_allowed(self):
        """Test trade validation when allowed."""
        limits = get_risk_profile("conservative")
        engine = RiskRuleEngine(limits)

        # Set portfolio state to avoid drawdown issues
        engine.update_portfolio_state(current_value=100000.0, volatility=0.1)

        allowed, reason = engine.validate_trade(
            trade_notional=1000.0,
            position_notional=10000.0,
            peak_value=100000.0,
        )

        assert allowed is True
        assert reason == ""

    def test_validate_trade_position_limit(self):
        """Test trade validation with position limit exceeded."""
        limits = get_risk_profile("conservative")
        engine = RiskRuleEngine(limits)

        # Set portfolio state
        engine.update_portfolio_state(current_value=100000.0, volatility=0.1)

        # Current position already exceeds max_position_notional (50000)
        allowed, reason = engine.validate_trade(
            trade_notional=1500.0,  # 1.5% of peak, within single trade limit
            position_notional=51000.0,  # Already exceeds 50000
            peak_value=100000.0,
        )

        assert allowed is False
        assert "position" in reason.lower()

    def test_validate_trade_single_trade_limit(self):
        """Test trade validation with single trade limit exceeded."""
        limits = get_risk_profile("conservative")
        engine = RiskRuleEngine(limits)

        # Set portfolio state
        engine.update_portfolio_state(current_value=100000.0, volatility=0.1)

        # Trade exceeds max_single_trade_pct (2%) of peak_value
        allowed, reason = engine.validate_trade(
            trade_notional=3000.0,  # 3% of 100000 > 2%
            position_notional=10000.0,
            peak_value=100000.0,
        )

        assert allowed is False
        assert "trade size" in reason.lower()


class TestRiskChecker:
    """Test cases for RiskChecker."""

    def test_risk_checker_init(self):
        """Test RiskChecker initialization."""
        limits = get_risk_profile("conservative")
        checker = RiskChecker(limits)

        assert isinstance(checker.engine, RiskRuleEngine)

    def test_pre_trade_check_allowed(self):
        """Test pre-trade check when allowed."""
        limits = get_risk_profile("conservative")
        checker = RiskChecker(limits)

        # Set portfolio state
        checker.post_trade_update(current_value=100000.0, volatility=0.1)

        allowed, reason = checker.pre_trade_check(
            trade_notional=1000.0,
            position_notional=10000.0,
            peak_value=100000.0,
        )

        assert allowed is True
        assert reason == ""

    def test_pre_trade_check_denied(self):
        """Test pre-trade check when denied."""
        limits = get_risk_profile("conservative")
        checker = RiskChecker(limits)

        # Set portfolio state
        checker.post_trade_update(current_value=100000.0, volatility=0.1)

        allowed, reason = checker.pre_trade_check(
            trade_notional=1500.0,  # Small trade within single limit
            position_notional=51000.0,  # Already exceeds position limit
            peak_value=100000.0,
        )

        assert allowed is False
        assert "position" in reason.lower()

    def test_post_trade_update(self):
        """Test post-trade update."""
        limits = get_risk_profile("conservative")
        checker = RiskChecker(limits)

        # Should not raise exception
        checker.post_trade_update(
            current_value=101000.0,
            volatility=0.15,
        )


class TestRiskManager:
    """Test cases for RiskManager."""

    def test_risk_manager_init(self):
        """Test RiskManager initialization."""
        manager = RiskManager("conservative")

        assert isinstance(manager.checker, RiskChecker)
        assert isinstance(manager.limits, RiskLimits)

    def test_validate_and_execute_trade_allowed(self):
        """Test trade validation and execution when allowed."""
        manager = RiskManager("conservative")

        # Set portfolio state
        manager.checker.post_trade_update(current_value=100000.0, volatility=0.1)

        def mock_trade_func():
            return "trade_executed"

        success, result, message = manager.validate_and_execute_trade(
            trade_func=mock_trade_func,
            trade_notional=1000.0,
            position_notional=10000.0,
            peak_value=100000.0,
        )

        assert success is True
        assert result == "trade_executed"
        assert "successfully" in message

    def test_validate_and_execute_trade_blocked(self):
        """Test trade validation and execution when blocked."""
        manager = RiskManager("conservative")

        # Set portfolio state
        manager.checker.post_trade_update(current_value=100000.0, volatility=0.1)

        def mock_trade_func():
            return "trade_executed"

        success, result, message = manager.validate_and_execute_trade(
            trade_func=mock_trade_func,
            trade_notional=1500.0,  # Small trade within single limit
            position_notional=51000.0,  # Already exceeds position limit
            peak_value=100000.0,
        )

        assert success is False
        assert result is None
        assert "position" in message.lower()

    def test_monitor_position(self):
        """Test position monitoring."""
        manager = RiskManager("conservative")

        status = manager.monitor_position(
            current_price=105.0,
            entry_price=100.0,
            position_side="long",
        )

        assert isinstance(status, dict)
        assert "trailing_stop" in status
        assert "take_profit" in status

    def test_get_status_report(self):
        """Test status report generation."""
        manager = RiskManager("conservative")

        report = manager.get_status_report()

        assert isinstance(report, dict)
        assert "profile" in report
        assert "current_status" in report
        assert "limits" in report
