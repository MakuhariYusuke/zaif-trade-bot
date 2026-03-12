"""Integration tests for risk management components (renamed to avoid collection clashes)."""

from unittest.mock import Mock

import pytest

from ztb.risk.checks import RiskChecker
from ztb.risk.profiles import RiskLimits
from ztb.risk.rules import RiskRuleEngine


@pytest.fixture
def integrated_risk_system():
    """Complete risk management system for integration testing."""
    # Create risk limits
    limits = RiskLimits(
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

    # Create shared engine instance
    engine = RiskRuleEngine(limits)

    # Create checker with shared engine
    checker = RiskChecker(limits)
    checker.engine = engine  # Override with shared engine

    manager = Mock()  # Mock for integration focus

    return {"engine": engine, "checker": checker, "manager": manager, "limits": limits}


class TestRiskSystemIntegration:
    """Integration tests for the complete risk management system."""

    def test_full_trade_workflow_allowed(self, integrated_risk_system):
        system = integrated_risk_system
        engine = system["engine"]
        checker = system["checker"]

        engine.portfolio_value = 100000.0
        engine.update_portfolio_state(100000.0, 0.08)

        allowed, reason = checker.pre_trade_check(
            trade_notional=3000.0,
            position_notional=50000.0,
            peak_value=105000.0,
            sharpe_ratio=1.2,
        )

        assert allowed is True
        assert reason == ""

    def test_full_trade_workflow_blocked(self, integrated_risk_system):
        system = integrated_risk_system
        engine = system["engine"]
        checker = system["checker"]

        engine.portfolio_value = 100000.0
        engine.update_portfolio_state(100000.0, 0.20)

        allowed, reason = checker.pre_trade_check(
            trade_notional=3000.0,
            position_notional=50000.0,
            peak_value=105000.0,
            sharpe_ratio=0.5,
        )

        assert allowed is False
        assert (
            "Portfolio volatility exceeds limit" in reason
            or "Sharpe ratio below threshold" in reason
        )

    def test_state_consistency_across_components(self, integrated_risk_system):
        system = integrated_risk_system
        engine = system["engine"]
        checker = system["checker"]

        initial_value = 100000.0
        engine.portfolio_value = initial_value

        assert checker.engine.portfolio_value == initial_value

        new_value = 95000.0
        engine.update_portfolio_state(new_value, 0.1)

        assert checker.engine.portfolio_value == new_value
        assert checker.engine.portfolio_volatility == 0.1

    def test_trade_recording_integration(self, integrated_risk_system):
        system = integrated_risk_system
        engine = system["engine"]

        for i in range(3):
            engine.record_trade({"pnl": -500.0})
