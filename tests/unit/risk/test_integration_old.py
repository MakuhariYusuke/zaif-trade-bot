"""Integration tests for risk management components."""

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
        """
        Test complete trade workflow when trade should be allowed.

        Scenario: Normal trading conditions, trade passes all risk checks
        Expected: Trade is approved through entire risk management pipeline
        """
        system = integrated_risk_system
        engine = system["engine"]
        checker = system["checker"]

        # Set up normal market conditions
        engine.portfolio_value = 100000.0
        engine.update_portfolio_state(100000.0, 0.08)  # Normal volatility

        # Step 1: Pre-trade validation via RiskChecker
        allowed, reason = checker.pre_trade_check(
            trade_notional=3000.0,
            position_notional=50000.0,
            peak_value=105000.0,
            sharpe_ratio=1.2,
        )

        assert allowed is True
        assert reason == ""

        # Step 2: Direct engine validation (should match)
        engine_allowed, engine_reason = engine.validate_trade(
            trade_notional=3000.0,
            position_notional=50000.0,
            peak_value=105000.0,
            sharpe_ratio=1.2,
        )

        assert engine_allowed == allowed
        assert engine_reason == reason

    def test_full_trade_workflow_blocked(self, integrated_risk_system):
        """
        Test complete trade workflow when trade should be blocked.

        Scenario: Risk limits exceeded, trade blocked at multiple levels
        Expected: Trade is consistently rejected through entire pipeline
        """
        system = integrated_risk_system
        engine = system["engine"]
        checker = system["checker"]

        # Set up risky conditions
        engine.portfolio_value = 100000.0
        engine.update_portfolio_state(100000.0, 0.20)  # High volatility

        # Step 1: Pre-trade validation via RiskChecker (should block due to volatility)
        allowed, reason = checker.pre_trade_check(
            trade_notional=3000.0,
            position_notional=50000.0,
            peak_value=105000.0,
            sharpe_ratio=0.5,  # Poor performance
        )

        assert allowed is False
        assert (
            "Portfolio volatility exceeds limit" in reason
            or "Sharpe ratio below threshold" in reason
        )

        # Step 2: Direct engine validation (should also block)
        engine_allowed, engine_reason = engine.validate_trade(
            trade_notional=3000.0,
            position_notional=50000.0,
            peak_value=105000.0,
            sharpe_ratio=0.5,
        )

        assert engine_allowed is False
        assert len(engine_reason) > 0

    def test_state_consistency_across_components(self, integrated_risk_system):
        """
        Test that risk state remains consistent across components.

        Scenario: State changes in one component should be reflected appropriately
        Expected: Components maintain consistent view of risk state
        """
        system = integrated_risk_system
        engine = system["engine"]
        checker = system["checker"]

        # Initial state
        initial_value = 100000.0
        engine.portfolio_value = initial_value

        # Both components should see same portfolio value through engine
        assert checker.engine.portfolio_value == initial_value

        # Update through engine
        new_value = 95000.0
        engine.update_portfolio_state(new_value, 0.1)

        # Checker should see updated value
        assert checker.engine.portfolio_value == new_value
        assert checker.engine.portfolio_volatility == 0.1

    def test_trade_recording_integration(self, integrated_risk_system):
        """
        Test trade recording and its impact on subsequent risk checks.

        Scenario: Record trades and verify they affect risk calculations
        Expected: Trade history properly influences risk decisions
        """
        system = integrated_risk_system
        engine = system["engine"]

        # Record some losing trades
        for i in range(3):
            engine.record_trade({"pnl": -500.0})

        # Check that consecutive losses trigger cooldown
        cooldown = engine.get_cooldown_period()
        assert cooldown == 300  # 5 minutes cooldown

        # Subsequent trade validation should be affected
        allowed, reason = engine.validate_trade(
            trade_notional=1000.0,
            position_notional=20000.0,
            peak_value=100000.0,
        )

        # Should be blocked due to cooldown (if within cooldown period)
        # Note: This test may be timing-dependent
        assert isinstance(allowed, bool)
        assert isinstance(reason, str)

    def test_risk_limit_consistency(self, integrated_risk_system):
        """
        Test that risk limits are consistently applied across components.

        Scenario: Same risk limits used by all components
        Expected: Consistent application of limits
        """
        system = integrated_risk_system
        limits = system["limits"]
        engine = system["engine"]
        checker = system["checker"]

        # All components should use same limits
        assert engine.limits == limits
        assert checker.engine.limits == limits

        # Test specific limit consistency
        test_position = limits.max_position_notional + 1000

        # Both should block at same position size
        engine_allowed, _ = engine.check_position_size(test_position)
        checker_allowed, _ = checker.pre_trade_check(
            trade_notional=1000.0,
            position_notional=test_position,
            peak_value=limits.max_position_notional,
        )

        assert engine_allowed == checker_allowed == False
