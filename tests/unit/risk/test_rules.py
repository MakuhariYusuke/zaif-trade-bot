"""Tests for RiskRuleEngine component."""

import time
from datetime import datetime, timedelta
from unittest.mock import patch

import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from ztb.risk.profiles import RiskLimits
from ztb.risk.rules import RiskRuleEngine


@pytest.fixture


@pytest.fixture
def risk_engine(sample_risk_limits):
    """RiskRuleEngine instance for testing."""
    return RiskRuleEngine(sample_risk_limits)


class TestRiskRuleEngineInitialization:
    """Test RiskRuleEngine initialization."""

    def test_initialization(self, risk_engine, sample_risk_limits):
        """Test proper initialization."""
        assert risk_engine.limits == sample_risk_limits
        assert risk_engine.daily_start_capital == 0.0
        assert risk_engine.daily_loss == 0.0
        assert risk_engine.portfolio_value == 0.0
        assert risk_engine.portfolio_volatility == 0.0
        assert risk_engine.trades_this_hour == 0
        assert risk_engine.trailing_stop_level is None
        assert risk_engine.trade_history == []


class TestRiskRuleEngineDailyTracking:
    """Test daily loss tracking functionality."""

    def test_reset_daily_tracking_same_day(self, risk_engine):
        """Test that daily tracking doesn't reset on same day."""
        initial_time = risk_engine.daily_start_time
        initial_capital = risk_engine.daily_start_capital

        risk_engine.reset_daily_tracking()

        assert risk_engine.daily_start_time == initial_time
        assert risk_engine.daily_start_capital == initial_capital

    @patch("ztb.risk.rules.datetime")
    def test_reset_daily_tracking_new_day(self, mock_datetime, risk_engine):
        """Test that daily tracking resets on new day."""
        # Set initial time to yesterday
        yesterday = datetime.now() - timedelta(days=1)
        yesterday_start = yesterday.replace(hour=0, minute=0, second=0, microsecond=0)
        risk_engine.daily_start_time = yesterday_start
        risk_engine.daily_start_capital = 100000.0
        risk_engine.daily_loss = 5000.0

        # Mock current time to today
        today = datetime.now()
        today_start = today.replace(hour=0, minute=0, second=0, microsecond=0)
        mock_datetime.now.return_value = today

        risk_engine.reset_daily_tracking()

        assert risk_engine.daily_start_time == today_start
        assert (
            risk_engine.daily_start_capital == 0.0
        )  # Reset to current portfolio_value
        assert risk_engine.daily_loss == 0.0

    def test_update_portfolio_state(self, risk_engine):
        """Test portfolio state updates."""
        risk_engine.daily_start_capital = 100000.0

        risk_engine.update_portfolio_state(95000.0, 0.12)

        assert risk_engine.portfolio_value == 95000.0
        assert risk_engine.portfolio_volatility == 0.12
        assert risk_engine.daily_loss == 5000.0


class TestRiskRuleEngineChecks:
    """Test individual risk rule checks."""

    def test_check_daily_loss_limit_no_capital(self, risk_engine):
        """Test daily loss check with no starting capital."""
        allowed, reason = risk_engine.check_daily_loss_limit()
        assert allowed is True
        assert reason == ""

    def test_check_daily_loss_limit_within_limit(self, risk_engine):
        """Test daily loss check within limit."""
        risk_engine.daily_start_capital = 100000.0
        risk_engine.portfolio_value = 96000.0  # 4% loss
        risk_engine.update_portfolio_state(96000.0, 0.1)

        allowed, reason = risk_engine.check_daily_loss_limit()
        assert allowed is True
        assert reason == ""

    def test_check_daily_loss_limit_exceeded(self, risk_engine):
        """Test daily loss check when limit exceeded."""
        risk_engine.daily_start_capital = 100000.0
        risk_engine.portfolio_value = 94000.0  # 6% loss
        risk_engine.update_portfolio_state(94000.0, 0.1)

        allowed, reason = risk_engine.check_daily_loss_limit()
        assert allowed is False
        assert "Daily loss limit exceeded" in reason

    def test_check_max_drawdown_within_limit(self, risk_engine):
        """Test max drawdown check within limit."""
        risk_engine.portfolio_value = 90500.0
        peak_value = 100000.0  # 9.5% drawdown (within 10% limit)

        allowed, reason = risk_engine.check_max_drawdown(peak_value)
        assert allowed is True
        assert reason == ""

    def test_check_max_drawdown_exceeded(self, risk_engine):
        """Test max drawdown check when exceeded."""
        risk_engine.portfolio_value = 85000.0
        peak_value = 100000.0  # 15% drawdown

        allowed, reason = risk_engine.check_max_drawdown(peak_value)
        assert allowed is False
        assert "Max drawdown exceeded" in reason

    def test_check_max_drawdown_boundary_exactly_at_limit(self, risk_engine):
        """Test max drawdown check at exactly the limit boundary."""
        risk_engine.portfolio_value = 90000.0
        peak_value = 100000.0  # Exactly 10% drawdown (should be allowed)

        allowed, reason = risk_engine.check_max_drawdown(peak_value)
        assert allowed is True
        assert reason == ""

    def test_check_max_drawdown_boundary_just_over_limit(self, risk_engine):
        """Test max drawdown check just over the limit boundary."""
        risk_engine.portfolio_value = 89999.0
        peak_value = 100000.0  # 10.001% drawdown (should be blocked)

        allowed, reason = risk_engine.check_max_drawdown(peak_value)
        assert allowed is False
        assert "Max drawdown exceeded" in reason

    def test_check_position_size_within_limit(self, risk_engine):
        """Test position size check within limit."""
        position_notional = 80000.0

        allowed, reason = risk_engine.check_position_size(position_notional)
        assert allowed is True
        assert reason == ""

    def test_check_position_size_exceeded(self, risk_engine):
        """Test position size check when exceeded."""
        position_notional = 120000.0

        allowed, reason = risk_engine.check_position_size(position_notional)
        assert allowed is False
        assert "Position size exceeds limit" in reason

    def test_check_position_size_boundary_exactly_at_limit(self, risk_engine):
        """Test position size check at exactly the limit boundary."""
        position_notional = 100000.0  # Exactly at limit (should be allowed)

        allowed, reason = risk_engine.check_position_size(position_notional)
        assert allowed is True
        assert reason == ""

    def test_check_position_size_boundary_just_over_limit(self, risk_engine):
        """Test position size check just over the limit boundary."""
        position_notional = 100001.0  # Just over limit (should be blocked)

        allowed, reason = risk_engine.check_position_size(position_notional)
        assert allowed is False
        assert "Position size exceeds limit" in reason

    def test_check_single_trade_size_within_limit(self, risk_engine):
        """Test single trade size check within limit."""
        risk_engine.portfolio_value = 100000.0
        trade_notional = 4000.0  # 4% of portfolio

        allowed, reason = risk_engine.check_single_trade_size(trade_notional)
        assert allowed is True
        assert reason == ""

    def test_check_single_trade_size_exceeded(self, risk_engine):
        """Test single trade size check when exceeded."""
        risk_engine.portfolio_value = 100000.0
        trade_notional = 6000.0  # 6% of portfolio

        allowed, reason = risk_engine.check_single_trade_size(trade_notional)
        assert allowed is False
        assert "Single trade size exceeds limit" in reason

    def test_check_trade_frequency_within_limit(self, risk_engine):
        """Test trade frequency check within limit."""
        risk_engine.trades_this_hour = 3

        allowed, reason = risk_engine.check_trade_frequency()
        assert allowed is True
        assert reason == ""

    def test_check_trade_frequency_exceeded(self, risk_engine):
        """Test trade frequency check when exceeded."""
        risk_engine.trades_this_hour = 5

        allowed, reason = risk_engine.check_trade_frequency()
        assert allowed is False
        assert "Trade frequency limit exceeded" in reason

    def test_check_trade_frequency_interval_violation(self, risk_engine):
        """Test trade frequency check with interval violation."""
        risk_engine.last_trade_time = time.time() - 300  # 5 minutes ago

        allowed, reason = risk_engine.check_trade_frequency()
        assert allowed is False
        assert "Minimum trade interval not met" in reason

    def test_check_trade_frequency_interval_boundary_exactly_at_limit(
        self, risk_engine
    ):
        """Test trade frequency check at exactly the interval limit."""
        risk_engine.last_trade_time = (
            time.time() - 600
        )  # Exactly 10 minutes ago (should be allowed)

        allowed, reason = risk_engine.check_trade_frequency()
        assert allowed is True
        assert reason == ""

    def test_check_trade_frequency_interval_boundary_just_under_limit(
        self, risk_engine
    ):
        """Test trade frequency check just under the interval limit."""
        risk_engine.last_trade_time = (
            time.time() - 599
        )  # 9 minutes 59 seconds ago (should be blocked)

        allowed, reason = risk_engine.check_trade_frequency()
        assert allowed is False
        assert "Minimum trade interval not met" in reason

    def test_check_volatility_limit_within_limit(self, risk_engine):
        """Test volatility check within limit."""
        risk_engine.portfolio_volatility = 0.12

        allowed, reason = risk_engine.check_volatility_limit()
        assert allowed is True
        assert reason == ""

    def test_check_volatility_limit_exceeded(self, risk_engine):
        """Test volatility check when exceeded."""
        risk_engine.portfolio_volatility = 0.18

        allowed, reason = risk_engine.check_volatility_limit()
        assert allowed is False
        assert "Portfolio volatility exceeds limit" in reason

    def test_check_performance_thresholds_above_threshold(self, risk_engine):
        """Test performance check above threshold."""
        sharpe_ratio = 1.2

        allowed, reason = risk_engine.check_performance_thresholds(sharpe_ratio)
        assert allowed is True
        assert reason == ""

    def test_check_performance_thresholds_below_threshold(self, risk_engine):
        """Test performance check below threshold."""
        sharpe_ratio = 0.8

        allowed, reason = risk_engine.check_performance_thresholds(sharpe_ratio)
        assert allowed is False
        assert "Sharpe ratio below threshold" in reason

    def test_check_single_trade_size_small_amount(self, risk_engine):
        """Test single trade size check with very small amount."""
        risk_engine.portfolio_value = 100000.0
        trade_notional = 50.0  # Very small trade ($50)

        allowed, reason = risk_engine.check_single_trade_size(trade_notional)
        assert allowed is True
        assert reason == ""

    def test_check_position_size_extreme_values(self, risk_engine):
        """Test position size check with extreme values."""
        # Very large position (should be blocked)
        position_notional = 1000000.0  # 10x the limit
        allowed, reason = risk_engine.check_position_size(position_notional)
        assert allowed is False
        assert "Position size exceeds limit" in reason

    def test_check_volatility_limit_edge_cases(self, risk_engine):
        """Test volatility check with edge case values."""
        # Exactly at limit
        risk_engine.portfolio_volatility = 0.15  # Exactly at limit
        allowed, reason = risk_engine.check_volatility_limit()
        assert allowed is True
        assert reason == ""

        # Just over limit
        risk_engine.portfolio_volatility = 0.15001
        allowed, reason = risk_engine.check_volatility_limit()
        assert allowed is False
        assert "Portfolio volatility exceeds limit" in reason


class TestRiskRuleEngineTrailingStop:
    """Test trailing stop functionality."""

    def test_update_trailing_stop_long_position_initial(self, risk_engine):
        """Test initial trailing stop setup for long position."""
        current_price = 100.0

        risk_engine.update_trailing_stop(current_price, "long")

        expected_stop = 100.0 * (1 - 0.03)  # 97.0
        assert risk_engine.trailing_stop_level == expected_stop

    def test_update_trailing_stop_short_position_initial(self, risk_engine):
        """Test initial trailing stop setup for short position."""
        current_price = 100.0

        risk_engine.update_trailing_stop(current_price, "short")

        expected_stop = 100.0 * (1 + 0.03)  # 103.0
        assert risk_engine.trailing_stop_level == expected_stop

    def test_update_trailing_stop_long_position_update(self, risk_engine):
        """Test trailing stop update for long position."""
        # Initial setup
        risk_engine.update_trailing_stop(100.0, "long")
        initial_stop = risk_engine.trailing_stop_level

        # Price increases, stop should trail up
        risk_engine.update_trailing_stop(110.0, "long")
        new_stop = risk_engine.trailing_stop_level

        assert new_stop > initial_stop

    def test_update_trailing_stop_short_position_update(self, risk_engine):
        """Test trailing stop update for short position."""
        # Initial setup
        risk_engine.update_trailing_stop(100.0, "short")
        initial_stop = risk_engine.trailing_stop_level

        # Price decreases, stop should trail down
        risk_engine.update_trailing_stop(90.0, "short")
        new_stop = risk_engine.trailing_stop_level

        assert new_stop < initial_stop

    def test_check_trailing_stop_long_not_hit(self, risk_engine):
        """Test trailing stop check for long position not hit."""
        risk_engine.update_trailing_stop(100.0, "long")
        current_price = 98.0  # Above stop level

        allowed, reason = risk_engine.check_trailing_stop(current_price, "long")
        assert allowed is True
        assert reason == ""

    def test_check_trailing_stop_long_hit(self, risk_engine):
        """Test trailing stop check for long position hit."""
        risk_engine.update_trailing_stop(100.0, "long")
        current_price = 96.0  # Below stop level

        allowed, reason = risk_engine.check_trailing_stop(current_price, "long")
        assert allowed is False
        assert "Trailing stop hit" in reason

    def test_check_trailing_stop_short_hit(self, risk_engine):
        """Test trailing stop check for short position hit."""
        risk_engine.update_trailing_stop(100.0, "short")
        current_price = 104.0  # Above stop level

        allowed, reason = risk_engine.check_trailing_stop(current_price, "short")
        assert allowed is False
        assert "Trailing stop hit" in reason


class TestRiskRuleEngineTakeProfit:
    """Test take profit functionality."""

    def test_check_take_profit_long_not_reached(self, risk_engine):
        """Test take profit check for long position not reached."""
        entry_price = 100.0
        current_price = 105.0  # 5% profit

        allowed, reason = risk_engine.check_take_profit(
            entry_price, current_price, "long"
        )
        assert allowed is True
        assert reason == ""

    def test_check_take_profit_long_reached(self, risk_engine):
        """Test take profit check for long position reached."""
        entry_price = 100.0
        current_price = 109.0  # 9% profit

        allowed, reason = risk_engine.check_take_profit(
            entry_price, current_price, "long"
        )
        assert allowed is False
        assert "Take profit target reached" in reason

    def test_check_take_profit_short_reached(self, risk_engine):
        """Test take profit check for short position reached."""
        entry_price = 100.0
        current_price = 91.0  # 9% profit

        allowed, reason = risk_engine.check_take_profit(
            entry_price, current_price, "short"
        )
        assert allowed is False
        assert "Take profit target reached" in reason


class TestRiskRuleEngineTradeRecording:
    """Test trade recording functionality."""

    def test_record_trade(self, risk_engine):
        """Test trade recording."""
        trade_data = {"pnl": 1000.0, "size": 50000.0}

        initial_trades = risk_engine.trades_this_hour
        initial_history_length = len(risk_engine.trade_history)

        risk_engine.record_trade(trade_data)

        assert len(risk_engine.trade_history) == initial_history_length + 1
        assert risk_engine.trades_this_hour == initial_trades + 1
        assert risk_engine.trade_history[-1]["pnl"] == 1000.0

    def test_get_cooldown_period_no_cooldown(self, risk_engine):
        """Test cooldown period with no recent losses."""
        cooldown = risk_engine.get_cooldown_period()
        assert cooldown == 0

    def test_get_cooldown_period_with_cooldown(self, risk_engine):
        """Test cooldown period with consecutive losses."""
        # Record 3 losing trades
        for _ in range(3):
            risk_engine.record_trade({"pnl": -100.0})

        cooldown = risk_engine.get_cooldown_period()
        assert cooldown == 300  # 5 minutes


class TestRiskRuleEngineValidateTrade:
    """Test comprehensive trade validation."""

    def test_validate_trade_all_checks_pass(self, risk_engine):
        """
        Test trade validation when all risk checks pass.

        Scenario: Normal trading conditions with all parameters within limits
        Expected: Trade should be allowed with no rejection reason
        """
        risk_engine.portfolio_value = 100000.0
        risk_engine.update_portfolio_state(100000.0, 0.1)

        allowed, reason = risk_engine.validate_trade(
            trade_notional=3000.0,
            position_notional=50000.0,
            peak_value=105000.0,
            sharpe_ratio=1.2,
        )

        assert allowed is True
        assert reason == ""

    def test_validate_trade_position_size_exceeded(self, risk_engine):
        """
        Test trade validation when position size exceeds limit.

        Scenario: Large position that exceeds max_position_notional limit
        Expected: Trade should be blocked with appropriate rejection reason
        """
        risk_engine.portfolio_value = 100000.0

        allowed, reason = risk_engine.validate_trade(
            trade_notional=1000.0,
            position_notional=120000.0,  # Exceeds limit
            peak_value=100000.0,
        )

        assert allowed is False
        assert "Position size exceeds limit" in reason

    def test_validate_trade_in_cooldown(self, risk_engine):
        """
        Test trade validation during cooldown period after consecutive losses.

        Scenario: Three consecutive losing trades trigger 5-minute cooldown
        Expected: Trade should be blocked until cooldown period expires
        """
        import time

        # Set up consecutive losses
        for _ in range(3):
            risk_engine.record_trade({"pnl": -100.0})

        # Temporarily reduce min_trade_interval to focus on cooldown test
        original_interval = risk_engine.limits.min_trade_interval_sec
        risk_engine.limits.min_trade_interval_sec = 0

        # Set last trade time to recent past to trigger cooldown check
        risk_engine.last_trade_time = time.time() - 100  # 100 seconds ago

        # Set portfolio value to avoid max drawdown trigger
        risk_engine.portfolio_value = 100000.0

        allowed, reason = risk_engine.validate_trade(
            trade_notional=1000.0,
            position_notional=50000.0,
            peak_value=100000.0,
        )

        # Restore original interval
        risk_engine.limits.min_trade_interval_sec = original_interval

        assert allowed is False
        assert "cooldown period" in reason


class TestRiskRuleEngineErrorHandling:
    """Test error handling and edge cases."""

    def test_update_portfolio_state_negative_value(self, risk_engine):
        """Test portfolio state update with negative value."""
        with pytest.raises(ValueError, match="Portfolio value cannot be negative"):
            risk_engine.update_portfolio_state(-1000.0, 0.1)

    def test_update_portfolio_state_zero_value(self, risk_engine):
        """Test portfolio state update with zero value."""
        risk_engine.update_portfolio_state(0.0, 0.1)
        assert risk_engine.portfolio_value == 0.0

    def test_update_trailing_stop_invalid_position_side(self, risk_engine):
        """Test trailing stop update with invalid position side."""
        with pytest.raises(ValueError, match="Invalid position side"):
            risk_engine.update_trailing_stop(100.0, "invalid_side")

    def test_check_max_drawdown_peak_less_than_current(self, risk_engine):
        """Test max drawdown check when peak is less than current value."""
        risk_engine.portfolio_value = 100000.0
        peak_value = 90000.0  # Peak is less than current (should be allowed)

        allowed, reason = risk_engine.check_max_drawdown(peak_value)
        assert allowed is True
        assert reason == ""

    def test_record_trade_invalid_data(self, risk_engine):
        """Test recording trade with invalid data."""
        # Empty trade data should not cause errors
        risk_engine.record_trade({})
        assert len(risk_engine.trade_history) == 1

    def test_validate_trade_extreme_values(self, risk_engine):
        """Test trade validation with extreme values."""
        # Very large values
        allowed, reason = risk_engine.validate_trade(
            trade_notional=1e10,  # Extremely large trade
            position_notional=1e10,
            peak_value=1e10,
        )
        # Should still work without crashing
        assert isinstance(allowed, bool)
        assert isinstance(reason, str)


class TestRiskRuleEnginePropertyBased:
    """Property-based tests using hypothesis."""

    @given(
        portfolio_value=st.floats(min_value=1000, max_value=10000000),
        peak_value=st.floats(min_value=1000, max_value=10000000),
    )
    @settings(suppress_health_check=[HealthCheck.function_scoped_fixture])
    def test_max_drawdown_calculation_properties(
        self, risk_engine, portfolio_value, peak_value
    ):
        """
        Property-based test for max drawdown calculation.

        Ensures that max drawdown logic is consistent across different value ranges.
        """
        # Update the engine's portfolio value to match the test parameter
        risk_engine.portfolio_value = portfolio_value

        # Calculate expected drawdown
        drawdown_pct = (
            (peak_value - portfolio_value) / peak_value if peak_value > 0 else 0.0
        )
        expected_allowed = drawdown_pct <= risk_engine.limits.max_drawdown_pct

        allowed, reason = risk_engine.check_max_drawdown(peak_value)

        # Result should match expectation
        assert allowed == expected_allowed
        if not allowed:
            assert "Max drawdown exceeded" in reason

    @given(
        position_notional=st.floats(min_value=0, max_value=1000000),
    )
    @settings(suppress_health_check=[HealthCheck.function_scoped_fixture])
    def test_position_size_limits_properties(self, risk_engine, position_notional):
        """
        Property-based test for position size limits.

        Verifies that position size checking works correctly across value ranges.
        """
        allowed, reason = risk_engine.check_position_size(position_notional)

        # Position should be allowed if within limit
        expected_allowed = position_notional <= risk_engine.limits.max_position_notional
        assert allowed == expected_allowed

        if not allowed:
            assert "Position size exceeds limit" in reason

    @given(
        trade_notional=st.floats(min_value=0, max_value=100000),
        portfolio_value=st.floats(min_value=10000, max_value=1000000),
    )
    @settings(suppress_health_check=[HealthCheck.function_scoped_fixture])
    def test_single_trade_size_percentage_properties(
        self, risk_engine, trade_notional, portfolio_value
    ):
        """
        Property-based test for single trade size as percentage of portfolio.

        Ensures trade size limits work correctly relative to portfolio value.
        """
        risk_engine.portfolio_value = portfolio_value

        allowed, reason = risk_engine.check_single_trade_size(trade_notional)

        # Calculate trade as percentage of portfolio
        if portfolio_value > 0:
            trade_pct = trade_notional / portfolio_value
            expected_allowed = trade_pct <= risk_engine.limits.max_single_trade_pct
        else:
            expected_allowed = True  # Edge case with zero portfolio

        assert allowed == expected_allowed
        if not allowed:
            assert "Single trade size exceeds limit" in reason


class TestRiskRuleEnginePerformance:
    """Performance tests for risk rule operations."""

    def test_validate_trade_performance(self, benchmark, risk_engine):
        """
        Benchmark trade validation performance.

        Measures execution time for comprehensive trade validation.
        """
        # Set up normal trading conditions
        risk_engine.portfolio_value = 100000.0
        risk_engine.update_portfolio_state(100000.0, 0.1)

        def run_validation():
            return risk_engine.validate_trade(
                trade_notional=3000.0,
                position_notional=50000.0,
                peak_value=105000.0,
                sharpe_ratio=1.2,
            )

        result = benchmark(run_validation)
        allowed, reason = result

        # Ensure the operation still works correctly
        assert allowed is True
        assert reason == ""

    def test_bulk_trade_recording_performance(self, benchmark, risk_engine):
        """
        Benchmark bulk trade recording performance.

        Tests performance when recording many trades in sequence.
        """
        trades = [{"pnl": 100.0, "timestamp": time.time() + i} for i in range(100)]

        def record_trades():
            for trade in trades:
                risk_engine.record_trade(trade)

        benchmark(record_trades)

        # Verify trades were recorded
        assert len(risk_engine.trade_history) >= 100

    def test_concurrent_risk_checks_performance(self, benchmark, risk_engine):
        """
        Benchmark concurrent risk checks performance.

        Tests performance when running multiple risk checks simultaneously.
        """
        # Set up various risk check scenarios
        risk_engine.portfolio_value = 100000.0
        risk_engine.update_portfolio_state(100000.0, 0.1)

        def run_multiple_checks():
            checks = [
                risk_engine.check_daily_loss_limit(),
                risk_engine.check_max_drawdown(105000.0),
                risk_engine.check_position_size(80000.0),
                risk_engine.check_single_trade_size(4000.0),
                risk_engine.check_trade_frequency(),
                risk_engine.check_volatility_limit(),
                risk_engine.check_performance_thresholds(1.2),
            ]
            return all(result[0] for result in checks)

        result = benchmark(run_multiple_checks)
        assert result is True  # All checks should pass
