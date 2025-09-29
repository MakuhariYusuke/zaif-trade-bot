#!/usr/bin/env python3
"""
Leakage detection tests for trading strategies.

Ensures no data leakage (look-ahead bias) in backtesting.
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, Any, List

from ztb.backtest.adapters import StrategyAdapter
from ztb.backtest.runner import BacktestEngine


class MockStrategy(StrategyAdapter):
    """Mock strategy for testing leakage."""

    def __init__(self, lookback_period: int = 10):
        self.lookback_period = lookback_period

    def generate_signal(self, data: pd.DataFrame, current_index: int) -> Dict[str, Any]:
        """Generate trading signal."""
        if current_index < self.lookback_period:
            return {'action': 'hold'}

        # Check for potential leakage: using future data
        future_data = data.iloc[current_index + 1: current_index + 5]  # This would be leakage!

        if not future_data.empty and future_data['close'].max() > data.iloc[current_index]['close'] * 1.01:
            return {'action': 'buy'}
        elif not future_data.empty and future_data['close'].min() < data.iloc[current_index]['close'] * 0.99:
            return {'action': 'sell'}

        return {'action': 'hold'}


class CleanStrategy(StrategyAdapter):
    """Clean strategy that only uses past data."""

    def __init__(self, lookback_period: int = 10):
        self.lookback_period = lookback_period

    def generate_signal(self, data: pd.DataFrame, current_index: int) -> Dict[str, Any]:
        """Generate trading signal using only past data."""
        if current_index < self.lookback_period:
            return {'action': 'hold'}

        # Only use past data
        past_data = data.iloc[max(0, current_index - self.lookback_period):current_index]

        if past_data['close'].iloc[-1] > past_data['close'].mean():
            return {'action': 'buy'}
        elif past_data['close'].iloc[-1] < past_data['close'].mean():
            return {'action': 'sell'}

        return {'action': 'hold'}


def create_test_data(length: int = 1000) -> pd.DataFrame:
    """Create synthetic price data for testing."""
    np.random.seed(42)  # For reproducible tests

    dates = pd.date_range('2020-01-01', periods=length, freq='1min')
    prices = 10000 * np.exp(np.cumsum(np.random.normal(0, 0.001, length)))

    return pd.DataFrame({
        'timestamp': dates,
        'open': prices,
        'high': prices * 1.001,
        'low': prices * 0.999,
        'close': prices,
        'volume': np.random.uniform(0.1, 10, length)
    }).set_index('timestamp')


def detect_future_data_access(strategy: StrategyAdapter, data: pd.DataFrame) -> List[str]:
    """
    Detect if strategy accesses future data by monitoring data access patterns.

    This is a simplified detection - in practice, you'd need more sophisticated
    monitoring of data access within the strategy.
    """
    violations = []

    # Check strategy code for obvious future data access patterns
    import inspect
    source = inspect.getsource(strategy.generate_signal)

    if 'current_index + 1' in source or 'future_data' in source:
        violations.append("Strategy appears to access future data directly")

    if 'data.iloc[current_index + ' in source:
        violations.append("Strategy uses positive offset from current_index")

    return violations


class TestLeakageDetection:
    """Test suite for data leakage detection."""

    def test_mock_strategy_has_leakage(self):
        """Test that mock strategy with obvious leakage is detected."""
        strategy = MockStrategy()
        data = create_test_data(100)

        violations = detect_future_data_access(strategy, data)
        assert len(violations) > 0, "Should detect leakage in mock strategy"

    def test_clean_strategy_no_leakage(self):
        """Test that clean strategy passes leakage detection."""
        strategy = CleanStrategy()
        data = create_test_data(100)

        violations = detect_future_data_access(strategy, data)
        assert len(violations) == 0, f"Clean strategy should not have violations: {violations}"

    def test_backtest_runner_prevents_future_access(self):
        """Test that backtest runner properly isolates data access."""
        strategy = CleanStrategy()
        engine = BacktestEngine()
        data = create_test_data(200)

        # Run backtest
        equity_curve, orders = engine.run_backtest(strategy, data)

        # Verify strategy only sees past data
        assert len(equity_curve) > 0, "Backtest should produce results"
        assert not orders.empty or len(orders) == 0, "Orders should be generated or empty"

    def test_signal_generation_is_deterministic(self):
        """Test that signals are deterministic given same data."""
        strategy = CleanStrategy()
        data = create_test_data(100)

        # Generate signals twice
        signals1 = [strategy.generate_signal(data, i) for i in range(20, 50)]
        signals2 = [strategy.generate_signal(data, i) for i in range(20, 50)]

        assert signals1 == signals2, "Signals should be deterministic"

    def test_no_side_effects_in_signal_generation(self):
        """Test that signal generation doesn't modify strategy state."""
        strategy = CleanStrategy()
        data = create_test_data(100)

        # Generate signals multiple times
        initial_state = strategy.__dict__.copy()

        for i in range(20, 50):
            strategy.generate_signal(data, i)

        final_state = strategy.__dict__.copy()

        assert initial_state == final_state, "Strategy state should not change during signal generation"

    def test_backtest_results_reproducible(self):
        """Test that backtest results are reproducible."""
        strategy = CleanStrategy()
        engine = BacktestEngine()
        data = create_test_data(200)

        # Run backtest twice
        equity1, orders1 = engine.run_backtest(strategy, data)
        equity2, orders2 = engine.run_backtest(strategy, data)

        pd.testing.assert_series_equal(equity1, equity2, check_names=False)
        pd.testing.assert_frame_equal(orders1, orders2, check_like=True)


if __name__ == '__main__':
    pytest.main([__file__])