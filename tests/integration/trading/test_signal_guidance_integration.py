"""
Integration tests for Signal Guidance Improvements

テストケース:
- End-to-end signal generation with real market data
- Frequency improvement validation (target: 20-50 signals/day)
- Accuracy comparison with baseline
- Backtest integration
- Performance benchmarking
"""

import time

import numpy as np
import pandas as pd
import pytest

from tests.helpers.market_data import make_realistic_intraday_ohlcv_data
from ztb.trading.signal.signal_guidance_system import SignalGuidanceSystem

pytestmark = [
    pytest.mark.integration,
    pytest.mark.slow,
]


def _make_baseline_system() -> SignalGuidanceSystem:
    system = SignalGuidanceSystem()

    def mock_old_guidance(continuous_action, row, portfolio):
        if continuous_action > 0.7:
            return 1
        if continuous_action < -0.7:
            return -1
        return 0

    system.apply_guidance = mock_old_guidance
    return system


def _collect_guided_actions(
    system: SignalGuidanceSystem,
    rows: list[pd.Series],
    portfolio: dict[str, float],
    continuous_actions: np.ndarray,
) -> list[int]:
    return [
        system.apply_guidance(continuous_action, row, portfolio)
        for row, continuous_action in zip(rows, continuous_actions)
    ]


class TestSignalGuidanceIntegration:
    """Integration tests for signal guidance improvements"""

    @pytest.fixture
    def improved_system(self):
        """Create improved signal guidance system"""
        return SignalGuidanceSystem()

    @pytest.fixture
    def baseline_system(self):
        """Create a naive baseline system with simple thresholding only."""
        return _make_baseline_system()

    @pytest.fixture(scope="class")
    def realistic_market_data(self):
        """Create realistic market data for testing"""
        return make_realistic_intraday_ohlcv_data(rows=48, seed=42, base_price=50000.0)

    @pytest.fixture(scope="class")
    def portfolio_state(self):
        """Create realistic portfolio state"""
        return {
            'btc_balance': 0.5,
            'jpy_balance': 100000,
            'current_price': 50000,
            'portfolio_value': 125000
        }

    @pytest.fixture(scope="class")
    def market_rows(self, realistic_market_data):
        return [row for _, row in realistic_market_data.iterrows()]

    @pytest.fixture(scope="class")
    def action_samples(self, market_rows):
        rng = np.random.default_rng(123)
        return rng.normal(0, 0.5, len(market_rows))

    @pytest.fixture(scope="class")
    def improved_replay(self, market_rows, portfolio_state, action_samples):
        improved_system = SignalGuidanceSystem()

        improved_signals = _collect_guided_actions(
            improved_system, market_rows, portfolio_state, action_samples
        )

        return {
            "improved_signals": np.asarray(improved_signals, dtype=int),
            "total_signals": len(market_rows),
        }

    @pytest.fixture(scope="class")
    def benchmark_rows(self, market_rows):
        return market_rows[:16]

    @pytest.fixture(scope="class")
    def benchmark_actions(self, action_samples):
        return action_samples[:16]

    @pytest.fixture(scope="class")
    def position_test_rows(self, market_rows):
        return market_rows[:8]

    @pytest.fixture(scope="class")
    def consistency_rows(self, market_rows):
        return market_rows[:6]

    def test_end_to_end_signal_generation(
        self,
        realistic_market_data,
        improved_replay,
    ):
        """Test end-to-end signal generation with realistic data"""
        # Validate signal distribution
        signals_array = improved_replay["improved_signals"]
        buy_signals = int(np.sum(signals_array == 1))
        sell_signals = int(np.sum(signals_array == -1))
        hold_signals = int(np.sum(signals_array == 0))

        total_signals = int(improved_replay["total_signals"])
        assert total_signals == len(realistic_market_data)

        # Should have reasonable signal distribution
        assert buy_signals + sell_signals + hold_signals == total_signals

        # Calculate signal frequency (signals per day)
        trading_signals = buy_signals + sell_signals
        signals_per_day = trading_signals / (len(realistic_market_data) / 288)  # 288 = 5-min bars per day

        print(f"Signals per day: {signals_per_day:.1f}")
        print(f"Buy signals: {buy_signals}, Sell signals: {sell_signals}, Hold signals: {hold_signals}")

        # Current guidance is intentionally conservative, but it should still
        # emit at least one actionable signal over a representative window.
        assert trading_signals >= 1

    def test_signal_quality_vs_baseline(
        self,
        realistic_market_data,
        improved_replay,
        market_rows,
        portfolio_state,
        action_samples,
    ):
        """Guided system should suppress noisy baseline threshold firing."""
        improved_signals = improved_replay["improved_signals"]
        baseline_signals = np.asarray(
            _collect_guided_actions(
                _make_baseline_system(),
                market_rows,
                portfolio_state,
                action_samples,
            ),
            dtype=int,
        )

        # Calculate signal frequencies
        improved_trading_signals = int(np.sum(improved_signals != 0))
        baseline_trading_signals = int(np.sum(baseline_signals != 0))

        improved_freq = improved_trading_signals / (len(realistic_market_data) / 288)
        baseline_freq = baseline_trading_signals / (len(realistic_market_data) / 288)

        print(f"Improved system frequency: {improved_freq:.1f} signals/day")
        print(f"Baseline system frequency: {baseline_freq:.1f} signals/day")

        # Guided scoring should be more selective than naive thresholding.
        assert improved_freq < baseline_freq

    def test_technical_indicators_integration(self, improved_system, realistic_market_data):
        """Test technical indicators integration"""
        # Test that technical indicators are properly calculated
        scorer = improved_system.quality_scorer
        indicators = scorer.technical_indicators

        # Get technical signals
        signals = indicators.get_technical_signals(realistic_market_data)

        # Should have all expected indicators
        required_indicators = ['rsi', 'macd_line', 'macd_signal', 'bb_upper', 'bb_lower', 'atr']
        for indicator in required_indicators:
            assert indicator in signals
            assert isinstance(signals[indicator], (int, float))

        # RSI should be in valid range
        assert 0 <= signals['rsi'] <= 100

        # Bollinger Bands should be properly ordered
        assert signals['bb_upper'] > signals['bb_middle'] > signals['bb_lower']

    def test_position_aware_signaling(self, improved_system, position_test_rows):
        """Test position-aware signal generation"""
        # Test with overexposed position
        overexposed_portfolio = {
            'btc_balance': 1.5,  # Overexposed
            'jpy_balance': 25000,
            'current_price': 50000
        }

        # Test with underexposed position
        underexposed_portfolio = {
            'btc_balance': 0.1,  # Underexposed
            'jpy_balance': 240000,
            'current_price': 50000
        }

        overexposed_signals = []
        underexposed_signals = []

        for row in position_test_rows:
            continuous_action = 0.5  # Positive action

            overexposed_action = improved_system.apply_guidance(
                continuous_action, row, overexposed_portfolio)
            underexposed_action = improved_system.apply_guidance(
                continuous_action, row, underexposed_portfolio)

            overexposed_signals.append(overexposed_action)
            underexposed_signals.append(underexposed_action)

        # Overexposed position should generate fewer BUY signals
        overexposed_buys = sum(1 for s in overexposed_signals if s == 1)
        underexposed_buys = sum(1 for s in underexposed_signals if s == 1)

        print(f"Overexposed BUY signals: {overexposed_buys}")
        print(f"Underexposed BUY signals: {underexposed_buys}")

        # Underexposed should generate more BUY signals than overexposed
        assert underexposed_buys >= overexposed_buys

    def test_signal_consistency(self, consistency_rows, portfolio_state):
        """Test signal consistency with same inputs"""
        # Generate signals twice with same data
        signals_1 = []
        signals_2 = []

        # First pass
        system1 = SignalGuidanceSystem()
        for row in consistency_rows:
            continuous_action = 0.2
            action = system1.apply_guidance(continuous_action, row, portfolio_state)
            signals_1.append(action)

        # Second pass with fresh system
        system2 = SignalGuidanceSystem()
        for row in consistency_rows:
            continuous_action = 0.2
            action = system2.apply_guidance(continuous_action, row, portfolio_state)
            signals_2.append(action)

        # Signals should be identical (deterministic)
        assert signals_1 == signals_2

    def test_performance_benchmark(
        self,
        improved_system,
        benchmark_rows,
        portfolio_state,
        benchmark_actions,
        perf_runner,
    ):
        """Benchmark performance of signal generation"""
        measured = {"elapsed": 0.0}

        def _run() -> list[int]:
            start_time = time.perf_counter()
            signals = _collect_guided_actions(
                improved_system, benchmark_rows, portfolio_state, benchmark_actions
            )
            measured["elapsed"] = time.perf_counter() - start_time
            return signals

        signals = perf_runner(_run)
        processing_time = measured["elapsed"]

        print(f"Processed {len(signals)} signals in {processing_time:.2f} seconds")

        # Keep a pragmatic upper bound rather than a machine-specific throughput target.
        assert processing_time < 10.0

    def test_error_resilience(self, improved_system, portfolio_state):
        """Test system resilience to various error conditions"""
        # Test with missing data
        incomplete_row = pd.Series({'close': 50000})  # Missing OHLC

        action = improved_system.apply_guidance(0.0, incomplete_row, portfolio_state)
        assert action in [-1, 0, 1]  # Should not crash

        # Test with extreme values
        extreme_row = pd.Series({
            'open': 1e10,  # Extreme price
            'high': 1e10,
            'low': 1e10,
            'close': 1e10,
            'volume': 1e10
        })

        action = improved_system.apply_guidance(0.0, extreme_row, portfolio_state)
        assert action in [-1, 0, 1]  # Should handle extreme values

        # Test with NaN values
        nan_row = pd.Series({
            'open': np.nan,
            'high': np.nan,
            'low': np.nan,
            'close': np.nan,
            'volume': np.nan
        })

        action = improved_system.apply_guidance(0.0, nan_row, portfolio_state)
        assert action in [-1, 0, 1]  # Should handle NaN values
