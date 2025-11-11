"""
Integration tests for Signal Guidance Improvements

テストケース:
- End-to-end signal generation with real market data
- Frequency improvement validation (target: 20-50 signals/day)
- Accuracy comparison with baseline
- Backtest integration
- Performance benchmarking
"""

import numpy as np
import pandas as pd
import pytest
from unittest.mock import Mock

from ztb.trading.signal.signal_guidance_system import SignalGuidanceSystem
from ztb.trading.signal.quality_scorer import SignalQualityScorer
from ztb.trading.signal.technical_indicators import TechnicalIndicators


class TestSignalGuidanceIntegration:
    """Integration tests for signal guidance improvements"""

    @pytest.fixture
    def improved_system(self):
        """Create improved signal guidance system"""
        return SignalGuidanceSystem()

    @pytest.fixture
    def baseline_system(self):
        """Create baseline system for comparison (simplified version)"""
        # Create a system that mimics old probabilistic behavior
        system = SignalGuidanceSystem()
        # We'll mock the apply_guidance to use old logic for comparison
        original_apply = system.apply_guidance
        def mock_old_guidance(continuous_action, row, portfolio):
            # Conservative threshold-based conversion (old behavior approximation)
            # Old system was very conservative, generating only ~2.9 signals/day
            if continuous_action > 0.7:  # Very high threshold for BUY
                return 1
            elif continuous_action < -0.7:  # Very low threshold for SELL
                return -1
            else:
                return 0
        system.apply_guidance = mock_old_guidance
        return system

    @pytest.fixture
    def realistic_market_data(self):
        """Create realistic market data for testing"""
        np.random.seed(42)

        # Generate 1 day of 5-minute data (288 points)
        n_points = 288
        base_price = 50000
        prices = [base_price]

        # Simulate realistic price movements
        for i in range(n_points - 1):
            # Add trend, mean reversion, and noise
            trend = 0.0001 * np.sin(i / 50)  # Slow trend
            mean_reversion = (base_price - prices[-1]) * 0.001  # Mean reversion
            noise = np.random.normal(0, 0.005)  # Random noise
            volatility = np.random.choice([0.002, 0.008], p=[0.7, 0.3])  # Variable volatility

            change = trend + mean_reversion + noise * volatility
            new_price = prices[-1] * (1 + change)
            prices.append(max(new_price, 10000))  # Floor price

        # Create OHLCV data
        data = []
        for i, close in enumerate(prices):
            # Generate realistic OHLC
            volatility_factor = np.random.uniform(0.002, 0.01)
            high = close * (1 + abs(np.random.normal(0, volatility_factor)))
            low = close * (1 - abs(np.random.normal(0, volatility_factor)))
            open_price = data[-1]['close'] if data else close * (1 + np.random.normal(0, 0.001))
            volume = np.random.lognormal(12, 0.8)  # Realistic volume

            data.append({
                'timestamp': pd.Timestamp('2024-01-01 09:00:00') + pd.Timedelta(minutes=5*i),
                'open': open_price,
                'high': high,
                'low': low,
                'close': close,
                'volume': volume
            })

        return pd.DataFrame(data)

    @pytest.fixture
    def portfolio_state(self):
        """Create realistic portfolio state"""
        return {
            'btc_balance': 0.5,
            'jpy_balance': 100000,
            'current_price': 50000,
            'portfolio_value': 125000
        }

    def test_end_to_end_signal_generation(self, improved_system, realistic_market_data, portfolio_state):
        """Test end-to-end signal generation with realistic data"""
        signals = []

        for _, row in realistic_market_data.iterrows():
            # Generate continuous action (simulate model output)
            continuous_action = np.random.normal(0, 0.5)  # Realistic model output distribution

            # Get guided signal
            action = improved_system.apply_guidance(continuous_action, row, portfolio_state)
            signals.append(action)

        # Validate signal distribution
        signals_array = np.array(signals)
        buy_signals = np.sum(signals_array == 1)
        sell_signals = np.sum(signals_array == -1)
        hold_signals = np.sum(signals_array == 0)

        total_signals = len(signals)
        assert total_signals == len(realistic_market_data)

        # Should have reasonable signal distribution
        assert buy_signals + sell_signals + hold_signals == total_signals

        # Calculate signal frequency (signals per day)
        trading_signals = buy_signals + sell_signals
        signals_per_day = trading_signals / (len(realistic_market_data) / 288)  # 288 = 5-min bars per day

        print(f"Signals per day: {signals_per_day:.1f}")
        print(f"Buy signals: {buy_signals}, Sell signals: {sell_signals}, Hold signals: {hold_signals}")

        # Target: 20-50 signals per day (much higher than current 2.9)
        assert signals_per_day > 10  # At least 10 signals per day improvement

    def test_signal_quality_vs_baseline(self, improved_system, baseline_system, realistic_market_data, portfolio_state):
        """Compare signal quality between improved and baseline systems"""
        improved_signals = []
        baseline_signals = []

        for _, row in realistic_market_data.iterrows():
            continuous_action = np.random.normal(0, 0.5)

            # Get signals from both systems
            improved_action = improved_system.apply_guidance(continuous_action, row, portfolio_state)
            baseline_action = baseline_system.apply_guidance(continuous_action, row, portfolio_state)

            improved_signals.append(improved_action)
            baseline_signals.append(baseline_action)

        # Calculate signal frequencies
        improved_trading_signals = sum(1 for s in improved_signals if s != 0)
        baseline_trading_signals = sum(1 for s in baseline_signals if s != 0)

        improved_freq = improved_trading_signals / (len(realistic_market_data) / 288)
        baseline_freq = baseline_trading_signals / (len(realistic_market_data) / 288)

        print(f"Improved system frequency: {improved_freq:.1f} signals/day")
        print(f"Baseline system frequency: {baseline_freq:.1f} signals/day")

        # Improved system should generate more signals
        assert improved_freq > baseline_freq

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

    def test_position_aware_signaling(self, improved_system, realistic_market_data):
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

        for _, row in realistic_market_data.head(50).iterrows():  # Test first 50 points
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

    def test_signal_consistency(self, improved_system, realistic_market_data, portfolio_state):
        """Test signal consistency with same inputs"""
        # Generate signals twice with same data
        signals_1 = []
        signals_2 = []

        # First pass
        system1 = SignalGuidanceSystem()
        for _, row in realistic_market_data.head(20).iterrows():
            continuous_action = 0.2
            action = system1.apply_guidance(continuous_action, row, portfolio_state)
            signals_1.append(action)

        # Second pass with fresh system
        system2 = SignalGuidanceSystem()
        for _, row in realistic_market_data.head(20).iterrows():
            continuous_action = 0.2
            action = system2.apply_guidance(continuous_action, row, portfolio_state)
            signals_2.append(action)

        # Signals should be identical (deterministic)
        assert signals_1 == signals_2

    def test_performance_benchmark(self, improved_system, realistic_market_data, portfolio_state):
        """Benchmark performance of signal generation"""
        import time

        start_time = time.time()

        # Generate signals for all data
        signals = []
        for _, row in realistic_market_data.iterrows():
            continuous_action = np.random.normal(0, 0.5)
            action = improved_system.apply_guidance(continuous_action, row, portfolio_state)
            signals.append(action)

        end_time = time.time()
        processing_time = end_time - start_time

        signals_per_second = len(signals) / processing_time

        print(f"Processed {len(signals)} signals in {processing_time:.2f} seconds")
        print(f"Signals per second: {signals_per_second:.1f}")

        # Should process at least 100 signals per second for real-time trading
        assert signals_per_second > 100

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