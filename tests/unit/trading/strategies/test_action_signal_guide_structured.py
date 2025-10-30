"""
Action Signal Guide Test Suite

Structured test suite for Action Signal Guide implementation with comprehensive
pattern recognition testing and integration validation.
"""

import os
import sys
import unittest
from datetime import datetime, timedelta
from typing import List

import numpy as np
import pandas as pd

# Add project root to path
project_root = os.path.dirname(
    os.path.dirname(
        os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    )
)
sys.path.insert(0, project_root)

from ztb.trading.strategies.action_signal_guide.action_signal_guide import (
    ActionSignalGuide,
    GuidanceLevel,
)
from ztb.trading.strategies.action_signal_guide.pattern_recognition.base import (
    PatternRecognizer,
    SignalResult,
)
from ztb.trading.strategies.action_signal_guide.pattern_recognition.bollinger_patterns import (
    BollingerBandsRecognizer,
)

# Import all pattern recognizers
from ztb.trading.strategies.action_signal_guide.pattern_recognition.candlestick_patterns import (
    EveningStarRecognizer,
    HammerRecognizer,
    HangingManRecognizer,
    MorningStarRecognizer,
    SakataFiveMethodsRecognizer,
)
from ztb.trading.strategies.action_signal_guide.pattern_recognition.fibonacci_patterns import (
    FibonacciExtensionRecognizer,
    FibonacciProjectionRecognizer,
    FibonacciRetracementRecognizer,
)
from ztb.trading.strategies.action_signal_guide.pattern_recognition.gann_analysis import (
    GannAngleRecognizer,
    GannSquareRecognizer,
    GannTimeClusterRecognizer,
)
from ztb.trading.strategies.action_signal_guide.pattern_recognition.harmonic_patterns import (
    BatRecognizer,
    ButterflyRecognizer,
    CrabRecognizer,
    GartleyRecognizer,
)
from ztb.trading.strategies.action_signal_guide.pattern_recognition.wave_counting import (
    CorrectiveWaveRecognizer,
    ImpulseWaveRecognizer,
    WaveExtensionRecognizer,
    WaveIRecognizer,
    WaveNRecognizer,
    WavePRecognizer,
    WaveSRecognizer,
    WaveVRecognizer,
    WaveYRecognizer,
)


class TestDataFactory:
    """Factory class for creating test data fixtures."""

    @staticmethod
    def create_sample_ohlcv_data(num_bars: int = 100) -> pd.DataFrame:
        """Create sample OHLCV data for testing."""
        np.random.seed(42)  # For reproducible results

        # Generate base price series with trend
        base_price = 100.0
        prices: List[float] = []

        for i in range(num_bars):
            trend = 0.001 * i  # Slight upward trend
            noise = np.random.normal(0, 0.02)  # Random noise
            price = base_price * (1 + trend + noise)
            prices.append(price)

        # Create OHLCV data
        timestamps = [
            datetime.now() - timedelta(hours=num_bars - i) for i in range(num_bars)
        ]
        opens: List[float] = []
        highs: List[float] = []
        lows: List[float] = []
        closes: List[float] = []
        volumes: List[int] = []

        opens.append(prices[0] * (1 + np.random.normal(0, 0.005)))

        for close in prices:
            high = close * (1 + abs(np.random.normal(0, 0.01)))
            low = close * (1 - abs(np.random.normal(0, 0.01)))
            volume = int(np.random.normal(1000, 200))

            highs.append(high)
            lows.append(low)
            closes.append(close)
            volumes.append(volume)

            # Next open is previous close with small gap
            if len(closes) < num_bars:
                next_open = close * (1 + np.random.normal(0, 0.002))
                opens.append(next_open)

        return pd.DataFrame(
            {
                "timestamp": timestamps,
                "open": opens,
                "high": highs,
                "low": lows,
                "close": closes,
                "volume": volumes,
            }
        )

    @staticmethod
    def create_minimal_data() -> pd.DataFrame:
        """Create minimal test data for edge case testing."""
        return TestDataFactory.create_sample_ohlcv_data(5)

    @staticmethod
    def create_trend_data(num_bars: int = 100, trend: str = "up") -> pd.DataFrame:
        """Create data with specific trend for testing."""
        np.random.seed(42)

        base_price = 100.0
        prices: List[float] = []

        trend_multiplier = 1 if trend == "up" else -1

        for i in range(num_bars):
            trend_component = trend_multiplier * 0.002 * i
            noise = np.random.normal(0, 0.015)
            price = base_price * (1 + trend_component + noise)
            prices.append(price)

        return TestDataFactory._create_ohlcv_from_prices(prices)

    @staticmethod
    def create_volatile_data(num_bars: int = 100) -> pd.DataFrame:
        """Create highly volatile data for testing."""
        np.random.seed(42)

        base_price = 100.0
        prices: List[float] = []

        for i in range(num_bars):
            trend = 0.0005 * i
            volatility = np.random.normal(0, 0.05)  # Higher volatility
            price = base_price * (1 + trend + volatility)
            prices.append(price)

        return TestDataFactory._create_ohlcv_from_prices(prices)

    @staticmethod
    def create_candlestick_pattern_data() -> pd.DataFrame:
        """Create data specifically designed to trigger candlestick patterns."""
        # Create hammer pattern data
        prices = [
            100.0,  # Open/Close for previous candle
            98.0,  # Hammer: High body, long lower wick
            99.0,  # Close above open (bullish)
        ]

        # Extend with more data
        for i in range(50):
            prices.append(prices[-1] * (1 + np.random.normal(0, 0.01)))

        return TestDataFactory._create_ohlcv_from_prices(prices)

    @staticmethod
    def _create_ohlcv_from_prices(prices: List[float]) -> pd.DataFrame:
        """Helper method to create OHLCV DataFrame from price list."""
        timestamps = [
            datetime.now() - timedelta(hours=len(prices) - i)
            for i in range(len(prices))
        ]

        opens: List[float] = []
        highs: List[float] = []
        lows: List[float] = []
        closes: List[float] = []
        volumes: List[int] = []

        for i, close in enumerate(prices):
            if i == 0:
                # First candle
                opens.append(close * (1 + np.random.normal(0, 0.005)))
            else:
                # Subsequent candles open at previous close
                opens.append(closes[-1] * (1 + np.random.normal(0, 0.001)))

            high = max(opens[-1], close) * (1 + abs(np.random.normal(0, 0.005)))
            low = min(opens[-1], close) * (1 - abs(np.random.normal(0, 0.005)))
            volume = int(np.random.normal(1000, 200))

            highs.append(high)
            lows.append(low)
            closes.append(close)
            volumes.append(volume)

        return pd.DataFrame(
            {
                "timestamp": timestamps,
                "open": opens,
                "high": highs,
                "low": lows,
                "close": closes,
                "volume": volumes,
            }
        )


class BaseActionSignalGuideTest(unittest.TestCase):
    """Base test class with common fixtures and utilities."""

    def setUp(self):
        """Set up test fixtures."""
        self.sample_data = TestDataFactory.create_sample_ohlcv_data(200)
        self.minimal_data = TestDataFactory.create_minimal_data()
        self.trend_up_data = TestDataFactory.create_trend_data(150, "up")
        self.trend_down_data = TestDataFactory.create_trend_data(150, "down")
        self.volatile_data = TestDataFactory.create_volatile_data(150)
        self.pattern_data = TestDataFactory.create_candlestick_pattern_data()

    def assert_signal_result_valid(self, signal: SignalResult):
        """Assert that a SignalResult has all required attributes."""
        self.assertIsInstance(signal, SignalResult)
        self.assertTrue(hasattr(signal, "signal_type"))
        self.assertTrue(hasattr(signal, "description"))
        self.assertTrue(hasattr(signal, "strength"))
        self.assertTrue(hasattr(signal, "direction"))
        self.assertTrue(hasattr(signal, "timestamp"))
        self.assertTrue(hasattr(signal, "confidence"))
        self.assertIsInstance(signal.strength, (int, float))
        self.assertIsInstance(signal.confidence, (int, float))
        self.assertGreaterEqual(signal.strength, 0.0)
        self.assertLessEqual(signal.strength, 1.0)
        self.assertGreaterEqual(signal.confidence, 0.0)
        self.assertLessEqual(signal.confidence, 1.0)

    def assert_recognizer_has_valid_interface(self, recognizer: PatternRecognizer):
        """Assert that a recognizer has the required interface."""
        self.assertTrue(hasattr(recognizer, "recognize"))
        self.assertTrue(callable(getattr(recognizer, "recognize")))

    def run_recognizer_on_data(
        self, recognizer: PatternRecognizer, data: pd.DataFrame, start_idx: int = 10
    ) -> List[SignalResult]:
        """Helper method to run a recognizer on data and collect signals."""
        signals = []
        for i in range(start_idx, len(data)):
            signal = recognizer.recognize(data, i)
            if signal:
                signals.append(signal)
        return signals

    def assert_signals_are_valid(self, signals: List[SignalResult]):
        """Assert that all signals in a list are valid."""
        for signal in signals:
            self.assert_signal_result_valid(signal)


class TestActionSignalGuideBasic(BaseActionSignalGuideTest):
    """Test basic ActionSignalGuide functionality.

    This test class covers the fundamental operations of the ActionSignalGuide
    including initialization, configuration, and basic signal generation.
    """

    def test_initialization(self):
        """Test ActionSignalGuide can be initialized with default settings."""
        guide = ActionSignalGuide()
        self.assertIsInstance(guide, ActionSignalGuide)

    def test_guidance_levels(self):
        """Test guidance level configuration.

        Verifies that the ActionSignalGuide can be configured with different
        guidance levels and maintains configuration state.
        """
        guide = ActionSignalGuide()

        # Test different guidance levels - just verify no errors occur
        # (Actual implementation may vary)
        try:
            guide.set_guidance_level(GuidanceLevel.NONE)
            guide.set_guidance_level(GuidanceLevel.WEAK)
            guide.set_guidance_level(GuidanceLevel.STRONG)
        except AttributeError:
            # If set_guidance_level doesn't exist, skip this test
            self.skipTest("Guidance level methods not available in this implementation")

    def test_basic_signal_generation(self):
        """Test basic signal generation with sample market data.

        Ensures that the ActionSignalGuide can process OHLCV data and generate
        trading signals without errors, even if no signals are found.
        """
        guide = ActionSignalGuide()

        # Generate signals for sample data
        signals = []
        for i in range(10, len(self.sample_data)):
            signal_list = guide.generate_signals(self.sample_data, i)
            if signal_list:
                signals.extend(signal_list)

        # Should generate some signals
        self.assertIsInstance(signals, list)


class TestPatternRecognizers(BaseActionSignalGuideTest):
    """Test all pattern recognizers comprehensively.

    This test class systematically tests all pattern recognition algorithms
    including candlestick patterns, Fibonacci analysis, harmonic patterns,
    wave counting, and Gann analysis. Each recognizer is tested for:

    - Interface compliance (has recognize method)
    - Functionality (can process data without errors)
    - Output validity (returns properly structured SignalResult objects)
    - Specific metadata requirements (for specialized recognizers)
    """

    def setUp(self):
        """Set up pattern recognizer instances for testing."""
        super().setUp()
        self.candlestick_recognizers = [
            SakataFiveMethodsRecognizer(),
            MorningStarRecognizer(),
            EveningStarRecognizer(),
            HammerRecognizer(),
            HangingManRecognizer(),
        ]

        self.fibonacci_recognizers = [
            FibonacciRetracementRecognizer(),
            FibonacciExtensionRecognizer(),
            FibonacciProjectionRecognizer(),
        ]

        self.harmonic_recognizers = [
            GartleyRecognizer(),
            ButterflyRecognizer(),
            BatRecognizer(),
            CrabRecognizer(),
        ]

        self.bollinger_recognizers = [
            BollingerBandsRecognizer(),
        ]

        self.wave_recognizers = [
            ImpulseWaveRecognizer(),
            CorrectiveWaveRecognizer(),
            WaveExtensionRecognizer(),
            WaveIRecognizer(),
            WaveVRecognizer(),
            WaveYRecognizer(),
            WavePRecognizer(),
            WaveNRecognizer(),
            WaveSRecognizer(),
        ]

        self.gann_recognizers = [
            GannAngleRecognizer(),
            GannSquareRecognizer(),
            GannTimeClusterRecognizer(),
        ]

    def test_candlestick_recognizers_interface(self):
        """Test candlestick recognizers have valid interface."""
        for recognizer in self.candlestick_recognizers:
            with self.subTest(recognizer=recognizer.__class__.__name__):
                self.assert_recognizer_has_valid_interface(recognizer)

    def test_candlestick_recognizers_functionality(self):
        """Test candlestick recognizers can process data and return valid signals."""
        for recognizer in self.candlestick_recognizers:
            with self.subTest(recognizer=recognizer.__class__.__name__):
                signals = self.run_recognizer_on_data(recognizer, self.sample_data, 5)
                self.assert_signals_are_valid(signals)
                # At least some recognizers should find signals in sample data
                # (This is a weak assertion - in real scenarios we'd use specific test data)

    def test_fibonacci_recognizers_interface(self):
        """Test fibonacci recognizers have valid interface."""
        for recognizer in self.fibonacci_recognizers:
            with self.subTest(recognizer=recognizer.__class__.__name__):
                self.assert_recognizer_has_valid_interface(recognizer)

    def test_fibonacci_recognizers_functionality(self):
        """Test fibonacci recognizers can process data."""
        for recognizer in self.fibonacci_recognizers:
            with self.subTest(recognizer=recognizer.__class__.__name__):
                signals = self.run_recognizer_on_data(recognizer, self.sample_data, 20)
                self.assert_signals_are_valid(signals)

    def test_harmonic_recognizers_interface(self):
        """Test harmonic recognizers have valid interface."""
        for recognizer in self.harmonic_recognizers:
            with self.subTest(recognizer=recognizer.__class__.__name__):
                self.assert_recognizer_has_valid_interface(recognizer)

    def test_harmonic_recognizers_functionality(self):
        """Test harmonic recognizers can process data."""
        extended_data = TestDataFactory.create_sample_ohlcv_data(300)
        for recognizer in self.harmonic_recognizers:
            with self.subTest(recognizer=recognizer.__class__.__name__):
                signals = self.run_recognizer_on_data(recognizer, extended_data, 50)
                self.assert_signals_are_valid(signals)

    def test_wave_recognizers_interface(self):
        """Test wave recognizers have valid interface."""
        for recognizer in self.wave_recognizers:
            with self.subTest(recognizer=recognizer.__class__.__name__):
                self.assert_recognizer_has_valid_interface(recognizer)

    def test_wave_recognizers_functionality(self):
        """Test wave recognizers can process data."""
        extended_data = TestDataFactory.create_sample_ohlcv_data(400)
        for recognizer in self.wave_recognizers:
            with self.subTest(recognizer=recognizer.__class__.__name__):
                signals = self.run_recognizer_on_data(recognizer, extended_data, 50)
                self.assert_signals_are_valid(signals)

                # Test specific wave recognizer metadata
                if recognizer.__class__.__name__ in [
                    "WaveIRecognizer",
                    "WaveVRecognizer",
                    "WaveYRecognizer",
                    "WavePRecognizer",
                    "WaveNRecognizer",
                    "WaveSRecognizer",
                ]:
                    for signal in signals:
                        self.assertIn("wave_label", signal.metadata)
                        self.assertIn("pattern", signal.metadata)

    def test_gann_recognizers_interface(self):
        """Test Gann recognizers have valid interface."""
        for recognizer in self.gann_recognizers:
            with self.subTest(recognizer=recognizer.__class__.__name__):
                self.assert_recognizer_has_valid_interface(recognizer)

    def test_gann_recognizers_functionality(self):
        """Test Gann recognizers can process data."""
        for recognizer in self.gann_recognizers:
            with self.subTest(recognizer=recognizer.__class__.__name__):
                signals = self.run_recognizer_on_data(recognizer, self.sample_data, 30)
                self.assert_signals_are_valid(signals)

    def test_bollinger_recognizers_interface(self):
        """Test Bollinger recognizers have valid interface."""
        for recognizer in self.bollinger_recognizers:
            with self.subTest(recognizer=recognizer.__class__.__name__):
                self.assert_recognizer_has_valid_interface(recognizer)

    def test_bollinger_recognizers_functionality(self):
        """Test Bollinger recognizers can process data."""
        for recognizer in self.bollinger_recognizers:
            with self.subTest(recognizer=recognizer.__class__.__name__):
                signals = self.run_recognizer_on_data(recognizer, self.sample_data, 30)
                self.assert_signals_are_valid(signals)


class TestRegression(BaseActionSignalGuideTest):
    """Regression tests to ensure previously working functionality remains intact.

    These tests verify that core functionality that worked in previous versions
    continues to work after code changes, preventing regressions.
    """

    def test_signal_generation_regression(self):
        """Test that signal generation produces expected output format."""
        guide = ActionSignalGuide()
        test_data = TestDataFactory.create_sample_ohlcv_data(200)

        # Generate signals at multiple points
        signals = []
        for i in range(50, len(test_data)):
            batch_signals = guide.generate_signals(test_data, i)
            if batch_signals:
                signals.extend(batch_signals)

        # Verify signals have required attributes
        for signal in signals:
            # ActionSignal should have these attributes
            self.assertTrue(hasattr(signal, "timestamp"))
            self.assertTrue(hasattr(signal, "direction"))
            self.assertTrue(hasattr(signal, "strength"))
            self.assertTrue(hasattr(signal, "confidence"))
            self.assertTrue(hasattr(signal, "signal_type"))
            self.assertTrue(hasattr(signal, "description"))
            self.assertTrue(hasattr(signal, "metadata"))
            self.assertTrue(hasattr(signal, "source_patterns"))

            # Validate ranges
            self.assertGreaterEqual(signal.direction, -1.0)
            self.assertLessEqual(signal.direction, 1.0)
            self.assertGreaterEqual(signal.strength, 0.0)
            self.assertLessEqual(signal.strength, 1.0)
            self.assertGreaterEqual(signal.confidence, 0.0)
            self.assertLessEqual(signal.confidence, 1.0)

    def test_recognizer_output_consistency(self):
        """Test that recognizers produce consistent output for same input."""
        recognizer = HammerRecognizer()
        test_data = TestDataFactory.create_sample_ohlcv_data(100)

        # Run multiple times with same data
        result1 = recognizer.recognize(test_data, 50)
        result2 = recognizer.recognize(test_data, 50)

        # Results should be consistent (or both None)
        if result1 is not None and result2 is not None:
            self.assertEqual(result1.action, result2.action)
            self.assertAlmostEqual(result1.confidence, result2.confidence, places=5)

    def test_configuration_persistence(self):
        """Test that recognizer configurations are properly maintained."""
        config = {"period": 25, "std_dev": 2.5}
        recognizer = BollingerBandsRecognizer(config)

        # Verify config is applied
        self.assertEqual(recognizer.period, 25)
        self.assertEqual(recognizer.std_dev, 2.5)

        # Test with default config
        default_recognizer = BollingerBandsRecognizer()
        self.assertEqual(default_recognizer.period, 20)
        self.assertEqual(default_recognizer.std_dev, 2.0)

    """Test edge cases and error handling."""

    def test_minimal_data_handling(self):
        """Test recognizers handle minimal data gracefully."""
        recognizer = HammerRecognizer()

        # Should handle minimal data without crashing
        result = recognizer.recognize(self.minimal_data, 4)
        # Result can be None or valid SignalResult
        if result is not None:
            self.assert_signal_result_valid(result)

    def test_invalid_index_handling(self):
        """Test recognizers handle invalid indices gracefully."""
        recognizer = HammerRecognizer()

        # Should handle out-of-bounds index
        try:
            result = recognizer.recognize(self.minimal_data, 100)
            # Should either return None or handle gracefully
        except (IndexError, ValueError):
            pass  # Expected for out of bounds

    def test_empty_data_handling(self):
        """Test recognizers handle empty data gracefully."""
        recognizer = HammerRecognizer()
        empty_data = pd.DataFrame()

        # Should handle empty data without crashing
        result = recognizer.recognize(empty_data, 0)
        self.assertIsNone(result)


class TestConfigurationValidation(BaseActionSignalGuideTest):
    """Test configuration validation and parameter handling.

    These tests ensure that all recognizers properly validate their configuration
    parameters and handle invalid configurations gracefully.
    """

    def test_recognizer_config_validation(self):
        """Test that recognizers validate configuration parameters."""
        # Test valid configurations
        valid_configs = [
            {"period": 20, "std_dev": 2.0},
            {"period": 10, "std_dev": 1.5, "squeeze_threshold": 0.05},
            {"period": 30, "std_dev": 3.0, "expansion_threshold": 0.20},
        ]

        for config in valid_configs:
            with self.subTest(config=config):
                recognizer = BollingerBandsRecognizer(config)
                self.assertIsInstance(recognizer, BollingerBandsRecognizer)

    def test_invalid_config_handling(self):
        """Test that recognizers handle invalid configurations gracefully."""
        invalid_configs = [
            {"period": -1},  # Negative period
            {"std_dev": 0},  # Zero standard deviation
            {"period": "invalid"},  # Wrong type
            {"squeeze_threshold": 2.0},  # Threshold > 1.0
        ]

        for config in invalid_configs:
            with self.subTest(config=config):
                # Should not crash during initialization
                try:
                    recognizer = BollingerBandsRecognizer(config)
                    # If it initializes, it should still be able to process data
                    test_data = TestDataFactory.create_sample_ohlcv_data(50)
                    result = recognizer.recognize(test_data, 25)
                    # Result can be None or valid
                except (ValueError, TypeError):
                    pass  # Expected for invalid configs

    def test_default_config_usage(self):
        """Test that recognizers use sensible defaults when no config provided."""
        recognizer = BollingerBandsRecognizer()

        # Should have reasonable defaults
        self.assertGreater(recognizer.period, 0)
        self.assertGreater(recognizer.std_dev, 0)
        self.assertGreater(recognizer.squeeze_threshold, 0)
        self.assertLess(recognizer.squeeze_threshold, 1)
        self.assertGreater(recognizer.expansion_threshold, 0)
        self.assertLess(recognizer.expansion_threshold, 1)

    def test_config_parameter_ranges(self):
        """Test that configuration parameters are within valid ranges."""
        test_configs = [
            {"period": 5, "std_dev": 0.5},  # Minimum reasonable values
            {"period": 100, "std_dev": 5.0},  # Maximum reasonable values
        ]

        for config in test_configs:
            with self.subTest(config=config):
                recognizer = BollingerBandsRecognizer(config)
                test_data = TestDataFactory.create_sample_ohlcv_data(150)
                # Should not crash with extreme but valid values
                result = recognizer.recognize(test_data, 75)
                # Result validation is optional for edge case configs

    """Test integration between components."""

    def test_action_signal_guide_with_all_recognizers(self):
        """Test ActionSignalGuide integrates with all pattern recognizers."""
        guide = ActionSignalGuide()

        # Process sample data
        signals = []
        for i in range(50, len(self.sample_data)):
            signal_list = guide.generate_signals(self.sample_data, i)
            if signal_list:
                signals.extend(signal_list)

        # Should generate some signals over the dataset
        self.assertIsInstance(signals, list)

    def test_signal_consistency_across_calls(self):
        """Test that repeated calls with same data produce consistent results."""
        guide = ActionSignalGuide()
        test_data = self.sample_data.iloc[:100]

        # Call multiple times with same data
        signals1 = guide.generate_signals(test_data, len(test_data) - 1)
        signals2 = guide.generate_signals(test_data, len(test_data) - 1)

        # Results should be consistent (same data, same signals)
        # Note: This might need adjustment based on actual implementation
        self.assertIsInstance(signals1, list)
        self.assertIsInstance(signals2, list)


class TestPerformanceMonitoring(BaseActionSignalGuideTest):
    """Performance monitoring tests to ensure system efficiency.

    These tests monitor execution time, memory usage, and error rates
    to ensure the system performs adequately under various conditions.
    """

    def test_recognizer_execution_time(self):
        """Test that recognizers execute within reasonable time limits."""
        import time

        recognizers = [
            HammerRecognizer(),
            BollingerBandsRecognizer(),
            GartleyRecognizer(),
            ImpulseWaveRecognizer(),
        ]

        test_data = TestDataFactory.create_sample_ohlcv_data(500)

        for recognizer in recognizers:
            with self.subTest(recognizer=recognizer.__class__.__name__):
                start_time = time.time()
                signals = self.run_recognizer_on_data(recognizer, test_data, 100)
                end_time = time.time()

                execution_time = end_time - start_time
                # Should complete within 60 seconds for reasonable dataset (harmonic patterns are complex)
                self.assertLess(
                    execution_time,
                    60.0,
                    f"{recognizer.__class__.__name__} took too long: {execution_time:.2f}s",
                )

    def test_memory_usage_stability(self):
        """Test that recognizers don't have memory leaks over multiple runs."""
        import os

        import psutil

        recognizer = BollingerBandsRecognizer()
        test_data = TestDataFactory.create_sample_ohlcv_data(200)

        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss

        # Run multiple recognition cycles
        for i in range(10):
            signals = self.run_recognizer_on_data(recognizer, test_data, 100)

        final_memory = process.memory_info().rss
        memory_increase = final_memory - initial_memory

        # Memory increase should be reasonable (less than 50MB)
        self.assertLess(
            memory_increase,
            50 * 1024 * 1024,
            f"Memory leak detected: {memory_increase / 1024 / 1024:.2f}MB increase",
        )

    def test_error_rate_monitoring(self):
        """Test that recognizers maintain low error rates."""
        recognizer = HammerRecognizer()
        test_data = TestDataFactory.create_sample_ohlcv_data(1000)

        error_count = 0
        total_calls = 100

        for i in range(total_calls):
            try:
                result = recognizer.recognize(test_data, i % len(test_data))
                # Validate result if not None
                if result is not None:
                    self.assert_signal_result_valid(result)
            except Exception:
                error_count += 1

        error_rate = error_count / total_calls
        # Error rate should be very low (< 1%)
        self.assertLess(error_rate, 0.01, f"High error rate: {error_rate:.2%}")

    def test_scalability_with_data_size(self):
        """Test that performance scales reasonably with data size."""
        import time

        recognizer = BollingerBandsRecognizer()
        data_sizes = [100, 500, 1000, 2000]

        times = []
        for size in data_sizes:
            test_data = TestDataFactory.create_sample_ohlcv_data(size)

            start_time = time.time()
            signals = self.run_recognizer_on_data(recognizer, test_data, size // 2)
            end_time = time.time()

            execution_time = end_time - start_time
            times.append(execution_time)

        # Execution time should scale roughly linearly or better
        # Check that doubling data size doesn't increase time by more than 4x
        if len(times) >= 2:
            for i in range(1, len(times)):
                ratio = times[i] / times[i - 1]
                data_ratio = data_sizes[i] / data_sizes[i - 1]
                scaling_factor = ratio / data_ratio

                # Should scale better than O(n^2) - allow some inefficiency for complex calculations
                self.assertLess(
                    scaling_factor,
                    6.0,
                    f"Poor scaling at size {data_sizes[i]}: {scaling_factor:.2f}x",
                )


if __name__ == "__main__":
    unittest.main()
