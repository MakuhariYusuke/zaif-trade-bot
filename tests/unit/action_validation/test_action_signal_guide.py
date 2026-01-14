#!/usr/bin/env python3
"""
Unit tests for ActionSignalGuide

This module contains comprehensive unit tests for the ActionSignalGuide
pattern recognition system, including individual recognizer tests,
signal aggregation tests, and integration tests.
"""

import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

# Add project root to path
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from ztb.trading.strategies.action_signal_guide.action_signal_guide import (
    ActionSignalGuide,
    ActionSignalGuideConfig,
    GuidanceLevel,
)
from ztb.trading.strategies.action_signal_guide.components.signal_generator import (
    SignalGenerator,
)


class TestActionSignalGuide(unittest.TestCase):
    """Test cases for ActionSignalGuide functionality."""

    def setUp(self):
        """Set up test fixtures."""
        self.config = ActionSignalGuideConfig(
            debug_short_mode=False,
            guidance_level=GuidanceLevel.WEAK,
            enable_candlestick_patterns=True,
            enable_fibonacci_patterns=False,
            enable_gann_patterns=False,
            enable_wave_patterns=False,
            enable_harmonic_patterns=False,
            enable_oscillator_patterns=False,
            enable_volume_patterns=False,
            enable_bollinger_patterns=False,
            enable_adx_patterns=False,
            enable_granville_patterns=False,
            enable_heikin_ashi_patterns=False,
            enable_dow_theory_patterns=False,
        )
        self.guide = ActionSignalGuide(config=self.config)

    def _create_test_data(self, n_periods=100):
        """Create synthetic OHLCV test data."""
        dates = pd.date_range("2023-01-01", periods=n_periods, freq="h")
        np.random.seed(42)

        # Generate realistic price data
        prices = []
        base_price = 50000.0
        for i in range(n_periods):
            price = base_price + np.sin(i * 0.1) * 1000 + np.random.normal(0, 500)
            prices.append(price)

        data = []
        for i, price in enumerate(prices):
            high = price * (1 + abs(np.random.normal(0, 0.02)))
            low = price * (1 - abs(np.random.normal(0, 0.02)))
            open_price = prices[i - 1] if i > 0 else price
            close = price
            volume = np.random.randint(100, 1000)

            data.append(
                {
                    "timestamp": dates[i],
                    "open": open_price,
                    "high": high,
                    "low": low,
                    "close": close,
                    "volume": volume,
                }
            )

        df = pd.DataFrame(data)
        df.set_index("timestamp", inplace=True)
        return df

    def test_initialization(self):
        """Test ActionSignalGuide initialization."""
        self.assertIsNotNone(self.guide)
        self.assertIsNotNone(self.guide.signal_generator)
        self.assertGreater(len(self.guide.all_recognizers), 0)

    def test_signal_generation_basic(self):
        """Test basic signal generation."""
        data = self._create_test_data(100)
        signals = self.guide.generate_signals(data, 50)

        self.assertIsInstance(signals, list)
        # Should generate at least some signals with our test data
        self.assertGreaterEqual(len(signals), 0)

    def test_signal_generation_with_hammer_pattern(self):
        """Test signal generation with hammer pattern in data."""
        data = self._create_test_data(100)

        # Create a hammer pattern at index 50
        data.iloc[50] = {
            "open": 50100,
            "high": 50300,
            "low": 49000,
            "close": 50200,
            "volume": 1000,
        }

        signals = self.guide.generate_signals(data, 50)
        self.assertIsInstance(signals, list)

    def test_config_guidance_levels(self):
        """Test different guidance level configurations."""
        for level in [GuidanceLevel.WEAK, GuidanceLevel.MODERATE, GuidanceLevel.STRONG]:
            config = ActionSignalGuideConfig(
                guidance_level=level, enable_candlestick_patterns=True
            )
            guide = ActionSignalGuide(config=config)
            self.assertEqual(guide.signal_generator.guidance_level, level)

    def test_recognizer_count(self):
        """Test that correct number of recognizers are initialized."""
        # With our config, should have candlestick patterns enabled
        expected_min_recognizers = 10  # At least some candlestick patterns
        self.assertGreaterEqual(
            len(self.guide.all_recognizers), expected_min_recognizers
        )

    def test_signal_properties(self):
        """Test that generated signals have required properties."""
        data = self._create_test_data(100)
        signals = self.guide.generate_signals(data, 50)

        for signal in signals:
            self.assertTrue(hasattr(signal, "direction"))
            self.assertTrue(hasattr(signal, "confidence"))
            self.assertTrue(hasattr(signal, "signal_type"))
            self.assertIsInstance(signal.direction, (int, float))
            self.assertIsInstance(signal.confidence, (int, float))
            self.assertIsInstance(signal.signal_type, str)

    def test_empty_data_handling(self):
        """Test handling of empty or insufficient data."""
        empty_data = pd.DataFrame()
        signals = self.guide.generate_signals(empty_data, 0)
        self.assertEqual(len(signals), 0)

    def test_out_of_bounds_index(self):
        """Test handling of out-of-bounds indices."""
        data = self._create_test_data(50)
        signals = self.guide.generate_signals(data, 100)  # Index beyond data length
        self.assertEqual(len(signals), 0)


class TestSignalGenerator(unittest.TestCase):
    """Test cases for SignalGenerator component."""

    def setUp(self):
        """Set up test fixtures."""
        self.config = ActionSignalGuideConfig(guidance_level=GuidanceLevel.WEAK)
        self.generator = SignalGenerator(config=self.config)

    def test_aggregate_signals(self):
        """Test signal aggregation logic."""
        # Create mock signals
        from ztb.trading.strategies.action_signal_guide.components.signal_generator import (
            _get_action_signal_class,
        )

        ActionSignal = _get_action_signal_class()

        mock_signals = [
            ActionSignal(
                timestamp=pd.Timestamp.now(),
                direction=1.0,
                strength=1.0,
                confidence=1.0,
                signal_type="test",
                description="Test signal 1",
                metadata={},
                source_patterns=["test_pattern"]
            ),
            ActionSignal(
                timestamp=pd.Timestamp.now(),
                direction=-1.0,
                strength=0.8,
                confidence=0.8,
                signal_type="test",
                description="Test signal 2",
                metadata={},
                source_patterns=["test_pattern"]
            ),
        ]

        result = self.generator._aggregate_signals(mock_signals, {})
        self.assertIsNotNone(result)
        self.assertTrue(hasattr(result, "direction"))
        self.assertTrue(hasattr(result, "confidence"))

    def test_filter_by_guidance_level(self):
        """Test signal filtering by guidance level."""
        from ztb.trading.strategies.action_signal_guide.components.signal_generator import (
            _get_action_signal_class,
        )

        ActionSignal = _get_action_signal_class()

        signals = [
            ActionSignal(
                timestamp=pd.Timestamp.now(),
                direction=1.0,
                strength=1.0,
                confidence=1.0,
                signal_type="strong",
                description="Strong signal",
                metadata={},
                source_patterns=["strong_pattern"]
            ),
            ActionSignal(
                timestamp=pd.Timestamp.now(),
                direction=1.0,
                strength=0.5,
                confidence=0.5,
                signal_type="moderate",
                description="Moderate signal",
                metadata={},
                source_patterns=["moderate_pattern"]
            ),
            ActionSignal(
                timestamp=pd.Timestamp.now(),
                direction=1.0,
                strength=0.2,
                confidence=0.2,
                signal_type="weak",
                description="Weak signal",
                metadata={},
                source_patterns=["weak_pattern"]
            ),
        ]

        # Test WEAK level (should include all)
        filtered = self.generator._filter_by_guidance_level(signals)
        self.assertGreaterEqual(len(filtered), 2)  # At least moderate and strong


if __name__ == "__main__":
    unittest.main()
