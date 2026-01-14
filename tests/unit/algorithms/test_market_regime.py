#!/usr/bin/env python3
"""
Unit tests for Market Regime Analysis Components.

This module contains comprehensive unit tests for the market regime analysis components
including MarketRegimeDetector, RegimeAdaptiveSignalProcessor, and MarketConditionAnalyzer.
"""

import sys
import unittest
from pathlib import Path

import pandas as pd
import numpy as np

# Add project root to path
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from ztb.trading.strategies.action_signal_guide.components.market_regime import (
    MarketRegimeDetector,
    RegimeAdaptiveSignalProcessor,
    MarketConditionAnalyzer,
    MarketRegime,
)


class MockActionSignal:
    """Mock ActionSignal for testing."""

    def __init__(self, action="BUY", confidence=0.8, pattern_type="fibonacci"):
        self.action = action
        self.confidence = confidence
        self.pattern_type = pattern_type
        self.price = 100.0
        self.timestamp = pd.Timestamp.now()


class TestMarketRegimeDetector(unittest.TestCase):
    """Test cases for MarketRegimeDetector."""

    def setUp(self):
        """Set up test fixtures."""
        self.detector = MarketRegimeDetector()

    def test_initialization(self):
        """Test detector initialization."""
        self.assertIsInstance(self.detector, MarketRegimeDetector)
        self.assertEqual(self.detector.transition_threshold, 0.6)
        self.assertEqual(self.detector.stability_window, 20)

    def test_detect_regime_trending_bullish(self):
        """Test detection of trending bullish regime."""
        # Create bullish trending data
        dates = pd.date_range(start="2023-01-01", periods=50, freq="D")
        prices = [100 + i * 0.5 + np.random.normal(0, 0.5) for i in range(50)]  # Upward trend

        data = pd.DataFrame({
            "open": prices,
            "high": [p + abs(np.random.normal(0, 1)) for p in prices],
            "low": [p - abs(np.random.normal(0, 1)) for p in prices],
            "close": prices,
            "volume": [1000 + np.random.normal(0, 100) for _ in range(50)]
        })

        regime = self.detector.detect_regime(data)

        self.assertIsInstance(regime, MarketRegime)

    def test_detect_regime_ranging(self):
        """Test detection of ranging regime."""
        # Create ranging data (no clear trend)
        dates = pd.date_range(start="2023-01-01", periods=50, freq="D")
        prices = [100 + np.random.normal(0, 2) for _ in range(50)]  # Random walk

        data = pd.DataFrame({
            "open": prices,
            "high": [p + abs(np.random.normal(0, 1)) for p in prices],
            "low": [p - abs(np.random.normal(0, 1)) for p in prices],
            "close": prices,
            "volume": [1000 + np.random.normal(0, 100) for _ in range(50)]
        })

        regime = self.detector.detect_regime(data)

        self.assertIsInstance(regime, MarketRegime)

    def test_detect_regime_high_volatility(self):
        """Test detection of high volatility regime."""
        # Create high volatility data
        dates = pd.date_range(start="2023-01-01", periods=50, freq="D")
        prices = [100]
        for _ in range(49):
            change = np.random.normal(0, 0.05)  # 5% daily volatility
            prices.append(prices[-1] * (1 + change))

        data = pd.DataFrame({
            "open": prices,
            "high": [p * 1.1 for p in prices],  # High volatility
            "low": [p * 0.9 for p in prices],
            "close": prices,
            "volume": [1000 + np.random.normal(0, 100) for _ in range(50)]
        })

        regime = self.detector.detect_regime(data)

        self.assertIsInstance(regime, MarketRegime)

    def test_get_regime_stability(self):
        """Test regime stability calculation."""
        # Add some regime history
        for _ in range(25):
            self.detector.regime_history.append({
                "regime": MarketRegime.TRENDING_BULLISH,
                "indicators": {"trend_strength": 0.8, "volatility": 0.02}
            })

        stability = self.detector.get_regime_stability()

        self.assertIsInstance(stability, float)
        self.assertGreaterEqual(stability, 0.0)
        self.assertLessEqual(stability, 1.0)

    def test_calculate_trend_strength(self):
        """Test trend strength calculation."""
        # Create upward trending data
        prices = [100, 101, 102, 103, 104, 105]
        data = pd.DataFrame({"close": prices})

        trend_strength = self.detector._calculate_trend_strength(data)

        self.assertIsInstance(trend_strength, float)

    def test_calculate_volatility(self):
        """Test volatility calculation."""
        # Create volatile data
        prices = [100, 105, 95, 110, 90, 115]
        data = pd.DataFrame({"close": prices})

        volatility = self.detector._calculate_volatility(data)

        self.assertIsInstance(volatility, float)
        self.assertGreaterEqual(volatility, 0.0)

    def test_classify_regime(self):
        """Test regime classification."""
        # Test various indicator combinations
        test_cases = [
            (0.05, 0.02, 0.3, 0.1, MarketRegime.TRENDING_BULLISH),  # Strong uptrend
            (-0.05, 0.02, 0.3, -0.1, MarketRegime.TRENDING_BEARISH),  # Strong downtrend
            (0.01, 0.08, 0.8, 0.0, MarketRegime.HIGH_VOLATILITY),  # High volatility
            (0.005, 0.005, 0.9, 0.0, MarketRegime.RANGING),  # Ranging
        ]

        for trend, vol, range_bound, momentum, expected in test_cases:
            with self.subTest(trend=trend, vol=vol, range_bound=range_bound, momentum=momentum):
                regime = self.detector._classify_regime(trend, vol, range_bound, momentum)
                self.assertIsInstance(regime, MarketRegime)

    def test_detect_regime_boundary_cases(self):
        """Test regime detection with boundary cases."""
        # Test with empty data
        empty_data = pd.DataFrame()
        regime = self.detector.detect_regime(empty_data)
        self.assertIsInstance(regime, MarketRegime)
        self.assertEqual(regime, MarketRegime.RANGING)  # Default for insufficient data

        # Test with single data point
        single_data = pd.DataFrame({
            "open": [100.0],
            "high": [105.0],
            "low": [95.0],
            "close": [100.0],
            "volume": [1000]
        })
        regime = self.detector.detect_regime(single_data)
        self.assertIsInstance(regime, MarketRegime)
        self.assertEqual(regime, MarketRegime.RANGING)  # Default for insufficient data

        # Test with extreme volatility (all same prices)
        flat_data = pd.DataFrame({
            "open": [100.0] * 50,
            "high": [100.0] * 50,
            "low": [100.0] * 50,
            "close": [100.0] * 50,
            "volume": [1000] * 50
        })
        regime = self.detector.detect_regime(flat_data)
        self.assertIsInstance(regime, MarketRegime)

        # Test with extreme volatility (alternating high/low)
        volatile_data = pd.DataFrame({
            "open": [100.0, 200.0] * 25,
            "high": [110.0, 210.0] * 25,
            "low": [90.0, 190.0] * 25,
            "close": [100.0, 200.0] * 25,
            "volume": [1000, 2000] * 25
        })
        regime = self.detector.detect_regime(volatile_data)
        self.assertIsInstance(regime, MarketRegime)


class TestRegimeAdaptiveSignalProcessor(unittest.TestCase):
    """Test cases for RegimeAdaptiveSignalProcessor."""

    def setUp(self):
        """Set up test fixtures."""
        self.processor = RegimeAdaptiveSignalProcessor()

    def test_initialization(self):
        """Test processor initialization."""
        self.assertIsInstance(self.processor, RegimeAdaptiveSignalProcessor)
        self.assertIsInstance(self.processor.regime_detector, MarketRegimeDetector)

    def test_process_signals_for_regime(self):
        """Test signal processing for specific regime."""
        signals = [
            MockActionSignal(action="BUY", confidence=0.8, pattern_type="fibonacci"),
            MockActionSignal(action="SELL", confidence=0.7, pattern_type="bollinger"),
        ]

        # Create trending bullish market data
        data = pd.DataFrame({
            "open": [100, 101, 102, 103, 104],
            "high": [105, 106, 107, 108, 109],
            "low": [95, 96, 97, 98, 99],
            "close": [103, 104, 105, 106, 107],
            "volume": [1000, 1100, 1200, 1300, 1400]
        })

        processed_signals = self.processor.process_signals_for_regime(signals, data)

        self.assertIsInstance(processed_signals, list)
        self.assertGreaterEqual(len(processed_signals), 0)

        # Check that signals have regime analysis metadata
        for signal in processed_signals:
            if hasattr(signal, 'regime_analysis'):
                self.assertIn('detected_regime', signal.regime_analysis)

    def test_get_regime_config(self):
        """Test regime configuration retrieval."""
        config = self.processor._get_regime_config(MarketRegime.TRENDING_BULLISH)

        self.assertIn("preferred_patterns", config)
        self.assertIn("boost_factor", config)
        self.assertIn("penalty_factor", config)
        self.assertIn("fibonacci", config["preferred_patterns"])

    def test_passes_regime_filter(self):
        """Test regime-specific signal filtering."""
        signal = MockActionSignal(action="BUY", confidence=0.8, pattern_type="fibonacci")
        market_data = pd.DataFrame({"close": [100, 101, 102]})

        # Test with trending bullish regime
        passes = self.processor._passes_regime_filter(signal, MarketRegime.TRENDING_BULLISH, market_data)
        self.assertTrue(passes)

        # Test with low confidence signal
        signal.confidence = 0.2
        passes = self.processor._passes_regime_filter(signal, MarketRegime.TRENDING_BULLISH, market_data)
        self.assertFalse(passes)

    def test_update_regime_performance(self):
        """Test regime performance tracking."""
        self.processor.update_regime_performance(MarketRegime.TRENDING_BULLISH, 0.02)

        self.assertIn(MarketRegime.TRENDING_BULLISH, self.processor.regime_performance)
        self.assertEqual(len(self.processor.regime_performance[MarketRegime.TRENDING_BULLISH]), 1)


class TestMarketConditionAnalyzer(unittest.TestCase):
    """Test cases for MarketConditionAnalyzer."""

    def setUp(self):
        """Set up test fixtures."""
        self.analyzer = MarketConditionAnalyzer()

    def test_initialization(self):
        """Test analyzer initialization."""
        self.assertIsInstance(self.analyzer, MarketConditionAnalyzer)
        self.assertIsInstance(self.analyzer.condition_indicators, dict)

    def test_analyze_market_conditions(self):
        """Test comprehensive market condition analysis."""
        data = pd.DataFrame({
            "open": [100, 101, 102, 103, 104],
            "high": [105, 106, 107, 108, 109],
            "low": [95, 96, 97, 98, 99],
            "close": [103, 104, 105, 106, 107],
            "volume": [1000, 1100, 1200, 1300, 1400]
        })

        analysis = self.analyzer.analyze_market_conditions(data)

        self.assertIn("trend", analysis)
        self.assertIn("volatility", analysis)
        self.assertIn("momentum", analysis)
        self.assertIn("volume", analysis)
        self.assertIn("support_resistance", analysis)
        self.assertIn("timestamp", analysis)

    def test_analyze_trend(self):
        """Test trend analysis."""
        # Upward trending data
        data = pd.DataFrame({
            "close": [100, 101, 102, 103, 104, 105]
        })

        trend = self.analyzer._analyze_trend(data)

        self.assertIn("direction", trend)
        self.assertIn("strength", trend)
        self.assertEqual(trend["direction"], "bullish")

    def test_analyze_volatility(self):
        """Test volatility analysis."""
        # Volatile data
        data = pd.DataFrame({
            "close": [100, 110, 90, 120, 80, 130]
        })

        volatility = self.analyzer._analyze_volatility(data)

        self.assertIn("level", volatility)
        self.assertIn("value", volatility)
        self.assertIsInstance(volatility["value"], float)

    def test_analyze_momentum(self):
        """Test momentum analysis."""
        # Strong upward momentum
        data = pd.DataFrame({
            "close": [100, 102, 105, 108, 112, 117]
        })

        momentum = self.analyzer._analyze_momentum(data)

        self.assertIn("value", momentum)
        self.assertIn("strength", momentum)

    def test_analyze_volume(self):
        """Test volume analysis."""
        data = pd.DataFrame({
            "close": [100, 101, 102],
            "volume": [1000, 1500, 1200]  # Increasing then decreasing
        })

        volume = self.analyzer._analyze_volume(data)

        self.assertIn("trend", volume)
        self.assertIn("confirmation", volume)

    def test_analyze_support_resistance(self):
        """Test support and resistance level analysis."""
        # Create data with clear pivot points
        data = pd.DataFrame({
            "high": [105, 110, 108, 112, 115],
            "low": [95, 98, 102, 105, 108],
            "close": [100, 105, 103, 108, 110]
        })

        sr_levels = self.analyzer._analyze_support_resistance(data)

        self.assertIn("nearby_levels", sr_levels)
        self.assertIn("resistance_levels", sr_levels)
        self.assertIn("support_levels", sr_levels)

    def test_insufficient_data_handling(self):
        """Test handling of insufficient data."""
        # Very small dataset
        data = pd.DataFrame({
            "open": [100],
            "high": [105],
            "low": [95],
            "close": [102],
            "volume": [1000]
        })

        analysis = self.analyzer.analyze_market_conditions(data)

        # Should return default conditions
        self.assertIn("trend", analysis)
        self.assertEqual(analysis["trend"]["direction"], "neutral")


if __name__ == "__main__":
    unittest.main()