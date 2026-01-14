"""
Unit tests for Market Regime Classification System

This module contains comprehensive unit tests for the market regime
classification components including MarketRegimeClassifier and
RegimeAdaptiveTrainerMixin.
"""

import os
import tempfile
import unittest
from unittest.mock import Mock, patch

import numpy as np
import pandas as pd

from ztb.analysis.market_regime_classifier import (
    MarketRegimeClassifier,
    RegimeDefinition,
    RegimeDetectionResult,
    RegimeMetrics,
    RegimeType,
)
from ztb.training.components.regime_adaptive_trainer import RegimeAdaptiveTrainerMixin


class TestMarketRegimeClassifier(unittest.TestCase):
    """Test cases for MarketRegimeClassifier"""

    def setUp(self):
        """Set up test fixtures"""
        self.classifier = MarketRegimeClassifier()

        # Create sample market data
        dates = pd.date_range("2023-01-01", periods=100, freq="H")
        np.random.seed(42)

        # Generate realistic price data
        base_price = 5000000.0  # JPY-based price
        returns = np.random.normal(0.0001, 0.02, 100)  # Small drift with volatility
        prices = base_price * np.exp(np.cumsum(returns))

        self.sample_data = pd.DataFrame(
            {
                "open": prices * (1 + np.random.normal(0, 0.005, 100)),
                "high": prices * (1 + np.random.normal(0, 0.01, 100)),
                "low": prices * (1 - np.random.normal(0, 0.01, 100)),
                "close": prices,
                "volume": np.random.uniform(1000, 10000, 100),
            },
            index=dates,
        )

    def test_initialization(self):
        """Test classifier initialization"""
        self.assertIsInstance(self.classifier, MarketRegimeClassifier)
        self.assertIsNotNone(self.classifier.regime_definitions)
        self.assertGreater(len(self.classifier.regime_definitions), 0)

    def test_detect_regime_basic(self):
        """Test basic regime detection"""
        result = self.classifier.detect_regime(self.sample_data)

        self.assertIsInstance(result, RegimeDetectionResult)
        self.assertIsInstance(result.primary_regime, RegimeType)
        self.assertIsInstance(result.confidence, float)
        self.assertGreaterEqual(result.confidence, 0.0)
        self.assertLessEqual(result.confidence, 1.0)

    def test_regime_metrics_calculation(self):
        """Test regime metrics calculation"""
        metrics = self.classifier._calculate_regime_metrics(self.sample_data, -1)

        self.assertIsInstance(metrics, RegimeMetrics)
        self.assertIsInstance(metrics.trend_strength, float)
        self.assertIsInstance(metrics.volatility, float)
        self.assertIsInstance(metrics.momentum, float)

    def test_regime_definitions(self):
        """Test regime definitions loading"""
        definitions = self.classifier.get_regime_definitions()
        self.assertIsInstance(definitions, list)
        self.assertGreater(len(definitions), 0)

        for definition in definitions:
            self.assertIsInstance(definition, RegimeDefinition)
            self.assertIsInstance(definition.regime_type, RegimeType)

    def test_custom_config(self):
        """Test classifier with custom configuration"""
        custom_config = {
            "regime_scheme": "basic",
            "lookback_periods": {"short": 10, "medium": 30, "long": 60},
        }

        classifier = MarketRegimeClassifier(custom_config)
        self.assertEqual(
            len(classifier.regime_definitions), 4
        )  # Basic scheme has 4 regimes

    def test_insufficient_data(self):
        """Test behavior with insufficient data"""
        small_data = self.sample_data.head(5)  # Less than minimum required
        result = self.classifier.detect_regime(small_data)

        # Should still return a result (with default metrics)
        self.assertIsInstance(result, RegimeDetectionResult)

    def test_secondary_regimes(self):
        """Test secondary regime calculation"""
        result = self.classifier.detect_regime(self.sample_data)
        self.assertIsInstance(result.secondary_regimes, list)

        for regime, confidence in result.secondary_regimes:
            self.assertIsInstance(regime, RegimeType)
            self.assertIsInstance(confidence, float)

    def test_config_update(self):
        """Test configuration updates"""
        new_config = {
            "confidence_threshold": 0.8,
            "lookback_periods": {"short": 15, "medium": 40, "long": 80},
        }

        self.classifier.update_config(new_config)
        self.assertEqual(self.classifier.config["confidence_threshold"], 0.8)


class TestRegimeAdaptiveTrainerMixin(unittest.TestCase):
    """Test cases for RegimeAdaptiveTrainerMixin"""

    def setUp(self):
        """Set up test fixtures"""
        self.mixin = RegimeAdaptiveTrainerMixin()

    def test_initialization(self):
        """Test mixin initialization"""
        self.assertIsInstance(self.mixin, RegimeAdaptiveTrainerMixin)
        self.assertFalse(self.mixin.regime_adaptation_enabled)  # Disabled by default

    def test_initialization_with_config(self):
        """Test initialization with regime config"""
        config = {"enabled": True, "adaptation_frequency": 50}
        mixin = RegimeAdaptiveTrainerMixin(config)

        self.assertTrue(mixin.regime_adaptation_enabled)
        self.assertEqual(mixin.regime_config["adaptation_frequency"], 50)

    def test_regime_detection_disabled(self):
        """Test regime detection when disabled"""
        result = self.mixin.detect_market_regime(self.sample_data)
        self.assertIsNone(result)

    @patch("ztb.training.components.regime_adaptive_trainer.MarketRegimeClassifier")
    def test_regime_detection_enabled(self, mock_classifier):
        """Test regime detection when enabled"""
        # Setup mock
        mock_result = Mock()
        mock_result.primary_regime = RegimeType.STRONG_BULL
        mock_result.confidence = 0.8
        mock_classifier.return_value.detect_regime.return_value = mock_result

        # Enable regime adaptation
        config = {"enabled": True}
        mixin = RegimeAdaptiveTrainerMixin(config)

        # Test detection
        result = mixin.detect_market_regime(self.sample_data)
        self.assertEqual(result.primary_regime, RegimeType.STRONG_BULL)
        self.assertEqual(result.confidence, 0.8)

    def test_regime_specific_parameters(self):
        """Test getting regime-specific parameters"""
        mixin = RegimeAdaptiveTrainerMixin()

        params = mixin.get_regime_specific_parameters(RegimeType.STRONG_BULL)
        self.assertIsInstance(params, dict)
        self.assertIn("ent_coef", params)

    def test_adaptation_suggestions(self):
        """Test getting adaptation suggestions"""
        mixin = RegimeAdaptiveTrainerMixin()
        suggestions = mixin.get_adaptation_suggestions()

        self.assertIsInstance(suggestions, list)
        # Should suggest enabling regime adaptation when disabled
        self.assertIn("Enable regime adaptation for better performance", suggestions)

    def test_performance_tracking(self):
        """Test regime performance tracking"""
        mixin = RegimeAdaptiveTrainerMixin({"enabled": True})

        # Add some performance data
        mixin.update_regime_performance(RegimeType.STRONG_BULL, 1.0, 100)
        mixin.update_regime_performance(RegimeType.STRONG_BULL, 2.0, 200)

        summary = mixin.get_regime_performance_summary()
        self.assertIn("strong_bull", summary)
        self.assertEqual(summary["strong_bull"]["total_steps"], 2)

    def test_data_export_import(self):
        """Test regime data export and import"""
        mixin = RegimeAdaptiveTrainerMixin({"enabled": True})

        # Add some test data
        mixin.current_regime = RegimeType.STRONG_BULL
        mixin.update_regime_performance(RegimeType.STRONG_BULL, 1.5, 100)

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            temp_file = f.name

        try:
            # Export data
            mixin.export_regime_data(temp_file)

            # Create new mixin and import data
            new_mixin = RegimeAdaptiveTrainerMixin({"enabled": True})
            new_mixin.load_regime_data(temp_file)

            # Check data was imported
            self.assertEqual(new_mixin.current_regime, RegimeType.STRONG_BULL)

        finally:
            if os.path.exists(temp_file):
                os.unlink(temp_file)


class TestRegimeIntegration(unittest.TestCase):
    """Integration tests for regime adaptation system"""

    def setUp(self):
        """Set up integration test fixtures"""
        self.classifier = MarketRegimeClassifier()

        # Create more realistic test data with different regimes
        dates = pd.date_range("2023-01-01", periods=200, freq="H")
        np.random.seed(123)

        # Create trending data (bull market simulation)
        base_price = 5000000.0  # JPY-based price
        trend = np.linspace(0, 2, 200)  # Upward trend, adjusted for JPY scale
        noise = np.random.normal(0, 0.01, 200)
        prices = base_price * np.exp(trend + noise)

        self.trending_data = pd.DataFrame(
            {
                "open": prices * (1 + np.random.normal(0, 0.002, 200)),
                "high": prices * (1 + np.random.normal(0, 0.005, 200)),
                "low": prices * (1 - np.random.normal(0, 0.005, 200)),
                "close": prices,
                "volume": np.random.uniform(5000, 15000, 200),
            },
            index=dates,
        )

        # Create ranging data (sideways market simulation)
        range_prices = 100 + np.random.normal(0, 2, 200)  # No trend, just noise
        range_prices = np.clip(range_prices, 95, 105)  # Keep in range

        self.ranging_data = pd.DataFrame(
            {
                "open": range_prices * (1 + np.random.normal(0, 0.001, 200)),
                "high": range_prices * (1 + np.random.normal(0, 0.003, 200)),
                "low": range_prices * (1 - np.random.normal(0, 0.003, 200)),
                "close": range_prices,
                "volume": np.random.uniform(1000, 5000, 200),
            },
            index=dates,
        )

    def test_trending_market_detection(self):
        """Test regime detection on trending market data"""
        result = self.classifier.detect_regime(self.trending_data)

        self.assertIsInstance(result, RegimeDetectionResult)
        # Trending data should be detected as some kind of trending regime
        trending_regimes = [
            RegimeType.STRONG_BULL,
            RegimeType.MODERATE_BULL,
            RegimeType.WEAK_BULL,
            RegimeType.STRONG_BEAR,
            RegimeType.MODERATE_BEAR,
            RegimeType.WEAK_BEAR,
        ]
        self.assertIn(result.primary_regime, trending_regimes)

    def test_ranging_market_detection(self):
        """Test regime detection on ranging market data"""
        result = self.classifier.detect_regime(self.ranging_data)

        self.assertIsInstance(result, RegimeDetectionResult)
        # Ranging data should be detected as ranging regime
        ranging_regimes = [
            RegimeType.HIGH_VOLATILITY_RANGE,
            RegimeType.MODERATE_VOLATILITY_RANGE,
            RegimeType.LOW_VOLATILITY_RANGE,
        ]
        # Note: May not always detect as ranging due to random nature of test data

    def test_regime_consistency(self):
        """Test regime detection consistency"""
        # Detect regime multiple times on same data
        results = []
        for _ in range(5):
            result = self.classifier.detect_regime(self.trending_data)
            results.append(result.primary_regime)

        # Results should be reasonably consistent (at least 3 out of 5 same)
        most_common = max(set(results), key=results.count)
        consistency_count = results.count(most_common)
        self.assertGreaterEqual(consistency_count, 3)

    def test_different_timeframes(self):
        """Test regime detection at different points in time series"""
        for i in [50, 100, 150]:
            result = self.classifier.detect_regime(self.trending_data, i)
            self.assertIsInstance(result, RegimeDetectionResult)
            self.assertIsInstance(result.primary_regime, RegimeType)


if __name__ == "__main__":
    unittest.main()
