"""
Unit tests for AdvancedRegimeDetector.
"""

import numpy as np

from ztb.analysis.regime.advanced_regime_detector import (
    AdvancedRegimeDetector,
    MarketRegime,
    TechnicalIndicators
)


class TestTechnicalIndicators:
    """Test technical indicators calculations."""

    def test_calculate_rsi(self):
        """Test RSI calculation."""
        prices = np.array([100, 101, 102, 103, 104, 105, 106, 107, 108, 109,
                          110, 111, 112, 113, 114, 115])

        rsi = TechnicalIndicators.calculate_rsi(prices)
        assert isinstance(rsi, float)
        assert 0 <= rsi <= 100

    def test_calculate_adx(self):
        """Test ADX calculation."""
        highs = np.array([105, 106, 107, 108, 109, 110, 111, 112, 113, 114,
                         115, 116, 117, 118, 119, 120, 121, 122, 123, 124])
        lows = np.array([95, 96, 97, 98, 99, 100, 101, 102, 103, 104,
                        105, 106, 107, 108, 109, 110, 111, 112, 113, 114])
        closes = np.array([100, 101, 102, 103, 104, 105, 106, 107, 108, 109,
                          110, 111, 112, 113, 114, 115, 116, 117, 118, 119])

        adx = TechnicalIndicators.calculate_adx(highs, lows, closes)
        assert isinstance(adx, float)
        assert 0 <= adx <= 100

    def test_calculate_volatility(self):
        """Test volatility calculation."""
        prices = np.array([100, 101, 102, 103, 104, 105, 106, 107, 108, 109,
                          110, 111, 112, 113, 114, 115])

        volatility = TechnicalIndicators.calculate_volatility(prices)
        assert isinstance(volatility, float)
        assert volatility >= 0


class TestAdvancedRegimeDetector:
    """Test AdvancedRegimeDetector functionality."""

    def test_initialization(self):
        """Test detector initialization."""
        detector = AdvancedRegimeDetector()

        assert detector.detection_window == 50
        assert detector.adaptation_frequency == 10
        assert detector.current_regime is None
        assert detector.regime_confidence == 0.0

    def test_insufficient_data_regime(self):
        """Test regime detection with insufficient data."""
        detector = AdvancedRegimeDetector()

        # Add minimal data
        for i in range(5):
            detector.update_price_data(100.0 + i)

        result = detector.detect_regime()

        assert result.regime == MarketRegime.CONSOLIDATION
        assert result.confidence == 0.5

    def test_regime_detection_with_sufficient_data(self):
        """Test regime detection with sufficient data."""
        detector = AdvancedRegimeDetector()

        # Add trending data (upward trend)
        for i in range(60):
            price = 100.0 + i * 0.5  # Gradual upward trend
            detector.update_price_data(price, price + 1, price - 1)

        result = detector.detect_regime()

        # Should detect some kind of bull trend
        assert result.regime in [MarketRegime.STRONG_BULL_TREND,
                                MarketRegime.MODERATE_BULL_TREND,
                                MarketRegime.WEAK_BULL_TREND,
                                MarketRegime.CONSOLIDATION]
        assert isinstance(result.confidence, float)
        assert 0 <= result.confidence <= 1

    def test_regime_statistics(self):
        """Test regime statistics generation."""
        detector = AdvancedRegimeDetector()

        # Add some data and detect regimes
        for i in range(30):
            detector.update_price_data(100.0 + np.sin(i * 0.2))

        # Force multiple detections
        for _ in range(3):
            detector.detect_regime()

        stats = detector.get_regime_statistics()

        assert "total_detections" in stats
        assert "regime_counts" in stats
        assert "average_confidence" in stats
        assert stats["total_detections"] >= 0

    def test_reset_functionality(self):
        """Test detector reset functionality."""
        detector = AdvancedRegimeDetector()

        # Add data
        for i in range(30):
            detector.update_price_data(100.0 + i)

        # Detect regime
        result = detector.detect_regime()
        assert len(detector.price_buffer) > 0

        # Reset detector
        detector.reset()

        # Check that state is cleared
        assert len(detector.price_buffer) == 0
        assert len(detector.regime_history) == 0
        assert detector.current_regime is None
        assert detector.regime_confidence == 0.0

    def test_indicator_calculation(self):
        """Test that indicators are properly calculated."""
        detector = AdvancedRegimeDetector()

        # Add price data
        for i in range(30):
            detector.update_price_data(100.0 + np.sin(i * 0.3))

        result = detector.detect_regime()

        # Check that key indicators are present
        required_indicators = ['rsi', 'adx', 'volatility', 'momentum']
        for indicator in required_indicators:
            assert indicator in result.indicators
            assert isinstance(result.indicators[indicator], (int, float))

    def test_all_regime_types_exist(self):
        """Test that all regime types are properly defined."""
        # Test that we can access all regime types
        regimes = [
            MarketRegime.STRONG_BULL_TREND,
            MarketRegime.MODERATE_BULL_TREND,
            MarketRegime.WEAK_BULL_TREND,
            MarketRegime.STRONG_BEAR_TREND,
            MarketRegime.MODERATE_BEAR_TREND,
            MarketRegime.WEAK_BEAR_TREND,
            MarketRegime.HIGH_VOLATILITY_RANGING,
            MarketRegime.MODERATE_VOLATILITY_RANGING,
            MarketRegime.LOW_VOLATILITY_RANGING,
            MarketRegime.EXTREME_VOLATILITY,
            MarketRegime.CONSOLIDATION,
            MarketRegime.BREAKOUT_SETUP,
            MarketRegime.BREAKDOWN_SETUP
        ]

        assert len(regimes) == 13  # Should have 13 regime types

        # Test that each regime has a string value
        for regime in regimes:
            assert isinstance(regime.value, str)
            assert len(regime.value) > 0