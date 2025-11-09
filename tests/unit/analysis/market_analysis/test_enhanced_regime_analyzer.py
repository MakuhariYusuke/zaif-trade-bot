"""
Unit tests for Enhanced Regime Analyzer.
"""

import numpy as np

from ztb.analysis.market_analysis import (
    EnhancedRegimeAnalyzer,
    EnhancedTechnicalIndicators,
    MarketRegime,
)


class TestEnhancedTechnicalIndicators:
    """Test enhanced technical indicators calculations."""

    def test_calculate_rsi_enhanced(self):
        """Test enhanced RSI calculation."""
        prices = np.array(
            [
                100,
                101,
                102,
                103,
                104,
                105,
                106,
                107,
                108,
                109,
                110,
                111,
                112,
                113,
                114,
                115,
                116,
                117,
                118,
                119,
            ]
        )

        rsi = EnhancedTechnicalIndicators.calculate_rsi(prices)
        assert isinstance(rsi, float)
        assert 0 <= rsi <= 100

        # Test with trending data
        trending_prices = np.array([100 + i * 0.5 for i in range(20)])
        rsi_trending = EnhancedTechnicalIndicators.calculate_rsi(trending_prices)
        assert rsi_trending > 50  # Should be high in uptrend

    def test_calculate_adx_enhanced(self):
        """Test enhanced ADX calculation."""
        highs = np.array(
            [
                105,
                106,
                107,
                108,
                109,
                110,
                111,
                112,
                113,
                114,
                115,
                116,
                117,
                118,
                119,
                120,
                121,
                122,
                123,
                124,
            ]
        )
        lows = np.array(
            [
                95,
                96,
                97,
                98,
                99,
                100,
                101,
                102,
                103,
                104,
                105,
                106,
                107,
                108,
                109,
                110,
                111,
                112,
                113,
                114,
            ]
        )
        closes = np.array(
            [
                100,
                101,
                102,
                103,
                104,
                105,
                106,
                107,
                108,
                109,
                110,
                111,
                112,
                113,
                114,
                115,
                116,
                117,
                118,
                119,
            ]
        )

        adx = EnhancedTechnicalIndicators.calculate_adx(highs, lows, closes)
        assert isinstance(adx, float)
        assert 0 <= adx <= 100

    def test_calculate_macd_enhanced(self):
        """Test enhanced MACD calculation."""
        prices = np.array([100 + np.sin(i * 0.1) for i in range(50)])

        macd, signal, histogram = EnhancedTechnicalIndicators.calculate_macd(prices)
        assert isinstance(macd, float)
        assert isinstance(signal, float)
        assert isinstance(histogram, float)

    def test_calculate_bollinger_bands(self):
        """Test Bollinger Bands calculation."""
        prices = np.array([100 + np.sin(i * 0.1) for i in range(30)])

        sma, upper, lower = EnhancedTechnicalIndicators.calculate_bollinger_bands(
            prices
        )
        assert isinstance(sma, float)
        assert isinstance(upper, float)
        assert isinstance(lower, float)
        assert upper > sma > lower

    def test_calculate_atr(self):
        """Test ATR calculation."""
        highs = np.array([105, 106, 107, 108, 109, 110])
        lows = np.array([95, 96, 97, 98, 99, 100])
        closes = np.array([100, 101, 102, 103, 104, 105])

        atr = EnhancedTechnicalIndicators.calculate_atr(highs, lows, closes)
        assert isinstance(atr, float)
        assert atr >= 0


class TestEnhancedRegimeAnalyzer:
    """Test EnhancedRegimeAnalyzer functionality."""

    def test_initialization(self):
        """Test analyzer initialization."""
        analyzer = EnhancedRegimeAnalyzer()

        assert analyzer.detection_window == 50
        assert analyzer.adaptation_frequency == 10
        assert analyzer.current_regime is None
        assert analyzer.regime_confidence == 0.0
        assert len(analyzer.price_buffer) == 0

    def test_insufficient_data_regime(self):
        """Test regime detection with insufficient data."""
        analyzer = EnhancedRegimeAnalyzer()

        # Add minimal data
        for i in range(5):
            analyzer.update_price_data(100.0 + i)

        result = analyzer.detect_regime()

        assert result.regime == MarketRegime.CONSOLIDATION
        assert result.confidence == 0.5
        assert "insufficient_data" in result.classification_path

    def test_extreme_volatility_detection(self):
        """Test extreme volatility regime detection."""
        analyzer = EnhancedRegimeAnalyzer()

        # Add highly volatile data
        base_price = 100.0
        for i in range(60):
            # Create extreme volatility
            volatility = 0.05 + 0.03 * np.sin(i * 0.5)  # High volatility
            price = base_price + np.random.normal(0, volatility * base_price)
            analyzer.update_price_data(price)

        result = analyzer.detect_regime()

        # Should detect extreme volatility
        assert result.regime in [
            MarketRegime.EXTREME_VOLATILITY,
            MarketRegime.HIGH_VOLATILITY_RANGING,
        ]
        assert result.confidence > 0.8

    def test_strong_trend_detection(self):
        """Test strong trend regime detection."""
        analyzer = EnhancedRegimeAnalyzer()

        # Add strong upward trending data
        for i in range(60):
            price = 100.0 + i * 0.8  # Strong upward trend
            high = price + 2
            low = price - 2
            analyzer.update_price_data(price, high, low)

        result = analyzer.detect_regime()

        # Should detect strong bull trend
        assert result.regime == MarketRegime.STRONG_BULL_TREND
        assert result.confidence > 0.8
        assert "strong_trend" in result.classification_path

    def test_consolidation_detection(self):
        """Test consolidation regime detection."""
        analyzer = EnhancedRegimeAnalyzer()

        # Add consolidating data (sideways movement)
        base_price = 100.0
        for i in range(60):
            price = base_price + np.sin(i * 0.1) * 0.5  # Small oscillations
            analyzer.update_price_data(price)

        result = analyzer.detect_regime()

        # Should detect consolidation
        assert result.regime in [
            MarketRegime.CONSOLIDATION,
            MarketRegime.LOW_VOLATILITY_RANGING,
        ]
        assert result.confidence > 0.4

    def test_adaptive_thresholds(self):
        """Test adaptive threshold adjustment."""
        analyzer = EnhancedRegimeAnalyzer()

        # Add data to build statistical baselines
        for i in range(120):  # More than minimum for adaptation
            volatility = 0.01 + 0.005 * np.sin(i * 0.1)
            price = 100.0 + np.random.normal(0, volatility * 100)
            analyzer.update_price_data(price)

        # Check that adaptive thresholds are updated
        assert hasattr(analyzer, "volatility_percentiles")
        assert hasattr(analyzer, "trend_thresholds")

        percentiles = analyzer.volatility_percentiles
        assert "p25" in percentiles
        assert "p50" in percentiles
        assert "p75" in percentiles
        assert "p90" in percentiles

    def test_regime_statistics(self):
        """Test regime statistics generation."""
        analyzer = EnhancedRegimeAnalyzer()

        # Add data and force multiple detections
        for i in range(30):
            analyzer.update_price_data(100.0 + np.sin(i * 0.2))

        # Force multiple detections
        for _ in range(5):
            analyzer.detect_regime()

        stats = analyzer.get_regime_statistics()

        assert "total_detections" in stats
        assert "regime_counts" in stats
        assert "average_confidence" in stats
        assert "adaptive_thresholds" in stats
        assert stats["total_detections"] >= 5

    def test_reset_functionality(self):
        """Test analyzer reset functionality."""
        analyzer = EnhancedRegimeAnalyzer()

        # Add data
        for i in range(30):
            analyzer.update_price_data(100.0 + i)

        # Detect regime
        result = analyzer.detect_regime()
        assert len(analyzer.price_buffer) > 0

        # Reset analyzer
        analyzer.reset()

        # Check that state is cleared
        assert len(analyzer.price_buffer) == 0
        assert len(analyzer.regime_history) == 0
        assert analyzer.current_regime is None
        assert analyzer.regime_confidence == 0.0

    def test_all_regime_types_accessible(self):
        """Test that all regime types are properly defined."""
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
            MarketRegime.BREAKDOWN_SETUP,
        ]

        assert len(regimes) == 13

        # Test that each regime has a string value
        for regime in regimes:
            assert isinstance(regime.value, str)
            assert len(regime.value) > 0

    def test_indicator_calculation_completeness(self):
        """Test that all indicators are properly calculated."""
        analyzer = EnhancedRegimeAnalyzer()

        # Add sufficient data
        for i in range(30):
            analyzer.update_price_data(100.0 + np.sin(i * 0.3))

        result = analyzer.detect_regime()

        # Check that key indicators are present
        required_indicators = [
            "rsi",
            "adx",
            "volatility",
            "momentum",
            "macd",
            "bb_position",
        ]
        for indicator in required_indicators:
            assert indicator in result.indicators
            assert isinstance(result.indicators[indicator], (int, float))

    def test_classification_path_tracking(self):
        """Test that classification paths are properly tracked."""
        analyzer = EnhancedRegimeAnalyzer()

        # Add trending data
        for i in range(60):
            price = 100.0 + i * 0.5
            analyzer.update_price_data(price)

        result = analyzer.detect_regime()

        # Should have classification path
        assert isinstance(result.classification_path, list)
        assert len(result.classification_path) > 0
        assert all(isinstance(step, str) for step in result.classification_path)
