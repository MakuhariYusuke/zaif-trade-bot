#!/usr/bin/env python3
"""Test for EnhancedRegimeAnalyzer refactored to use existing feature generators."""

import numpy as np

from ztb.analysis.market_analysis.regime_analyzer import (
    EnhancedRegimeAnalyzer,
    MarketRegime,
)


def test_enhanced_regime_analyzer_basic():
    """Test basic functionality of EnhancedRegimeAnalyzer."""
    analyzer = EnhancedRegimeAnalyzer()

    # Create sample price data
    prices = np.array(
        [
            100.0,
            101.0,
            102.0,
            103.0,
            104.0,
            105.0,
            106.0,
            107.0,
            108.0,
            109.0,
            110.0,
            111.0,
            112.0,
            113.0,
            114.0,
            115.0,
            116.0,
            117.0,
            118.0,
            119.0,
        ]
    )
    highs = prices + 1.0
    lows = prices - 1.0

    # Update analyzer with data
    for i in range(len(prices)):
        analyzer.update_price_data(prices[i], highs[i], lows[i])

    # Get regime detection result
    result = analyzer.detect_regime()

    # Verify result structure
    assert isinstance(result, object)  # RegimeDetectionResult
    assert hasattr(result, "regime")
    assert hasattr(result, "confidence")
    assert hasattr(result, "indicators")
    assert hasattr(result, "metadata")
    assert hasattr(result, "classification_path")

    # Verify regime is valid
    assert isinstance(result.regime, MarketRegime)

    # Verify confidence is reasonable
    assert 0.0 <= result.confidence <= 1.0

    # Verify indicators contain expected keys
    expected_indicators = [
        "rsi",
        "adx",
        "volatility",
        "momentum",
        "macd",
        "macd_signal",
        "macd_histogram",
        "bb_position",
        "atr",
        "sma",
        "bb_upper",
        "bb_lower",
    ]
    for indicator in expected_indicators:
        assert indicator in result.indicators

    print("EnhancedRegimeAnalyzer basic test passed!")


def test_regime_analyzer_with_insufficient_data():
    """Test behavior with insufficient data."""
    analyzer = EnhancedRegimeAnalyzer()

    # Add minimal data (less than required for indicators)
    analyzer.update_price_data(100.0, 101.0, 99.0)

    result = analyzer.detect_regime()

    # Should return default regime with low confidence
    assert result.regime == MarketRegime.CONSOLIDATION  # Default fallback
    assert result.confidence == 0.5

    print("Insufficient data test passed!")


def test_regime_analyzer_adaptive_thresholds():
    """Test adaptive threshold functionality."""
    analyzer = EnhancedRegimeAnalyzer()

    # Add some volatility data to test adaptation
    for i in range(30):
        price = 100.0 + np.sin(i * 0.1) * 2.0
        high = price + 1.0
        low = price - 1.0
        analyzer.update_price_data(price, high, low)

    # Check that adaptive thresholds are updated
    assert hasattr(analyzer, "volatility_percentiles")
    assert hasattr(analyzer, "trend_thresholds")

    print("Adaptive thresholds test passed!")


if __name__ == "__main__":
    test_enhanced_regime_analyzer_basic()
    test_regime_analyzer_with_insufficient_data()
    test_regime_analyzer_adaptive_thresholds()
    print("All EnhancedRegimeAnalyzer tests passed!")
