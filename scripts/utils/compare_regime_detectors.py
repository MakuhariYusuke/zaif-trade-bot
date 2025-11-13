#!/usr/bin/env python3
"""
Comparison script between old and new regime detection implementations.
"""

import os
import sys

import numpy as np

# Add project root to path
sys.path.insert(
    0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
)

from ztb.analysis.market_analysis import EnhancedRegimeAnalyzer
from ztb.analysis.regime.advanced_regime_detector import AdvancedRegimeDetector


def generate_test_data(scenario: str, length: int = 60) -> list:
    """Generate test data for different market scenarios."""
    np.random.seed(42)  # For reproducible results

    if scenario == "strong_bull_trend":
        # Strong upward trend
        prices = [5000000.0 + i * 40000 for i in range(length)]  # Remove noise, JPY scale
    elif scenario == "extreme_volatility":
        # Extreme volatility - very extreme
        base_price = 5000000.0  # JPY-based
        prices = []
        for i in range(length):
            volatility = 0.15 + 0.10 * np.sin(i * 0.5)  # Much higher volatility
            price = base_price + np.random.normal(0, volatility * base_price)
            prices.append(price)
            base_price = price
    elif scenario == "consolidation":
        # Sideways consolidation - very stable
        base_price = 5000000.0  # JPY-based
        prices = [
            base_price + np.sin(i * 0.1) * 50000 for i in range(length)
        ]  # Much smaller oscillations, JPY scale
    elif scenario == "high_volatility_ranging":
        # High volatility ranging - moderate
        base_price = 5000000.0  # JPY-based
        prices = []
        for i in range(length):
            change = np.random.normal(0, 50000)  # Moderate volatility, JPY scale
            price = base_price + change
            prices.append(price)
            base_price = price
    else:
        # Default: moderate trend
        prices = [100.0 + i * 0.2 for i in range(length)]  # Remove noise

    return prices


def compare_regime_detectors():
    """Compare old and new regime detection implementations."""
    scenarios = [
        "strong_bull_trend",
        "extreme_volatility",
        "consolidation",
        "high_volatility_ranging",
        "moderate_trend",
    ]

    print("=== Regime Detection Comparison ===\n")

    for scenario in scenarios:
        print(f"Testing scenario: {scenario}")
        print("-" * 40)

        # Generate test data
        prices = generate_test_data(scenario)

        # Test old implementation
        old_detector = AdvancedRegimeDetector()
        for price in prices:
            old_detector.update_price_data(price)

        old_result = old_detector.detect_regime()

        # Test new implementation
        new_analyzer = EnhancedRegimeAnalyzer()
        for price in prices:
            new_analyzer.update_price_data(price)

        new_result = new_analyzer.detect_regime()

        # Debug: Print volatility values
        if "volatility" in new_result.indicators:
            print(f"  New Volatility: {new_result.indicators['volatility']:.6f}")

        # Compare results
        print("Old Implementation:")
        print(f"  Regime: {old_result.regime.value}")
        print(f"  Confidence: {old_result.confidence:.3f}")
        print(f"  Indicators: {len(old_result.indicators)} calculated")

        print("New Implementation:")
        print(f"  Regime: {new_result.regime.value}")
        print(f"  Confidence: {new_result.confidence:.3f}")
        print(f"  Classification Path: {' -> '.join(new_result.classification_path)}")
        print(f"  Indicators: {len(new_result.indicators)} calculated")

        # Check if regimes match
        regime_match = old_result.regime == new_result.regime
        print(f"  Regime Match: {'✓' if regime_match else '✗'}")

        # Check confidence difference
        confidence_diff = abs(old_result.confidence - new_result.confidence)
        print(f"  Confidence Difference: {confidence_diff:.3f}")

        print()


def test_indicator_accuracy():
    """Test indicator calculation accuracy."""
    print("=== Indicator Accuracy Test ===\n")

    # Generate known test data
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

    from ztb.analysis.market_analysis import EnhancedTechnicalIndicators

    # Test RSI calculation
    rsi = EnhancedTechnicalIndicators.calculate_rsi(prices)
    print(f"RSI (20 prices): {rsi:.2f}")

    # Test with trending data
    trending_prices = np.array([100 + i * 0.5 for i in range(20)])
    rsi_trending = EnhancedTechnicalIndicators.calculate_rsi(trending_prices)
    print(f"RSI (trending): {rsi_trending:.2f}")

    # Test MACD
    macd, signal, histogram = EnhancedTechnicalIndicators.calculate_macd(prices)
    print(f"MACD: {macd:.4f}, Signal: {signal:.4f}, Histogram: {histogram:.4f}")

    # Test Bollinger Bands
    sma, upper, lower = EnhancedTechnicalIndicators.calculate_bollinger_bands(prices)
    print(f"Bollinger Bands - SMA: {sma:.2f}, Upper: {upper:.2f}, Lower: {lower:.2f}")

    print()


if __name__ == "__main__":
    try:
        compare_regime_detectors()
        test_indicator_accuracy()
        print("Comparison completed successfully!")
    except Exception as e:
        print(f"Error during comparison: {e}")
        import traceback

        traceback.print_exc()
