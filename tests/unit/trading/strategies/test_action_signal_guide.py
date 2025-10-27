"""
Test script for Action Signal Guide implementation.

This script tests the basic functionality of the ActionSignalGuide class
and its pattern recognition systems.
"""

import os
import sys

# Add project root to path: tests/unit/trading/strategies -> root
project_root = os.path.dirname(
    os.path.dirname(
        os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    )
)
sys.path.insert(0, project_root)

from datetime import datetime, timedelta
from typing import List

import numpy as np
import pandas as pd

from ztb.trading.strategies.action_signal_guide.action_signal_guide import (
    ActionSignalGuide,
    GuidanceLevel,
)
from ztb.trading.strategies.action_signal_guide.pattern_recognition.base import (
    PatternRecognizer,
    SignalResult,
)

# Import all pattern recognizers for comprehensive testing
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


def create_sample_data(num_bars: int = 100) -> pd.DataFrame:
    """Create sample OHLCV data for testing."""
    np.random.seed(42)  # For reproducible results

    # Generate base price series with trend - pre-allocate for memory efficiency
    base_price = 100.0
    prices: List[float] = [0.0] * num_bars

    for i in range(num_bars):
        # Add some trend and noise
        trend = 0.001 * i  # Slight upward trend
        noise = np.random.normal(0, 0.02)  # Random noise
        price = base_price * (1 + trend + noise)
        prices[i] = price

    # Create OHLCV data more efficiently - pre-allocate all lists
    timestamps = [
        datetime.now() - timedelta(hours=num_bars - i) for i in range(num_bars)
    ]
    opens: List[float] = [0.0] * num_bars
    highs: List[float] = [0.0] * num_bars
    lows: List[float] = [0.0] * num_bars
    closes: List[float] = [0.0] * num_bars
    volumes: List[int] = [0] * num_bars

    opens[0] = prices[0] * (1 + np.random.normal(0, 0.005))

    for i, close in enumerate(prices):
        high = close * (1 + abs(np.random.normal(0, 0.01)))
        low = close * (1 - abs(np.random.normal(0, 0.01)))
        if i < num_bars - 1:
            opens[i + 1] = (
                opens[i] if i == 0 else close * (1 + np.random.normal(0, 0.005))
            )
        volume = np.random.randint(1000, 10000)

        highs[i] = max(opens[i], high)
        lows[i] = min(opens[i], low)
        closes[i] = close
        volumes[i] = volume

    df = pd.DataFrame(
        {
            "timestamp": timestamps,
            "open": opens,
            "high": highs,
            "low": lows,
            "close": closes,
            "volume": volumes,
        }
    )
    df.set_index("timestamp", inplace=True)
    return df


def test_basic_functionality() -> None:
    """Test basic functionality of ActionSignalGuide."""
    print("Testing ActionSignalGuide basic functionality...")

    # Create sample data
    data = create_sample_data(200)
    print(f"Created sample data with {len(data)} bars")

    # Initialize ActionSignalGuide
    guide = ActionSignalGuide(guidance_level=GuidanceLevel.STRONG)

    # Test recognizer status
    status = guide.get_recognizer_status()
    print(f"Recognizer status: {status}")

    # Generate signals for the last 50 bars
    total_signals = []
    for i in range(50, len(data)):
        signals = guide.generate_signals(data, i)
        total_signals.extend(signals)

    print(f"Generated {len(total_signals)} signals")

    # Show some signal examples
    if total_signals:
        print("\nSample signals:")
        for i, signal in enumerate(total_signals[:5]):
            print(
                f"  {i+1}. {signal.signal_type}: {signal.description} "
                f"(strength: {signal.strength:.2f}, direction: {signal.direction})"
            )

    # Test consolidated signal
    if total_signals:
        recent_signals = total_signals[-10:]  # Last 10 signals
        consolidated = guide.get_consolidated_signal(recent_signals)
        if consolidated:
            print(
                f"\nConsolidated signal: {consolidated.description} "
                f"(strength: {consolidated.strength:.2f}, direction: {consolidated.direction})"
            )

    # Test guidance level changes
    print(f"\nCurrent guidance level: {guide.guidance_level.value}")
    guide.set_guidance_level(GuidanceLevel.WEAK)
    print(f"Changed to: {guide.guidance_level.value}")

    print("Basic functionality test completed successfully!")


def test_pattern_recognizers() -> None:
    """Test individual pattern recognizers."""
    print("\nTesting individual pattern recognizers...")

    data = create_sample_data(100)

    # Test a few recognizers directly
    from ztb.trading.strategies.action_signal_guide.pattern_recognition.candlestick_patterns import (
        HammerRecognizer,
    )
    from ztb.trading.strategies.action_signal_guide.pattern_recognition.fibonacci_patterns import (
        FibonacciRetracementRecognizer,
    )

    hammer_recognizer = HammerRecognizer()
    fib_recognizer = FibonacciRetracementRecognizer()

    hammer_signals = []
    fib_signals = []

    for i in range(20, len(data)):
        hammer_signal = hammer_recognizer.recognize(data, i)
        if hammer_signal:
            hammer_signals.append(hammer_signal)

        fib_signal = fib_recognizer.recognize(data, i)
        if fib_signal:
            fib_signals.append(fib_signal)

    print(f"Hammer recognizer found {len(hammer_signals)} signals")
    print(f"Fibonacci retracement recognizer found {len(fib_signals)} signals")

    if hammer_signals:
        signal = hammer_signals[0]
        print(
            f"Sample hammer signal: {signal.description} (strength: {signal.strength:.2f})"
        )

    if fib_signals:
        signal = fib_signals[0]
        print(
            f"Sample fib signal: {signal.description} (strength: {signal.strength:.2f})"
        )


# ===== Comprehensive Unit Tests =====


def test_candlestick_pattern_recognizers():
    """Test all candlestick pattern recognizers."""
    print("\nTesting candlestick pattern recognizers...")

    data = create_sample_data(200)
    recognizers = [
        SakataFiveMethodsRecognizer(),
        MorningStarRecognizer(),
        EveningStarRecognizer(),
        HammerRecognizer(),
        HangingManRecognizer(),
    ]

    total_signals = 0
    for recognizer in recognizers:
        signals = []
        for i in range(10, len(data)):
            signal = recognizer.recognize(data, i)
            if signal:
                signals.append(signal)

        print(f"{recognizer.__class__.__name__}: {len(signals)} signals")
        total_signals += len(signals)

        # Test method signature consistency
        assert hasattr(
            recognizer, "recognize"
        ), f"{recognizer.__class__.__name__} missing recognize method"

    print(f"Total candlestick signals: {total_signals}")


def test_fibonacci_pattern_recognizers():
    """Test all fibonacci pattern recognizers."""
    print("\nTesting fibonacci pattern recognizers...")

    data = create_sample_data(200)
    recognizers = [
        FibonacciRetracementRecognizer(),
        FibonacciExtensionRecognizer(),
        FibonacciProjectionRecognizer(),
    ]

    total_signals = 0
    for recognizer in recognizers:
        signals = []
        for i in range(20, len(data)):  # Need more data for fibonacci patterns
            signal = recognizer.recognize(data, i)
            if signal:
                signals.append(signal)

        print(f"{recognizer.__class__.__name__}: {len(signals)} signals")
        total_signals += len(signals)

    print(f"Total fibonacci signals: {total_signals}")


def test_harmonic_pattern_recognizers():
    """Test all harmonic pattern recognizers."""
    print("\nTesting harmonic pattern recognizers...")

    data = create_sample_data(300)  # Need more data for harmonic patterns
    recognizers = [
        GartleyRecognizer(),
        ButterflyRecognizer(),
        BatRecognizer(),
        CrabRecognizer(),
    ]

    total_signals = 0
    for recognizer in recognizers:
        signals = []
        for i in range(50, len(data)):  # Need significant data for harmonic patterns
            signal = recognizer.recognize(data, i)
            if signal:
                signals.append(signal)

        print(f"{recognizer.__class__.__name__}: {len(signals)} signals")
        total_signals += len(signals)

    print(f"Total harmonic signals: {total_signals}")


def test_wave_counting_recognizers():
    """Test all wave counting recognizers including new specific wave recognizers."""
    print("\nTesting wave counting recognizers...")

    data = create_sample_data(400)  # Need lots of data for wave patterns
    recognizers = [
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

    total_signals = 0
    for recognizer in recognizers:
        signals = []
        for i in range(50, len(data)):  # Start earlier for new recognizers
            signal = recognizer.recognize(data, i)
            if signal:
                signals.append(signal)

        print(f"{recognizer.__class__.__name__}: {len(signals)} signals")
        total_signals += len(signals)

        # Test method signature consistency
        assert hasattr(
            recognizer, "recognize"
        ), f"{recognizer.__class__.__name__} missing recognize method"
        assert callable(
            getattr(recognizer, "recognize")
        ), f"{recognizer.__class__.__name__} recognize not callable"

        # Test with valid data
        result = recognizer.recognize(data, len(data) - 1)
        if result:
            assert isinstance(
                result, SignalResult
            ), f"{recognizer.__class__.__name__} should return SignalResult"
            assert hasattr(result, "signal_type"), "Signal should have signal_type"
            assert hasattr(result, "metadata"), "Signal should have metadata"

            # Test specific wave recognizer metadata
            if recognizer.__class__.__name__ in [
                "WaveIRecognizer",
                "WaveVRecognizer",
                "WaveYRecognizer",
                "WavePRecognizer",
                "WaveNRecognizer",
                "WaveSRecognizer",
            ]:
                assert (
                    "wave_label" in result.metadata
                ), f"{recognizer.__class__.__name__} should have wave_label in metadata"
                assert (
                    "pattern" in result.metadata
                ), f"{recognizer.__class__.__name__} should have pattern in metadata"

    print(f"Total wave counting signals: {total_signals}")
    print("Wave counting recognizers test completed!")


def test_gann_analysis_recognizers():
    """Test all Gann analysis recognizers."""
    print("\nTesting Gann analysis recognizers...")

    data = create_sample_data(200)
    recognizers = [
        GannAngleRecognizer(),
        GannSquareRecognizer(),
        GannTimeClusterRecognizer(),
    ]

    total_signals = 0
    for recognizer in recognizers:
        signals = []
        for i in range(30, len(data)):
            signal = recognizer.recognize(data, i)
            if signal:
                signals.append(signal)

        print(f"{recognizer.__class__.__name__}: {len(signals)} signals")
        total_signals += len(signals)

    print(f"Total Gann analysis signals: {total_signals}")


def test_method_signature_consistency():
    """Test that all pattern recognizers have consistent method signatures."""
    print("\nTesting method signature consistency...")

    data = create_sample_data(100)

    # Test all recognizer classes
    all_recognizers = [
        # Candlestick
        SakataFiveMethodsRecognizer(),
        MorningStarRecognizer(),
        EveningStarRecognizer(),
        HammerRecognizer(),
        HangingManRecognizer(),
        # Fibonacci
        FibonacciRetracementRecognizer(),
        FibonacciExtensionRecognizer(),
        FibonacciProjectionRecognizer(),
        # Harmonic
        GartleyRecognizer(),
        ButterflyRecognizer(),
        BatRecognizer(),
        CrabRecognizer(),
        # Wave counting
        ImpulseWaveRecognizer(),
        CorrectiveWaveRecognizer(),
        WaveExtensionRecognizer(),
        # Gann
        GannAngleRecognizer(),
        GannSquareRecognizer(),
        GannTimeClusterRecognizer(),
    ]

    for recognizer in all_recognizers:
        # Test that recognize method exists and has correct signature
        assert hasattr(
            recognizer, "recognize"
        ), f"{recognizer.__class__.__name__} missing recognize method"

        # Test method signature by calling with different parameters
        try:
            # Test with default parameter
            result1 = recognizer.recognize(data, -1)
            # Test with explicit index
            result2 = recognizer.recognize(data, 50)
            print(f"✓ {recognizer.__class__.__name__} method signature OK")
        except Exception as e:
            print(f"✗ {recognizer.__class__.__name__} method signature error: {e}")
            raise

    print("All method signatures are consistent!")


def test_edge_cases():
    """Test edge cases and error handling."""
    print("\nTesting edge cases...")

    # Test with minimal data
    minimal_data = create_sample_data(5)
    recognizer = HammerRecognizer()

    # Should handle minimal data gracefully
    result = recognizer.recognize(minimal_data, 4)
    assert result is None or isinstance(
        result, SignalResult
    ), "Should handle minimal data"

    # Test with invalid index
    try:
        result = recognizer.recognize(minimal_data, 100)  # Index out of bounds
        # Should either return None or handle gracefully
    except (IndexError, ValueError):
        pass  # Expected for out of bounds

    print("Edge case handling OK!")


def test_signal_result_structure():
    """Test that all signals have proper SignalResult structure."""
    print("\nTesting SignalResult structure...")

    data = create_sample_data(200)
    recognizer = HammerRecognizer()

    signals_found = 0
    for i in range(10, len(data)):
        signal = recognizer.recognize(data, i)
        if signal:
            signals_found += 1
            # Verify SignalResult structure
            assert hasattr(signal, "signal_type"), "Signal missing signal_type"
            assert hasattr(signal, "description"), "Signal missing description"
            assert hasattr(signal, "strength"), "Signal missing strength"
            assert hasattr(signal, "direction"), "Signal missing direction"
            assert hasattr(signal, "timestamp"), "Signal missing timestamp"
            assert hasattr(signal, "confidence"), "Signal missing confidence"

            # Verify types
            assert isinstance(
                signal.strength, (int, float)
            ), "Strength should be numeric"
            assert isinstance(
                signal.confidence, (int, float)
            ), "Confidence should be numeric"
            assert signal.direction in [-1, 0, 1], "Direction should be -1, 0, or 1"

            if signals_found >= 3:  # Test first few signals
                break

    print(f"SignalResult structure validation passed for {signals_found} signals!")


def test_wave_counting_recognizers():
    """Test all wave counting recognizers including new specific wave recognizers."""
    print("\nTesting wave counting recognizers...")

    data = create_sample_data(300)  # Need more data for wave analysis
    recognizers = [
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

    total_signals = 0
    for recognizer in recognizers:
        signals = []
        for i in range(30, len(data)):  # Start from 30 to allow lookback
            signal = recognizer.recognize(data, i)
            if signal:
                signals.append(signal)

        print(f"{recognizer.__class__.__name__}: {len(signals)} signals")
        total_signals += len(signals)

        # Test method signature consistency
        assert hasattr(
            recognizer, "recognize"
        ), f"{recognizer.__class__.__name__} missing recognize method"
        assert callable(
            getattr(recognizer, "recognize")
        ), f"{recognizer.__class__.__name__} recognize not callable"

        # Test with valid data
        result = recognizer.recognize(data, len(data) - 1)
        if result:
            assert isinstance(
                result, SignalResult
            ), f"{recognizer.__class__.__name__} should return SignalResult"
            assert hasattr(result, "signal_type"), "Signal should have signal_type"
            assert hasattr(result, "metadata"), "Signal should have metadata"

            # Test specific wave recognizer metadata
            if recognizer.__class__.__name__ in [
                "WaveIRecognizer",
                "WaveVRecognizer",
                "WaveYRecognizer",
                "WavePRecognizer",
                "WaveNRecognizer",
                "WaveSRecognizer",
            ]:
                assert (
                    "wave_label" in result.metadata
                ), f"{recognizer.__class__.__name__} should have wave_label in metadata"
                assert (
                    "pattern" in result.metadata
                ), f"{recognizer.__class__.__name__} should have pattern in metadata"

    print(f"Total wave counting signals found: {total_signals}")
    print("Wave counting recognizers test completed!")


def test_specific_wave_recognizer_details():
    """Test specific details of new wave recognizers."""
    print("\nTesting specific wave recognizer details...")

    # Create data with specific wave patterns for testing
    dates = pd.date_range("2023-01-01", periods=200, freq="D")

    # Create impulse wave pattern (Wave I and Wave V)
    prices = []
    base_price = 100.0

    # Wave I: Strong upward move
    for i in range(50):
        base_price += 0.5 + np.random.normal(0, 0.1)
        prices.append(base_price)

    # Wave II: Correction
    for i in range(30):
        base_price -= 0.3 + np.random.normal(0, 0.1)
        prices.append(base_price)

    # Wave III: Strongest move
    for i in range(40):
        base_price += 0.8 + np.random.normal(0, 0.1)
        prices.append(base_price)

    # Wave IV: Correction
    for i in range(25):
        base_price -= 0.2 + np.random.normal(0, 0.1)
        prices.append(base_price)

    # Wave V: Final move
    for i in range(55):
        base_price += 0.4 + np.random.normal(0, 0.1)
        prices.append(base_price)

    # Create DataFrame
    data = pd.DataFrame(
        {
            "open": prices,
            "high": [p + abs(np.random.normal(0, 0.5)) for p in prices],
            "low": [p - abs(np.random.normal(0, 0.5)) for p in prices],
            "close": [p + np.random.normal(0, 0.2) for p in prices],
            "volume": np.random.randint(1000, 10000, len(prices)),
        },
        index=dates[: len(prices)],
    )

    # Test Wave I recognizer
    wave_i_recognizer = WaveIRecognizer()
    wave_i_signals = []
    for i in range(30, len(data)):
        signal = wave_i_recognizer.recognize(data, i)
        if signal:
            wave_i_signals.append(signal)

    print(f"Wave I recognizer found {len(wave_i_signals)} signals")
    if wave_i_signals:
        signal = wave_i_signals[0]
        assert signal.signal_type == "wave_i", "Should identify as wave_i"
        assert (
            signal.metadata["pattern"] == "wave_i"
        ), "Should have correct pattern metadata"
        print(
            f"Sample Wave I signal: {signal.description} (strength: {signal.strength:.2f})"
        )

    # Test Wave V recognizer
    wave_v_recognizer = WaveVRecognizer()
    wave_v_signals = []
    for i in range(50, len(data)):  # Start later for 5-wave structure
        signal = wave_v_recognizer.recognize(data, i)
        if signal:
            wave_v_signals.append(signal)

    print(f"Wave V recognizer found {len(wave_v_signals)} signals")
    if wave_v_signals:
        signal = wave_v_signals[0]
        assert signal.signal_type == "wave_v", "Should identify as wave_v"
        assert (
            signal.metadata["pattern"] == "wave_v"
        ), "Should have correct pattern metadata"
        print(
            f"Sample Wave V signal: {signal.description} (strength: {signal.strength:.2f})"
        )

    # Test Wave P recognizer (irregular correction)
    # Create irregular correction pattern
    irregular_prices = []
    base_price = 100.0

    # Initial move
    for i in range(20):
        base_price += 0.3
        irregular_prices.append(base_price)

    # Irregular correction (overshoots)
    for i in range(15):
        base_price -= 0.5  # Overshoot correction
        irregular_prices.append(base_price)

    # Recovery
    for i in range(20):
        base_price += 0.2
        irregular_prices.append(base_price)

    irregular_data = pd.DataFrame(
        {
            "open": irregular_prices,
            "high": [p + 0.5 for p in irregular_prices],
            "low": [p - 0.5 for p in irregular_prices],
            "close": irregular_prices,
            "volume": np.random.randint(1000, 10000, len(irregular_prices)),
        },
        index=dates[: len(irregular_prices)],
    )

    wave_p_recognizer = WavePRecognizer()
    wave_p_signals = []
    for i in range(25, len(irregular_data)):
        signal = wave_p_recognizer.recognize(irregular_data, i)
        if signal:
            wave_p_signals.append(signal)

    print(f"Wave P recognizer found {len(wave_p_signals)} signals")
    if wave_p_signals:
        signal = wave_p_signals[0]
        assert signal.signal_type == "wave_p", "Should identify as wave_p"
        assert (
            "overshoot_ratio" in signal.metadata
        ), "Should have overshoot_ratio in metadata"
        print(
            f"Sample Wave P signal: {signal.description} (overshoot: {signal.metadata['overshoot_ratio']:.2f})"
        )

    print("Specific wave recognizer details test completed!")


def test_wave_recognizer_edge_cases():
    """Test edge cases for wave recognizers."""
    print("\nTesting wave recognizer edge cases...")

    # Test with minimal data
    minimal_data = create_sample_data(10)

    recognizers = [
        WaveIRecognizer(),
        WaveVRecognizer(),
        WaveYRecognizer(),
        WavePRecognizer(),
        WaveNRecognizer(),
        WaveSRecognizer(),
    ]

    for recognizer in recognizers:
        # Should handle minimal data gracefully
        result = recognizer.recognize(minimal_data, 5)
        assert (
            result is None
        ), f"{recognizer.__class__.__name__} should return None for minimal data"

        # Test with None data
        try:
            result = recognizer.recognize(None, 5)
            assert (
                result is None
            ), f"{recognizer.__class__.__name__} should handle None data"
        except (AttributeError, TypeError):
            pass  # Expected for None data

    # Test with empty DataFrame
    empty_data = pd.DataFrame()
    for recognizer in recognizers:
        result = recognizer.recognize(empty_data, 0)
        assert (
            result is None
        ), f"{recognizer.__class__.__name__} should handle empty data"

    print("Wave recognizer edge cases test completed!")


def test_action_signal_guide_integration():
    """Test ActionSignalGuide integration with all recognizers."""
    print("\nTesting ActionSignalGuide integration...")

    data = create_sample_data(300)
    guide = ActionSignalGuide(guidance_level=GuidanceLevel.STRONG)

    # Test that all recognizers are initialized
    status = guide.get_recognizer_status()
    assert "total_recognizers" in status, "Missing total_recognizers in status"
    assert status["total_recognizers"] > 0, "Should have recognizers initialized"

    # Test signal generation
    signals = guide.generate_signals(data, len(data) - 1)
    assert isinstance(signals, list), "generate_signals should return list"

    # Test consolidated signal
    if signals:
        consolidated = guide.get_consolidated_signal(signals)
        if consolidated:
            assert hasattr(
                consolidated, "signal_type"
            ), "Consolidated signal missing signal_type"
            assert hasattr(
                consolidated, "strength"
            ), "Consolidated signal missing strength"

    # Test guidance level changes
    original_level = guide.guidance_level
    guide.set_guidance_level(GuidanceLevel.WEAK)
    assert guide.guidance_level == GuidanceLevel.WEAK, "Guidance level should change"
    guide.set_guidance_level(original_level)  # Reset

    print("ActionSignalGuide integration test passed!")


def test_pattern_recognizer_inheritance():
    """Test that all pattern recognizers properly inherit from PatternRecognizer."""
    print("\nTesting pattern recognizer inheritance...")

    recognizers = [
        # Candlestick
        SakataFiveMethodsRecognizer(),
        MorningStarRecognizer(),
        EveningStarRecognizer(),
        HammerRecognizer(),
        HangingManRecognizer(),
        # Fibonacci
        FibonacciRetracementRecognizer(),
        FibonacciExtensionRecognizer(),
        FibonacciProjectionRecognizer(),
        # Harmonic
        GartleyRecognizer(),
        ButterflyRecognizer(),
        BatRecognizer(),
        CrabRecognizer(),
        # Wave counting
        ImpulseWaveRecognizer(),
        CorrectiveWaveRecognizer(),
        WaveExtensionRecognizer(),
        WaveIRecognizer(),
        WaveVRecognizer(),
        WaveYRecognizer(),
        WavePRecognizer(),
        WaveNRecognizer(),
        WaveSRecognizer(),
        # Gann
        GannAngleRecognizer(),
        GannSquareRecognizer(),
        GannTimeClusterRecognizer(),
    ]

    for recognizer in recognizers:
        # Test inheritance
        assert isinstance(
            recognizer, PatternRecognizer
        ), f"{recognizer.__class__.__name__} should inherit from PatternRecognizer"

        # Test required methods exist
        assert hasattr(
            recognizer, "recognize"
        ), f"{recognizer.__class__.__name__} missing recognize method"

    print("All pattern recognizers properly inherit from base class!")


if __name__ == "__main__":
    try:
        test_basic_functionality()
        test_pattern_recognizers()

        # Run comprehensive tests
        test_candlestick_pattern_recognizers()
        test_fibonacci_pattern_recognizers()
        test_harmonic_pattern_recognizers()
        test_wave_counting_recognizers()
        test_specific_wave_recognizer_details()
        test_wave_recognizer_edge_cases()
        test_gann_analysis_recognizers()
        test_method_signature_consistency()
        test_edge_cases()
        test_signal_result_structure()
        test_action_signal_guide_integration()
        test_pattern_recognizer_inheritance()

        print("\n🎉 All comprehensive tests completed successfully!")
    except Exception as e:
        print(f"❌ Test failed with error: {e}")
        import traceback

        traceback.print_exc()
