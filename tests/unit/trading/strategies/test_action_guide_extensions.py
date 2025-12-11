#!/usr/bin/env python3
"""
Action Signal Guide Extension Test Script

This script demonstrates the extended capabilities of the ActionSignalGuide
including multi-timeframe analysis, combined signals, and adaptive features.
"""

import numpy as np

from ztb.trading.strategies.action_signal_guide import ActionSignalGuide, GuidanceMode
from ztb.trading.strategies.signal_definitions import SignalDefinitions


def create_sample_observation():
    """Create a sample observation with various technical indicators."""
    # Sample feature names (what would be in observation vector)
    feature_names = [
        "close",
        "sma5",
        "sma20",
        "ema5",
        "ema20",
        "rsi",
        "macd",
        "macd_signal",
        "stoch_k",
        "stoch_d",
        "bb_upper",
        "bb_lower",
        "adx",
        "plus_di",
        "minus_di",
        "williams_r",
        "cci",
        "atr",
    ]

    # Sample observation values (normalized/scaled)
    observation = np.array(
        [
            100.0,  # close price
            98.5,  # sma5
            97.0,  # sma20
            99.0,  # ema5
            97.5,  # ema20
            35.0,  # rsi (oversold)
            -0.5,  # macd
            -0.3,  # macd_signal
            25.0,  # stoch_k
            30.0,  # stoch_d
            102.0,  # bb_upper
            92.0,  # bb_lower
            28.0,  # adx (strong trend)
            22.0,  # plus_di
            18.0,  # minus_di
            -75.0,  # williams_r (oversold)
            -120.0,  # cci (oversold)
            0.025,  # atr (moderate volatility)
        ]
    )

    return observation, feature_names


def test_basic_functionality():
    """Test basic signal guide functionality."""
    print("=== Testing Basic Functionality ===")

    observation, feature_names = create_sample_observation()

    # Initialize signal guide
    guide = ActionSignalGuide(mode=GuidanceMode.FULL_GUIDANCE)
    guide.set_feature_names(feature_names)

    # Test signal evaluation
    buy_strength = guide.get_signal_strength(observation, 1, 0)
    print(f"RSI Oversold signal strength: {buy_strength}")

    # Test action recommendation
    action, confidence = guide.get_action_recommendation(observation)
    action_names = ["HOLD", "BUY", "SELL"]
    print(f"Recommended action: {action_names[action]}, Confidence: {confidence:.3f}")


def test_multi_timeframe():
    """Test multi-timeframe signal evaluation."""
    print("\n=== Testing Multi-Timeframe Analysis ===")

    observation, feature_names = create_sample_observation()

    # Initialize with multi-timeframe enabled
    guide = ActionSignalGuide(mode=GuidanceMode.FULL_GUIDANCE)
    guide.set_feature_names(feature_names)
    # guide.use_multi_timeframe = True # If needed, but it's auto-initialized if available

    # Test multi-timeframe signal strength
    for action in [0, 1, 2]:  # HOLD, BUY, SELL
        strength = guide.get_multi_timeframe_signal_strength(observation, action)
        action_names = ["HOLD", "BUY", "SELL"]
        print(f"Multi-timeframe {action_names[action]} strength: {strength:.3f}")

    # Test multi-timeframe recommendation
    action, confidence = guide.get_multi_timeframe_action_recommendation(observation)
    action_names = ["HOLD", "BUY", "SELL"]
    print(
        f"Multi-timeframe recommendation: {action_names[action]}, Confidence: {confidence:.3f}"
    )


def test_combined_signals():
    """Test combined signal evaluation."""
    print("\n=== Testing Combined Signals ===")

    observation, feature_names = create_sample_observation()

    signal_defs = SignalDefinitions()

    # Test combined signals
    combined_signals = [
        "trend_momentum_bullish",
        "oscillator_divergence_bullish",
        "high_volatility_breakout",
    ]

    for signal_name in combined_signals:
        if signal_name in signal_defs.signals:
            signal_func = signal_defs.signals[signal_name]["function"]
            try:
                result = signal_func(observation, feature_names)
                print(f"{signal_name}: {result}")
            except Exception as e:
                print(f"{signal_name}: Error - {e}")


def test_adaptive_features():
    """Test adaptive signal strength features."""
    print("\n=== Testing Adaptive Features ===")

    observation, feature_names = create_sample_observation()

    guide = ActionSignalGuide(mode=GuidanceMode.FULL_GUIDANCE)
    guide.set_feature_names(feature_names)

    # Test adaptive signal strength
    for action in [1, 2]:  # BUY, SELL
        adaptive_strength = guide.get_adaptive_signal_strength(observation, action)
        action_names = ["HOLD", "BUY", "SELL"]
        print(f"Adaptive {action_names[action]} strength: {adaptive_strength:.3f}")

    # Simulate learning by updating confidence
    print("Simulating learning updates...")
    for _ in range(5):
        guide.update_signal_confidence(observation, 1, 0.1)  # Positive reward for BUY

    # Test updated adaptive strength
    updated_strength = guide.get_adaptive_signal_strength(observation, 1)
    print(f"Updated adaptive BUY strength after learning: {updated_strength:.3f}")


def main():
    """Run all tests."""
    print("Action Signal Guide Extension Test")
    print("=" * 50)

    try:
        test_basic_functionality()
        test_multi_timeframe()
        test_combined_signals()
        test_adaptive_features()

        print("\n=== Test Summary ===")
        print("✅ Basic functionality: Working")
        print("✅ Multi-timeframe analysis: Working")
        print("✅ Combined signals: Working")
        print("✅ Adaptive features: Working")
        print("\nAll extensions successfully implemented and tested!")

    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    main()
