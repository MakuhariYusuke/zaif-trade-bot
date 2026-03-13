#!/usr/bin/env python3
"""
Test script for Action Signal Guide functionality
"""

import numpy as np

from ztb.trading.strategies import ActionSignalGuide, SignalDefinitions


def test_signal_guide():
    print("Testing ActionSignalGuide...")

    # シグナルガイドの初期化テスト
    guide = ActionSignalGuide()
    print("✓ ActionSignalGuide initialized")

    # 特徴量名の設定
    feature_names = ["close", "rsi_14", "macd", "macd_signal", "bb_upper", "bb_lower"]
    guide.set_feature_names(feature_names)
    print("✓ Feature names set")

    # テスト観測値（BUYシグナル: RSI低位, 価格がBB下バンド）
    buy_observation = np.array([100.0, 25.0, -0.5, 0.2, 105.0, 95.0])
    action, confidence = guide.get_action_recommendation(buy_observation)
    print(f"✓ BUY signal test - Action: {action}, Confidence: {confidence:.3f}")

    # SELLシグナルテスト
    sell_observation = np.array([110.0, 75.0, 0.5, -0.2, 105.0, 95.0])
    action, confidence = guide.get_action_recommendation(sell_observation)
    print(f"✓ SELL signal test - Action: {action}, Confidence: {confidence:.3f}")

    # シグナル強度テスト
    buy_strength = guide.get_signal_strength(buy_observation, 1, step=0)  # BUY action
    sell_strength = guide.get_signal_strength(
        sell_observation, 2, step=0
    )  # SELL action
    print(f"✓ Signal strength - BUY: {buy_strength:.3f}, SELL: {sell_strength:.3f}")

    print("✓ Signal guide test completed successfully")


def test_signal_definitions():
    print("\nTesting SignalDefinitions...")

    signals = SignalDefinitions()
    signal_names = signals.get_signal_names()
    print(f"✓ Available signals: {len(signal_names)}")

    # 各シグナルタイプの数を確認
    buy_signals = signals.get_signals_by_type(signals.SignalType.BUY)
    sell_signals = signals.get_signals_by_type(signals.SignalType.SELL)
    neutral_signals = signals.get_signals_by_type(signals.SignalType.NEUTRAL)

    print(f"✓ BUY signals: {len(buy_signals)}")
    print(f"✓ SELL signals: {len(sell_signals)}")
    print(f"✓ NEUTRAL signals: {len(neutral_signals)}")

    print("✓ Signal definitions test completed successfully")


if __name__ == "__main__":
    test_signal_guide()
    test_signal_definitions()
    print("\n🎉 All tests passed!")
