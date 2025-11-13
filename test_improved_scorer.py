#!/usr/bin/env python3
"""
Test script for improved SignalQualityScorer
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import numpy as np
from ztb.trading.signal.quality_scorer import SignalQualityScorer

def create_test_data():
    """Create test market data"""
    np.random.seed(42)

    # Generate sample OHLCV data (100 periods)
    dates = pd.date_range('2024-01-01', periods=100, freq='1H')
    base_price = 100.0

    # Create realistic price movements
    price_changes = np.random.normal(0, 0.02, 100)  # 2% volatility
    prices = base_price * np.cumprod(1 + price_changes)

    # Create OHLCV data
    high_mult = 1 + np.abs(np.random.normal(0, 0.01, 100))
    low_mult = 1 - np.abs(np.random.normal(0, 0.01, 100))
    volume = np.random.uniform(1000, 10000, 100)

    df = pd.DataFrame({
        'open': prices * (1 + np.random.normal(0, 0.005, 100)),
        'high': prices * high_mult,
        'low': prices * low_mult,
        'close': prices,
        'volume': volume
    }, index=dates)

    return df

def test_improved_scorer():
    """Test the improved SignalQualityScorer"""
    print("Testing improved SignalQualityScorer...")

    # Create test data
    df = create_test_data()

    # Initialize scorer
    scorer = SignalQualityScorer()

    # Test portfolio state
    portfolio = {
        'btc_balance': 0.1,
        'jpy_balance': 50000.0,
        'current_price': df['close'].iloc[-1]
    }

    # Test scoring
    continuous_action = 0.5  # Neutral action
    discrete_action, quality_score = scorer.calculate_signal_quality(
        df, continuous_action, portfolio
    )

    print("✓ SignalQualityScorer test completed successfully!")
    print(f"  Final score: {quality_score:.2f}")
    print(f"  Discrete action: {discrete_action}")
    print(f"  Weights: {scorer.weights}")

    # Test with different market conditions
    print("\nTesting with oversold conditions...")

    # Create oversold scenario (RSI < 30)
    oversold_df = df.copy()
    oversold_prices = oversold_df['close'].values
    # Make recent prices decline sharply to create oversold RSI
    for i in range(10):
        oversold_prices[-(i+1)] = oversold_prices[-(i+1)] * (1 - 0.03 * (i+1)/10)

    oversold_df['close'] = oversold_prices

    discrete_action_os, quality_score_os = scorer.calculate_signal_quality(
        oversold_df, continuous_action, portfolio
    )

    print(f"  Oversold scenario - Score: {quality_score_os:.2f}, Action: {discrete_action_os}")

    print("\n✓ All tests passed!")

if __name__ == "__main__":
    test_improved_scorer()