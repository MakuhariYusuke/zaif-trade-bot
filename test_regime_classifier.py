#!/usr/bin/env python3
"""Quick test of market regime classifier functionality."""

from ztb.analysis.market_regime_classifier import MarketRegimeClassifier, RegimeType
import pandas as pd
import numpy as np

def test_classifier():
    # Create sample data
    dates = pd.date_range('2023-01-01', periods=100, freq='H')
    np.random.seed(42)
    close = 100 + np.sin(np.linspace(0, 4*np.pi, 100)) * 5 + np.random.normal(0, 1, 100)
    data = pd.DataFrame({
        'close': close,
        'high': close + 1,
        'low': close - 1,
        'volume': np.random.uniform(1000, 10000, 100)
    }, index=dates)

    # Test classifier
    config = {
        'adaptation': {
            'enabled': True,
            'regime_reward_multipliers': {RegimeType.STRONG_BULL: 1.5}
        }
    }
    classifier = MarketRegimeClassifier(config)
    result = classifier.detect_regime(data)
    print(f'Regime: {result.primary_regime}, Confidence: {result.confidence:.2f}')
    print(f'Reward multiplier: {classifier.get_regime_multiplier(result.primary_regime, "reward")}')
    print('Market regime classifier test passed!')

if __name__ == '__main__':
    test_classifier()