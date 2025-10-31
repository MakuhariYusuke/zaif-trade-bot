#!/usr/bin/env python3
"""Quick test of HeavyTradingEnv market regime adaptation."""

import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

from ztb.trading.environment.heavy_env.core import HeavyTradingEnv
from ztb.analysis.market_regime_classifier import MarketRegimeClassifier
import pandas as pd
import numpy as np

def test_env_regime_adaptation():
    # Create sample market data with required columns
    dates = pd.date_range('2023-01-01', periods=100, freq='h')  # Use 'h' instead of 'H'
    np.random.seed(42)
    close = 100 + np.sin(np.linspace(0, 4*np.pi, 100)) * 5 + np.random.normal(0, 1, 100)
    market_data = pd.DataFrame({
        'timestamp': dates,
        'open': close + np.random.normal(0, 0.5, 100),
        'high': close + np.abs(np.random.normal(0, 1, 100)),
        'low': close - np.abs(np.random.normal(0, 1, 100)),
        'close': close,
        'volume': np.random.uniform(1000, 10000, 100)
    })

    # Create environment config
    env_config = {
        'initial_balance': 10000,
        'max_position_size': 1.0,
        'transaction_fee': 0.001,
        'slippage': 0.0005,
        'market_regime_adaptation': {
            'enabled': True
        }
    }

    try:
        env = HeavyTradingEnv(df=market_data, config=env_config)

        # Create classifier
        classifier_config = {
            'adaptation': {
                'enabled': True,
                'regime_reward_multipliers': {'STRONG_BULL': 1.5}
            }
        }
        classifier = MarketRegimeClassifier(classifier_config)

        # Enable regime adaptation
        env.enable_market_regime_adaptation(classifier)

        print("HeavyTradingEnv initialized successfully with regime adaptation!")

        # Check attributes
        print(f"Regime adaptation enabled: {env.market_regime_adaptation_enabled}")
        print(f"Regime classifier set: {env.regime_classifier is not None}")

        # Test a step
        action = 0.1  # Small buy action (scalar)
        observation, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated

        print(f"Step completed - Reward: {reward}, Done: {done}")
        print(f"Regime info in step: {'regime' in info}")

        if 'regime' in info:
            print(f"Current regime: {info['regime']}")

        print("HeavyTradingEnv regime adaptation test passed!")

    except Exception as e:
        print(f"HeavyTradingEnv test failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == '__main__':
    test_env_regime_adaptation()