"""
Test script for SignalRewardIntegrator with new Bollinger Bands and ADX integration
"""

import pandas as pd
import numpy as np
from ztb.trading.strategies.action_signal_guide import ActionSignalGuide
from ztb.trading.strategies.signal_reward_integrator import SignalRewardIntegrator

def test_new_integrator():
    """Test SignalRewardIntegrator with new indicators."""

    # Create ActionSignalGuide with new indicators
    guide = ActionSignalGuide()

    # Create SignalRewardIntegrator with new weights
    integrator = SignalRewardIntegrator(
        signal_guide=guide,
        signal_bonus_weight=0.1,
        signal_penalty_weight=0.05,
        bollinger_weight=1.3,  # Higher weight for volatility signals
        adx_weight=1.4,       # Higher weight for trend strength signals
        enable_advanced_integration=True
    )

    # Create sample observation (OHLCV data)
    dates = pd.date_range('2023-01-01', periods=100, freq='1H')
    data = pd.DataFrame({
        'open': np.random.uniform(100, 110, 100),
        'high': np.random.uniform(105, 115, 100),
        'low': np.random.uniform(95, 105, 100),
        'close': np.random.uniform(100, 110, 100),
        'volume': np.random.uniform(1000, 10000, 100)
    }, index=dates)

    # Convert to observation format (simplified)
    observation = np.array([
        data.iloc[-1]['open'],
        data.iloc[-1]['high'],
        data.iloc[-1]['low'],
        data.iloc[-1]['close'],
        data.iloc[-1]['volume']
    ])

    # Test different actions
    actions = [0, 1, 2]  # HOLD, BUY, SELL
    base_reward = 1.0

    print("Testing SignalRewardIntegrator with new indicators...")
    print("=" * 60)

    for action in actions:
        action_name = {0: "HOLD", 1: "BUY", 2: "SELL"}[action]

        # Test integration
        modified_reward = integrator.integrate_signal_reward(
            reward=base_reward,
            observation=observation,
            action=action,
            step=1
        )

        print(f"Action {action_name}: Base reward {base_reward:.3f} -> Modified {modified_reward:.3f}")

    # Get integration stats
    stats = integrator.get_integration_stats()
    print("\nIntegration Statistics:")
    print(f"Total steps: {stats['total_steps']}")
    print(f"Signal bonuses applied: {stats['signal_bonuses_applied']}")
    print(f"Signal penalties applied: {stats['signal_penalties_applied']}")
    print(f"Bollinger signals used: {stats['bollinger_signals_used']}")
    print(f"ADX signals used: {stats['adx_signals_used']}")

    print("\nPattern Weights:")
    print(f"Bollinger Bands: {stats['pattern_weights']['bollinger']}")
    print(f"ADX: {stats['pattern_weights']['adx']}")

    # Test system status
    system_status = guide.get_system_status()
    print("\nActionSignalGuide Status:")
    print(f"Bollinger recognizers: {len(system_status['recognizers'].get('bollinger', []))}")
    print(f"ADX recognizers: {len(system_status['recognizers'].get('adx', []))}")

    print("\nTest completed successfully! ✅")

if __name__ == "__main__":
    test_new_integrator()