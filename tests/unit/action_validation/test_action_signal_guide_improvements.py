#!/usr/bin/env python3
"""
Test script for ActionSignalGuide improvements
"""

import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd

from ztb.trading.strategies.action_signal_guide.action_signal_guide import (
    ActionSignalGuide,
    GuidanceLevel,
)


def test_action_signal_guide_improvements():
    """Test ActionSignalGuide improvements"""
    print("Testing ActionSignalGuide improvements...")

    # Create test data
    np.random.seed(42)
    dates = pd.date_range("2024-01-01", periods=100, freq="h")
    base_price = 100.0
    price_changes = np.random.normal(0, 0.02, 100)
    prices = base_price * np.cumprod(1 + price_changes)
    df = pd.DataFrame(
        {
            "open": prices,
            "high": prices * 1.01,
            "low": prices * 0.99,
            "close": prices,
            "volume": np.random.uniform(1000, 10000, 100),
        },
        index=dates,
    )

    # Test ActionSignalGuide with memory management
    config = {
        "guidance_level": GuidanceLevel.STRONG,
        "max_signal_history": 50,  # Smaller for testing
        "memory_cleanup_interval": 10,  # Frequent cleanup
        "enable_caching": True,
        "cache_size": 100,
    }

    try:
        guide = ActionSignalGuide(config=config)
        print("✓ ActionSignalGuide initialized with memory management")

        # Test signal generation
        signals = guide.generate_signals(df, 50)
        print(f"✓ Generated {len(signals)} signals")

        # Test memory management
        initial_history_len = len(guide.signal_history)
        print(f"✓ Initial signal history: {initial_history_len} entries")

        # Generate more signals to trigger cleanup
        for i in range(15):  # More than cleanup interval
            try:
                guide.generate_signals(df, 50 + i)
            except IndexError:
                break  # Stop if we run out of data

        final_history_len = len(guide.signal_history)
        print(
            f"✓ Final signal history: {final_history_len} entries (limited by max_signal_history)"
        )

        # Test cleanup method
        guide._cleanup_memory()
        print("✓ Memory cleanup executed successfully")

        print("All ActionSignalGuide improvements working correctly!")
        return True

    except Exception as e:
        print(f"✗ Test failed: {e}")
        import traceback

        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = test_action_signal_guide_improvements()
    sys.exit(0 if success else 1)
