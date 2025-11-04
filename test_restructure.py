#!/usr/bin/env python3
"""
Basic functionality test for restructured ActionSignalGuide.
"""

import pandas as pd
from ztb.trading.strategies.action_signal_guide import ActionSignalGuide, GuidanceMode

# Create a simple config
class SimpleConfig:
    def __init__(self):
        self.guidance_level = GuidanceMode.FULL_GUIDANCE
        self.enable_parallel_processing = False
        self.enable_adaptive_algorithms = True
        self.enable_market_analysis = True
        self.enable_sac_integration = True
        self.enable_signal_validation = True
        self.enable_data_sanitization = True
        self.enable_performance_tracking = True

def main():
    print("Testing restructured ActionSignalGuide...")

    config = SimpleConfig()

    try:
        # Create ActionSignalGuide instance
        asg = ActionSignalGuide(config)
        print('✓ ActionSignalGuide instance created successfully')

        # Create sample data
        data = pd.DataFrame({
            'open': [100, 101, 102],
            'high': [105, 106, 107],
            'low': [95, 96, 97],
            'close': [103, 104, 105],
            'volume': [1000, 1100, 1200]
        })

        # Test signal generation
        signals = asg.generate_signals(data, current_index=2)
        print(f'✓ Signals generated: {len(signals)} signals')
        if signals:
            print(f'✓ First signal: {signals[0]}')
        print('✓ Basic functionality test passed!')

    except Exception as e:
        print(f'✗ Test failed with error: {e}')
        import traceback
        traceback.print_exc()
        return False

    return True

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)