import os
import sys

sys.path.insert(0, os.path.dirname(__file__))

try:
    from ztb.trading.strategies.action_signal_guide.action_signal_guide import (
        ActionSignalGuide,
    )

    print("Direct import successful")

    # Try instantiation
    guide = ActionSignalGuide()
    print("Instantiation successful")

    # Try basic functionality
    import pandas as pd

    data = pd.DataFrame(
        {
            "open": [100, 101, 102, 103, 104],
            "high": [105, 106, 107, 108, 109],
            "low": [95, 96, 97, 98, 99],
            "close": [102, 103, 104, 105, 106],
            "volume": [1000, 1100, 1200, 1300, 1400],
        }
    )

    signals = guide.generate_signals(data, 4)
    print(f"Generated {len(signals)} signals successfully")

except Exception as e:
    print(f"Error: {e}")
    import traceback

    traceback.print_exc()
