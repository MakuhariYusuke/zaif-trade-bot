import pandas as pd
import numpy as np
from ztb.trading.strategies.action_signal_guide.action_signal_guide import ActionSignalGuide

# Create test data
data = pd.DataFrame({
    'open': [100, 101, 102, 103, 104],
    'high': [105, 106, 107, 108, 109],
    'low': [95, 96, 97, 98, 99],
    'close': [102, 103, 104, 105, 106],
    'volume': [1000, 1100, 1200, 1300, 1400]
})

guide = ActionSignalGuide()
signals = guide.generate_signals(data, 4)
print(f'Generated {len(signals)} signals successfully')