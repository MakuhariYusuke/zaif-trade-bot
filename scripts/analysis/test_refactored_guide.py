import pandas as pd
import numpy as np
from ztb.trading.strategies.action_signal_guide import ActionSignalGuide

# Create sample data
dates = pd.date_range('2023-01-01', periods=100, freq='1H')
data = pd.DataFrame({
    'open': np.random.uniform(100, 110, 100),
    'high': np.random.uniform(105, 115, 100),
    'low': np.random.uniform(95, 105, 100),
    'close': np.random.uniform(100, 110, 100),
    'volume': np.random.uniform(1000, 10000, 100)
}, index=dates)

guide = ActionSignalGuide()
signals = guide.generate_signals(data, 50)
print(f'Generated {len(signals)} signals')
stats = guide.get_performance_stats()
print(f'Cache hits: {stats["cache_hits"]}, Cache misses: {stats["cache_misses"]}')
print('Signal generation test completed successfully!')