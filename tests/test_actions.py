import sys
sys.path.insert(0, '.')
from scripts.test_signal_improvement import SignalGuidanceBacktestValidator
import pandas as pd
import numpy as np

# Create test data
dates = pd.date_range('2024-01-01', periods=100, freq='5min')
market_data = pd.DataFrame({
    'open': np.random.uniform(50000, 51000, 100),
    'high': np.random.uniform(50500, 51500, 100),
    'low': np.random.uniform(49500, 50500, 100),
    'close': np.random.uniform(50000, 51000, 100),
    'volume': np.random.uniform(100, 1000, 100)
}, index=dates)

validator = SignalGuidanceBacktestValidator()
actions = validator.simulate_model_actions(market_data)
print(f'Generated {len(actions)} actions')
print(f'Buy signals (>0.2): {sum(1 for a in actions if a > 0.2)}')
print(f'Sell signals (<-0.2): {sum(1 for a in actions if a < -0.2)}')
print(f'Neutral (-0.2 to 0.2): {sum(1 for a in actions if -0.2 <= a <= 0.2)}')
print(f'Sample actions: {actions[:10]}')