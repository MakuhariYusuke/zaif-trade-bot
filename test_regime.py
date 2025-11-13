import pandas as pd
import numpy as np
from ztb.trading.signal.regime.classifier import MarketRegimeClassifier

# Create test data
dates = pd.date_range('2024-01-01', periods=100, freq='h')
np.random.seed(42)
close_prices = []
base_price = 100.0
for i in range(100):
    trend = 0.0005 * i
    volatility = 0.01 * np.random.normal(0, 1)
    price = base_price * (1 + trend + volatility)
    close_prices.append(price)
    base_price = price

data = pd.DataFrame({
    'timestamp': dates,
    'open': close_prices,
    'high': [p * (1 + abs(np.random.normal(0, 0.005))) for p in close_prices],
    'low': [p * (1 - abs(np.random.normal(0, 0.005))) for p in close_prices],
    'close': close_prices,
    'volume': np.random.uniform(1000, 2000, 100)
})
data.set_index('timestamp', inplace=True)

classifier = MarketRegimeClassifier()
try:
    result = classifier.detect_regime(data)
    print('Success:', result['primary_regime'])
except Exception as e:
    print('Error:', e)
    import traceback
    traceback.print_exc()