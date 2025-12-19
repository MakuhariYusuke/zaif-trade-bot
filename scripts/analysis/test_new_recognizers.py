from ztb.trading.strategies.action_signal_guide import ActionSignalGuide
from ztb.trading.strategies.action_signal_guide.pattern_recognition.bollinger_patterns import BollingerBandsRecognizer
from ztb.trading.strategies.action_signal_guide.pattern_recognition.adx_patterns import ADXRecognizer
import pandas as pd
import numpy as np

print('All imports successful!')

# Test basic instantiation
guide = ActionSignalGuide()
status = guide.get_system_status()
print('Bollinger recognizers:', len(status['recognizers'].get('bollinger', [])))
print('ADX recognizers:', len(status['recognizers'].get('adx', [])))

# Test signal generation with sample data
dates = pd.date_range('2023-01-01', periods=100, freq='1H')
data = pd.DataFrame({
    'open': np.random.uniform(100, 110, 100),
    'high': np.random.uniform(105, 115, 100),
    'low': np.random.uniform(95, 105, 100),
    'close': np.random.uniform(100, 110, 100),
    'volume': np.random.uniform(1000, 10000, 100)
}, index=dates)

signals = guide.generate_signals(data, 50)
print(f'Generated {len(signals)} signals')

# Test individual recognizers
bollinger_recognizer = BollingerBandsRecognizer()
adx_recognizer = ADXRecognizer()

bollinger_signal = bollinger_recognizer.recognize(data, 50)
adx_signal = adx_recognizer.recognize(data, 50)

print(f'Bollinger signal: {bollinger_signal.signal_type if bollinger_signal else None}')
print(f'ADX signal: {adx_signal.signal_type if adx_signal else None}')

print('Test completed successfully!')