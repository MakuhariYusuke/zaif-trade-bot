from ztb.trading.environment.heavy_env.core import HeavyTradingEnv
import pandas as pd
import numpy as np

# Create sample market data like in the test
dates = pd.date_range('2023-01-01', periods=2000, freq='1min')
np.random.seed(42)

# Create realistic market data with multiple regimes
base_price = 100

# First 500 points: strong bull trend
bull_trend = np.linspace(0, 15, 500)
bull_noise = np.random.normal(0, 0.8, 500)
bull_prices = base_price + bull_trend + bull_noise

# Next 500 points: high volatility ranging
range_center = base_price + 15
range_noise = np.random.normal(0, 3.0, 500)
range_prices = range_center + range_noise

# Next 500 points: strong bear trend
bear_trend = np.linspace(0, -12, 500)
bear_noise = np.random.normal(0, 1.0, 500)
bear_prices = range_center + bear_trend + bear_noise

# Last 500 points: low volatility ranging
final_center = range_center - 12
final_noise = np.random.normal(0, 0.3, 500)
final_prices = final_center + final_noise

close = np.concatenate([bull_prices, range_prices, bear_prices, final_prices])

# Create OHLC data
high = close + np.abs(np.random.normal(0, 0.5, 2000))
low = close - np.abs(np.random.normal(0, 0.5, 2000))
open_price = np.roll(close, 1)
open_price[0] = base_price
volume = np.random.uniform(100, 2000, 2000)

df = pd.DataFrame({
    'timestamp': dates,
    'open': open_price,
    'high': high,
    'low': low,
    'close': close,
    'volume': volume
})

env = HeavyTradingEnv(df=df, config={'initial_balance': 10000})
print('hasattr(env, enable_market_regime_adaptation):', hasattr(env, 'enable_market_regime_adaptation'))