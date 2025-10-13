#!/usr/bin/env python3
"""
Debug environment initialization
"""

import sys
sys.path.insert(0, '.')
from ztb.trading.environment.heavy_env.core import HeavyTradingEnv
from ztb.trading.environment.utils.config import EnvironmentConfig
import pandas as pd

config = EnvironmentConfig(use_continuous_actions=True, transaction_cost=0.0)
df = pd.read_csv('btc_jpy_real_dataset.csv')
df['timestamp'] = pd.to_datetime(df['timestamp'])
env = HeavyTradingEnv(df=df, config=config)

print("Environment initialized")
print(f"Action space: {env.action_space}")
print(f"Observation space: {env.observation_space}")

obs = env.reset()
print('Reset obs:', obs)
print('Current step:', env.current_step)
print('Position:', env.position)
print('Portfolio value:', env.portfolio_value)

# Check current price
current_price = env.df.iloc[env.current_step]['close']
print('Current price at step', env.current_step, ':', current_price)

# Try a step
print("\nTrying a step with action 0.1...")
try:
    next_obs, reward, done, truncated, info = env.step(0.1)
    print('Step successful')
    print('Reward:', reward)
    print('New position:', env.position)
except Exception as e:
    print('Step failed:', e)
    import traceback
    traceback.print_exc()