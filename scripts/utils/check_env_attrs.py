#!/usr/bin/env python3
"""Check HeavyTradingEnv attributes"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path.cwd()))

import pandas as pd

from ztb.trading.environment.heavy_env.core import HeavyTradingEnv
from ztb.trading.environment.utils.config import EnvironmentConfig

# Load data
df = pd.read_csv("data/btc_jpy_real_dataset.csv")
env_config = EnvironmentConfig(
    initial_portfolio_value=10000.0, reward_scaling=1.0, use_continuous_actions=True
)
env = HeavyTradingEnv(df=df.head(100), config=env_config, use_continuous_actions=True)

print("HeavyTradingEnv attributes:")
attrs = [attr for attr in dir(env) if not attr.startswith("_")]
for attr in sorted(attrs):
    try:
        value = getattr(env, attr)
        if not callable(value):
            print(f"{attr}: {type(value)} = {value}")
    except:
        print(f"{attr}: <error accessing>")
