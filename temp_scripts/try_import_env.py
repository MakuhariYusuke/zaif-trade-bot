import traceback
from ztb.trading.environment import HeavyTradingEnv
import pandas as pd
import numpy as np


def create_synthetic_df(rows=200):
    rng = np.random.default_rng(42)
    price_trend = np.linspace(100, 110, rows) + rng.normal(0, 0.5, rows)
    return pd.DataFrame(
        {
            "open": price_trend + rng.normal(0, 0.1, rows),
            "high": price_trend + rng.normal(0, 0.2, rows),
            "low": price_trend - rng.normal(0, 0.2, rows),
            "close": price_trend + rng.normal(0, 0.05, rows),
            "volume": rng.normal(1000, 50, rows),
        }
    )

try:
    df = create_synthetic_df(200)
    env = HeavyTradingEnv(df=df, config={"feature_set":"minimal","curriculum_stage":"forced_balance","curriculum_learning":{"enabled":True,"auto_progression":False}})
    print('Instantiated OK')
except Exception as e:
    traceback.print_exc()