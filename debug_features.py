
import sys
import os
import pandas as pd

# Add project root to path
sys.path.append(os.getcwd())

from ztb.trading.environment.heavy_env.core import HeavyTradingEnv
from ztb.trading.environment.utils.config import EnvironmentConfig

def check_features():
    df = pd.read_csv("data/btc_jpy_1m_v451.csv", parse_dates=["timestamp"], index_col="timestamp")
    # Use a small chunk
    df = df.iloc[:1000]
    
    config = EnvironmentConfig(
        feature_set="default",
        use_continuous_actions=True,
        target_feature_count=138,
        correlation_reduction=True
    )
    
    env = HeavyTradingEnv(df=df, config=config)
    print(f"Number of features with target=138: {len(env.features)}")
    # print(f"Features: {env.features}")

if __name__ == "__main__":
    check_features()
