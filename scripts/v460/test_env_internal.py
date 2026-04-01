import sys
import numpy as np
import pandas as pd
from pathlib import Path

project_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(project_root))

from configs.v460.base import EnvironmentConfig
from ztb.trading.environment.heavy_env.core import HeavyTradingEnv

def main() -> None:
    try:
        import torch  # noqa: F401
    except Exception:
        pass

    cfg = EnvironmentConfig(
        transaction_cost=0.001,
        max_position_size=0.01,
        initial_portfolio_value=10000000.0,
        use_continuous_actions=True,
        action_space_type="continuous_1d",
        exchange="coincheck",
        timeframe="1m"
    )
    
    # 5 rows is enough
    df = pd.DataFrame({
        "timestamp": pd.date_range("2021-01-01", periods=5, freq="1Min"),
        "open": [100]*5,
        "high": [100]*5,
        "low": [100]*5,
        "close": [100]*5,
        "volume": [1]*5,
        "feature1": [1.0, 2.0, 3.0, 4.0, 5.0],
    })
    
    env = HeavyTradingEnv(df=df, config=cfg)
    obs, info = env.reset()
    
    print(f"Num config features: {len(env.features)}")
    print(f"Observation shape: {obs.shape}")
    print(f"Action space: {env.action_space}")
    
    # Take some steps
    # continuous_1d uses values between [-1, +1]. So 0.8 is buy
    obs, reward, done, trunc, info = env.step(np.array([0.8]))
    print(f"After BUY: obs shape {obs.shape}, position {env.position}")
    
    obs, reward, done, trunc, info = env.step(np.array([0.0]))
    print(f"After HOLD: obs shape {obs.shape}, position {env.position}")
    
    obs, reward, done, trunc, info = env.step(np.array([-0.9]))
    print(f"After SELL: obs shape {obs.shape}, position {env.position}")

if __name__ == "__main__":
    main()
