import sys
from pathlib import Path
import pandas as pd
import numpy as np
import logging

# Add workspace root to path
workspace_root = Path(__file__).parent.parent.parent
sys.path.append(str(workspace_root))

from ztb.features.base_features_v456 import calculate_base_features
from ztb.trading.environment.utils.fast_intraday_env_v456_utils import (
    create_fast_intraday_env_v456,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("debug_env")

def main():
    # Mock dataframe with enough data to avoid window errors
    dates = pd.date_range(start='2021-01-01', periods=500, freq='1min')
    df = pd.DataFrame({
        'timestamp': dates,
        'open': np.random.uniform(100, 200, 500),
        'high': np.random.uniform(200, 210, 500),
        'low': np.random.uniform(90, 100, 500),
        'close': np.random.uniform(100, 200, 500),
        'volume': np.random.uniform(1000, 5000, 500)
    })

    logger.info("Created mock dataframe")

    df = calculate_base_features(df, copy=False)

    logger.info("Creating env...")
    env = create_fast_intraday_env_v456(df=df, env_config={})
    if env is None:
        logger.error("Failed to create environment.")
        return
    del df
    
    logger.info(f"Env type: {type(env)}")
    
    if env:
        logger.info(f"Has df attribute: {hasattr(env, 'df')}")
        if hasattr(env, 'df'):
            logger.info(f"Env df type: {type(env.df)}")
            if env.df is not None:
                logger.info(f"Env df len: {len(env.df)}")
            else:
                logger.error("Env df is None")
        else:
            logger.error("Env has no df attribute")
            
        # Check step return
        obs, info = env.reset()
        logger.info("Reset done.")
        action = env.action_space.sample()
        step_result = env.step(action)
        logger.info(f"Step result length: {len(step_result)}")
        logger.info(f"Step result types: {[type(x) for x in step_result]}")

if __name__ == "__main__":
    main()
