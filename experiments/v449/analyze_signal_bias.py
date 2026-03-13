import logging
import os
import sys

import numpy as np
import pandas as pd
from stable_baselines3 import SAC

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

from ztb.config.manager import ConfigManager
from ztb.features.models.sac.sac_v427_feature_engineering import (
    SACv427FeatureEngineering,
)
from ztb.training.environments.heavy_trading_env import HeavyTradingEnv

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def analyze_bias():
    # 1. Load Config
    config_path = "config/sac_v446_fixed.json"
    config_manager = ConfigManager(config_path)
    config = config_manager.config

    # 2. Load Data (Uptrend Period)
    data_path = "data/yahoo_finance/btc_jpy_1m_converted.csv"
    df = pd.read_csv(data_path, index_col=0, parse_dates=True)

    # Filter for uptrend: 2025-11-05 01:31:00 to 2025-11-06 13:00:00
    start_date = "2025-11-05 01:31:00"
    end_date = "2025-11-06 13:00:00"
    df = df[(df.index >= start_date) & (df.index <= end_date)]

    logger.info(f"Loaded {len(df)} rows for analysis.")

    # 3. Feature Engineering
    fe = SACv427FeatureEngineering()
    featured_data = fe.transform(df)

    # 4. Setup Environment
    env_config = config.environment
    # Ensure feature set matches
    env_config.feature_set = "sac_v427"

    env = HeavyTradingEnv(data=df, featured_data=featured_data, config=env_config)

    # 5. Load Model
    model_path = "models/sac_v446_fixed.zip"
    model = SAC.load(model_path)

    # 6. Run Analysis Loop
    obs, _ = env.reset()

    actions = []
    thresholds = []
    rewards = []
    pnls = []

    print("\n--- Detailed Step Analysis ---")
    print(
        f"{'Step':<6} | {'Price':<10} | {'Action (Raw)':<12} | {'Threshold':<10} | {'Decision':<8} | {'Reward':<10}"
    )

    for i in range(min(100, len(df))):  # Analyze first 100 steps
        action, _ = model.predict(obs, deterministic=True)
        action_val = float(action[0])

        # Get threshold before step (it might change after step, but we want current)
        threshold = env.threshold_manager.get_threshold(
            env.data.iloc[env.current_step].get("atr_14", 0),
            env.data.iloc[env.current_step]["close"],
        )

        # Step
        obs, reward, done, truncated, info = env.step(action)

        # Determine decision
        decision = "HOLD"
        if action_val > threshold:
            decision = "BUY"
        elif action_val < -threshold:
            decision = "SELL"

        print(
            f"{i:<6} | {info['current_price']:<10.1f} | {action_val:<12.4f} | {threshold:<10.4f} | {decision:<8} | {reward:<10.4f}"
        )

        actions.append(action_val)
        thresholds.append(threshold)
        rewards.append(reward)

        if done:
            break

    # Summary Statistics
    actions = np.array(actions)
    print("\n--- Summary ---")
    print(f"Mean Action: {np.mean(actions):.4f}")
    print(f"Min Action:  {np.min(actions):.4f}")
    print(f"Max Action:  {np.max(actions):.4f}")
    print(f"Std Action:  {np.std(actions):.4f}")
    print(f"Threshold Mean: {np.mean(thresholds):.4f}")


if __name__ == "__main__":
    analyze_bias()
