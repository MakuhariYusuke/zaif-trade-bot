#!/usr/bin/env python3
"""
Test untrained model predictions.
"""

import pandas as pd
from stable_baselines3 import PPO

from ztb.trading.environment import HeavyTradingEnv

# Load test data
df = pd.read_csv("ml-dataset-enhanced.csv")

# Create environment
env_config = {
    "reward_scaling": 1.0,
    "transaction_cost": 0.0,
    "max_position_size": 1.0,
    "feature_set": "full",
    "curriculum_stage": "full",
}

env = HeavyTradingEnv(df=df, config=env_config, random_start=False)
obs, info = env.reset()

# Create untrained model to see random actions
model = PPO("MlpPolicy", env, verbose=0)

print("Testing untrained model predictions:")
for i in range(10):
    obs_reshaped = obs.reshape(1, -1)
    action, _ = model.predict(obs_reshaped, deterministic=False)
    print(f"Step {i}: obs[:3]={obs[:3]}, action={action[0]}")
    obs, reward, terminated, truncated, info = env.step(action[0])
    if terminated or truncated:
        break
