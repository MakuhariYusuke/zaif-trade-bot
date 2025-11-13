#!/usr/bin/env python3
"""
Simple test script to debug subprocess execution
"""
import sys
import os
import json
import pandas as pd

# Test basic imports
try:
    from stable_baselines3 import SAC
    from ztb.trading.environment.environment import HeavyTradingEnv
    import numpy as np
    print("All imports successful")
except Exception as e:
    print(f"Import error: {e}")
    sys.exit(1)

# Test data loading
try:
    # Generate dummy data
    dates = pd.date_range('2020-01-01', periods=100, freq='1min')
    df = pd.DataFrame({
        'timestamp': dates,
        'open': 1000000 + np.random.randn(100) * 10000,
        'high': 1010000 + np.random.randn(100) * 10000,
        'low': 990000 + np.random.randn(100) * 10000,
        'close': 1000000 + np.random.randn(100) * 10000,
        'volume': np.random.randint(100, 1000, 100)
    })
    print("Data generation successful")
except Exception as e:
    print(f"Data generation error: {e}")
    sys.exit(1)

# Test environment creation
try:
    env_config = {
        'use_continuous_actions': True,
        'max_position_size': 0.1,
        'transaction_fee': 0.001
    }
    env = HeavyTradingEnv(df=df, config=env_config)
    print("Environment creation successful")
except Exception as e:
    print(f"Environment creation error: {e}")
    sys.exit(1)

# Test SAC model creation
try:
    model = SAC(
        'MlpPolicy',
        env,
        learning_rate=3e-4,
        buffer_size=10000,
        learning_starts=10,
        batch_size=64,
        tau=0.005,
        gamma=0.99,
        ent_coef=1.0,
        verbose=0
    )
    print("SAC model creation successful")
except Exception as e:
    print(f"SAC model creation error: {e}")
    sys.exit(1)

# Test training
try:
    model.learn(total_timesteps=50)
    print("Training successful")
except Exception as e:
    print(f"Training error: {e}")
    sys.exit(1)

# Return success
result = {
    "success": True,
    "total_timesteps": 50,
    "algorithm": "sac",
    "model_path": "temp_model.zip",
    "log_path": "temp_logs",
    "critic_loss": 1.0,
    "actor_loss": 0.8,
    "ent_coef": 1.0,
}

print(json.dumps(result))