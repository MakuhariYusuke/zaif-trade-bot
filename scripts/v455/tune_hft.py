#!/usr/bin/env python3
"""
HFT / Fast Intraday Hyperparameter Tuning Script (v455)
"""

import sys
import os
from pathlib import Path
import pandas as pd
import numpy as np
import optuna
from stable_baselines3 import SAC
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

# Add project root to path
project_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(project_root))

from ztb.features.hft_proxies import add_hft_features
from ztb.trading.environment.fast_intraday_env import FastIntradayEnv
from ztb.utils.logging_utils import setup_logging, get_logger

setup_logging()
logger = get_logger(__name__)

DATA_PATH = "data/btc_jpy_1m_v454.csv"
LOG_DIR = "logs/v455_hft_tuning"
os.makedirs(LOG_DIR, exist_ok=True)

# Load Data Once
if os.path.exists(DATA_PATH):
    df_raw = pd.read_csv(DATA_PATH, parse_dates=["timestamp"], index_col=0)
    df_processed = add_hft_features(df_raw)
    feature_columns = ["clv", "vol_pressure", "impact_proxy", "vol_regime", "trend_persistence"]
else:
    logger.error(f"Data file not found: {DATA_PATH}")
    sys.exit(1)

def objective(trial):
    # Hyperparameters
    alpha = trial.suggest_float("alpha", 0.01, 1.0, log=True)
    beta = trial.suggest_float("beta", 0.0001, 0.01, log=True)
    learning_rate = trial.suggest_float("learning_rate", 1e-5, 1e-3, log=True)
    batch_size = trial.suggest_categorical("batch_size", [64, 128, 256, 512])
    buffer_size = trial.suggest_categorical("buffer_size", [10000, 50000, 100000])
    ent_coef = trial.suggest_categorical("ent_coef", ["auto", 0.01, 0.05, 0.1])

    base_reward_params = {
        "min_edge_mult": 1.2,
        "edge_penalty_rate": 1.0,
        "vol_floor": 0.001,
        "vol_floor_penalty": 50.0,
        "hold_grace": 10,
        "hold_ramp": 0.01
    }
    
    # Create Environment
    def make_env():
        env = FastIntradayEnv(
            df=df_processed,
            feature_columns=feature_columns,
            max_steps=1000, # Short episodes for tuning
            prewarm_steps=100,
            max_ttl_steps=60,
            cooldown_steps=5,
            reward_params={"alpha": alpha, "beta": beta, **base_reward_params}
        )
        env = Monitor(env, None) # No log file for tuning trials to save space
        return env
    
    env = DummyVecEnv([make_env])
    env = VecNormalize(env, norm_obs=True, norm_reward=True, clip_obs=10.0)
    
    # Initialize Model
    model = SAC(
        "MlpPolicy",
        env,
        verbose=0,
        learning_rate=learning_rate,
        buffer_size=buffer_size,
        batch_size=batch_size,
        ent_coef=ent_coef,
        train_freq=1,
        gradient_steps=1,
        learning_starts=100
    )
    
    # Train for a short period
    try:
        model.learn(total_timesteps=5000)
    except Exception as e:
        logger.error(f"Trial failed: {e}")
        return -float("inf")
    
    # Evaluate
    mean_reward = 0.0
    eval_episodes = 5
    for _ in range(eval_episodes):
        obs = env.reset()
        done = False
        episode_reward = 0.0
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, info = env.step(action)
            episode_reward += reward[0]
        mean_reward += episode_reward
        
    return mean_reward / eval_episodes

def main():
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=20)
    
    logger.info("Best trial:")
    trial = study.best_trial
    logger.info(f"  Value: {trial.value}")
    logger.info("  Params: ")
    for key, value in trial.params.items():
        logger.info(f"    {key}: {value}")
        
    # Save best params
    import json
    with open(os.path.join(LOG_DIR, "best_params.json"), "w") as f:
        json.dump(trial.params, f, indent=4)

if __name__ == "__main__":
    main()
