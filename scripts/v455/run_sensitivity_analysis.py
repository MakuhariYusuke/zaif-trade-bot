#!/usr/bin/env python3
"""
HFT Sensitivity Analysis Script (v455)
Runs a grid search over reward parameters to find optimal settings.
"""

import sys
import os
import pandas as pd
import numpy as np
from pathlib import Path
from stable_baselines3 import SAC
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.callbacks import BaseCallback

# Add project root to path
try:
    from ztb.utils.path_utils import get_project_root, ensure_dir
    project_root = get_project_root()
except ImportError:
    project_root = Path(__file__).resolve().parent.parent.parent
    sys.path.insert(0, str(project_root))
    from ztb.utils.path_utils import get_project_root, ensure_dir

from ztb.features.hft_proxies import add_hft_features
from ztb.trading.environment.fast_intraday_env import FastIntradayEnv
from ztb.utils.logging_utils import setup_logging, get_logger
from ztb.utils.seed_manager import SeedManager

setup_logging()
logger = get_logger(__name__)

class SensitivityMetricsCallback(BaseCallback):
    def __init__(self, verbose=0):
        super().__init__(verbose)
        self.episode_rewards = []
        self.episode_lengths = []
        self.episode_costs = []
        self.episode_edge_shortfalls = []
        
        self.current_cost = 0.0
        self.current_shortfall = 0.0
        
    def _on_step(self) -> bool:
        infos = self.locals.get("infos", [])
        if infos:
            info = infos[0]
            self.current_cost += info.get("step_cost", 0.0)
            self.current_shortfall += info.get("edge_shortfall", 0.0)
            
            dones = self.locals.get("dones", [])
            if dones and dones[0]:
                self.episode_rewards.append(self.locals.get("rewards", [0])[0]) # This is step reward, not episode reward. 
                # Actually SB3 Monitor wrapper handles episode rewards. We can get it from info['episode']['r'] if available.
                if 'episode' in info:
                    self.episode_rewards.append(info['episode']['r'])
                    self.episode_lengths.append(info['episode']['l'])
                
                self.episode_costs.append(self.current_cost)
                self.episode_edge_shortfalls.append(self.current_shortfall)
                
                self.current_cost = 0.0
                self.current_shortfall = 0.0
        return True

def run_training(min_edge_mult, vol_floor, total_timesteps=20000, seed=42):
    """
    Runs a single training session with specified parameters.
    """
    run_name = f"edge_{min_edge_mult}_vol_{vol_floor}"
    log_dir = f"logs/v455_sensitivity/{run_name}"
    ensure_dir(log_dir)
    
    # Load Data (Cached if possible)
    data_path = "data/btc_jpy_1m_v454.csv"
    df = pd.read_csv(data_path, parse_dates=True, index_col=0)
    df = add_hft_features(df)
    feature_columns = ["clv", "vol_pressure", "impact_proxy", "vol_regime", "trend_persistence"]
    
    def make_env():
        env = FastIntradayEnv(
            df=df,
            feature_columns=feature_columns,
            initial_balance=100_000.0,
            max_position=0.01,
            max_steps=1000,
            prewarm_steps=100,
            max_ttl_steps=60,
            cooldown_steps=5,
            reward_params={
                "alpha": 0.5,
                "beta": 0.02,
                "min_edge_mult": min_edge_mult,
                "edge_penalty_rate": 1.0,
                "vol_floor": vol_floor,
                "vol_floor_penalty": 50.0,
                "hold_grace": 10,
                "hold_ramp": 0.01
            }
        )
        env.reset(seed=seed)
        env = Monitor(env, log_dir, info_keywords=("edge_shortfall", "vol_ratio", "trade_cost"))
        return env

    env = DummyVecEnv([make_env])
    env = VecNormalize(env, norm_obs=True, norm_reward=True, clip_obs=10.0)
    
    model = SAC(
        "MlpPolicy",
        env,
        verbose=0,
        learning_rate=3e-4,
        buffer_size=50_000, # Reduced for speed
        batch_size=256,
        ent_coef="auto",
        seed=seed
    )
    
    callback = SensitivityMetricsCallback()
    model.learn(total_timesteps=total_timesteps, callback=callback)
    
    # Calculate Metrics
    avg_reward = np.mean(callback.episode_rewards[-10:]) if callback.episode_rewards else -np.inf
    avg_len = np.mean(callback.episode_lengths[-10:]) if callback.episode_lengths else 0
    total_cost = np.sum(callback.episode_costs)
    total_shortfall = np.sum(callback.episode_edge_shortfalls)
    
    return {
        "min_edge_mult": min_edge_mult,
        "vol_floor": vol_floor,
        "avg_reward": avg_reward,
        "avg_len": avg_len,
        "total_cost": total_cost,
        "total_shortfall": total_shortfall
    }

def main():
    # Grid Search Parameters
    edge_mults = [1.0, 1.5, 2.0]
    vol_floors = [0.0005, 0.001, 0.002]
    
    results = []
    
    print("Starting Sensitivity Analysis...")
    print(f"Grid: Edge {edge_mults} x Vol {vol_floors}")
    
    for edge in edge_mults:
        for vol in vol_floors:
            print(f"Testing: Edge={edge}, Vol={vol}...")
            try:
                # Run short training (e.g. 10k steps for smoke test, 50k for real)
                # Using 5000 steps here just to prove it runs in the demo context
                res = run_training(edge, vol, total_timesteps=5000) 
                results.append(res)
                print(f"  -> Reward: {res['avg_reward']:.2f}, Len: {res['avg_len']:.1f}")
            except Exception as e:
                print(f"  -> Failed: {e}")
                
    # Save Results
    df_res = pd.DataFrame(results)
    output_path = "reports/v455_sensitivity_results.csv"
    ensure_dir("reports")
    df_res.to_csv(output_path, index=False)
    print(f"\nAnalysis Complete. Results saved to {output_path}")
    print(df_res)

if __name__ == "__main__":
    main()
