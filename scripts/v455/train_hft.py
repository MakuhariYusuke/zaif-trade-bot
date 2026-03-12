#!/usr/bin/env python3
"""
HFT / Fast Intraday Training Script (v455)
"""

import sys
import os
import gc
from pathlib import Path
import pandas as pd
import numpy as np
from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import CheckpointCallback, BaseCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

# Add project root to path
# Use ztb.utils.path_utils if available, otherwise fallback
try:
    # Try to import from local ztb if already in path (unlikely but possible)
    from ztb.utils.path_utils import get_project_root, ensure_dir
    project_root = get_project_root()
except ImportError:
    # Fallback to manual resolution
    project_root = Path(__file__).resolve().parent.parent.parent
    sys.path.insert(0, str(project_root))
    from ztb.utils.path_utils import get_project_root, ensure_dir

from ztb.features.hft_proxies import add_hft_features
from ztb.trading.environment.fast_intraday_env import FastIntradayEnv
from ztb.utils.logging_utils import setup_logging, get_logger
from ztb.utils import format_time, format_number, format_currency
from ztb.utils.seed_manager import SeedManager
from ztb.utils.notify import DiscordNotifier

setup_logging()
logger = get_logger(__name__)

class HFTMetricsCallback(BaseCallback):
    """
    Custom callback for logging HFT-specific metrics to TensorBoard.
    """
    def __init__(self, verbose=0):
        super().__init__(verbose)
        self.current_episode_cost = 0.0
        
    def _on_step(self) -> bool:
        # Throttle logging to every 100 steps to save I/O and processing
        if self.n_calls % 100 != 0:
            # Still need to accumulate cost?
            # Accessing infos every step might be slow if infos is large.
            # But we need to accumulate cost for the episode.
            # Let's check if we can optimize this.
            # If we skip accessing infos, we miss step_cost.
            # So we must access infos.
            infos = self.locals.get("infos", [])
            if infos:
                self.current_episode_cost += infos[0].get("step_cost", 0)
            return True

        # Access the info dict of the last step
        infos = self.locals.get("infos", [])
        if infos:
            # Assuming single env for now
            info = infos[0]
            
            # Log instantaneous metrics
            # Use format_number/currency for readable console output
            # Note: SB3 might treat strings as text logs (not plotted in TB scalars)
            # So we might want to keep raw values if plotting is needed, 
            # but the requirement is to use format_number.
            # We will log both for safety: raw for graphs, formatted for display.
            
            balance = info.get("balance", 0)
            position = info.get("position", 0)
            ttl = info.get("ttl", 0)
            drawdown = info.get("drawdown", 0)
            edge_shortfall = info.get("edge_shortfall", 0.0)
            trade_cost = info.get("trade_cost", info.get("step_cost", 0.0))
            vol_ratio = info.get("vol_ratio", 0.0)
            
            # Raw for TensorBoard
            self.logger.record("hft/balance_val", balance)
            self.logger.record("hft/position_val", position)
            self.logger.record("hft/drawdown_val", drawdown)
            self.logger.record("hft/edge_shortfall_val", edge_shortfall)
            self.logger.record("hft/trade_cost_val", trade_cost)
            self.logger.record("hft/vol_ratio_val", vol_ratio)
            
            # New Reward Metrics
            self.logger.record("hft/edge_shortfall", info.get("edge_shortfall", 0))
            self.logger.record("hft/vol_ratio", info.get("vol_ratio", 0))
            self.logger.record("hft/trade_cost", info.get("trade_cost", 0))
            
            # Formatted for Console
            self.logger.record("hft/balance", format_number(balance))
            self.logger.record("hft/position", format_number(position))
            self.logger.record("hft/ttl", ttl)
            self.logger.record("hft/drawdown", f"{drawdown:.1%}")
            
            # Accumulate cost (already done above if skipped, but here we do it if not skipped)
            # Wait, if I added the check above, I shouldn't double count.
            # Let's restructure.
            pass
            
        # Always accumulate cost
        if infos:
             self.current_episode_cost += infos[0].get("step_cost", 0)
             
             # Check for episode end
             dones = self.locals.get("dones", [])
             if dones and dones[0]:
                self.logger.record("hft/episode_cost", format_number(self.current_episode_cost))
                self.current_episode_cost = 0.0
                
        return True

def main():
    # Configuration
    DATA_PATH = "data/btc_jpy_1m_v454.csv"
    TOTAL_TIMESTEPS = 300_000  # Main training run
    CHECKPOINT_FREQ = 50_000
    MODEL_DIR = "models/v455_hft_main"
    LOG_DIR = "logs/v455_hft_main"
    SEED = 42
    
    # Initialize Utilities
    seed_manager = SeedManager()
    seed_manager.set_seed(SEED)
    
    ensure_dir(MODEL_DIR)
    ensure_dir(LOG_DIR)
    
    # Notification
    discord_webhook = os.environ.get("DISCORD_WEBHOOK_URL")
    notifier = DiscordNotifier(webhook_url=discord_webhook) if discord_webhook else None
    
    if notifier:
        notifier.send_notification(
            title="HFT Training Started",
            message=f"Training started for {format_number(TOTAL_TIMESTEPS)} steps.",
            color="info"
        )
    
    # Load Data
    logger.info(f"Loading data from {DATA_PATH}...")
    if not os.path.exists(DATA_PATH):
        logger.error(f"Data file not found: {DATA_PATH}")
        return
        
    # Use parse_dates=True to parse the index (column 0) as dates
    df = pd.read_csv(DATA_PATH, parse_dates=True, index_col=0)
    
    # Add Features
    logger.info("Adding HFT features...")
    df = add_hft_features(df)
    
    feature_columns = ["clv", "vol_pressure", "impact_proxy", "vol_regime", "trend_persistence"]
    
    # Create Environment
    def make_env():
        env = FastIntradayEnv(
            df=df,
            feature_columns=feature_columns,
            initial_balance=100_000.0, # Small account (100k JPY)
            max_position=0.01, # 0.01 BTC (~137k JPY) -> ~1.37x Leverage (Safe for bitFlyer)
            max_steps=1000, # Episode length
            prewarm_steps=100,
            max_ttl_steps=60,
            cooldown_steps=5,
            reward_params={
                "alpha": 0.5, # Churn penalty
                "beta": 0.02, # Time holding penalty
                "min_edge_mult": 1.5, # Require 1.5x edge above cost (Optimized)
                "edge_penalty_rate": 1.0, # Penalize trades with insufficient edge
                "vol_floor": 0.002, # Avoid trading below 0.2% ATR/price (Optimized)
                "vol_floor_penalty": 50.0, # Penalize holding in low vol
                "hold_grace": 10, # Grace period before extra time decay
                "hold_ramp": 0.01 # Extra decay per step after grace
            }
        )
        # Seed the environment
        env.reset(seed=SEED)
        env = Monitor(env, LOG_DIR, info_keywords=("edge_shortfall", "vol_ratio", "trade_cost", "balance", "drawdown"))
        return env
    
    # Vectorize and Normalize
    env = DummyVecEnv([make_env])
    env = VecNormalize(env, norm_obs=True, norm_reward=True, clip_obs=10.0)
    
    # Clean up large dataframe to free memory
    del df
    gc.collect()
    logger.info("Garbage collection completed.")
    
    # Initialize Model
    logger.info("Initializing SAC model...")
    model = SAC(
        "MlpPolicy",
        env,
        verbose=1,
        tensorboard_log=LOG_DIR,
        learning_rate=3e-4,
        buffer_size=100_000,
        batch_size=256,
        ent_coef="auto",
        train_freq=1,
        gradient_steps=1,
        learning_starts=1000,
        seed=SEED
    )
    
    # Callbacks
    checkpoint_callback = CheckpointCallback(
        save_freq=CHECKPOINT_FREQ,
        save_path=MODEL_DIR,
        name_prefix="sac_hft"
    )
    metrics_callback = HFTMetricsCallback()
    
    # Train
    logger.info(f"Starting training for {format_number(TOTAL_TIMESTEPS)} timesteps...")
    model.learn(
        total_timesteps=TOTAL_TIMESTEPS,
        callback=[checkpoint_callback, metrics_callback],
        progress_bar=True
    )
    
    # Save Final Model
    final_path = os.path.join(MODEL_DIR, "sac_hft_final")
    model.save(final_path)
    env.save(os.path.join(MODEL_DIR, "vec_normalize.pkl"))
    logger.info(f"Model saved to {final_path}")
    
    if notifier:
        notifier.send_notification(
            title="HFT Training Completed",
            message=f"Training completed successfully. Model saved to {final_path}",
            color="success"
        )

if __name__ == "__main__":
    main()
