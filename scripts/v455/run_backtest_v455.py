#!/usr/bin/env python3
"""
Backtest SAC v455 (MTF + Online Scaler)
Uses the v454 model but runs in the v455 environment context.
"""

import sys
import os
import json
import logging
import warnings
import numpy as np
import pandas as pd
from pathlib import Path
from stable_baselines3 import SAC

# Add project root to path
project_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(project_root))

# Suppress warnings
warnings.filterwarnings("ignore", category=FutureWarning, module="pandas")
warnings.filterwarnings("ignore", category=UserWarning, module="gymnasium")

from ztb.config.unified_config import UnifiedConfig
from ztb.trading.environment.utils.config import EnvironmentConfig
from ztb.trading.environment.heavy_env.core import HeavyTradingEnv
from ztb.trading.environment.constants import continuous_to_discrete_action
from ztb.trading.constants import ACTION_BUY, ACTION_HOLD, ACTION_SELL
from ztb.utils.logging_utils import setup_logging, get_logger
from ztb.utils.training_utils import load_model
from utils.backtest_init_utils import initialize_backtest_components
from ztb.utils.data_utils import load_csv_data

logger = get_logger(__name__)

def run_backtest_v455_custom(model_name, config_path):
    """
    Custom backtest runner for v455 that ensures correct feature count.
    """
    logger.info("🚀 SAC v455 Backtest (MTF + Online Scaler)")
    logger.info(f"🔍 Model: {model_name}")
    logger.info(f"Config: {config_path}")

    try:
        # Load Config
        unified_config = UnifiedConfig.from_file(config_path)
        logger.info("✅ UnifiedConfig loaded successfully")
        
        if hasattr(unified_config, "to_dict"):
            config = unified_config.to_dict()
        else:
            # Fallback
            with open(config_path, "r") as f:
                config = json.load(f)

        # Load Data
        # We use the same data loading logic as simple_backtest_v444
        # Assuming data path is in config or default
        data_path = "data/btc_jpy_1m_v454.csv" # Default for v454
        if not os.path.exists(data_path):
            logger.error(f"Data file not found: {data_path}")
            return None
            
        logger.info(f"Loading data from {data_path}")
        df = load_csv_data(data_path, parse_dates=["timestamp"], index_col=0)
        logger.info(f"✅ Data loaded: {len(df)} rows")

        # Load Model
        model_path = f"models/{model_name}.zip"
        logger.info(f"Loading model from {model_path}")
        model = load_model(model_path, algorithm="SAC")
        logger.info(f"✅ Model loaded: {model_name}")
        
        obs_dim = model.observation_space.shape[0]
        logger.info(f"   Model expects {obs_dim} features")

        # Prepare Environment Config
        env_config = config.get("environment", {})
        
        # Create EnvironmentConfig object
        env_config_obj = EnvironmentConfig.from_dict(env_config)
        
        # CRITICAL: Set target_feature_count to match model
        env_config_obj.target_feature_count = obs_dim
        logger.info(f"✅ Set target_feature_count to {obs_dim}")

        # Create Environment
        logger.info("Creating HeavyTradingEnv...")
        # We pass the raw DF. The environment will generate features (including MTF)
        # and then reduce them to target_feature_count.
        
        env = HeavyTradingEnv(
            df=df,
            config=env_config_obj,
        )
        logger.info("✅ HeavyTradingEnv created successfully")
        logger.info(f"Environment observation space: {env.observation_space}")
        
        if env.observation_space.shape[0] != obs_dim:
            logger.warning(f"⚠️ Environment observation space ({env.observation_space.shape[0]}) does not match model ({obs_dim})!")
            # This might cause a crash, but let's proceed and see.
            # If correlation reduction failed to reach exact count, we might have issues.
        
        # Run Backtest Loop
        logger.info("Starting backtest loop...")
        obs, _ = env.reset()
        
        # Skip warmup
        warmup_steps = 100
        env.current_step = warmup_steps - 1
        
        total_reward = 0
        trades = []
        position = 0
        entry_price = 0
        initial_balance = 10000
        balance = initial_balance
        
        # Track Scaler updates
        scaler_n_start = env.online_scaler.n if hasattr(env, "online_scaler") else 0
        
        # Run for a subset of steps to be quick, or full if needed. 
        # Let's run 2000 steps for verification.
        steps_to_run = 2000
        end_step = min(warmup_steps + steps_to_run, len(df))
        
        for i in range(warmup_steps, end_step):
            action, _ = model.predict(obs, deterministic=True)
            
            obs, reward, terminated, truncated, info = env.step(action)
            
            # Track trades from info
            env_position = info.get("position", 0)
            current_price = df["close"].iloc[i]
            
            if env_position != position:
                if env_position > 0 and position <= 0: # Buy
                    trades.append({"type": "BUY", "price": current_price, "step": i})
                elif env_position < 0 and position >= 0: # Sell
                    trades.append({"type": "SELL", "price": current_price, "step": i})
                elif env_position == 0: # Close
                    trades.append({"type": "CLOSE", "price": current_price, "step": i})
            
            position = env_position
            balance = info.get("balance", balance)
            
            if terminated or truncated:
                break
                
        scaler_n_end = env.online_scaler.n if hasattr(env, "online_scaler") else 0
        
        # Results
        total_return_pct = (balance - initial_balance) / initial_balance * 100
        
        results = {
            "total_return_pct": total_return_pct,
            "final_balance": balance,
            "trades_count": len(trades),
            "scaler_updates": scaler_n_end - scaler_n_start
        }
        
        return results

    except Exception as e:
        logger.error(f"Backtest failed: {e}")
        import traceback
        traceback.print_exc()
        return None

def main():
    config_path = "config/v454/sac_v454_config.json"
    model_name = "sac_v454_inverse_confidence"
    
    results = run_backtest_v455_custom(model_name, config_path)
    
    if results:
        print("\n" + "="*60)
        print("BACKTEST RESULTS")
        print("="*60)
        print(f"Total Return: {results['total_return_pct']:.2f}%")
        print(f"Final Balance: {results['final_balance']:.2f}")
        print(f"Total Trades: {results['trades_count']}")
        print(f"Online Scaler Updates: {results['scaler_updates']}")
        print("="*60)
        
        if results['scaler_updates'] > 0:
            print("✅ Online Scaler is working correctly (updated during backtest).")
        else:
            print("❌ Online Scaler did not update!")
            
        return True
    return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
