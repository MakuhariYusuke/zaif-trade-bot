#!/usr/bin/env python3
"""
Backtest script for v452 Optimized Model.
"""

import os
import sys
import json
import logging
from pathlib import Path
import pandas as pd
import numpy as np
from stable_baselines3 import SAC

# Add project root to path
project_root = Path(__file__).resolve().parents[2]
sys.path.append(str(project_root))

from ztb.trading.environment.heavy_env.core import HeavyTradingEnv
from ztb.trading.environment.utils.config import EnvironmentConfig
from ztb.analysis.v444_regime_classifier import V444RegimeClassifier
from ztb.metrics.metrics import calculate_all_metrics

# Configure logging to reduce spam
logging.basicConfig(level=logging.WARNING)
logging.getLogger("ztb").setLevel(logging.WARNING)

# Configuration
MODEL_PATH = "models/sac_v452_optimized_10k.zip"
DATA_PATH = "data/btc_jpy_1m_merged.csv"
THRESHOLDS_PATH = "config/v452/threshold_optimized.json"
RESULTS_DIR = "backtest_results/v452_optimized"

def run_backtest():
    print("Starting backtest for v452 Optimized")
    print(f"Model: {MODEL_PATH}")
    
    if not os.path.exists(MODEL_PATH):
        print(f"Error: Model not found: {MODEL_PATH}")
        return

    # Load optimized thresholds
    with open(THRESHOLDS_PATH, 'r') as f:
        optimized_thresholds = json.load(f)
    print(f"Loaded thresholds: {optimized_thresholds}")

    # Load Data
    print("Loading data...")
    print(f"Using merged data: {DATA_PATH}")
    df = pd.read_csv(DATA_PATH, parse_dates=["timestamp"], index_col="timestamp")
    
    # For speed, use last 20% (which now includes the latest data)
    start_idx = int(len(df) * 0.8)
    df = df.iloc[start_idx:]
    print(f"Data truncated for speed: {len(df)} rows (Last 20%)")
    
    # Create Environment Config
    env_config = EnvironmentConfig(
        feature_set="default",
        use_continuous_actions=True,
        target_feature_count=138,
        correlation_reduction=True,
        adaptive_threshold_mode=True,  # Enable adaptive thresholds for HFT logic
        threshold_volatility_multiplier=1.0,
    )
    
    # Inject optimized thresholds
    env_config.regime_threshold_multipliers = optimized_thresholds
    
    # Create Environment
    print("Creating environment...")
    env = HeavyTradingEnv(df=df, config=env_config)
    
    # Enable Regime Adaptation
    print("Enabling regime adaptation...")
    regime_classifier = V444RegimeClassifier()
    env.enable_market_regime_adaptation(regime_classifier=regime_classifier)
    
    # Load Model
    print("Loading model...")
    model = SAC.load(MODEL_PATH, env=env)
    
    # Run Backtest
    print("Running backtest loop...")
    obs, _ = env.reset()
    done = False
    
    history = []
    
    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        
        # Collect data for analysis
        current_price = env._resolve_price(env.current_step)
        
        # Handle regime enum
        regime_val = info.get("market_regime", "unknown")
        if hasattr(regime_val, "value"):
            regime_val = regime_val.value
        
        step_data = {
            "timestamp": info.get("timestamp", df.index[env.current_step - 1]),
            "portfolio_value": env.portfolio_value,
            "price": current_price,
            "action": action[0] if isinstance(action, (list, np.ndarray)) else action,
            "pnl": env.total_pnl,
            "reward": reward,
            "regime": str(regime_val),
            "action_type": info.get("action_type") # None if missing
        }
        
        # If action_type is not in info, derive it (assuming continuous action [-1, 1])
        if step_data["action_type"] is None:
             act_val = step_data["action"]
             if act_val > 0.33:
                 step_data["action_type"] = "BUY"
             elif act_val < -0.33:
                 step_data["action_type"] = "SELL"
             else:
                 step_data["action_type"] = "HOLD"

        history.append(step_data)
        
        if env.current_step % 1000 == 0:
            print(f"Step {env.current_step}, PnL: {env.total_pnl:.2f}")
            
    print(f"Backtest finished. Final PnL: {env.total_pnl:.2f}")
    
    # Save Results
    os.makedirs(RESULTS_DIR, exist_ok=True)
    results_df = pd.DataFrame(history)
    
    csv_path = os.path.join(RESULTS_DIR, "backtest_results.csv")
    results_df.to_csv(csv_path, index=False)
    print(f"Results saved to {csv_path}")
    
    # Calculate Metrics
    print("\nCalculating performance metrics...")
    # Calculate returns (percentage change of portfolio value)
    results_df["return"] = results_df["portfolio_value"].pct_change().fillna(0)
    
    # Calculate comprehensive metrics
    # period_per_year for 1m data: 365 days * 24 hours * 60 minutes = 525600
    metrics = calculate_all_metrics(results_df["return"], period_per_year=525600)
    
    # Print Metrics
    print("\n=== Performance Metrics ===")
    print(f"Total Return: {metrics['total_return']:.2%}")
    print(f"Sharpe Ratio: {metrics['sharpe_ratio']:.4f}")
    print(f"Sortino Ratio: {metrics['sortino_ratio']:.4f}")
    print(f"Max Drawdown: {metrics['max_drawdown']:.2%}")
    print(f"Win Rate: {metrics['win_rate']:.2%}")
    print(f"Profit Factor: {metrics['profit_factor']:.4f}")
    print(f"Volatility: {metrics['volatility']:.4f}")
    
    # Save summary JSON
    summary = {
        "initial_balance": env.initial_portfolio_value,
        "final_balance": env.portfolio_value,
        "total_pnl": env.total_pnl,
        "total_steps": env.current_step,
        "metrics": metrics
    }
    
    # Handle non-serializable types in metrics (like numpy types)
    def default_converter(o):
        if isinstance(o, np.integer): return int(o)
        if isinstance(o, np.floating): return float(o)
        if isinstance(o, np.ndarray): return o.tolist()
        return str(o)

    with open(os.path.join(RESULTS_DIR, "backtest_results.json"), "w") as f:
        json.dump(summary, f, indent=4, default=default_converter)

if __name__ == "__main__":
    run_backtest()
