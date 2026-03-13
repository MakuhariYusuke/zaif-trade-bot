"""
Integration Test for FastIntradayEnv
"""

import os
import sys
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Add project root to path
project_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(project_root))

from ztb.features.hft_proxies import add_hft_features
from ztb.trading.environment.fast_intraday_env import FastIntradayEnv

def main():
    # Load Data
    data_path = "data/btc_jpy_1m_v454.csv"
    if not os.path.exists(data_path):
        print(f"Data file not found: {data_path}")
        # Create dummy data
        print("Creating dummy data...")
        dates = pd.date_range(start="2024-01-01", periods=10000, freq="1min")
        df = pd.DataFrame({
            "open": np.linspace(10000, 11000, 10000) + np.random.randn(10000)*10,
            "high": np.linspace(10010, 11010, 10000) + np.random.randn(10000)*10,
            "low": np.linspace(9990, 10990, 10000) + np.random.randn(10000)*10,
            "close": np.linspace(10000, 11000, 10000) + np.random.randn(10000)*10,
            "volume": np.random.rand(10000) * 100
        }, index=dates)
    else:
        df = pd.read_csv(data_path, parse_dates=["timestamp"], index_col=0)
        
    # Add Features
    print("Adding HFT features...")
    df = add_hft_features(df)
    print(f"Features added. Columns: {df.columns}")
    
    feature_columns = ["clv", "vol_pressure", "impact_proxy", "vol_regime", "trend_persistence"]
    
    # Initialize Env
    print("Initializing FastIntradayEnv...")
    env = FastIntradayEnv(
        df=df,
        feature_columns=feature_columns,
        max_steps=1000,
        prewarm_steps=100,
        max_ttl_steps=30,
        cooldown_steps=5,
        reward_params={"alpha": 0.1, "beta": 0.001}
    )
    
    # Run Episode
    print("Running episode...")
    obs, info = env.reset(seed=42)
    
    rewards = []
    positions = []
    balances = []
    ttls = []
    
    terminated = False
    truncated = False
    
    while not (terminated or truncated):
        # Random Action
        action = env.action_space.sample()
        # Bias towards holding/small moves to simulate realistic agent?
        # Or just random.
        
        obs, reward, terminated, truncated, info = env.step(action)
        
        rewards.append(reward)
        positions.append(info["position"])
        balances.append(info["balance"])
        ttls.append(info["ttl"])
        
    print(f"Episode finished. Total Reward: {sum(rewards):.4f}")
    print(f"Final Balance: {balances[-1]:.2f}")
    print(f"Total Trades: {len([p for i, p in enumerate(positions) if i > 0 and p != positions[i-1]])}")
    
    # Plot
    plt.figure(figsize=(12, 8))
    plt.subplot(3, 1, 1)
    plt.plot(balances, label="Balance")
    plt.legend()
    plt.title("Balance")
    
    plt.subplot(3, 1, 2)
    plt.plot(positions, label="Position")
    plt.plot(np.array(ttls) / 30.0, label="TTL (Norm)", alpha=0.5)
    plt.legend()
    plt.title("Position & TTL")
    
    plt.subplot(3, 1, 3)
    plt.plot(rewards, label="Reward")
    plt.legend()
    plt.title("Reward")
    
    output_path = "scripts/v455/hft_env_test_plot.png"
    plt.savefig(output_path)
    print(f"Plot saved to {output_path}")

if __name__ == "__main__":
    main()
