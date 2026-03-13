#!/usr/bin/env python3
"""SafeIntradayEnvWrapper テスト"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from ztb.trading.environment.fast_intraday_env_v456 import FastIntradayEnvV456
from scripts.v456.train_mlp_v456_fixed import SafeIntradayEnvWrapper

# データ読み込み
data_path = Path(__file__).parent.parent.parent / 'data' / 'btc_jpy_1m_v454.csv'
market_data = pd.read_csv(data_path, index_col=0, parse_dates=True)

# 特徴量準備
df = market_data.copy()
for i in range(30):
    if f'base_{i}' not in df.columns:
        df[f'base_{i}'] = np.random.randn(len(df))
for i in range(27):
    if f'mtf_{i}' not in df.columns:
        df[f'mtf_{i}'] = np.random.randn(len(df))
for i in range(13):
    if f'regime_{i}' not in df.columns:
        df[f'regime_{i}'] = np.random.randn(len(df))

if 'atr' not in df.columns:
    df['atr'] = np.random.rand(len(df)) + 1.0
if 'impact_proxy' not in df.columns:
    df['impact_proxy'] = np.random.rand(len(df)) + 1.0

print("="*70)
print("SafeIntradayEnvWrapper Test")
print("="*70)
print()

# 環境作成
print("Creating base environment...")
base_env = FastIntradayEnvV456(
    df=df,
    base_feature_columns=[f'base_{i}' for i in range(30)],
    mtf_feature_columns=[f'mtf_{i}' for i in range(27)],
    regime_feature_columns=[f'regime_{i}' for i in range(13)],
    initial_balance=100000,
    max_position=0.01,  # 小数BTC単位
    max_steps=500,
    drawdown_limit=0.3,
)

print("Creating wrapper environment...")
env = SafeIntradayEnvWrapper(
    base_env=base_env,
    warmup_steps=10,
    initial_drawdown_limit=0.5,
    final_drawdown_limit=0.3,
)

print("✓ Environment created successfully")
print()

# リセット
print("Resetting environment...")
obs, info = env.reset()
print(f"✓ Reset successful: obs shape={obs.shape}")
print()

# ステップテスト
print("Step Test (20 steps):")
print("-"*70)

episode_done = False
for i in range(20):
    action = env.action_space.sample()
    obs, reward, done, truncated, info = env.step(action)
    
    balance = info.get('balance', 0)
    warmup_status = "WARMUP" if env.warmup_counter < env.warmup_steps else "NORMAL"
    
    print(f"Step {i+1:2d}: R={reward:7.4f} | Balance={balance:10.2f} | {warmup_status:6s} | Done={done}")
    
    if done:
        episode_done = True
        print(f"         🔴 Episode terminated at step {i+1}")
        break

print("-"*70)

if not episode_done:
    print("✓ SUCCESS: 20 steps completed without early termination!")
else:
    print("⚠️  WARNING: Episode terminated early")

env.close()
print()
print("Test completed")
