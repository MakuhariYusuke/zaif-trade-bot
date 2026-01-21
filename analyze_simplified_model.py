#!/usr/bin/env python3
"""
訓練されたモデルのアクション分析
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import numpy as np
import pandas as pd
from stable_baselines3 import SAC

from ztb.features.base_features_v456 import calculate_base_features
from ztb.trading.environment.utils.fast_intraday_env_v456_utils import (
    create_fast_intraday_env_v456,
)

# データ読み込み
df = pd.read_csv('data/btc_jpy_1m_v451.csv')
print(f"Loaded {len(df):,} bars")

# 特徴量の前計算
df = calculate_base_features(df, copy=False)

# 環境作成
env = create_fast_intraday_env_v456(
    df=df,
    env_config={
        "reward_settings": {
            "alpha": 0.0,
            "beta": 0.0,
            "gamma": 0.0,
        }
    },
)
if env is None:
    print("❌ Failed to create environment")
    sys.exit(1)
del df

print(f"Env observation space: {env.observation_space.shape}")
print(f"Env action space: {env.action_space}")

# モデル読み込み
model_path = "models/v456/final/v456_simplified_1768526891.zip"
try:
    model = SAC.load(model_path, env=env, device="cpu")
    print(f"✓ Model loaded: {model_path}")
except Exception as e:
    print(f"❌ Failed to load: {e}")
    sys.exit(1)

# アクション分析
n_steps = 1000
actions_list = []

obs, _ = env.reset()
for i in range(n_steps):
    action, _ = model.predict(obs, deterministic=True)
    actions_list.append(action.copy())
    obs, reward, done, truncated, info = env.step(action)
    
    if done or truncated:
        obs, _ = env.reset()

actions = np.array(actions_list)

print("\n" + "="*70)
print("Action Distribution Analysis")
print("="*70)
print(f"Total steps: {len(actions)}")
print(f"\naction[0] (target_position): [-1, 1]")
print(f"  Mean: {actions[:, 0].mean():.4f}")
print(f"  Std: {actions[:, 0].std():.4f}")
print(f"  Min: {actions[:, 0].min():.4f}")
print(f"  Max: {actions[:, 0].max():.4f}")

# Buy/Sell 分類
buy_mask = actions[:, 0] > 0.0
sell_mask = actions[:, 0] < 0.0
neutral_mask = np.abs(actions[:, 0]) < 0.001

buy_count = buy_mask.sum()
sell_count = sell_mask.sum()
neutral_count = neutral_mask.sum()

print(f"\nPosition distribution:")
print(f"  Buy (> 0): {buy_count} ({100*buy_count/len(actions):.1f}%)")
print(f"  Sell (< 0): {sell_count} ({100*sell_count/len(actions):.1f}%)")
print(f"  Neutral: {neutral_count} ({100*neutral_count/len(actions):.1f}%)")

if buy_count > 0:
    print(f"\nBuy actions stats:")
    buy_actions = actions[buy_mask, 0]
    print(f"  Mean: {buy_actions.mean():.4f}")
    print(f"  Std: {buy_actions.std():.4f}")

if sell_count > 0:
    print(f"\nSell actions stats:")
    sell_actions = actions[sell_mask, 0]
    print(f"  Mean: {sell_actions.mean():.4f}")
    print(f"  Std: {sell_actions.std():.4f}")
