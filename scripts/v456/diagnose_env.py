#!/usr/bin/env python3
"""
環境診断スクリプト
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from ztb.trading.environment.fast_intraday_env_v456 import FastIntradayEnvV456

print("=" * 70)
print("環境診断: FastIntradayEnvV456")
print("=" * 70)

# データ読み込み
data_path = PROJECT_ROOT / 'data' / 'btc_jpy_1m_v454.csv'
market_data = pd.read_csv(data_path, index_col=0, parse_dates=True)

print(f"\n✓ データロード: {len(market_data)} rows")
print(f"  Columns: {list(market_data.columns)}")

# 特徴量準備
df = market_data.copy()

base_cols = [f'base_{i}' for i in range(30)]
mtf_cols = [f'mtf_{i}' for i in range(27)]
regime_cols = [f'regime_{i}' for i in range(13)]

# ダミーデータ補完
for col_list in [base_cols, mtf_cols, regime_cols]:
    for col in col_list:
        if col not in df.columns:
            df[col] = np.random.randn(len(df))

for col in ['atr', 'impact_proxy']:
    if col not in df.columns:
        df[col] = np.random.rand(len(df)) + 1.0

print(f"\n✓ 特徴量準備完了: {len(df)} rows, {len(df.columns)} cols")

# 環境作成
print(f"\n環境作成中...")
env = FastIntradayEnvV456(
    df=df,
    base_feature_columns=base_cols[:30],
    mtf_feature_columns=mtf_cols[:27],
    regime_feature_columns=regime_cols[:13],
    initial_balance=50000.0,
    max_position=500,
    max_steps=500,
    drawdown_limit=0.3,
    prewarm_steps=100,
    commission_rate=0.001
)

print(f"✓ 環境作成成功")
print(f"  Observation space: {env.observation_space}")
print(f"  Action space: {env.action_space}")

# リセットテスト
print(f"\nリセット中...")
obs = env.reset()
print(f"✓ リセット成功")
print(f"  Initial obs shape: {obs.shape if isinstance(obs, np.ndarray) else type(obs)}")

# 単一ステップテスト
print(f"\n単一ステップテスト...")
for i in range(5):
    action = env.action_space.sample()  # ランダムアクション
    result = env.step(action)
    
    # Gym 0.26+ では5値返却 (obs, reward, terminated, truncated, info)
    if len(result) == 5:
        obs, reward, terminated, truncated, info = result
        done = terminated or truncated
    else:
        obs, reward, done, info = result
    
    print(f"  Step {i+1}: reward={reward:.6f}, done={done}, pos={info.get('position', 0):.3f}")
    
    if done:
        print(f"  ⚠️  エピソード終了 (ステップ {i+1})")
        break

print("\n✓ 診断完了")
env.close()
