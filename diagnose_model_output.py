#!/usr/bin/env python3
"""v456 モデル出力値診断"""

import os
import sys
from pathlib import Path
import numpy as np
from stable_baselines3 import SAC
import pandas as pd

project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

# データ読み込み
df = pd.read_csv('data/btc_jpy_1m_v451.csv', index_col=0, parse_dates=True)
prices = df['close']

# モデル読み込み
model_path = 'models/v456/final/v456_trained_1768486770.zip'
model = SAC.load(model_path)

print("🔍 モデル出力値診断\n")

# 1000ステップ分のアクション値を収集
actions_collected = []

for idx in range(min(1000, len(df))):
    current_price = float(prices.iloc[idx])
    
    # ダミー観測値
    obs = np.zeros(88, dtype=np.float32)
    obs[0] = current_price
    obs[1] = current_price
    
    # 予測
    action, _ = model.predict(obs, deterministic=True)
    action_val = float(action[0]) if isinstance(action, np.ndarray) else float(action)
    
    actions_collected.append(action_val)

actions_arr = np.array(actions_collected)

print(f"📊 アクション値統計 (最初の1000ステップ):")
print(f"  平均: {np.mean(actions_arr):.4f}")
print(f"  中央値: {np.median(actions_arr):.4f}")
print(f"  最小: {np.min(actions_arr):.4f}")
print(f"  最大: {np.max(actions_arr):.4f}")
print(f"  標準偏差: {np.std(actions_arr):.4f}")

print(f"\n📈 分布:")
print(f"  > 0.3 (買い): {np.sum(actions_arr > 0.3)} ({np.sum(actions_arr > 0.3)/len(actions_arr)*100:.1f}%)")
print(f"  -0.3 ~ 0.3 (ホールド): {np.sum((actions_arr >= -0.3) & (actions_arr <= 0.3))} ({np.sum((actions_arr >= -0.3) & (actions_arr <= 0.3))/len(actions_arr)*100:.1f}%)")
print(f"  < -0.3 (売り): {np.sum(actions_arr < -0.3)} ({np.sum(actions_arr < -0.3)/len(actions_arr)*100:.1f}%)")

print(f"\n🔢 最初の20値:")
for i, val in enumerate(actions_arr[:20]):
    action_type = "買い" if val > 0.3 else ("売り" if val < -0.3 else "ホールド")
    print(f"  Step {i+1}: {val:7.4f} ({action_type})")

# モデルのアクション空間を確認
print(f"\n🎯 モデル設定:")
print(f"  Action space: {model.action_space}")
print(f"  Action space bounds: low={model.action_space.low}, high={model.action_space.high}")
