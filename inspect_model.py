#!/usr/bin/env python3
"""モデル内部検査"""

import os
import sys
from pathlib import Path
from zipfile import ZipFile
import json
from stable_baselines3 import SAC

model_path = 'models/v456/final/v456_trained_1768486770.zip'

# ZIPの内容を確認
print(f"📦 {model_path} の内容:\n")
with ZipFile(model_path, 'r') as z:
    print("Files in archive:")
    for name in z.namelist():
        info = z.getinfo(name)
        print(f"  {name} ({info.file_size} bytes)")
    
    # system_info.json を読む
    try:
        with z.open('system_info.json') as f:
            info = json.load(f)
            print(f"\n📋 System Info:")
            for k, v in info.items():
                if isinstance(v, dict):
                    print(f"  {k}:")
                    for k2, v2 in v.items():
                        print(f"    {k2}: {v2}")
                else:
                    print(f"  {k}: {v}")
    except Exception as e:
        print(f"  system_info.json 読み込み失敗: {e}")

# モデルをロードして詳細を確認
print(f"\n🔍 モデル詳細:")
try:
    model = SAC.load(model_path)
    print(f"  Actor network: {model.actor}")
    print(f"  Critic network: {model.critic}")
    print(f"  Learning rate: {model.learning_rate}")
    print(f"  Gamma: {model.gamma}")
    print(f"  Tau: {model.tau}")
except Exception as e:
    print(f"  ロード失敗: {e}")
    import traceback
    traceback.print_exc()
