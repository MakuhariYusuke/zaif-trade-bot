#!/usr/bin/env python3
import fnmatch

import pandas as pd

from ztb.features.feature_set_config import FeatureSetConfig

# データ読み込み
df = pd.read_csv("data/btc_jpy_featured_dataset.csv")
print("Data loaded, shape:", df.shape)

# 特徴量設定
fsc = FeatureSetConfig()
fsc.set_feature_set("high_quality")
exclude_patterns = fsc.current_config.get("excluded_features", [])
print("Exclude patterns:", exclude_patterns)

# 数値列を取得
numeric_cols = df.select_dtypes(include=["number"]).columns
print("Numeric columns count:", len(numeric_cols))
print("First 10 numeric columns:", list(numeric_cols[:10]))

# 特徴量フィルタリング
available_features = []
for col in numeric_cols:
    excluded = False
    for pattern in exclude_patterns:
        if fnmatch.fnmatch(col, pattern):
            excluded = True
            break
    if not excluded:
        available_features.append(col)

print("Available features count:", len(available_features))
print("First 10 available features:", available_features[:10])
print("Final 8 features:", available_features[:8])
