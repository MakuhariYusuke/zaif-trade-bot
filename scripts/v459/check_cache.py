#!/usr/bin/env python3
import pandas as pd

df = pd.read_feather("data/btc_jpy_1m_v451.cached.feather")
print(f"Shape: {df.shape}")
print(f"Columns: {list(df.columns)}")
print(f"Timestamp dtype: {df['timestamp'].dtype}")
print(f"First timestamp: {df['timestamp'].iloc[0]}")
print(f"Is datetime64: {pd.api.types.is_datetime64_any_dtype(df['timestamp'])}")
