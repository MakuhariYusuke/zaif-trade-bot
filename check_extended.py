#!/usr/bin/env python3
"""
Check extended dataset features
"""

import pandas as pd

# Load extended dataset
df = pd.read_csv('data/btc_jpy_extended_dataset.csv')
features = [col for col in df.columns if col not in ['timestamp']]

print(f'Extended dataset features: {len(features)}')
print('First 20 features:', features[:20])
print('Total features:', len(features))