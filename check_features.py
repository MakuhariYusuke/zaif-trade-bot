#!/usr/bin/env python3
"""
Check available features in dataset
"""

import pandas as pd

# Load dataset
df = pd.read_csv('data/btc_jpy_featured_dataset.csv')
features = [col for col in df.columns if col not in ['timestamp']]

print(f'Available features: {len(features)}')
print('Features:', features)

# Save available features
with open('available_features.txt', 'w') as f:
    for feature in features:
        f.write(f'"{feature}",\n')

print('Saved to available_features.txt')