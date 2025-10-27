#!/usr/bin/env python3
"""
Check signal distribution in training data
"""

import numpy as np
import pandas as pd

from ztb.trading.strategies import ActionSignalGuide, SignalType

# Load training data directly from CSV
data_path = "data/btc_jpy_featured_dataset.csv"
try:
    df = pd.read_csv(data_path)
    print(f"Loaded {len(df)} data samples from {data_path}")
except FileNotFoundError:
    print(f"Data file not found: {data_path}")
    exit(1)

# Check signal distribution
guide = ActionSignalGuide()
# Use standard feature names that should be in the dataset
feature_names = ["close", "rsi", "macd", "bb_upper", "bb_lower"]
guide.set_feature_names(feature_names)

sell_signal_counts = 0
buy_signal_counts = 0
total_samples = min(1000, len(df))  # Check up to 1000 samples

print(f"Checking signal distribution in first {total_samples} samples...")

for i in range(total_samples):
    row = df.iloc[i]

    # Extract relevant features
    features = []
    for feat in feature_names:
        if feat in row:
            features.append(row[feat])
        else:
            # Use default values if feature not found
            if feat == "close":
                features.append(100.0)
            elif feat == "rsi":
                features.append(50.0)
            elif feat == "macd":
                features.append(0.0)
            elif feat == "bb_upper":
                features.append(105.0)
            elif feat == "bb_lower":
                features.append(95.0)

    test_obs = np.array(features)

    buy_signals = guide._evaluate_signals_for_action(test_obs, SignalType.BUY)
    sell_signals = guide._evaluate_signals_for_action(test_obs, SignalType.SELL)

    if sell_signals > 0.3:  # Lower threshold for detection
        sell_signal_counts += 1
    if buy_signals > 0.3:
        buy_signal_counts += 1

print(
    f"BUY signals > 0.3: {buy_signal_counts}/{total_samples} samples ({buy_signal_counts/total_samples*100:.1f}%)"
)
print(
    f"SELL signals > 0.3: {sell_signal_counts}/{total_samples} samples ({sell_signal_counts/total_samples*100:.1f}%)"
)

# Show some example signal strengths
print("\nExample signal strengths (first 5 samples):")
for i in range(min(5, len(df))):
    row = df.iloc[i]
    features = []
    for feat in feature_names:
        if feat in row:
            features.append(row[feat])
        else:
            features.append(50.0)  # default

    test_obs = np.array(features)
    buy_signals = guide._evaluate_signals_for_action(test_obs, SignalType.BUY)
    sell_signals = guide._evaluate_signals_for_action(test_obs, SignalType.SELL)

    print(f"Sample {i}: BUY={buy_signals:.3f}, SELL={sell_signals:.3f}")
