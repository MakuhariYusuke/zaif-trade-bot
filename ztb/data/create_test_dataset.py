"""
Test dataset creation for SELL mitigation.

Creates a simple synthetic dataset with proper action labels for testing.
"""

from typing import cast

import numpy as np
import pandas as pd

# Set random seed
np.random.seed(42)

# Create synthetic data
n_samples = 1000

data = {
    "close": np.random.randn(n_samples).cumsum() + 100,
    "volume": np.random.randint(100, 1000, n_samples),
    "rsi": np.random.uniform(20, 80, n_samples),
    "sma_short": np.random.randn(n_samples).cumsum() + 100,
    "sma_long": np.random.randn(n_samples).cumsum() + 100,
    "return_1": np.random.randn(n_samples) * 0.01,
    "return_3": np.random.randn(n_samples) * 0.02,
    "momentum_5": np.random.randn(n_samples) * 0.03,
}

df = pd.DataFrame(data)

# Create action labels with intentional imbalance (typical problem)
# HOLD: 70%, BUY: 20%, SELL: 10%
actions = np.random.choice([0, 1, 2], size=n_samples, p=[0.7, 0.2, 0.1])
df["action"] = actions

# Save
output_path = "test_synthetic_dataset.csv"
df.to_csv(output_path, index=False)

print(f"Created synthetic dataset: {output_path}")
print(f"Shape: {df.shape}")
print("\nAction distribution:")
action_names = {0: "HOLD", 1: "BUY", 2: "SELL"}
for action, count in df["action"].value_counts().sort_index().items():
    action = cast(int, action)
    name = action_names[action]
    pct = count / len(df) * 100
    print(f"  {name}: {count} ({pct:.1f}%)")
