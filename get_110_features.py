#!/usr/bin/env python3
"""
Get 110-dimensional feature set
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from ztb.features.curated_features import CURATED_FEATURES, FEATURES_TO_REMOVE

# Combine features and remove duplicates
all_features = list(set(CURATED_FEATURES + FEATURES_TO_REMOVE))
all_features.sort()  # Sort for consistency
print(f"Total unique features: {len(all_features)}")
print("First 20 features:", all_features[:20])
print("Last 20 features:", all_features[-20:])

# Save to file
with open("110_features.txt", "w") as f:
    for feature in all_features:
        f.write(f'"{feature}",\n')

print("Saved to 110_features.txt")