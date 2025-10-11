import numpy as np
import json
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from ztb.utils.config import TypedConfig

# Load scaler feature names - use config-based path
config = TypedConfig()
scaler_path = f"{config.get_model_dir()}/scaler.npz"
scaler_data = np.load(scaler_path, allow_pickle=True)
feature_names = scaler_data['feature_names'].tolist()

print(f"Total features in model: {len(feature_names)}\n")
print(f"First 20 features:")
for i, name in enumerate(feature_names[:20]):
    print(f"  {i+1:3d}. {name}")

print(f"\nLast 20 features:")
for i, name in enumerate(feature_names[-20:], start=len(feature_names)-19):
    print(f"  {i:3d}. {name}")

# Save to file for reference
with open('model_features_110.txt', 'w') as f:
    for i, name in enumerate(feature_names, 1):
        f.write(f"{i:3d}. {name}\n")

print(f"\n✅ Full feature list saved to: model_features_110.txt")
