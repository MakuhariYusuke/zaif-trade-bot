import sys
from pathlib import Path

import numpy as np

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from ztb.utils.config import TypedConfig

# Load scaler - use config-based path
config = TypedConfig()
scaler_path = f"{config.get_model_dir()}/scaler.npz"
scaler_data = np.load(scaler_path)
print(f"Scaler arrays: {list(scaler_data.keys())}")

if "mean" in scaler_data:
    print(f"Mean shape: {scaler_data['mean'].shape}")
    print(f"Feature count from scaler: {len(scaler_data['mean'])}")

if "std" in scaler_data:
    print(f"Std shape: {scaler_data['std'].shape}")
