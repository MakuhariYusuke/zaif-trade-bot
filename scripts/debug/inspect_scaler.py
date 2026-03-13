import os

import numpy as np

scaler_path = "models/scaler.npz"

if os.path.exists(scaler_path):
    data = np.load(scaler_path)
    print(f"Keys in {scaler_path}: {list(data.keys())}")
    if "mean" in data:
        print(f"Mean shape: {data['mean'].shape}")
        print(f"Mean sample: {data['mean'][:5]}")
    if "scale" in data:
        print(f"Scale shape: {data['scale'].shape}")
        print(f"Scale sample: {data['scale'][:5]}")
else:
    print(f"File not found: {scaler_path}")
