#!/usr/bin/env python3
"""
Investigate SB3 PPO internals to understand where to hook our custom components.
"""

import inspect
from stable_baselines3 import PPO
from sb3_contrib import MaskablePPO

print("=" * 80)
print("SB3 PPO Investigation")
print("=" * 80)

# Check PPO location
print(f"\nPPO location: {inspect.getfile(PPO)}")
print(f"MaskablePPO location: {inspect.getfile(MaskablePPO)}")

# Check train method signature
print("\n" + "-" * 80)
print("PPO.train() method signature:")
print("-" * 80)
train_method = getattr(PPO, 'train', None)
if train_method:
    sig = inspect.signature(train_method)
    print(f"Signature: {sig}")
    
    # Get source code
    try:
        source = inspect.getsource(train_method)
        print("\nFirst 50 lines of train() method:")
        lines = source.split('\n')[:50]
        for i, line in enumerate(lines, 1):
            print(f"{i:3d}: {line}")
    except Exception as e:
        print(f"Could not get source: {e}")

# Check for compute_returns_and_advantage
print("\n" + "-" * 80)
print("Looking for advantage computation methods:")
print("-" * 80)

for name in dir(PPO):
    if 'advantage' in name.lower() or 'return' in name.lower():
        print(f"  - {name}")

# Check RolloutBuffer
print("\n" + "-" * 80)
print("RolloutBuffer methods:")
print("-" * 80)

try:
    from stable_baselines3.common.buffers import RolloutBuffer
    print(f"RolloutBuffer location: {inspect.getfile(RolloutBuffer)}")
    
    for name in dir(RolloutBuffer):
        if not name.startswith('_'):
            print(f"  - {name}")
except Exception as e:
    print(f"Error: {e}")

print("\n" + "=" * 80)
print("Investigation complete")
print("=" * 80)
