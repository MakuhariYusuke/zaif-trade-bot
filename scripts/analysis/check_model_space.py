#!/usr/bin/env python3
"""
Check model observation space
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from stable_baselines3 import SAC


def main():
    model_path = "checkpoints/sac_v435_test_1000_steps.zip"

    try:
        model = SAC.load(model_path)
        print(f"Model observation space: {model.observation_space}")
        print(f"Observation shape: {model.observation_space.shape}")
        print(f"Observation space type: {type(model.observation_space)}")
    except Exception as e:
        print(f"Error loading model: {e}")


if __name__ == "__main__":
    main()
