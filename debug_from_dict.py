#!/usr/bin/env python
"""Debug from_dict method to see if curriculum_stage is loaded correctly."""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from ztb.utils.v4xx_config_converter import V4XXConfigConverter
from ztb.trading.environment.utils.config import EnvironmentConfig

# Load and convert config
config = json.load(open('config/sac_v444_3_balanced_penalty_scale_200.json'))
converted = V4XXConfigConverter.convert_v444_to_unified(config)

# Get the environment dict
env_dict = converted.get('training', {}).get('environment', {})
print(f"env_dict keys: {list(env_dict.keys())}")
print(f"env_dict['curriculum_stage']: {env_dict.get('curriculum_stage')}")

# Test from_dict method
print("\n" + "="*70)
print("Testing EnvironmentConfig.from_dict()")
print("="*70)

env_cfg = EnvironmentConfig.from_dict(env_dict)
print(f"Result: curriculum_stage = {env_cfg.curriculum_stage}")

# Compare with direct initialization
print("\n" + "="*70)
print("Testing EnvironmentConfig() then setattr")
print("="*70)

env_cfg2 = EnvironmentConfig()
env_cfg2.curriculum_stage = env_dict.get('curriculum_stage')
print(f"Result: curriculum_stage = {env_cfg2.curriculum_stage}")
