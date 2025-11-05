#!/usr/bin/env python
"""Debug script to check curriculum_stage value at runtime."""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from ztb.config.loader import ConfigLoader
from ztb.utils.v4xx_config_converter import V4XXConfigConverter
from ztb.trading.environment.utils.config import EnvironmentConfig

print("\n" + "=" * 70)
print("DEBUG: Curriculum Stage Value at Each Stage of Initialization")
print("=" * 70)

# Step 1: Load config
config = json.load(open('config/sac_v444_3_balanced_penalty_scale_200.json'))
stage_1 = config.get('training', {}).get('curriculum_learning', {}).get('curriculum_stage')
print(f"\n[1] Raw config: curriculum_stage = {stage_1}")

# Step 2: Convert
converted = V4XXConfigConverter.convert_v444_to_unified(config)
stage_2 = converted.get('training', {}).get('environment', {}).get('curriculum_stage')
print(f"[2] After V4XXConfigConverter: curriculum_stage = {stage_2}")

# Step 3: Validate with ConfigLoader
try:
    validated = ConfigLoader.validate_config(converted)
    stage_3 = validated.get('training', {}).get('environment', {}).get('curriculum_stage')
    print(f"[3] After ConfigLoader.validate_config(): curriculum_stage = {stage_3}")
except Exception as e:
    print(f"[3] ConfigLoader error: {e}")
    stage_3 = None

# Step 4: Try creating EnvironmentConfig directly
try:
    env_cfg = EnvironmentConfig(
        curriculum_stage=stage_2,
        exchange='coincheck'
    )
    stage_4 = env_cfg.curriculum_stage
    print(f"[4] Direct EnvironmentConfig instantiation: curriculum_stage = {stage_4}")
except Exception as e:
    print(f"[4] EnvironmentConfig error: {e}")
    stage_4 = None

# Step 5: Try from dict with **kwargs (how it's likely used in actual code)
try:
    env_dict = converted.get('training', {}).get('environment', {})
    
    # Filter to only include valid EnvironmentConfig fields (this is the fix)
    from dataclasses import fields as dataclass_fields
    valid_env_keys = {f.name for f in dataclass_fields(EnvironmentConfig)}
    filtered_env_dict = {k: v for k, v in env_dict.items() if k in valid_env_keys}
    
    env_cfg2 = EnvironmentConfig(**filtered_env_dict)
    stage_5 = env_cfg2.curriculum_stage
    print(f"[5] EnvironmentConfig(**filtered_env_dict): curriculum_stage = {stage_5}")
except Exception as e:
    print(f"[5] EnvironmentConfig error: {e}")
    stage_5 = None

print("\n" + "=" * 70)
print("SUMMARY")
print("=" * 70)
print(f"Raw config:          {stage_1}")
print(f"After conversion:    {stage_2}")
print(f"After validation:    {stage_3}")
print(f"Direct init:         {stage_4}")
print(f"From dict unpacking: {stage_5}")

if stage_5 == 'balanced_penalty':
    print("\n✅ curriculum_stage correctly set to 'balanced_penalty'")
elif stage_5 is None:
    print("\n⚠️  curriculum_stage is None (default)")
else:
    print(f"\n❌ curriculum_stage is '{stage_5}' (unexpected)")
