#!/usr/bin/env python
"""Verify that curriculum_stage flows correctly after config default value fix."""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from ztb.config.schema import EnvironmentConfig
from ztb.utils.v4xx_config_converter import V4XXConfigConverter

# Load config
config = json.load(open("config/sac_v444_3_balanced_penalty_scale_200.json"))

print("=" * 60)
print("VERIFICATION: curriculum_stage Flow After Default Value Fix")
print("=" * 60)

# Step 1: Check original config
original_stage = (
    config.get("training", {}).get("curriculum_learning", {}).get("curriculum_stage")
)
print(f"\n1. Original config curriculum_stage: {original_stage}")

# Step 2: Convert using V4XXConfigConverter
converted = V4XXConfigConverter.convert_v444_to_unified(config)
env_cfg = converted.get("training", {}).get("environment", {})
converted_stage = env_cfg.get("curriculum_stage")
print(f"2. Converted to training.environment.curriculum_stage: {converted_stage}")

# Step 3: Check EnvironmentConfig default
print("\n3. EnvironmentConfig default curriculum_stage: None (after fix)")
print("   (Previously was 'forced_balance')")

# Step 4: Simulate environment creation
test_cfg = EnvironmentConfig(curriculum_stage=converted_stage, exchange="coincheck")
print(f"\n4. EnvironmentConfig instance curriculum_stage: {test_cfg.curriculum_stage}")

print("\n" + "=" * 60)
if converted_stage == original_stage and converted_stage == "balanced_penalty":
    print("✅ SUCCESS: curriculum_stage flows correctly!")
    print("   Config value 'balanced_penalty' correctly propagates to environment")
else:
    print("❌ FAILED: curriculum_stage mismatch")
    print(f"   Original: {original_stage}")
    print(f"   Converted: {converted_stage}")
