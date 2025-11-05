#!/usr/bin/env python
"""Debug action_bonuses loading from config."""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from ztb.utils.v4xx_config_converter import V4XXConfigConverter
from ztb.trading.environment.utils.config import EnvironmentConfig

# Load config
config = json.load(open('config/sac_v444_3_balanced_penalty_scale_200.json'))
converted = V4XXConfigConverter.convert_v444_to_unified(config)
env_dict = converted.get('training', {}).get('environment', {})

print("="*70)
print("Action Bonuses Loading Test")
print("="*70)

print("\n1. env_dict keys:")
for k in sorted(env_dict.keys()):
    print(f"   {k}: {env_dict.get(k)}")

print("\n2. Creating EnvironmentConfig from dict...")
env_cfg = EnvironmentConfig.from_dict(env_dict)

print(f"\n3. env_cfg.action_bonuses = {env_cfg.action_bonuses}")
print(f"   Type: {type(env_cfg.action_bonuses)}")

# Check if buy_action_bonus is in the dict
print(f"\n4. Looking for individual bonus keys:")
print(f"   env_cfg.action_bonuses.get('buy_action_bonus') = {env_cfg.action_bonuses.get('buy_action_bonus')}")
print(f"   env_cfg.action_bonuses.get('sell_action_bonus') = {env_cfg.action_bonuses.get('sell_action_bonus')}")
print(f"   env_cfg.action_bonuses.get('hold_action_bonus') = {env_cfg.action_bonuses.get('hold_action_bonus')}")

# Check if values are at root level
print(f"\n5. Checking root level:")
print(f"   env_dict.get('buy_action_bonus') = {env_dict.get('buy_action_bonus')}")
print(f"   env_dict.get('sell_action_bonus') = {env_dict.get('sell_action_bonus')}")
print(f"   env_dict.get('hold_action_bonus') = {env_dict.get('hold_action_bonus')}")
