#!/usr/bin/env python
"""Debug: Check what keys are in the environment config dict."""

import json
from ztb.utils.v4xx_config_converter import V4XXConfigConverter

config = json.load(open('config/sac_v444_3_balanced_penalty_scale_200.json'))
converted = V4XXConfigConverter.convert_v444_to_unified(config)
env_dict = converted.get('training', {}).get('environment', {})

print("\n" + "=" * 70)
print("Environment Config Dict Keys")
print("=" * 70)

for key in sorted(env_dict.keys()):
    print(f"  {key}: {type(env_dict[key]).__name__}")

print("\n" + "=" * 70)
print("EnvironmentConfig valid fields:")
print("=" * 70)

from dataclasses import fields
from ztb.trading.environment.utils.config import EnvironmentConfig

for field in fields(EnvironmentConfig):
    print(f"  {field.name}: {field.type}")
