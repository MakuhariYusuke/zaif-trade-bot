#!/usr/bin/env python3
import json

config_path = "config/sac_v444_advanced_regime_adaptation_config.json"
with open(config_path, "r", encoding="utf-8") as f:
    config = json.load(f)
print("Config loaded successfully")
print("reward_settings keys:", list(config.get("reward_settings", {}).keys()))
print(
    "behavior_optimization in reward_settings:",
    "behavior_optimization" in config.get("reward_settings", {}),
)
