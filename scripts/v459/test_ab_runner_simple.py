#!/usr/bin/env python3
"""
Simple test for AB runner
"""

import sys
from pathlib import Path

project_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(project_root))

print("✅ Imports successful")
print(f"Project root: {project_root}")

from ztb.training.unified_trainer.trainer import UnifiedTrainer
from ztb.training.reward_config_schema import load_reward_config

print("✅ Module imports successful")

# Test reward config loading
reward_config_path = project_root / "configs/rewards/stage1_basic.yaml"
print(f"Loading reward config from: {reward_config_path}")

reward_settings = load_reward_config(reward_config_path)
print(f"✅ Reward config loaded: {type(reward_settings)}")

# Test config generation
BASE_CONFIG = {
    "training": {
        "algorithm": "SAC",
        "total_timesteps": 5000,
        "data_config": {
            "data_path": str(project_root / "data" / "btc_jpy_1m_v451.csv"),
            "window_size": 60
        },
        "environment": {
            "use_continuous_actions": True,
            "action_space_type": "continuous",
        },
        "walk_forward": {
            "enabled": True,
            "n_splits": 4
        }
    }
}

config = BASE_CONFIG.copy()
config["seed"] = 42
config["training"]["reward_settings"] = reward_settings
config["experiment_name"] = "test_exp"
config["training"]["model_name"] = "test_model"
config["training"]["output_dir"] = str(project_root / "results/test_ab")

print("✅ Config generation successful")
print(f"Config keys: {list(config.keys())}")
print(f"Training keys: {list(config['training'].keys())}")

# Test trainer creation
try:
    trainer = UnifiedTrainer(config)
    print("✅ Trainer created successfully")
except Exception as e:
    print(f"❌ Trainer creation failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("\n🎉 All basic tests passed!")
