#!/usr/bin/env python3
"""
Train SAC v500 Equalized with balanced action ratios.
"""

import json
import os
import sys
from pathlib import Path

# Ensure we're using the correct Python environment
if sys.version_info < (3, 11):
    print("Error: Python 3.11+ required")
    sys.exit(1)

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from ztb.training.unified_trainer.main import main as train_main


def main():
    """Train SAC v500 model with equalized action bonuses."""
    config_path = "config/sac_v503_fine_tune_3_config.json"

    if not os.path.exists(config_path):
        print(f"Config file not found: {config_path}")
        return

    # Load and validate config
    with open(config_path, 'r', encoding='utf-8') as f:
        config = json.load(f)

    print("Training SAC v500 Equalized:")
    print("📈 Profit Bonuses:")
    for k, v in config['reward_settings']['profit_bonuses'].items():
        print(f"   {k}: {v}")
    print("🎯 Action Bonuses:")
    for k, v in config['reward_settings']['action_bonuses'].items():
        print(f"   {k}: {v}")
    print("⚖️  Behavior Penalties:")
    for k, v in config['reward_settings']['behavior_penalties'].items():
        print(f"   {k}: {v}")
    print("⚠️  Risk Penalties:")
    for k, v in config['reward_settings']['risk_penalties'].items():
        print(f"   {k}: {v}")

    print("Starting training with unified trainer...")
    # Use unified trainer
    sys.argv = ['train_sac_v500_equalized.py', '--config', config_path]
    train_main()


if __name__ == "__main__":
    main()