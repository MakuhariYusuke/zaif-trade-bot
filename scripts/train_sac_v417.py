#!/usr/bin/env python3
"""
Train SAC v417 with comprehensive reward parameters.
"""

import json
import os
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from ztb.training.unified_trainer.main import main as train_main


def main():
    """Train SAC v417 model with comprehensive reward parameters."""
    config_path = "config/sac_v417_comprehensive_trading_config.json"

    if not os.path.exists(config_path):
        print(f"Config file not found: {config_path}")
        return

    # Load and validate config
    with open(config_path, 'r') as f:
        config = json.load(f)

    print("Training SAC v417 with comprehensive reward parameters:")
    print(f"- win_rate_bonus: {config['reward_settings']['win_rate_bonus']}")
    print(f"- momentum_bonus: {config['reward_settings']['momentum_bonus']}")
    print(f"- volatility_penalty: {config['reward_settings']['volatility_penalty']}")
    print(f"- action_frequency_penalty: {config['reward_settings']['action_frequency_penalty']}")
    print(f"- diversity_bonus: {config['reward_settings']['diversity_bonus']}")

    # Use unified trainer
    sys.argv = ['train_sac_v417.py', '--config', config_path]
    train_main()


if __name__ == "__main__":
    main()