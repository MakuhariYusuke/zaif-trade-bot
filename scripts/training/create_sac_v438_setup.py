#!/usr/bin/env python3
"""
SAC v438 - Enhanced Bear Market Performance Setup

Creates configuration and scripts for SAC v438 with improved bear market handling.
"""

import json
from pathlib import Path


def create_v438_reward_config():
    """Create enhanced reward config for better bear market performance."""

    config = {
        "reward_function": {
            # Improved asymmetric scaling for bear markets
            "long_position_reward_multiplier": 1.3,  # Reduced from 1.5
            "short_position_reward_multiplier": 1.1,  # Increased from 0.7 (57% boost)
            "long_position_penalty_multiplier": 0.9,  # Increased from 0.8
            "short_position_penalty_multiplier": 0.95,  # Decreased from 1.2 (21% reduction)
            # Bear market incentives
            "bear_market_bonus": 0.1,
            "profit_bonus_multiplier": 1.2,
            "loss_penalty_multiplier": 0.8,
        }
    }

    return config


def save_configs():
    """Save v438 configurations."""

    config_dir = Path("config")
    config_dir.mkdir(exist_ok=True)

    # Save reward config
    reward_config = create_v438_reward_config()
    reward_path = config_dir / "sac_v438_reward_config.json"

    with open(reward_path, "w", encoding="utf-8") as f:
        json.dump(reward_config, f, indent=2, ensure_ascii=False)

    print(f"✅ Saved v438 reward config: {reward_path}")
    return reward_path


def main():
    print("🐻 SAC v438 - Bear Market Enhancement Setup")
    print("=" * 50)

    # Save configurations
    config_path = save_configs()

    print("\n✅ Configuration created!")
    print(f"📁 Config: {config_path}")
    print("\n🎯 Key improvements:")
    print("• Short position rewards: 0.7 → 1.1 (+57%)")
    print("• Short position penalties: 1.2 → 0.95 (-21%)")
    print("• Added bear market bonus incentives")

    print("\n📋 Next: Update reward_calculator.py with these values")


if __name__ == "__main__":
    main()
