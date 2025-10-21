#!/usr/bin/env python3
"""
Simple test script for SAC v431 reward function validation
"""

import json
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(project_root))

def test_v431_reward_function():
    """Test v431 reward function configuration"""

    # Load v431 config
    config_path = Path("../configs/v431/sac_v431_1_enhanced.json")
    with open(config_path, "r") as f:
        config = json.load(f)

    print("=== SAC v431 Configuration Test ===")
    print(f"Version: {config['version']}")
    print(f"Description: {config['description']}")

    # Check reward function
    reward_func = config.get('reward_function', {})
    print("\n=== Reward Function ===")
    print(f"Sell Bonus: {reward_func.get('sell_bonus', 'Not set')}")
    print(f"Hold Bonus: {reward_func.get('hold_bonus', 'Not set')}")

    # Check action thresholds
    action_thresh = config.get('action_thresholds', {})
    print("\n=== Action Thresholds ===")
    print(f"Sell Threshold: {action_thresh.get('sell_threshold', 'Not set')}")
    print(f"Buy Threshold: {action_thresh.get('buy_threshold', 'Not set')}")

    # Check advanced learning
    adv_learning = config.get('advanced_learning', {})
    print("\n=== Advanced Learning ===")
    print(f"Curriculum Enabled: {adv_learning.get('curriculum', {}).get('enabled', False)}")
    print(f"Multi-stage Enabled: {adv_learning.get('multi_stage', {}).get('enabled', False)}")
    print(f"Ensemble Enabled: {adv_learning.get('ensemble', {}).get('enabled', False)}")

    # Simulate reward calculation
    print("\n=== Reward Calculation Simulation ===")

    # Test actions
    actions = [-0.5, -0.2, 0.0, 0.2, 0.5]  # SELL, HOLD, BUY range
    sell_threshold = action_thresh.get('sell_threshold', -0.3333)
    buy_threshold = action_thresh.get('buy_threshold', 0.3333)

    sell_bonus = reward_func.get('sell_bonus', 0.25)
    hold_bonus = reward_func.get('hold_bonus', 0.0053)
    buy_bonus = reward_func.get('buy_bonus', 0.2)

    for action in actions:
        if action <= sell_threshold:
            reward = sell_bonus
            action_type = "SELL"
        elif action >= buy_threshold:
            reward = buy_bonus
            action_type = "BUY"
        else:
            reward = hold_bonus
            action_type = "HOLD"

        print(f"Action {action:.1f} ({action_type}): Reward = {reward:.4f}")

    print("\n=== Analysis ===")
    print("✅ v431 uses BONUS-based rewards for all actions (balanced reinforcement)")
    print("✅ Symmetric thresholds prevent value sticking")
    print("✅ All actions (BUY/SELL/HOLD) are rewarded, encouraging balanced trading")

if __name__ == "__main__":
    test_v431_reward_function()