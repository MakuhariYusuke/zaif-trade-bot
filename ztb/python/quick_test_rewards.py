#!/usr/bin/env python3
"""
Quick test script to compare reward function improvements (v378, v379, v380)

Tests three reward configurations with short training runs:
- v378: Scale-adjusted (4x HOLD penalty, 3x profit, 3x trading bonus)
- v379: Dynamic (v378 + market-adaptive scaling)
- v380: Aggressive (10x HOLD penalty, 5x profit, 6x trading bonus)

Usage:
    python quick_test_rewards.py
"""

import json
import subprocess
import sys
from typing import Dict

# Test configurations
CONFIGS = [
    {
        "name": "v378_scale_adjusted",
        "path": "configs/training/ppo_reward_v378_scale.json",
        "description": "Scale-adjusted rewards (Medium risk)",
        "expected_hold_rate": "45-55%",
    },
    {
        "name": "v379_dynamic",
        "path": "configs/training/ppo_reward_v379_dynamic.json",
        "description": "Dynamic market-adaptive rewards (Medium-High risk)",
        "expected_hold_rate": "40-50%",
    },
    {
        "name": "v380_aggressive",
        "path": "configs/training/ppo_reward_v380_aggressive.json",
        "description": "Aggressive anti-HOLD rewards (High risk)",
        "expected_hold_rate": "30-40%",
    },
]

# Quick test settings (override total_timesteps for fast validation)
QUICK_TEST_TIMESTEPS = 2000
QUICK_TEST_EVAL_FREQ = 500


def modify_config_for_quick_test(config_path: str) -> Dict:
    """Load config and modify for quick testing."""
    with open(config_path, "r", encoding="utf-8") as f:
        config = json.load(f)

    # Override training settings for quick test
    config["training"]["total_timesteps"] = QUICK_TEST_TIMESTEPS
    config["training"]["eval_freq"] = QUICK_TEST_EVAL_FREQ
    config["training"]["checkpoint_interval"] = 1000

    # Add quick test marker
    config["session_id"] = config["session_id"] + "_quicktest"

    return config


def run_quick_test(config_info: Dict) -> bool:
    """Run a quick test with the given config."""
    print(f"\n{'='*80}")
    print(f"Testing: {config_info['name']}")
    print(f"Description: {config_info['description']}")
    print(f"Expected HOLD rate: {config_info['expected_hold_rate']}")
    print(f"Config: {config_info['path']}")
    print(f"{'='*80}\n")

    # Load and modify config
    config = modify_config_for_quick_test(config_info["path"])

    # Save temporary config
    temp_config_path = f"configs/training/temp_{config_info['name']}_quicktest.json"
    with open(temp_config_path, "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2, ensure_ascii=False)

    print(f"✅ Created temporary config: {temp_config_path}")
    print(
        f"⏱️ Quick test: {QUICK_TEST_TIMESTEPS} timesteps (eval every {QUICK_TEST_EVAL_FREQ} steps)"
    )
    print(f"\n{'─'*80}")
    print("Starting training...\n")

    # Run training
    cmd = [
        sys.executable,
        "-m",
        "ztb.training.ppo_trainer",
        "--config",
        temp_config_path,
    ]

    try:
        result = subprocess.run(cmd, check=True, capture_output=False, text=True)
        print(f"\n✅ Test completed: {config_info['name']}")
        return True
    except subprocess.CalledProcessError as e:
        print(f"\n❌ Test failed: {config_info['name']}")
        print(f"Error: {e}")
        return False
    except KeyboardInterrupt:
        print(f"\n⚠️ Test interrupted by user: {config_info['name']}")
        return False


def main():
    """Run quick tests for all reward configurations."""
    print("=" * 80)
    print("Quick Test: Reward Function Improvements (v378, v379, v380)")
    print("=" * 80)
    print("\nTest settings:")
    print(f"  - Timesteps per test: {QUICK_TEST_TIMESTEPS}")
    print(f"  - Evaluation frequency: {QUICK_TEST_EVAL_FREQ}")
    print(f"  - Number of configs: {len(CONFIGS)}")
    print("\nConfigurations to test:")
    for i, config in enumerate(CONFIGS, 1):
        print(f"  {i}. {config['name']}: {config['description']}")

    print(f"\n{'='*80}\n")

    # Confirmation
    response = input("Run all tests? (y/n): ").strip().lower()
    if response != "y":
        print("Test cancelled.")
        return

    # Run tests
    results = []
    for config in CONFIGS:
        success = run_quick_test(config)
        results.append(
            {
                "name": config["name"],
                "success": success,
            }
        )

        if not success:
            print("\n⚠️ Test failed or was interrupted.")
            response = input("Continue with remaining tests? (y/n): ").strip().lower()
            if response != "y":
                break

    # Summary
    print(f"\n{'='*80}")
    print("Test Summary")
    print(f"{'='*80}\n")

    for result in results:
        status = "✅ SUCCESS" if result["success"] else "❌ FAILED"
        print(f"{status}: {result['name']}")

    successful = sum(1 for r in results if r["success"])
    print(f"\nTotal: {successful}/{len(results)} tests successful")

    print(f"\n{'='*80}")
    print("Next steps:")
    print("1. Check training reports in outputs/training/")
    print("2. Compare HOLD rates across v378, v379, v380")
    print("3. Analyze reward trajectories and action distributions")
    print("4. Select best configuration for full 30k timesteps training")
    print(f"{'='*80}\n")


if __name__ == "__main__":
    main()
