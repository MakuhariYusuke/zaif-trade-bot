#!/usr/bin/env python3
"""
Deep verification: Check if curriculum_stage flows through the entire configuration chain.
"""

import json
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from ztb.config.loader import ConfigLoader
from ztb.utils.v4xx_config_converter import V4XXConfigConverter


def verify_curriculum_stage_flow():
    """Verify curriculum_stage flows through config conversion."""
    print("\n" + "=" * 70)
    print("Deep Verification: curriculum_stage Flow Through Config Chain")
    print("=" * 70 + "\n")

    config_path = "config/sac_v444_3_balanced_penalty_scale_200.json"

    print("Step 1: Load raw configuration")
    with open(config_path) as f:
        raw_config = json.load(f)

    training = raw_config.get("training", {})
    curriculum_learning = training.get("curriculum_learning", {})
    curriculum_stage = curriculum_learning.get("curriculum_stage")

    print(f"  ✓ Raw config curriculum_stage: {curriculum_stage}")
    if curriculum_stage != "balanced_penalty":
        print("  ❌ FAIL: curriculum_stage not in raw config")
        return False

    print("\nStep 2: Test direct conversion (without ConfigLoader)")
    # Test direct conversion first
    unified_config_direct = V4XXConfigConverter.convert_to_unified(raw_config)

    training_env_direct = unified_config_direct.get("training", {}).get(
        "environment", {}
    )
    env_curriculum_stage_direct = training_env_direct.get("curriculum_stage")

    print(
        f"  ✓ Direct conversion training.environment.curriculum_stage: {env_curriculum_stage_direct}"
    )
    if env_curriculum_stage_direct != "balanced_penalty":
        print("  ❌ FAIL: curriculum_stage not mapped in direct conversion")

    print("\nStep 3: Test with ConfigLoader validation")
    config_loader = ConfigLoader()
    validated_config = config_loader.validate_config(raw_config)

    unified_config = V4XXConfigConverter.convert_to_unified(validated_config)

    # Check if curriculum_stage is in training.environment
    training_env = unified_config.get("training", {}).get("environment", {})
    env_curriculum_stage = training_env.get("curriculum_stage")

    print(
        f"  ✓ With ConfigLoader training.environment.curriculum_stage: {env_curriculum_stage}"
    )
    if env_curriculum_stage != "balanced_penalty":
        print("  ❌ FAIL: curriculum_stage not mapped to training.environment")
        return False

    print("\nStep 3: Verify curriculum_learning preservation")
    training_curriculum_learning = unified_config.get("training", {}).get(
        "curriculum_learning", {}
    )
    curriculum_stage_preserved = training_curriculum_learning.get("curriculum_stage")

    print(
        f"  ✓ Unified config training.curriculum_learning.curriculum_stage: {curriculum_stage_preserved}"
    )
    if curriculum_stage_preserved != "balanced_penalty":
        print("  ❌ FAIL: curriculum_learning not preserved")
        return False

    print("\nStep 4: Check other environment config parameters")
    env_keys = list(training_env.keys())
    print(f"  ✓ training.environment keys: {env_keys[:5]}... ({len(env_keys)} total)")

    important_keys = [
        "initial_balance",
        "commission",
        "balance_penalty",
        "curriculum_stage",
        "buy_action_bonus",
    ]

    missing_keys = [k for k in important_keys if k not in training_env]
    if missing_keys:
        print(f"  ⚠ Missing keys in environment config: {missing_keys}")

    present_keys = [k for k in important_keys if k in training_env]
    print(f"  ✓ Present important keys: {present_keys}")

    print("\n" + "-" * 70)
    print("✓ DEEP VERIFICATION PASSED")
    print("-" * 70)
    print("\ncurriculum_stage Flow Summary:")
    print(
        "  1. Raw config: training.curriculum_learning.curriculum_stage = 'balanced_penalty' ✅"
    )
    print(
        "  2. Unified config: training.environment.curriculum_stage = 'balanced_penalty' ✅"
    )
    print("  3. Unified config: training.curriculum_learning preserved ✅")
    print("\nThis means RewardCalculator will receive curriculum_stage correctly!")

    return True


if __name__ == "__main__":
    success = verify_curriculum_stage_flow()
    sys.exit(0 if success else 1)
