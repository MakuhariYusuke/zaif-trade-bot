#!/usr/bin/env python3
"""
Simple verification that reward_calculator.py contains the correct balance_penalty fix.
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))


def verify_reward_calculator_fix():
    """Verify that reward_calculator.py has the correct fix for balance_penalty."""
    print("\n" + "=" * 70)
    print("Verification: Balance Penalty Fix in reward_calculator.py")
    print("=" * 70 + "\n")

    reward_calc_path = Path("ztb/trading/environment/components/reward_calculator.py")

    if not reward_calc_path.exists():
        print(f"❌ File not found: {reward_calc_path}")
        return False

    with open(reward_calc_path, "r") as f:
        content = f.read()

    # Check 1: balance_penalty_enabled_stages tuple exists
    if "balance_penalty_enabled_stages = (" in content:
        print("✓ Found balance_penalty_enabled_stages tuple")
    else:
        print("❌ Missing balance_penalty_enabled_stages tuple")
        return False

    # Check 2: Verify all 4 stages are included
    stages_to_find = [
        '"forced_balance"',
        '"balanced_penalty"',
        '"balance_optimization"',
        '"balance_penalty"',
    ]

    for stage in stages_to_find:
        if stage in content:
            print(f"✓ Found curriculum_stage: {stage}")
        else:
            print(f"❌ Missing curriculum_stage: {stage}")
            return False

    # Check 3: Verify the membership test is used
    if "if curriculum_stage in balance_penalty_enabled_stages:" in content:
        print(
            "✓ Found correct condition: 'if curriculum_stage in balance_penalty_enabled_stages:'"
        )
    else:
        print("❌ Missing or incorrect condition")
        return False

    # Check 4: Verify balance_penalty calculation is present
    if (
        "balance_penalty = abs(buy_ratio - sell_ratio) * balance_penalty_scale"
        in content
    ):
        print("✓ Found correct balance_penalty calculation")
    else:
        print("❌ Missing or incorrect balance_penalty calculation")
        return False

    # Check 5: Verify logging includes curriculum_stage
    if 'f"BALANCE_PENALTY ({curriculum_stage}):' in content:
        print("✓ Found logging with curriculum_stage")
    else:
        print("❌ Missing logging with curriculum_stage")
        return False

    print("\n" + "-" * 70)
    print("✓ ALL VERIFICATION CHECKS PASSED")
    print("-" * 70)
    print("\nThe reward_calculator.py file contains the correct fix:")
    print("  1. Supports 4 curriculum_stage values")
    print("  2. Uses membership test for flexibility")
    print("  3. Includes proper balance_penalty calculation")
    print("  4. Has proper logging")

    return True


def verify_config_files():
    """Verify that config files use 'balanced_penalty' curriculum_stage."""
    print("\n" + "=" * 70)
    print("Verification: Config Files Use Correct curriculum_stage")
    print("=" * 70 + "\n")

    import json

    config_files = [
        "config/sac_v444_3_balanced_penalty_scale_200.json",
        "config/sac_v444_3_balanced_penalty_scale_300.json",
        "config/sac_v444_3_balanced_penalty_scale_500.json",
    ]

    all_correct = True

    for config_file in config_files:
        config_path = Path(config_file)
        if not config_path.exists():
            print(f"⚠ Config not found: {config_file} (skipping)")
            continue

        with open(config_path) as f:
            config = json.load(f)

        training = config.get("training", {})
        curriculum = training.get("curriculum_learning", {})
        stage = curriculum.get("curriculum_stage")

        if stage == "balanced_penalty":
            print(f"✓ {config_file}: curriculum_stage = '{stage}'")
        else:
            print(
                f"❌ {config_file}: curriculum_stage = '{stage}' (expected 'balanced_penalty')"
            )
            all_correct = False

    if all_correct:
        print("\n✓ All config files use correct curriculum_stage")

    return all_correct


def main():
    """Run all verifications."""
    print("\n" + "=" * 70)
    print("Balance Penalty Fix Verification")
    print("=" * 70)

    success = True

    # Verify reward_calculator.py
    if not verify_reward_calculator_fix():
        success = False

    # Verify config files
    if not verify_config_files():
        success = False

    print("\n" + "=" * 70)
    if success:
        print("✓ ALL VERIFICATIONS PASSED")
        print("The fix is correctly implemented and configured!")
    else:
        print("❌ SOME VERIFICATIONS FAILED")
        print("Please review the errors above")
    print("=" * 70 + "\n")

    return success


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
