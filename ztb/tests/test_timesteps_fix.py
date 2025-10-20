#!/usr/bin/env python3
"""
Test script to verify total_timesteps override works correctly.
This tests Bug #53 fix.
"""
import json
from pathlib import Path


def test_timesteps_override():
    """Test that --timesteps override works correctly."""

    # Test config with training section
    test_config = {
        "session_id": "test_timesteps",
        "algorithm": "ppo",
        "training": {"total_timesteps": 999999},  # Should be overridden
        "data_path": "ml-dataset-enhanced-balanced.csv",
    }

    # Save test config
    config_path = Path("configs/training/test_timesteps_override.json")
    with open(config_path, "w") as f:
        json.dump(test_config, f, indent=2)

    print("=" * 80)
    print("TESTING: total_timesteps override")
    print("=" * 80)
    print(f"Config file: {config_path}")
    print(f"Config total_timesteps: {test_config['training']['total_timesteps']:,}")
    print("Command-line override: 1000")
    print()

    # Test with UnifiedTrainer
    from ztb.training.unified_trainer import UnifiedTrainer

    # Test 1: Without override
    print("Test 1: WITHOUT override (should use config value)")
    print("-" * 80)
    trainer = UnifiedTrainer(config=test_config, force=True, dry_run=True)
    unified = trainer.build_unified_config()

    result1 = unified.get("total_timesteps")
    ppo_result1 = unified.get("ppo", {}).get("total_timesteps")

    print(f"  unified['total_timesteps']: {result1:,}")
    print(f"  unified['ppo']['total_timesteps']: {ppo_result1:,}")

    if result1 == 999999 and ppo_result1 == 999999:
        print("  ✅ PASS: Config value used correctly")
    else:
        print(f"  ❌ FAIL: Expected 999999, got top={result1}, ppo={ppo_result1}")

    print()

    # Test 2: With override
    print("Test 2: WITH override (should use 1000)")
    print("-" * 80)
    trainer2 = UnifiedTrainer(
        config=test_config.copy(),
        force=True,
        dry_run=True,
        total_timesteps=1000,  # Override to 1000
    )
    unified2 = trainer2.build_unified_config()

    result2 = unified2.get("total_timesteps")
    ppo_result2 = unified2.get("ppo", {}).get("total_timesteps")

    print(f"  unified['total_timesteps']: {result2:,}")
    print(f"  unified['ppo']['total_timesteps']: {ppo_result2:,}")

    if result2 == 1000 and ppo_result2 == 1000:
        print("  ✅ PASS: Override worked correctly")
    else:
        print(f"  ❌ FAIL: Expected 1000, got top={result2}, ppo={ppo_result2}")

    print()

    # Test 3: Cache invalidation
    print("Test 3: CACHE INVALIDATION (second call should still use override)")
    print("-" * 80)
    unified3 = trainer2.build_unified_config()  # Call again

    result3 = unified3.get("total_timesteps")
    ppo_result3 = unified3.get("ppo", {}).get("total_timesteps")

    print(f"  unified['total_timesteps']: {result3:,}")
    print(f"  unified['ppo']['total_timesteps']: {ppo_result3:,}")

    if result3 == 1000 and ppo_result3 == 1000:
        print("  ✅ PASS: Cache invalidation works")
    else:
        print(f"  ❌ FAIL: Expected 1000, got top={result3}, ppo={ppo_result3}")

    print()
    print("=" * 80)

    # Summary
    all_pass = (
        result1 == 999999
        and ppo_result1 == 999999
        and result2 == 1000
        and ppo_result2 == 1000
        and result3 == 1000
        and ppo_result3 == 1000
    )

    if all_pass:
        print("✅ ALL TESTS PASSED - Bug #53 is fixed!")
    else:
        print("❌ SOME TESTS FAILED - Bug #53 still exists")

    print("=" * 80)

    # Cleanup
    config_path.unlink()

    return all_pass


if __name__ == "__main__":
    import sys

    success = test_timesteps_override()
    sys.exit(0 if success else 1)
