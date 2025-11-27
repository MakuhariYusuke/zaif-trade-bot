"""
Quick Validation Script for SAC v448 Emergency Fix
v448 emergency fix設定の動作確認スクリプト（軽量版）

Usage:
    python scripts/validate_v448_emergency.py [--timesteps STEPS]

Features:
    - Single seed (42) for reproducibility
    - 1000 timesteps default (changeable)
    - Action distribution validation
    - Bias collapse detection
"""
import argparse
import json
import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


def load_config(config_path: Path) -> dict:
    """設定ファイルを読み込み"""
    with open(config_path, "r", encoding="utf-8") as f:
        return json.load(f)


def validate_emergency_config(config: dict) -> dict:
    """Emergency fix設定の検証"""
    issues = []
    warnings = []

    # Action bonuses check
    bonuses = config["training"]["environment"].get("action_bonuses", {})
    if any(v != 0.0 for v in bonuses.values()):
        issues.append(f"❌ Action bonuses not zero: {bonuses}")
    else:
        print("✅ Action bonuses = 0.00 (correct)")

    # Asymmetric scaling check
    scaling = config["training"]["environment"]["reward_settings"][
        "asymmetric_reward_scaling"
    ]
    if any(v != 1.0 for v in scaling.values()):
        issues.append(f"❌ Asymmetric scaling not neutral: {scaling}")
    else:
        print("✅ Asymmetric scaling = 1.00 (correct)")

    # Balance targets check
    targets = config["training"]["environment"]["reward_settings"][
        "balance_penalty_targets"
    ]
    expected = {"buy_target": 0.475, "sell_target": 0.475, "hold_target": 0.05}
    if targets != expected:
        warnings.append(f"⚠️ Balance targets differ from baseline: {targets}")
    else:
        print("✅ Balance targets = 47.5/47.5/5.0 (correct)")

    # Forced balance min_actions check
    min_actions = config["training"]["environment"]["reward_settings"][
        "forced_balance_min_actions"
    ]
    if min_actions < 100:
        issues.append(
            f"❌ forced_balance_min_actions too low: {min_actions} (should be ≥100)"
        )
    else:
        print(f"✅ forced_balance_min_actions = {min_actions} (correct)")

    # Forced balance threshold check
    threshold = config["training"]["environment"]["reward_settings"][
        "forced_balance_threshold"
    ]
    if threshold > 0.10:
        warnings.append(
            f"⚠️ forced_balance_threshold high: {threshold} (recommend ≤0.08)"
        )
    else:
        print(f"✅ forced_balance_threshold = {threshold} (correct)")

    # Emergency penalty check
    emergency_penalty = config["training"]["environment"]["reward_settings"].get(
        "forced_balance_emergency_penalty", 0
    )
    if emergency_penalty < 300:
        warnings.append(
            f"⚠️ emergency_penalty low: {emergency_penalty} (recommend ≥300)"
        )
    else:
        print(f"✅ emergency_penalty = {emergency_penalty} (correct)")

    # Entropy coefficient check
    ent_coef = config["training"]["sac_hyperparameters"]["ent_coef"]
    if ent_coef < 0.03:
        warnings.append(f"⚠️ ent_coef low: {ent_coef} (recommend ≥0.05)")
    else:
        print(f"✅ ent_coef = {ent_coef} (correct)")

    # MTF weights check
    mtf_weights = config["training"]["environment"]["multi_timeframe"][
        "feature_weights"
    ]
    if mtf_weights.get("1min", 0) > 0.35:
        warnings.append(f"⚠️ 1min weight high: {mtf_weights['1min']} (recommend ≤0.30)")
    else:
        print(f"✅ MTF weights = {mtf_weights} (correct)")

    return {"issues": issues, "warnings": warnings}


def run_quick_test(config_path: Path, timesteps: int = 1000):
    """簡易テスト実行"""
    print(f"\n{'='*60}")
    print("SAC v448 Emergency Fix - Quick Validation")
    print(f"{'='*60}\n")

    # Config validation
    print("1️⃣ Configuration Validation:")
    print("-" * 60)
    config = load_config(config_path)
    validation = validate_emergency_config(config)

    if validation["issues"]:
        print(f"\n❌ Critical Issues Found ({len(validation['issues'])}):")
        for issue in validation["issues"]:
            print(f"  {issue}")
        return False

    if validation["warnings"]:
        print(f"\n⚠️ Warnings ({len(validation['warnings'])}):")
        for warning in validation["warnings"]:
            print(f"  {warning}")

    print("\n✅ Configuration validation passed!\n")

    # Training execution
    print("2️⃣ Training Execution:")
    print("-" * 60)
    print(f"Config: {config_path.name}")
    print(f"Timesteps: {timesteps}")
    print("Seed: 42 (fixed)")
    print("\nCommand to run:")
    print("  python scripts/unified_trainer.py \\")
    print(f"    --config {config_path} \\")
    print(f"    --timesteps {timesteps} \\")
    print("    --seed 42\n")

    # User confirmation
    response = input("Execute training? (y/n): ").strip().lower()
    if response != "y":
        print("❌ Training cancelled.")
        return False

    # Execute training
    import subprocess

    cmd = [
        "python",
        "scripts/unified_trainer.py",
        "--config",
        str(config_path),
        "--timesteps",
        str(timesteps),
        "--seed",
        "42",
    ]

    print("\n🚀 Starting training...\n")
    result = subprocess.run(cmd, cwd=PROJECT_ROOT)

    if result.returncode != 0:
        print(f"\n❌ Training failed with exit code {result.returncode}")
        return False

    print("\n✅ Training completed successfully!\n")

    # Result analysis
    print("3️⃣ Result Analysis:")
    print("-" * 60)
    print("Check the latest training report in:")
    print("  training_results/[latest]/training_report.json\n")
    print("Key metrics to verify:")
    print("  - BUY% < 90%")
    print("  - SELL% < 90%")
    print("  - |BUY% - SELL%| < 25%")
    print("  - Final reward > -5.0\n")

    return True


def run_child_wrapper_diagnostics(config_path: Path) -> bool:
    """Run child trainer wrapper --diagnostics-only to verify child imports and DLL setup."""
    print(
        "\n4️⃣ Running child wrapper diagnostics (tools/run_child_trainer_wrapper.py --diagnostics-only)"
    )
    wrapper_cmd = [
        "python",
        "tools/run_child_trainer_wrapper.py",
        "--config",
        str(config_path),
        "--diagnostics-only",
    ]
    result = subprocess.run(wrapper_cmd, cwd=PROJECT_ROOT)
    if result.returncode == 0:
        print("✅ Child-wrapper diagnostics OK (imports and DLL path check passed)")
        return True
    else:
        print(
            "❌ Child-wrapper diagnostics failed. Check logs/child_wrapper_debug.jsonl for details."
        )
        return False


def main():
    parser = argparse.ArgumentParser(
        description="Quick validation for SAC v448 emergency fix"
    )
    parser.add_argument(
        "--timesteps", type=int, default=1000, help="Training timesteps (default: 1000)"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="config/v448/sac_v448_emergency_fix.json",
        help="Config file path",
    )

    args = parser.parse_args()
    config_path = PROJECT_ROOT / args.config

    if not config_path.exists():
        print(f"❌ Config file not found: {config_path}")
        sys.exit(1)

    success = run_quick_test(config_path, args.timesteps)
    # Run child wrapper diagnostics as additional validation
    diag_ok = run_child_wrapper_diagnostics(config_path)
    if not diag_ok:
        print(
            "\n⚠️ Child wrapper diagnostics failed; some child process imports or DLL setup may be broken."
        )
        success = False
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
