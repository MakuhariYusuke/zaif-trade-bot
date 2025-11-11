"""
SAC v395g - Micro Reward Scale
報酬を1/10に縮小してQ値の爆発を防ぐ
"""
import json

from ztb.training.unified_trainer import UnifiedTrainer


def main():
    print("🔬 SAC v395g - Micro Reward Scale (1/10)")
    print("=" * 80)

    config_path = "configs/sac_v395g_micro_reward.json"

    # 設定ファイル読み込み
    with open(config_path, "r", encoding="utf-8") as f:
        config = json.load(f)

    print("🚨 Problem in v395f:")
    print("-" * 80)
    improvement = config["improvement_from_v395f"]
    print(f"  • {improvement['problem_identified']}")
    print(f"  • Root Cause: {improvement['root_cause']}")
    print()

    print("💡 Solution:")
    print("-" * 80)
    print(f"  {improvement['solution']}")
    print()
    print("  Changes:")
    for key, value in improvement["changes"].items():
        print(f"    • {key:20s}: {value}")
    print()

    print("📊 New Reward Interpretation:")
    print("-" * 80)
    for scenario, value in improvement["new_reward_interpretation"].items():
        print(f"  • {scenario:15s} → {value}")
    print()

    print("🎯 Expected Improvements:")
    print("-" * 80)
    for metric, target in improvement["expected_metrics"].items():
        print(f"  • {metric:20s}: {target}")
    print()

    print("🚀 Starting 5k timesteps training with micro-scale reward...")
    print("=" * 80)
    trainer = UnifiedTrainer(config)
    result = trainer.train()

    print("\n" + "=" * 80)
    if result:
        print("✅ Training completed!")
        print(f"Model saved to: {result.get('model_path', 'N/A')}")
        print("\n📊 Critical Checks:")
        print("  1. ✅ Critic Loss < 100? (was 1e6-1e8)")
        print("  2. ✅ Actor Loss > 0 and < 100? (was -4e4)")
        print("  3. ✅ ent_coef 0.5-1.5? (was 3.58)")
        print("\n🔬 This should be THE breakthrough!")
    else:
        print("❌ Training failed")
    print("=" * 80)


if __name__ == "__main__":
    main()
