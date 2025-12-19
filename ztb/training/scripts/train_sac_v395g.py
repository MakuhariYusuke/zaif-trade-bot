"""
SAC v395g - Micro Reward Scale
報酬を1/10に縮小してQ値の爆発を防ぐ
"""
import time
from pathlib import Path

from ztb.training.unified_trainer import UnifiedTrainer
from ztb.utils.file_utils import safe_json_load
from ztb.utils.training_utils import display_training_complete


def main():
    print("🔬 SAC v395g - Micro Reward Scale (1/10)")
    print("=" * 80)

    start_time = time.time()

    config_path = "configs/sac_v395g_micro_reward.json"

    # 設定ファイル読み込み
    config = safe_json_load(Path(config_path))

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

    training_time = time.time() - start_time
    final_metrics = {
        "model_path": result.get('model_path', 'N/A') if result else None,
        "training_success": bool(result),
    }
    display_training_complete(final_metrics, training_time)
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
