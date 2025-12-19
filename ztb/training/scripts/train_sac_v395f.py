"""
SAC v395f - Revolutionary Simple Reward Function
複雑な報酬関数を捨て、純粋なPnLベースの報酬へ
"""
import json
import time

from ztb.training.unified_trainer import UnifiedTrainer
from ztb.utils.training_utils import display_training_complete


def main():
    print("🔥 SAC v395f - Simple Reward Revolution 🔥")
    print("=" * 80)

    start_time = time.time()

    config_path = "configs/sac_v395f_simple_reward.json"

    # 設定ファイル読み込み
    with open(config_path, "r", encoding="utf-8") as f:
        config = json.load(f)

    print("💡 Reward Function Redesign:")
    print("-" * 80)
    redesign = config["reward_redesign"]
    print(f"  Motivation: {redesign['motivation']}")
    print(f"  Solution: {redesign['solution']}")
    print()

    print("🗑️  Removed Complexity:")
    for change in redesign["key_changes"]:
        print(f"  ❌ {change}")
    print()

    print("📊 Reward Formula:")
    print(f"  {redesign['reward_formula']}")
    print()
    print("  Interpretation:")
    for scenario, value in redesign["reward_interpretation"].items():
        print(f"    • {scenario:15s} → {value}")
    print()

    print("🎯 Expected Improvements:")
    print("-" * 80)
    for metric, target in redesign["expected_improvements"].items():
        print(f"  • {metric:20s}: {target}")
    print()

    print("🚀 Starting 5k timesteps training with simple reward...")
    print("=" * 80)
    trainer = UnifiedTrainer(config)
    result = trainer.train()

    training_time = time.time() - start_time
    final_metrics = {
        "model_path": result.get('model_path', 'N/A') if result else None,
        "training_success": bool(result),
    }
    display_training_complete(final_metrics, training_time)
        print("  2. ent_coef should be 0.5-2.0 (not 3-4)")
        print("  3. Actor Loss should be stable")
        print("  4. Training should be smooth (no explosions)")
        print("\n🔬 Next: Run compare_sac_sessions.py to verify improvements")
    else:
        print("❌ Training failed")
    print("=" * 80)


if __name__ == "__main__":
    main()
