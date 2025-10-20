"""
SAC v395b (Stable) - 5k timesteps
積極的なパラメータ調整でCritic Loss安定化
"""
import json

from ztb.training.unified_trainer import UnifiedTrainer


def main():
    print("🧪 SAC v395b Test - Stabilized Parameters")
    print("=" * 80)

    config_path = "configs/sac_v395b_stable.json"

    # 設定ファイル読み込み
    with open(config_path, "r", encoding="utf-8") as f:
        config = json.load(f)

    print("📊 Key Changes from v395a:")
    for change in config.get("changes_from_v395a", []):
        print(f"  • {change}")
    print()

    print("🎯 Expected Improvements:")
    print("  • Critic Loss < 1e7 (was 4.34e7)")
    print("  • ent_coef stable in 0.5-2.0 range (was 4.03)")
    print("  • Smoother learning curve")
    print()

    print("🚀 Starting 5k timesteps training...")
    trainer = UnifiedTrainer(config)
    result = trainer.train()

    print("\n" + "=" * 80)
    if result:
        print("✅ Training completed!")
        print(f"Model saved to: {result.get('model_path', 'N/A')}")
        print("\n📊 Next: Run analyze_sac_logs.py to compare with v395a")
    else:
        print("❌ Training failed")
    print("=" * 80)


if __name__ == "__main__":
    main()
