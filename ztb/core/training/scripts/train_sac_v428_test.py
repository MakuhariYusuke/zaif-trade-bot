"""
SAC v428 Position Optimized Test - 5k timesteps
"""
import json

from ztb.training.unified_trainer import UnifiedTrainer


def main():
    print("🧪 SAC v428 Position Optimized Test - 5k timesteps")
    print("=" * 60)

    config_path = "configs/sac_v428_position_optimized.json"

    # 設定ファイル読み込み
    with open(config_path, "r", encoding="utf-8") as f:
        config = json.load(f)

    # 短いテスト用にタイムステップを5kに設定
    config["total_timesteps"] = 5000
    config["logging"]["eval_interval"] = 1000
    config["logging"]["save_interval"] = 1000

    print("🚀 Starting 5k timesteps test run...")
    trainer = UnifiedTrainer(config)
    result = trainer.run()

    print("\n" + "=" * 60)
    if result:
        print("✅ Test run completed!")
        print("Model saved successfully")
    else:
        print("❌ Test run failed")
    print("=" * 60)


if __name__ == "__main__":
    main()
