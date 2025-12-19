"""
SAC v395a Test - 5k timesteps
"""
from pathlib import Path

from ztb.training.unified_trainer import UnifiedTrainer
from ztb.utils.file_utils import safe_json_load


def main():
    print("🧪 SAC v395a Test - 5k timesteps")
    print("=" * 60)

    config_path = "configs/sac_v395a_test_5k.json"

    # 設定ファイル読み込み
    config = safe_json_load(Path(config_path))

    print("🚀 Starting 5k timesteps test run...")
    trainer = UnifiedTrainer(config)
    result = trainer.train()

    print("\n" + "=" * 60)
    if result:
        print("✅ Test run completed!")
        print(f"Model saved to: {result.get('model_path', 'N/A')}")
    else:
        print("❌ Test run failed")
    print("=" * 60)


if __name__ == "__main__":
    main()
