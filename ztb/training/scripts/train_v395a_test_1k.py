"""
SAC v395a Test - 1k timesteps quick iteration
連続行動空間での動作確認用
"""
# ruff: noqa: E402
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from ztb.training.core.config_manager import ConfigManager
from ztb.training.trainers.sac_trainer import SACAlgorithmTrainer


def main():
    print("🧪 SAC v395a Test - 1k timesteps quick iteration")
    print("=" * 60)

    config_path = "configs/sac_v395a_test_1k.json"

    # ConfigManagerを使用して設定を読み込み
    config_manager = ConfigManager(config_path)
    unified_config = config_manager.load_unified_config()

    # SACAlgorithmTrainerを初期化して訓練
    trainer = SACAlgorithmTrainer(config_manager=config_manager)
    trainer.train(unified_config)

    print("\n🚀 Starting 1k timesteps test run...")
    trainer.train()

    print("\n✅ Test run completed!")
    print("=" * 60)


if __name__ == "__main__":
    main()
